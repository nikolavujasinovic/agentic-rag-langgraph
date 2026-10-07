"""Graph nodes.

Nodes never raise: every external call (LLM or tool) is wrapped, failures are
recorded in ``state["errors"]`` and a deterministic fallback keeps the turn
moving toward an answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal, Sequence

from langchain_core.messages import AIMessage, AnyMessage

from agent.config import Settings
from agent.llm import Reasoner, fallback_answer, fallback_grade, fallback_plan
from agent.state import AgentState, Document
from agent.tools import CircuitBreaker, Hit, Tool, call_tool

log = logging.getLogger(__name__)

_CITATION = re.compile(r"\[((?:doc|web)-\d+)\]")


@dataclass
class AgentDeps:
    """Everything the graph needs from the outside world (dependency injection)."""

    reasoner: Reasoner
    rag_tool: Tool
    web_tool: Tool
    settings: Settings = field(default_factory=Settings)
    rag_breaker: CircuitBreaker | None = None
    web_breaker: CircuitBreaker | None = None

    def __post_init__(self) -> None:
        s = self.settings
        self.rag_breaker = self.rag_breaker or CircuitBreaker(
            "internal_docs", s.breaker_failure_threshold, s.breaker_recovery_s
        )
        self.web_breaker = self.web_breaker or CircuitBreaker(
            "web_search", s.breaker_failure_threshold, s.breaker_recovery_s
        )


class AgentNodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        self.settings = deps.settings

    # ------------------------------------------------------------------ #
    def analyze_query(self, state: AgentState) -> AgentState:
        """Plan the turn and reset all per-turn state (only `messages` persists)."""
        messages = state["messages"]
        query = messages[-1].text
        history = self._history(messages[:-1])
        errors: list[str] = []
        try:
            plan = self.deps.reasoner.plan(query, history)
        except Exception as exc:
            log.warning("planner failed, using heuristic plan: %s", exc)
            errors.append(f"planner: {type(exc).__name__}: {exc}")
            plan = fallback_plan(query)
        if not plan.standalone_query.strip():
            plan = plan.model_copy(update={"standalone_query": query})

        return {
            "query": query,
            "search_query": plan.standalone_query,
            "plan": plan,
            "documents": [],
            "grade": None,
            "tool_call_count": 0,
            "rewrite_count": 0,
            "last_tool": None,
            "web_tried": False,
            "rag_unavailable": False,
            "errors": errors,
            "citations": [],
            "final_answer": "",
        }

    # ------------------------------------------------------------------ #
    def rag_search(self, state: AgentState) -> AgentState:
        return self._search(state, "rag")

    def web_search(self, state: AgentState) -> AgentState:
        return self._search(state, "web")

    def _search(self, state: AgentState, kind: Literal["rag", "web"]) -> AgentState:
        d, s = self.deps, self.settings
        tool, breaker = (d.rag_tool, d.rag_breaker) if kind == "rag" else (d.web_tool, d.web_breaker)
        result = call_tool(
            "internal_docs" if kind == "rag" else "web_search",
            tool,
            state["search_query"],
            breaker,
            timeout_s=s.tool_timeout_s,
            retries=s.tool_retries,
            backoff_s=s.tool_backoff_s,
        )
        update: AgentState = {
            # Counted on every visit (even skipped/failed calls) so the budget is a hard bound.
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "last_tool": kind,
            "documents": _merge_documents(state.get("documents", []), result.hits, kind),
        }
        if kind == "web":
            update["web_tried"] = True
        if result.error:
            update["errors"] = [*state.get("errors", []), result.error]
            if kind == "rag":
                update["rag_unavailable"] = True
        return update

    # ------------------------------------------------------------------ #
    def grade_context(self, state: AgentState) -> AgentState:
        documents = state.get("documents", [])
        if state.get("tool_call_count", 0) >= self.settings.max_tool_calls:
            return {"grade": None}  # budget spent; routing will synthesize -- skip the LLM call
        if not documents:
            return {"grade": fallback_grade(documents).model_copy(update={"reason": "no documents retrieved"})}
        try:
            grade = self.deps.reasoner.grade(state["query"], documents)
        except Exception as exc:
            log.warning("grader failed, failing open: %s", exc)
            return {
                "grade": fallback_grade(documents),
                "errors": [*state.get("errors", []), f"grader: {type(exc).__name__}: {exc}"],
            }
        return {"grade": grade}

    # ------------------------------------------------------------------ #
    def rewrite_query(self, state: AgentState) -> AgentState:
        update: AgentState = {"rewrite_count": state.get("rewrite_count", 0) + 1}
        try:
            new_query = self.deps.reasoner.rewrite(state["search_query"], state.get("documents", []))
            if new_query:
                update["search_query"] = new_query
        except Exception as exc:
            log.warning("rewrite failed, keeping query: %s", exc)
            update["errors"] = [*state.get("errors", []), f"rewrite: {type(exc).__name__}: {exc}"]
        return update

    # ------------------------------------------------------------------ #
    def synthesize(self, state: AgentState) -> AgentState:
        query, documents = state["query"], state.get("documents", [])
        history = self._history(state["messages"][:-1])
        errors = list(state.get("errors", []))

        if not documents and state["plan"].route == "direct":
            try:
                answer = self.deps.reasoner.chat(query, history)
            except Exception as exc:
                errors.append(f"chat: {type(exc).__name__}: {exc}")
                answer = "Sorry, I can't respond right now. Please try again shortly."
            citations: list[Document] = []
        elif not documents:
            answer = _no_context_answer(state)
            citations = []
        else:
            try:
                raw = self.deps.reasoner.answer(query, documents, history)
            except Exception as exc:
                log.warning("answer generation failed: %s", exc)
                errors.append(f"answer: {type(exc).__name__}: {exc}")
                raw = fallback_answer(documents)
            answer, citations = _validate_citations(raw, documents)
            if state.get("rag_unavailable") and all(d.source == "web" for d in documents):
                answer += (
                    "\n\nNote: internal documentation was unavailable for this request, "
                    "so this answer relies on external sources only."
                )

        return {
            "final_answer": answer,
            "citations": citations,
            "errors": errors,
            "messages": [AIMessage(content=answer)],
        }

    # ------------------------------------------------------------------ #
    def _history(self, messages: Sequence[AnyMessage]) -> list[AnyMessage]:
        window = self.settings.history_window
        return [m for m in messages if m.type in ("human", "ai")][-window:]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _merge_documents(existing: list[Document], hits: list[Hit], kind: Literal["rag", "web"]) -> list[Document]:
    """Append new hits with stable per-turn ids (doc-1, doc-2, web-1...), skipping duplicates."""
    prefix, source = ("doc", "internal") if kind == "rag" else ("web", "web")
    seen = {(d.location, d.content) for d in existing}
    next_n = sum(d.id.startswith(prefix) for d in existing) + 1
    merged = list(existing)
    for hit in hits:
        key = (hit["location"], hit["content"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(Document(id=f"{prefix}-{next_n}", source=source, **hit))
        next_n += 1
    return merged


def _validate_citations(answer: str, documents: list[Document]) -> tuple[str, list[Document]]:
    """Strip citation ids the model invented; return the documents actually cited."""
    by_id = {d.id: d for d in documents}
    cited: dict[str, Document] = {}

    def keep_or_drop(match: re.Match) -> str:
        doc_id = match.group(1)
        if doc_id in by_id:
            cited.setdefault(doc_id, by_id[doc_id])
            return match.group(0)
        log.warning("dropping hallucinated citation %s", doc_id)
        return ""

    cleaned = _CITATION.sub(keep_or_drop, answer)
    return re.sub(r"[ \t]+([.,;:])", r"\1", cleaned), list(cited.values())


def _no_context_answer(state: AgentState) -> str:
    if state.get("rag_unavailable"):
        return (
            "Internal documentation is temporarily unavailable, so I can't answer this reliably right now. "
            "Please try again in a few minutes."
        )
    return "I couldn't find information to answer that in the internal docs or other available sources."
