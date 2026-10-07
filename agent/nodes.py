"""Graph nodes.

Nodes never raise: every external call (LLM or tool) is wrapped, failures are
recorded in ``state["errors"]`` and a deterministic fallback keeps the turn
moving toward an answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Sequence

from langchain_core.messages import AIMessage, AnyMessage

from agent.config import Settings
from agent.llm import (
    REFUSAL_ANSWER,
    LLMRefusalError,
    Reasoner,
    classify_error,
    fallback_answer,
    fallback_grade,
    fallback_plan,
)
from agent.state import AgentState, Document
from agent.tools import CircuitBreaker, Hit, Tool, call_tool

log = logging.getLogger(__name__)


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
        refused = False
        try:
            plan = self.deps.reasoner.plan(query, history)
        except LLMRefusalError as exc:
            errors.append(_llm_failure("planner", exc))
            plan, refused = fallback_plan(query), True
        except Exception as exc:
            errors.append(_llm_failure("planner", exc, "using heuristic plan"))
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
            "refused": refused,
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
            return {
                "grade": fallback_grade(documents),
                "errors": [*state.get("errors", []), _llm_failure("grader", exc, "failing open")],
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
            update["errors"] = [*state.get("errors", []), _llm_failure("rewrite", exc, "keeping query")]
        return update

    # ------------------------------------------------------------------ #
    def synthesize(self, state: AgentState) -> AgentState:
        query, documents = state["query"], state.get("documents", [])
        history = self._history(state["messages"][:-1])
        errors = list(state.get("errors", []))

        citations: list[Document] = []

        if state.get("refused"):
            answer = REFUSAL_ANSWER
        elif not documents and state["plan"].route == "direct":
            try:
                answer = self.deps.reasoner.chat(query, history)
            except LLMRefusalError as exc:
                errors.append(_llm_failure("chat", exc))
                answer = REFUSAL_ANSWER
            except Exception as exc:
                errors.append(_llm_failure("chat", exc))
                answer = "Sorry, I can't respond right now. Please try again shortly."
        elif not documents:
            answer = _no_context_answer(state)
        else:
            try:
                result = self.deps.reasoner.answer(query, documents, history)
            except LLMRefusalError as exc:
                errors.append(_llm_failure("answer", exc))
                result = None
            except Exception as exc:
                errors.append(_llm_failure("answer", exc, "returning source list"))
                result = fallback_answer(documents)
            if result is None:
                answer = REFUSAL_ANSWER
            else:
                # Citations come from the API (search_result_location), so every id is valid by construction.
                by_id = {d.id: d for d in documents}
                answer = result.text
                citations = [by_id[i] for i in result.cited_ids if i in by_id]
            if result is not None and state.get("rag_unavailable") and all(d.source == "web" for d in documents):
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


def _llm_failure(step: str, exc: BaseException, action: str = "") -> str:
    """Log an LLM failure at a severity matching its kind and return the error record."""
    kind, transient = classify_error(exc)
    suffix = f"; {action}" if action else ""
    if kind == "refusal" or transient:
        log.warning("%s %s%s: %s", step, kind, suffix, exc)
    else:  # bad request / auth / unexpected: a bug or misconfiguration, keep the traceback
        log.error("%s %s%s: %s", step, kind, suffix, exc, exc_info=exc)
    return f"{step}: {kind}: {exc}"


def _no_context_answer(state: AgentState) -> str:
    if state.get("rag_unavailable"):
        return (
            "Internal documentation is temporarily unavailable, so I can't answer this reliably right now. "
            "Please try again in a few minutes."
        )
    return "I couldn't find information to answer that in the internal docs or other available sources."
