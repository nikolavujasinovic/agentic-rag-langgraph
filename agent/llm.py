"""LLM-backed reasoning steps.

The graph depends only on the ``Reasoner`` protocol, so the Claude implementation
can be swapped for a fake in tests (or another provider) without touching the
orchestration code.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol, Sequence

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from agent.state import Document, Grade, Plan

log = logging.getLogger(__name__)


class Reasoner(Protocol):
    def plan(self, query: str, history: Sequence[AnyMessage]) -> Plan: ...
    def grade(self, query: str, documents: Sequence[Document]) -> Grade: ...
    def rewrite(self, query: str, documents: Sequence[Document]) -> str: ...
    def answer(self, query: str, documents: Sequence[Document], history: Sequence[AnyMessage]) -> str: ...
    def chat(self, query: str, history: Sequence[AnyMessage]) -> str: ...


PLAN_PROMPT = """You are the planning step of a knowledge assistant for Acme Cloud employees.
Internal documentation covers: refund policy, pricing tiers and plan limits, SSO/SAML setup, \
and the on-call / incident response runbook.

Decide how to answer the latest user message:
- route='rag' when internal docs may contain the answer (default for anything about Acme).
- route='web' only when the question is purely about public or third-party information.
- route='direct' for greetings, thanks, or questions fully answered by the conversation so far.
Set allow_web_fallback=false only when public web results would be misleading \
(e.g. strictly confidential internal matters).
Write standalone_query as a self-contained search query; resolve pronouns and references \
using the conversation history."""

GRADE_PROMPT = """You grade retrieved context for a question-answering system.
sufficient=true only if the documents contain the key facts needed to answer the question \
accurately. needs_external=true if what is missing is public or third-party knowledge that \
internal company documents are unlikely to contain."""

REWRITE_PROMPT = """The search query below did not retrieve documents that answer the question.
Rewrite it to improve semantic retrieval over internal company docs: use different wording, \
synonyms, and more specific terms. Reply with the new query only."""

ANSWER_PROMPT = """You answer questions for Acme Cloud employees using ONLY the sources provided.
Rules:
- Cite every factual claim with its source id in square brackets, e.g. [doc-1] or [web-2].
- Use only ids that appear in the sources. Never invent facts or ids.
- Clearly distinguish internal policy (doc-*) from external guidance (web-*).
- Citation ids are scoped to this turn's sources; ignore any citation markers from earlier turns.
- If the sources only partially answer the question, say what is missing.
- Be concise: answer the question directly, no preamble."""

CHAT_PROMPT = """You are a concise assistant for Acme Cloud employees. Reply to the latest \
message using the conversation history. Do not state company facts that are not in the history."""


class ClaudeReasoner:
    """Reasoner backed by Claude via ``langchain-anthropic``."""

    def __init__(self, model: str, timeout_s: float = 60.0, max_retries: int = 2) -> None:
        self.llm = ChatAnthropic(
            model=model, max_tokens=4096, timeout=timeout_s, max_retries=max_retries
        )
        # Native JSON-schema structured output: the response is constrained to the schema.
        self._planner = self.llm.with_structured_output(Plan, method="json_schema")
        self._grader = self.llm.with_structured_output(Grade, method="json_schema")

    def plan(self, query: str, history: Sequence[AnyMessage]) -> Plan:
        prompt = f"Conversation so far:\n{format_history(history)}\n\nLatest user message:\n{query}"
        return self._planner.invoke([SystemMessage(PLAN_PROMPT), HumanMessage(prompt)])

    def grade(self, query: str, documents: Sequence[Document]) -> Grade:
        prompt = f"Question:\n{query}\n\nRetrieved documents:\n{format_documents(documents)}"
        return self._grader.invoke([SystemMessage(GRADE_PROMPT), HumanMessage(prompt)])

    def rewrite(self, query: str, documents: Sequence[Document]) -> str:
        titles = ", ".join(d.title for d in documents) or "none"
        prompt = f"Query: {query}\nTitles of the unhelpful results: {titles}"
        reply = self.llm.invoke([SystemMessage(REWRITE_PROMPT), HumanMessage(prompt)])
        return reply.text.strip().strip('"')

    def answer(self, query: str, documents: Sequence[Document], history: Sequence[AnyMessage]) -> str:
        prompt = (
            f"Conversation so far:\n{format_history(history)}\n\n"
            f"Sources:\n{format_documents(documents)}\n\nQuestion:\n{query}"
        )
        reply = self.llm.invoke([SystemMessage(ANSWER_PROMPT), HumanMessage(prompt)])
        text = reply.text.strip()
        if reply.response_metadata.get("stop_reason") == "max_tokens":
            log.warning("answer truncated at max_tokens")
            text += "\n\n(Answer truncated due to length.)"
        return text

    def chat(self, query: str, history: Sequence[AnyMessage]) -> str:
        prompt = f"Conversation so far:\n{format_history(history)}\n\nLatest message:\n{query}"
        return self.llm.invoke([SystemMessage(CHAT_PROMPT), HumanMessage(prompt)]).text.strip()


_CITATION_MARKER = re.compile(r"\s?\[(?:doc|web)-\d+\]")


def format_history(history: Sequence[AnyMessage]) -> str:
    """Render prior turns. Citation ids are per-turn, so old markers are stripped to avoid confusion."""
    if not history:
        return "(no previous messages)"
    role = {"human": "User", "ai": "Assistant"}
    return "\n".join(
        f"{role.get(m.type, m.type)}: {_CITATION_MARKER.sub('', m.text)}" for m in history
    )


def format_documents(documents: Sequence[Document]) -> str:
    return "\n\n".join(
        f"[{d.id}] {d.title} ({d.location})\n{d.content}" for d in documents
    )


# --------------------------------------------------------------------------- #
# Deterministic fallbacks used when the LLM call itself fails
# --------------------------------------------------------------------------- #
_SMALL_TALK = {"hi", "hello", "hey", "thanks", "thank you", "thx", "bye"}


def fallback_plan(query: str) -> Plan:
    """Safe default: search internal docs, allow the web as backup."""
    if query.strip().lower().strip("!.") in _SMALL_TALK:
        return Plan(route="direct", allow_web_fallback=False, standalone_query=query)
    return Plan(route="rag", allow_web_fallback=True, standalone_query=query)


def fallback_grade(documents: Sequence[Document]) -> Grade:
    """Fail open: if we cannot grade, answer with what we have instead of looping."""
    return Grade(
        sufficient=bool(documents),
        needs_external=False,
        reason="grader unavailable; proceeding with retrieved context",
    )


def fallback_answer(documents: Sequence[Document]) -> str:
    listing = "\n".join(f"- [{d.id}] {d.title} ({d.location})" for d in documents)
    return (
        "I couldn't generate a full answer right now (language model unavailable). "
        f"These sources look most relevant:\n{listing}"
    )
