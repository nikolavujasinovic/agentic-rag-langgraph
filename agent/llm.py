"""LLM-backed reasoning steps, implemented with the official Anthropic SDK.

The graph depends only on the ``Reasoner`` protocol, so the Claude implementation
can be swapped for a fake in tests (or another provider) without touching the
orchestration code.

Claude API features used (see https://platform.claude.com/docs):
- Structured outputs via ``messages.parse(output_format=...)`` for plan and grade.
- Search result content blocks with native citations for the final answer.
- ``output_config.effort`` per step (``low`` for routing steps).
- Server-side refusal fallbacks (``fallbacks="default"``) and explicit stop-reason checks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import anthropic
from langchain_core.messages import AnyMessage

from agent.state import Document, Grade, Plan

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class AnswerResult:
    """Answer text with inline ``[doc-x]`` markers derived from API citations."""

    text: str
    cited_ids: list[str] = field(default_factory=list)
    truncated: bool = False


class Reasoner(Protocol):
    def plan(self, query: str, history: Sequence[AnyMessage]) -> Plan: ...
    def grade(self, query: str, documents: Sequence[Document]) -> Grade: ...
    def rewrite(self, query: str, documents: Sequence[Document]) -> str: ...
    def answer(self, query: str, documents: Sequence[Document], history: Sequence[AnyMessage]) -> AnswerResult: ...
    def chat(self, query: str, history: Sequence[AnyMessage]) -> str: ...


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class LLMRefusalError(Exception):
    """Claude (and any server-side fallback model) declined the request."""

    def __init__(self, category: str | None) -> None:
        super().__init__(f"request declined (category={category})")
        self.category = category


class LLMTruncatedError(Exception):
    """Output hit max_tokens; structured output may be incomplete or invalid."""


class LLMParseError(Exception):
    """A structured-output call returned no parsed object."""


def classify_error(exc: BaseException) -> tuple[str, bool]:
    """Map an exception to ``(kind, transient)``, most specific first.

    Transient failures (rate limits, overload, network) are expected in production.
    Non-transient ones (bad request, auth, missing model) indicate a bug or misconfiguration.
    """
    if isinstance(exc, LLMRefusalError):
        return "refusal", False
    if isinstance(exc, LLMTruncatedError):
        return "truncated", False
    if isinstance(exc, LLMParseError):
        return "parse_error", False
    if isinstance(exc, anthropic.RateLimitError):
        return "rate_limited", True
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return "auth", False
    if isinstance(exc, (anthropic.BadRequestError, anthropic.NotFoundError)):
        return "bad_request", False
    if isinstance(exc, anthropic.APIStatusError):
        return ("server_error", True) if exc.status_code >= 500 else ("api_error", False)
    if isinstance(exc, anthropic.APIConnectionError):  # includes APITimeoutError
        return "unavailable", True
    return "unexpected", False


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
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

ANSWER_PROMPT = """You answer questions for Acme Cloud employees using only the search results \
provided in the latest message.
- Results titled "Internal:" are Acme policy; results titled "Web:" are external guidance. \
Keep the two clearly distinguished.
- If the results only partially answer the question, say what is missing. Never invent facts.
- Be concise: answer the question directly, no preamble."""

CHAT_PROMPT = """You are a concise assistant for Acme Cloud employees, replying to small talk or \
to a question already answered earlier in the conversation. Your earlier answers were grounded in \
cited company documents that are not repeated here; treat them as reliable and do not retract them. \
Reply briefly. Do not introduce new company facts that are not in the conversation."""


# --------------------------------------------------------------------------- #
# Claude implementation
# --------------------------------------------------------------------------- #
class ClaudeReasoner:
    """Reasoner backed by the official ``anthropic`` SDK."""

    def __init__(
        self,
        model: str,
        *,
        step_effort: str = "low",
        answer_effort: str = "medium",
        max_tokens: int = 16000,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.step_effort = step_effort
        self.answer_effort = answer_effort
        self.max_tokens = max_tokens
        # The SDK retries 408/409/429/5xx and connection errors with exponential backoff.
        self.client = client or anthropic.Anthropic(timeout=timeout_s, max_retries=max_retries)

    # -- public steps ------------------------------------------------------ #
    def plan(self, query: str, history: Sequence[AnyMessage]) -> Plan:
        transcript = format_transcript(history)
        prompt = f"Conversation so far:\n{transcript}\n\nLatest user message:\n{query}"
        return self._call(PLAN_PROMPT, [user(prompt)], self.step_effort, output_format=Plan).parsed_output

    def grade(self, query: str, documents: Sequence[Document]) -> Grade:
        prompt = f"Question:\n{query}\n\nRetrieved documents:\n{format_documents(documents)}"
        return self._call(GRADE_PROMPT, [user(prompt)], self.step_effort, output_format=Grade).parsed_output

    def rewrite(self, query: str, documents: Sequence[Document]) -> str:
        titles = ", ".join(d.title for d in documents) or "none"
        prompt = f"Query: {query}\nTitles of the unhelpful results: {titles}"
        response = self._call(REWRITE_PROMPT, [user(prompt)], self.step_effort)
        return response_text(response).strip().strip('"')

    def answer(self, query: str, documents: Sequence[Document], history: Sequence[AnyMessage]) -> AnswerResult:
        # Empty text blocks are rejected by the API, so blank documents are skipped.
        content = [search_result_block(d) for d in documents if d.content.strip()]
        content.append({"type": "text", "text": query.strip() or "(empty question)"})
        messages = history_to_messages(history) + [{"role": "user", "content": content}]
        response = self._call(ANSWER_PROMPT, messages, self.answer_effort, allow_truncation=True)
        return parse_cited_answer(response, documents)

    def chat(self, query: str, history: Sequence[AnyMessage]) -> str:
        messages = history_to_messages(history) + [user(query.strip() or "(empty message)")]
        return response_text(self._call(CHAT_PROMPT, messages, self.step_effort)).strip()

    # -- transport --------------------------------------------------------- #
    def _call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        effort: str,
        *,
        output_format: type | None = None,
        allow_truncation: bool = False,
    ) -> Any:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": effort},
            # On a policy decline, the API re-runs the request on Anthropic's recommended
            # fallback model for that refusal category, inside the same call.
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if output_format is not None:
            response = self.client.beta.messages.parse(output_format=output_format, **kwargs)
        else:
            response = self.client.beta.messages.create(**kwargs)
        check_stop_reason(response, allow_truncation=allow_truncation)
        if output_format is not None and response.parsed_output is None:
            raise LLMParseError(f"no parsed {output_format.__name__} in response")
        return response


# --------------------------------------------------------------------------- #
# Request/response helpers (pure functions, unit tested)
# --------------------------------------------------------------------------- #
def user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def check_stop_reason(response: Any, *, allow_truncation: bool = False) -> None:
    """Branch on stop_reason before reading content (refusal / max_tokens)."""
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise LLMRefusalError(getattr(details, "category", None))
    if response.stop_reason == "max_tokens" and not allow_truncation:
        raise LLMTruncatedError(f"output hit max_tokens ({response.usage.output_tokens} tokens)")


def response_text(response: Any) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def search_result_block(doc: Document) -> dict[str, Any]:
    """Wrap a retrieved document as a citable ``search_result`` block.

    Content is split into one text block per paragraph: Claude cites whole blocks,
    so smaller blocks give more precise citations.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.content) if p.strip()] or [doc.content]
    label = "Internal" if doc.source == "internal" else "Web"
    return {
        "type": "search_result",
        "source": doc.location,
        "title": f"{label}: {doc.title}",
        "content": [{"type": "text", "text": p} for p in paragraphs],
        "citations": {"enabled": True},  # all-or-nothing across the request
    }


def parse_cited_answer(response: Any, documents: Sequence[Document]) -> AnswerResult:
    """Build answer text with ``[doc-x]`` markers from ``search_result_location`` citations.

    ``search_result_index`` counts search_result blocks in request order. Only the current
    turn's message carries them, so the index maps directly onto ``documents``.
    """
    parts: list[str] = []
    cited: list[str] = []
    for block in response.content:
        if block.type != "text":
            continue  # thinking / fallback blocks
        markers = []
        for citation in getattr(block, "citations", None) or []:
            if citation.type != "search_result_location":
                continue
            idx = citation.search_result_index
            if 0 <= idx < len(documents):
                doc_id = documents[idx].id
                if doc_id not in markers:
                    markers.append(doc_id)
                if doc_id not in cited:
                    cited.append(doc_id)
        # Place markers before trailing whitespace so they stay on the cited line.
        body = block.text.rstrip()
        parts.append(body + "".join(f" [{m}]" for m in markers) + block.text[len(body):])
    text = "".join(parts).strip()
    truncated = response.stop_reason == "max_tokens"
    if truncated:
        log.warning("answer truncated at max_tokens")
        text += "\n\n(Answer truncated due to length.)"
    return AnswerResult(text=text, cited_ids=cited, truncated=truncated)


_CITATION_MARKER = re.compile(r"\s?\[(?:doc|web)-\d+\]")


def history_to_messages(history: Sequence[AnyMessage]) -> list[dict[str, Any]]:
    """Convert prior turns to Messages API format (must start with a user turn).

    Citation markers are stripped: ids are scoped to the turn that produced them,
    and that turn's search results are not resent.
    """
    messages = []
    for m in history:
        role = {"human": "user", "ai": "assistant"}.get(m.type)
        if role and m.text.strip():
            messages.append({"role": role, "content": _CITATION_MARKER.sub("", m.text)})
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return messages


def format_transcript(history: Sequence[AnyMessage]) -> str:
    """Plain-text transcript, used where the history is input data (planning), not a dialogue."""
    if not history:
        return "(no previous messages)"
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history_to_messages(history)
    ) or "(no previous messages)"


def format_documents(documents: Sequence[Document]) -> str:
    return "\n\n".join(f"[{d.id}] {d.title} ({d.location})\n{d.content}" for d in documents)


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


def fallback_answer(documents: Sequence[Document]) -> AnswerResult:
    listing = "\n".join(f"- {d.title} ({d.location}) [{d.id}]" for d in documents)
    return AnswerResult(
        text=(
            "I couldn't generate a full answer right now (language model unavailable). "
            f"These sources look most relevant:\n{listing}"
        ),
        cited_ids=[d.id for d in documents],
    )


REFUSAL_ANSWER = "I can't help with that request."
