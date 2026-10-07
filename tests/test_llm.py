"""Pure request/response helpers of the Claude reasoner (no network)."""

from types import SimpleNamespace as NS

import anthropic
import httpx2
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.llm import (
    ClaudeReasoner,
    LLMParseError,
    LLMRefusalError,
    LLMTruncatedError,
    check_stop_reason,
    classify_error,
    history_to_messages,
    parse_cited_answer,
    search_result_block,
)
from agent.state import Document, Plan

DOCS = [
    Document(id="doc-1", source="internal", title="Refunds", location="docs/refund.md", content="A.\n\nB."),
    Document(id="web-1", source="web", title="Okta", location="https://okta.example", content="C."),
]


def citation(index):
    return NS(type="search_result_location", search_result_index=index, cited_text="...")


def response(content, stop_reason="end_turn"):
    return NS(content=content, stop_reason=stop_reason, stop_details=None, usage=NS(output_tokens=10))


def test_search_result_block_shape_and_paragraph_split():
    block = search_result_block(DOCS[0])
    assert block["type"] == "search_result"
    assert block["source"] == "docs/refund.md" and block["title"] == "Internal: Refunds"
    assert block["content"] == [{"type": "text", "text": "A."}, {"type": "text", "text": "B."}]
    assert block["citations"] == {"enabled": True}
    assert search_result_block(DOCS[1])["title"] == "Web: Okta"


def test_api_citations_become_inline_markers_and_cited_ids():
    resp = response([
        NS(type="thinking", thinking=""),
        NS(type="text", text="Intro. ", citations=None),
        NS(type="text", text="Refunds in 30 days.", citations=[citation(0), citation(0)]),
        NS(type="text", text=" Okta signs both.", citations=[citation(1), citation(7)]),  # 7: out of range
    ])
    result = parse_cited_answer(resp, DOCS)
    assert result.text == "Intro. Refunds in 30 days. [doc-1] Okta signs both. [web-1]"
    assert result.cited_ids == ["doc-1", "web-1"]
    assert not result.truncated


def test_truncated_answer_is_flagged():
    result = parse_cited_answer(response([NS(type="text", text="Partial", citations=None)], "max_tokens"), DOCS)
    assert result.truncated and "truncated" in result.text


def test_stop_reason_checks():
    with pytest.raises(LLMRefusalError):
        check_stop_reason(NS(stop_reason="refusal", stop_details=NS(category="cyber")))
    with pytest.raises(LLMTruncatedError):
        check_stop_reason(response([], "max_tokens"))
    check_stop_reason(response([], "max_tokens"), allow_truncation=True)
    check_stop_reason(response([]))


def test_history_to_messages_strips_markers_and_starts_with_user():
    history = [AIMessage("orphan"), HumanMessage("q1"), AIMessage("a1 [doc-2]."), HumanMessage("q2")]
    assert history_to_messages(history) == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1."},
        {"role": "user", "content": "q2"},
    ]


def test_classify_error_distinguishes_transient_from_bugs():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    def status_error(cls, code):
        return cls("x", response=httpx2.Response(code, request=request), body=None)

    assert classify_error(status_error(anthropic.RateLimitError, 429)) == ("rate_limited", True)
    assert classify_error(status_error(anthropic.BadRequestError, 400)) == ("bad_request", False)
    assert classify_error(status_error(anthropic.InternalServerError, 500)) == ("server_error", True)
    assert classify_error(anthropic.APIConnectionError(request=request)) == ("unavailable", True)
    assert classify_error(LLMRefusalError("bio")) == ("refusal", False)
    assert classify_error(ValueError()) == ("unexpected", False)


def test_marker_stays_on_cited_line():
    resp = response([NS(type="text", text="- Item one.\n", citations=[citation(0)]), NS(type="text", text="- Next", citations=None)])
    assert parse_cited_answer(resp, DOCS).text == "- Item one. [doc-1]\n- Next"


# --------------------------------------------------------------------------- #
# ClaudeReasoner request wiring, with a fake SDK client
# --------------------------------------------------------------------------- #
class FakeMessages:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(("parse", kwargs))
        return self.reply

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return self.reply


def reasoner_with(reply):
    messages = FakeMessages(reply)
    client = NS(beta=NS(messages=messages))
    return ClaudeReasoner("claude-sonnet-5-5", step_effort="low", answer_effort="medium", client=client), messages


def test_structured_steps_use_parse_with_low_effort_and_fallbacks():
    plan = Plan(route="rag", allow_web_fallback=True, standalone_query="q")
    reasoner, messages = reasoner_with(NS(stop_reason="end_turn", parsed_output=plan))

    assert reasoner.plan("q", []) == plan
    method, kwargs = messages.calls[0]
    assert method == "parse" and kwargs["output_format"] is Plan
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["fallbacks"] == "default" and kwargs["betas"] == ["server-side-fallback-2026-07-01"]


def test_missing_parsed_output_raises_typed_error():
    reasoner, _ = reasoner_with(NS(stop_reason="end_turn", parsed_output=None))
    with pytest.raises(LLMParseError):
        reasoner.grade("q", DOCS)


def test_answer_sends_history_and_search_results_and_allows_truncation():
    reply = response([NS(type="text", text="Partial", citations=[citation(1)])], "max_tokens")
    reasoner, messages = reasoner_with(reply)
    blank = Document(id="doc-2", source="internal", title="Blank", location="x", content="  ")

    result = reasoner.answer("q2", [*DOCS, blank], [HumanMessage("q1"), AIMessage("a1 [doc-1]")])

    method, kwargs = messages.calls[0]
    assert method == "create" and kwargs["output_config"] == {"effort": "medium"}
    assert [m["role"] for m in kwargs["messages"]] == ["user", "assistant", "user"]
    last = kwargs["messages"][-1]["content"]
    assert [b["type"] for b in last] == ["search_result", "search_result", "text"]  # blank doc skipped
    assert result.truncated and result.cited_ids == ["web-1"]
