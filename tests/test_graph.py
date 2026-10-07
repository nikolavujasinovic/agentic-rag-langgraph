"""End-to-end graph tests with fake LLM and tools (no network)."""

from agent import AgentDeps, Settings, build_graph, run_turn
from agent.state import Grade, Plan
from tests.fakes import FakeReasoner, FakeTool

FAST = Settings(max_tool_calls=4, max_rewrites=1, tool_retries=0, tool_backoff_s=0)
NEVER_ENOUGH = lambda docs: Grade(sufficient=False, needs_external=False, reason="never enough")


def make(reasoner=None, rag=None, web=None, settings=FAST):
    reasoner = reasoner or FakeReasoner()
    rag, web = rag or FakeTool("rag"), web or FakeTool("web")
    graph = build_graph(AgentDeps(reasoner=reasoner, rag_tool=rag, web_tool=web, settings=settings))
    return graph, reasoner, rag, web


def test_happy_path_answers_from_internal_docs_with_citations():
    reasoner = FakeReasoner(answer_fn=lambda docs: "Refunds within 30 days [doc-1]. Also [doc-99].")
    graph, _, rag, web = make(reasoner)

    result = run_turn(graph, "t1", "refund policy?")

    assert result.path == ["analyze_query", "rag_search", "grade_context", "synthesize"]
    assert result.tool_calls == 1 and web.calls == []
    assert "[doc-99]" not in result.answer  # hallucinated citation stripped
    assert [d.id for d in result.citations] == ["doc-1"]


def test_corrective_loop_rewrites_then_falls_back_to_web():
    grades = iter([False, False, True])
    reasoner = FakeReasoner(grade_fn=lambda docs: Grade(sufficient=next(grades), needs_external=False, reason=""))
    graph, _, rag, web = make(reasoner)

    result = run_turn(graph, "t1", "question")

    assert result.path == [
        "analyze_query", "rag_search", "grade_context", "rewrite_query",
        "rag_search", "grade_context", "web_search", "grade_context", "synthesize",
    ]
    assert rag.calls[1].endswith("(rephrased)")
    assert {d.id for d in result.citations} == {"doc-1", "doc-2", "web-1"}


def test_loop_is_bounded_by_tool_budget_even_if_grader_never_satisfied():
    # Unlimited rewrites + a grader that is never satisfied = a would-be infinite loop.
    settings = Settings(max_tool_calls=4, max_rewrites=100, tool_retries=0, tool_backoff_s=0)
    graph, _, rag, web = make(FakeReasoner(grade_fn=NEVER_ENOUGH), settings=settings)

    result = run_turn(graph, "t1", "question")

    assert result.tool_calls == 4
    assert len(rag.calls) + len(web.calls) == 4
    assert result.path[-1] == "synthesize" and result.answer


def test_recursion_limit_is_a_graceful_backstop():
    settings = Settings(max_tool_calls=100, max_rewrites=100, tool_retries=0, tool_backoff_s=0)
    graph, *_ = make(FakeReasoner(grade_fn=NEVER_ENOUGH), settings=settings)

    result = run_turn(graph, "t1", "question", recursion_limit=8)

    assert "allowed number of steps" in result.answer
    assert any("recursion limit" in e for e in result.errors)


def test_rag_outage_degrades_to_web():
    graph, _, rag, web = make(rag=FakeTool("rag", error=ConnectionError("db down")))

    result = run_turn(graph, "t1", "question")

    assert "rewrite_query" not in result.path  # rephrasing is pointless when the tool is down
    assert len(web.calls) == 1
    assert any("internal_docs" in e and "db down" in e for e in result.errors)
    assert [d.id for d in result.citations] == ["web-1"]


def test_all_sources_down_gives_honest_answer():
    graph, *_ = make(
        rag=FakeTool("rag", error=ConnectionError("db down")),
        web=FakeTool("web", error=TimeoutError("slow")),
    )
    result = run_turn(graph, "t1", "question")
    assert "temporarily unavailable" in result.answer and result.citations == []


def test_planner_failure_falls_back_to_heuristic_plan():
    graph, *_ = make(FakeReasoner(plan_result=RuntimeError("LLM down")))
    result = run_turn(graph, "t1", "refund policy?")
    assert result.path[1] == "rag_search"
    assert any(e.startswith("planner") for e in result.errors)


def test_direct_route_skips_tools():
    reasoner = FakeReasoner(plan_result=Plan(route="direct", allow_web_fallback=False, standalone_query="hi"))
    graph, _, rag, web = make(reasoner)
    result = run_turn(graph, "t1", "hi")
    assert result.path == ["analyze_query", "synthesize"] and result.answer == "Hello!"
    assert rag.calls == [] and web.calls == []


def test_checkpointer_keeps_history_per_thread_and_resets_turn_state():
    graph, reasoner, *_ = make(FakeReasoner(grade_fn=NEVER_ENOUGH))

    first = run_turn(graph, "thread-a", "first question")
    second = run_turn(graph, "thread-a", "follow-up")
    other = run_turn(graph, "thread-b", "unrelated")

    history_a = reasoner.plan_calls[1][1]
    assert [m.text for m in history_a] == ["first question", first.answer]
    assert reasoner.plan_calls[2][1] == []          # threads are isolated
    assert second.tool_calls == first.tool_calls    # per-turn budget was reset
    assert other.tool_calls == first.tool_calls
