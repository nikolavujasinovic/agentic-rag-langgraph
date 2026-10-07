from agent.config import Settings
from agent.routing import route_after_analyze, route_after_grade
from agent.state import Grade, Plan

SETTINGS = Settings(max_tool_calls=4, max_rewrites=1)
INSUFFICIENT = Grade(sufficient=False, needs_external=False, reason="missing")


def state(**overrides):
    base = {
        "plan": Plan(route="rag", allow_web_fallback=True, standalone_query="q"),
        "grade": INSUFFICIENT,
        "tool_call_count": 1,
        "rewrite_count": 0,
        "last_tool": "rag",
        "web_tried": False,
        "rag_unavailable": False,
    }
    return {**base, **overrides}


def test_analyze_routes_by_plan():
    for route, node in [("rag", "rag_search"), ("web", "web_search"), ("direct", "synthesize")]:
        assert route_after_analyze(state(plan=Plan(route=route, allow_web_fallback=True, standalone_query="q"))) == node


def test_budget_exhausted_always_synthesizes():
    # Even though every other condition says "keep searching", the budget wins.
    assert route_after_grade(state(tool_call_count=4), SETTINGS) == "synthesize"


def test_sufficient_context_synthesizes():
    assert route_after_grade(state(grade=Grade(sufficient=True, needs_external=False, reason="")), SETTINGS) == "synthesize"


def test_skipped_grade_synthesizes():
    assert route_after_grade(state(grade=None), SETTINGS) == "synthesize"


def test_weak_internal_results_trigger_rewrite():
    assert route_after_grade(state(), SETTINGS) == "rewrite_query"


def test_external_gap_goes_straight_to_web():
    grade = Grade(sufficient=False, needs_external=True, reason="third-party info")
    assert route_after_grade(state(grade=grade), SETTINGS) == "web_search"


def test_rewrites_exhausted_escalate_to_web():
    assert route_after_grade(state(rewrite_count=1), SETTINGS) == "web_search"


def test_rag_outage_skips_rewrite_and_uses_web():
    assert route_after_grade(state(rag_unavailable=True), SETTINGS) == "web_search"


def test_web_not_allowed_and_nothing_left_synthesizes():
    plan = Plan(route="rag", allow_web_fallback=False, standalone_query="q")
    assert route_after_grade(state(plan=plan, rewrite_count=1), SETTINGS) == "synthesize"


def test_web_is_never_called_twice():
    assert route_after_grade(state(last_tool="web", web_tried=True), SETTINGS) == "synthesize"
