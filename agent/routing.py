"""Conditional-edge routing functions.

These are pure functions of the state, so they can be unit tested without an LLM.
The tool budget is checked FIRST in every decision that could lead to another
tool call -- this is what guarantees the corrective loop always terminates.
"""

from __future__ import annotations

from typing import Literal

from agent.config import Settings
from agent.state import AgentState

AfterAnalyze = Literal["rag_search", "web_search", "synthesize"]
AfterGrade = Literal["rewrite_query", "web_search", "synthesize"]


def route_after_analyze(state: AgentState) -> AfterAnalyze:
    route = state["plan"].route
    if route == "rag":
        return "rag_search"
    if route == "web":
        return "web_search"
    return "synthesize"


def route_after_grade(state: AgentState, settings: Settings) -> AfterGrade:
    # 1. Circuit breaker on the loop itself: budget spent -> answer with what we have.
    if state.get("tool_call_count", 0) >= settings.max_tool_calls:
        return "synthesize"

    grade = state.get("grade")
    if grade is None or grade.sufficient:
        return "synthesize"

    can_web = state["plan"].allow_web_fallback and not state.get("web_tried", False)
    can_rewrite = (
        state.get("last_tool") == "rag"
        and not state.get("rag_unavailable", False)
        and state.get("rewrite_count", 0) < settings.max_rewrites
    )

    # 2. Missing info is external -> go straight to the web instead of rephrasing.
    if grade.needs_external and can_web:
        return "web_search"
    # 3. Internal docs probably have it but the query missed -> rephrase and retry.
    if can_rewrite:
        return "rewrite_query"
    # 4. Internal options exhausted (or RAG is down) -> escalate to the web.
    if can_web:
        return "web_search"
    # 5. Nothing left to try -> answer honestly with partial/no context.
    return "synthesize"
