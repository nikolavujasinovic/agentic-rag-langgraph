"""Graph wiring and the turn runner.

    START -> analyze_query --(rag)----> rag_search --> grade_context --(sufficient | budget)--> synthesize -> END
                           --(web)----> web_search --> grade_context --(weak, rewrites left)-> rewrite_query -> rag_search
                           --(direct)-> synthesize                   --(insufficient)-------> web_search
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes import AgentDeps, AgentNodes
from agent.routing import route_after_analyze, route_after_grade
from agent.state import AgentState, Document, Grade, Plan

log = logging.getLogger(__name__)


def make_checkpointer() -> BaseCheckpointSaver:
    """In-memory checkpointer that explicitly allow-lists our state models for deserialization.

    Production: use ``PostgresSaver`` (langgraph-checkpoint-postgres) with the same serializer.
    """
    allowed = [(model.__module__, model.__name__) for model in (Plan, Grade, Document)]
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=allowed))


def build_graph(deps: AgentDeps, checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Compile the agent graph with a checkpointer for multi-turn conversation persistence."""
    nodes = AgentNodes(deps)
    graph = StateGraph(AgentState)

    graph.add_node("analyze_query", nodes.analyze_query)
    graph.add_node("rag_search", nodes.rag_search)
    graph.add_node("web_search", nodes.web_search)
    graph.add_node("grade_context", nodes.grade_context)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("synthesize", nodes.synthesize)

    graph.add_edge(START, "analyze_query")
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analyze,
        ["rag_search", "web_search", "synthesize"],
    )
    graph.add_edge("rag_search", "grade_context")
    graph.add_edge("web_search", "grade_context")
    graph.add_conditional_edges(
        "grade_context",
        partial(route_after_grade, settings=deps.settings),
        ["rewrite_query", "web_search", "synthesize"],
    )
    graph.add_edge("rewrite_query", "rag_search")
    graph.add_edge("synthesize", END)

    return graph.compile(checkpointer=checkpointer or make_checkpointer())


@dataclass
class TurnResult:
    answer: str
    citations: list[Document] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    tool_calls: int = 0
    plan: Plan | None = None
    errors: list[str] = field(default_factory=list)


def run_turn(graph: CompiledStateGraph, thread_id: str, text: str, recursion_limit: int = 25) -> TurnResult:
    """Run one user turn on a conversation thread. Never raises to the caller."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    path: list[str] = []
    try:
        for update in graph.stream({"messages": [HumanMessage(content=text)]}, config, stream_mode="updates"):
            path.extend(update.keys())
    except GraphRecursionError:
        # Last line of defense: should be unreachable because routing enforces the tool budget.
        log.error("recursion limit hit on thread %s; path=%s", thread_id, path)
        return TurnResult(
            answer="Sorry, I couldn't complete this request within the allowed number of steps.",
            path=path,
            errors=[f"recursion limit ({recursion_limit}) reached"],
        )
    except Exception as exc:
        log.exception("unexpected failure on thread %s", thread_id)
        return TurnResult(
            answer="Sorry, something went wrong while processing your request.",
            path=path,
            errors=[f"{type(exc).__name__}: {exc}"],
        )

    state = graph.get_state(config).values
    return TurnResult(
        answer=state.get("final_answer", ""),
        citations=state.get("citations", []),
        path=path,
        tool_calls=state.get("tool_call_count", 0),
        plan=state.get("plan"),
        errors=state.get("errors", []),
    )
