"""Deterministic test doubles -- the test suite needs no API keys or network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from langchain_core.messages import AnyMessage

from agent.state import Document, Grade, Plan
from agent.tools import Hit


@dataclass
class FakeReasoner:
    plan_result: Plan | Exception = field(
        default_factory=lambda: Plan(route="rag", allow_web_fallback=True, standalone_query="q")
    )
    grade_fn: Callable[[Sequence[Document]], Grade] = lambda docs: Grade(
        sufficient=True, needs_external=False, reason="ok"
    )
    answer_fn: Callable[[Sequence[Document]], str] = lambda docs: " ".join(f"[{d.id}]" for d in docs)
    plan_calls: list[tuple[str, list[AnyMessage]]] = field(default_factory=list)

    def plan(self, query: str, history: Sequence[AnyMessage]) -> Plan:
        self.plan_calls.append((query, list(history)))
        if isinstance(self.plan_result, Exception):
            raise self.plan_result
        return self.plan_result

    def grade(self, query: str, documents: Sequence[Document]) -> Grade:
        return self.grade_fn(documents)

    def rewrite(self, query: str, documents: Sequence[Document]) -> str:
        return f"{query} (rephrased)"

    def answer(self, query: str, documents: Sequence[Document], history: Sequence[AnyMessage]) -> str:
        return self.answer_fn(documents)

    def chat(self, query: str, history: Sequence[AnyMessage]) -> str:
        return "Hello!"


class FakeTool:
    """Returns one unique hit per call, or raises if ``error`` is set."""

    def __init__(self, name: str, error: Exception | None = None) -> None:
        self.name = name
        self.error = error
        self.calls: list[str] = []

    def __call__(self, query: str) -> list[Hit]:
        self.calls.append(query)
        if self.error:
            raise self.error
        n = len(self.calls)
        return [Hit(title=f"{self.name} {n}", location=f"{self.name}/{n}", content=f"fact {n}", score=0.9)]
