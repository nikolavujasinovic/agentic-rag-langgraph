"""Graph state and the structured objects that flow through it."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Output of the planning step: which tool to start with and what to search for."""

    route: Literal["rag", "web", "direct"] = Field(
        description=(
            "'rag' if internal company docs may answer the question (default for anything "
            "company-specific); 'web' if only public/external information is needed; "
            "'direct' for greetings, thanks, or questions fully answered by the conversation."
        )
    )
    allow_web_fallback: bool = Field(
        description="Whether public web results may be used if internal docs are insufficient."
    )
    standalone_query: str = Field(
        description="The latest user message rewritten as a self-contained search query, "
        "resolving references to earlier turns."
    )


class Grade(BaseModel):
    """Output of the grading step: is the retrieved context good enough to answer?"""

    sufficient: bool = Field(description="True only if the documents contain the key facts needed.")
    needs_external: bool = Field(
        description="True if the missing information is public/third-party knowledge that "
        "internal docs are unlikely to contain."
    )
    reason: str = Field(description="One short sentence explaining the verdict.")


class Document(BaseModel):
    """A retrieved piece of evidence that can be cited in the final answer."""

    id: str                                   # e.g. "doc-1", "web-2"
    source: Literal["internal", "web"]
    title: str
    location: str                             # file path or URL
    content: str
    score: float | None = None


class AgentState(TypedDict, total=False):
    # Persisted across turns by the checkpointer (conversation memory).
    messages: Annotated[list[AnyMessage], add_messages]

    # Per-turn working state -- reset by `analyze_query` at the start of every turn.
    query: str
    search_query: str
    plan: Plan
    documents: list[Document]
    grade: Grade | None
    tool_call_count: int
    rewrite_count: int
    last_tool: Literal["rag", "web"] | None
    web_tried: bool
    rag_unavailable: bool
    refused: bool                              # the model declined the request (policy refusal)
    errors: list[str]
    citations: list[Document]
    final_answer: str
