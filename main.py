"""CLI entry point.

    uv run python main.py                 # interactive chat (multi-turn, one thread)
    uv run python main.py --demo          # scripted scenarios showing routing, memory and failures
    uv run python main.py --thread abc    # resume/continue a named conversation thread
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid

from langchain_openai import OpenAIEmbeddings

from agent import AgentDeps, Settings, TurnResult, build_graph, load_environment, run_turn
from agent.llm import ClaudeReasoner
from agent.tools import InternalDocsRetriever, MockWebSearch, Tool


def make_deps(settings: Settings, *, rag_tool: Tool | None = None, web_fail: bool = False) -> AgentDeps:
    retriever = InternalDocsRetriever(
        settings.docs_dir, OpenAIEmbeddings(model=settings.embedding_model), k=settings.rag_top_k
    )
    return AgentDeps(
        reasoner=ClaudeReasoner(settings.llm_model, timeout_s=settings.llm_timeout_s),
        rag_tool=rag_tool or retriever,
        web_tool=MockWebSearch(fail=web_fail),
        settings=settings,
    )


def print_turn(question: str, result: TurnResult, max_tool_calls: int) -> None:
    print(f"\nUser: {question}")
    if result.plan:
        p = result.plan
        print(f"  plan:       route={p.route}, web_fallback={p.allow_web_fallback}, query={p.standalone_query!r}")
    print(f"  path:       {' -> '.join(result.path)}")
    print(f"  tool calls: {result.tool_calls}/{max_tool_calls}")
    for err in result.errors:
        print(f"  error:      {err}")
    print(f"\nAssistant: {result.answer}")
    if result.citations:
        print("\nSources:")
        for doc in result.citations:
            print(f"  [{doc.id}] {doc.title} - {doc.location}")
    print("-" * 80)


def run_demo(settings: Settings) -> None:
    graph = build_graph(make_deps(settings))

    print("\n=== 1. Internal question (RAG) ===")
    q = "What is our refund policy for annual plans?"
    print_turn(q, run_turn(graph, "demo-memory", q, settings.recursion_limit), settings.max_tool_calls)

    print("\n=== 2. Follow-up on the same thread (checkpointed memory) ===")
    q = "Does that also apply to Enterprise customers?"
    print_turn(q, run_turn(graph, "demo-memory", q, settings.recursion_limit), settings.max_tool_calls)

    print("\n=== 3. Internal + external question (corrective fallback to web) ===")
    q = "How does our SSO setup compare to Okta's recommended SAML best practices?"
    print_turn(q, run_turn(graph, "demo-mixed", q, settings.recursion_limit), settings.max_tool_calls)

    print("\n=== 4. Internal docs outage (retries, circuit breaker, graceful degradation) ===")

    def broken_vector_db(query: str):
        raise ConnectionError("vector database unreachable (simulated)")

    outage = Settings(breaker_failure_threshold=2, tool_backoff_s=0.2)
    outage_graph = build_graph(make_deps(outage, rag_tool=broken_vector_db))
    for q in (
        "How does our SEV1 incident process compare to PagerDuty's recommendations?",
        "How many users does the Pro tier include?",
        "Who has to approve a refund above $5,000?",  # breaker opened after 2 failures: RAG skipped instantly
    ):
        print_turn(q, run_turn(outage_graph, "demo-outage", q, outage.recursion_limit), outage.max_tool_calls)


def run_chat(settings: Settings, thread_id: str) -> None:
    graph = build_graph(make_deps(settings))
    print(f"Thread: {thread_id}  (Ctrl+D or 'exit' to quit)")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if q.lower() in {"exit", "quit"}:
            return
        if q:
            print_turn(q, run_turn(graph, thread_id, q, settings.recursion_limit), settings.max_tool_calls)


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic RAG with LangGraph")
    parser.add_argument("--demo", action="store_true", help="run scripted demo scenarios")
    parser.add_argument("--thread", default=None, help="conversation thread id")
    parser.add_argument("-v", "--verbose", action="store_true", help="show info logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    load_environment()
    missing = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not os.getenv(k)]
    if missing:
        print(f"Missing API keys: {', '.join(missing)}. Copy .env.example to .env and fill it in.")
        return 1
    settings = Settings()
    if args.demo:
        run_demo(settings)
    else:
        run_chat(settings, args.thread or f"cli-{uuid.uuid4().hex[:8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
