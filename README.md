# Agentic RAG with LangGraph

An agent that answers employee questions about internal company documentation. It:

1. **Analyzes** the query and plans which tools are needed.
2. **Searches internal docs** with RAG: OpenAI embeddings over an in-memory vector store.
3. **Optionally searches the web.** Web search is mocked, since the focus here is orchestration.
4. **Grades** the retrieved context. If the context is weak, it **corrects course** by rewriting the query or escalating to the web.
5. **Synthesizes** a final answer with validated citations.

It keeps **multi-turn conversation state** through LangGraph checkpointing. Every loop is **bounded** by a tool budget, a per-tool circuit breaker and a recursion-limit backstop.

Claude handles planning, grading, rewriting and answer generation. OpenAI `text-embedding-3-small` handles retrieval embeddings.

## Architecture

```
START ─► analyze_query ──(rag)────► rag_search ──► grade_context ──(sufficient │ budget spent)──► synthesize ─► END
              │                        ▲                 │
              │                        │                 ├──(weak results, rewrites left)──► rewrite_query ─┐
              │                        └─────────────────┼──────────────────────────────────────────────────┘
              ├──(web)────► web_search ◄─────────────────┘ (insufficient / external info / RAG down)
              │                │
              │                └──► grade_context
              └──(direct)─────────────────────────────────────────────────────────────────► synthesize
```

| Node | Responsibility |
|---|---|
| `analyze_query` | Uses Claude with structured output (`Plan`) to choose the route: `rag`, `web` or `direct`. It also decides whether web fallback is allowed and rewrites follow-up questions into standalone queries using the conversation history. It resets all per-turn state. |
| `rag_search` / `web_search` | Call the tool through `call_tool`, which applies the timeout, retries, the circuit breaker and error capture. Results are merged with stable per-turn ids (`doc-1`, `web-1`). |
| `grade_context` | Claude with structured output (`Grade`) decides whether the context is sufficient. If not, it says whether the missing information is external. |
| `rewrite_query` | Rephrases the search query for a second retrieval attempt (corrective RAG). |
| `synthesize` | Writes a grounded answer with `[doc-x]` / `[web-x]` citations. It strips citation ids that don't exist, and says honestly when no context was found. |

**Why a planner plus a bounded corrective loop, instead of a free-form ReAct agent?**
- In a ReAct loop the LLM decides at every step whether to call another tool. That is flexible, but it is hard to predict, hard to test, and it can loop.
- Here the LLM makes the *judgments*: what to search, whether the context is good enough, and how to rephrase. The *control flow* lives in explicit, pure routing functions (`agent/routing.py`) that can be unit tested and that always terminate.

## Preventing infinite agent loops

The corrective loop (`grade → rewrite → search → grade …`) is the only cycle in the graph. It is bounded by several independent layers:

| # | Guard | Where | What it stops |
|---|---|---|---|
| 1 | **Tool budget** (`max_tool_calls=4` per turn) | `route_after_grade` checks it *first* | Any number of search/grade cycles. Once the budget is spent the agent answers with what it has. Every search-node visit counts, including skipped or failed calls, so this is a hard bound. |
| 2 | **Rewrite cap** (`max_rewrites=1`) | `route_after_grade` | Endless rephrasing of the same query |
| 3 | **Web search at most once** (`web_tried`) | `route_after_grade` | Repeated web calls |
| 4 | **Per-tool circuit breaker** | `agent/tools.py` | Hammering a failing dependency. After N consecutive failures the tool is skipped instantly. The breaker goes half-open after a cool-down. |
| 5 | **`recursion_limit`** | `run_turn` | A bug in routing. `GraphRecursionError` is caught and turned into a graceful reply. |
| + | **Timeouts and bounded retries** | `call_tool` | A single hung or flaky tool call |

The grader also **fails open**: if the grading LLM call errors, the agent answers with the context it has instead of searching again.

`tests/test_graph.py::test_loop_is_bounded_by_tool_budget_even_if_grader_never_satisfied` proves this. It sets up a grader that is never satisfied and allows unlimited rewrites, and the turn still ends after exactly 4 tool calls with an answer.

## Error handling

- **Nodes never raise.** Every LLM and tool call is wrapped. A failure is appended to `state["errors"]` and a deterministic fallback keeps the turn moving:
  - Planner fails → heuristic plan: search internal docs, allow the web.
  - Grader fails → fail open and answer with the retrieved context.
  - Rewrite fails → keep the original query.
  - Answer generation fails → return the list of the most relevant sources.
- **Graceful degradation:**
  - If RAG is down, rewriting is skipped (pointless) and the agent goes to the web, if the plan allows it.
  - The answer then says it is based on external sources only.
  - If nothing is available, the agent says so instead of guessing.
- **Citation validation:** ids the model invents are removed, and only real documents are returned as sources.
- **Structured output:** Claude's native JSON-schema mode (`method="json_schema"`) is used for `Plan` and `Grade`.
- **Truncation detection:** if Claude stops at `max_tokens`, the answer is marked as truncated.
- **Checkpoint safety:** state models are explicitly allow-listed for checkpoint deserialization.
- **Top-level guard:** `run_turn` catches anything unexpected, logs it with a traceback, and returns a safe message.

## State and multi-turn memory

- `AgentState` (`agent/state.py`) separates two kinds of fields:
  - `messages` is persisted across turns by the checkpointer (`InMemorySaver`, keyed by `thread_id`).
  - Everything else is **per-turn working state** and is reset by `analyze_query`, so a new question never inherits the previous turn's documents or tool budget.
- The planner sees the most recent history window. It uses it to resolve follow-ups like *"Does that also apply to Enterprise customers?"* into a standalone search query.
- Citation markers from earlier turns are stripped from the history, because citation ids are scoped to a single turn.

## Project layout

```
agent/
  config.py    Settings (loop limits, timeouts, models) + .env loading
  state.py     AgentState, Plan, Grade, Document
  llm.py       Reasoner protocol, Claude implementation, prompts, fallbacks
  tools.py     CircuitBreaker, call_tool (timeout/retry/breaker), RAG retriever, mock web search
  routing.py   Pure conditional-edge functions
  nodes.py     Graph nodes + dependency container (AgentDeps)
  graph.py     Graph wiring, checkpointer, run_turn
data/docs/     Sample internal documentation (markdown)
tests/         Unit + end-to-end graph tests with fakes (no API keys needed)
main.py        Interactive chat + scripted demo
```

The graph depends only on the `Reasoner` protocol and plain `Tool` callables, which are injected through `AgentDeps`. That makes it easy to test with fakes or to swap providers.

## Running it

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env               # fill in ANTHROPIC_API and OPEN_AI_API

uv run pytest                      # 26 tests, no API keys or network needed
uv run python main.py --demo       # scripted scenarios against the real APIs
uv run python main.py              # interactive multi-turn chat
uv run python main.py --thread my-thread -v
```

The demo covers four scenarios:
1. An internal question answered from RAG with citations.
2. A follow-up on the same thread, which shows checkpointed memory and query rewriting.
3. A mixed internal + external question, which shows corrective escalation to the web.
4. A simulated vector-DB outage: retries, web fallback, then the circuit breaker opening so that RAG is skipped instantly.

Every turn prints the plan, the path taken through the graph and the tool-call count.

## Production next steps

- `PostgresSaver` (or Redis) instead of `InMemorySaver`, so conversations survive restarts and scale horizontally.
- A real vector database (pgvector, Qdrant…) with ingestion and re-indexing pipelines. Add metadata filters and a relevance threshold.
- A real web search provider (Tavily, Brave) behind the same `Tool` interface.
- Parallel RAG and web retrieval with `Send` when the plan needs both.
- Async nodes, streaming answers, and LangSmith / OpenTelemetry tracing. Track metrics per node (latency, breaker state, budget exhaustion rate).
- A shared circuit-breaker state (e.g. Redis) when running multiple workers.
- An evaluation set for answer faithfulness and citation precision.
