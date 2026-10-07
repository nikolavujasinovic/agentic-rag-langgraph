# Agentic RAG with LangGraph

An agent that answers employee questions about internal company documentation. It:

1. **Analyzes** the query and plans which tools are needed.
2. **Searches internal docs** with RAG: OpenAI embeddings over an in-memory vector store.
3. **Optionally searches the web.** Web search is mocked, since the focus here is orchestration.
4. **Grades** the retrieved context. If the context is weak, it **corrects course** by rewriting the query or escalating to the web.
5. **Synthesizes** a final answer with **native Claude citations**, via search result content blocks.

It keeps **multi-turn conversation state** through LangGraph checkpointing. Every loop is **bounded** by a tool budget, a per-tool circuit breaker and a recursion-limit backstop.

- **LangGraph** handles orchestration: state, routing and checkpointing.
- Claude is called through the **official `anthropic` SDK**, for planning, grading, rewriting and answer generation.
- OpenAI `text-embedding-3-small` handles retrieval embeddings.

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
| `analyze_query` | Uses Claude with structured output (`messages.parse` → `Plan`) to choose the route: `rag`, `web` or `direct`. It also decides whether web fallback is allowed and rewrites follow-up questions into standalone queries using the conversation history. It resets all per-turn state. |
| `rag_search` / `web_search` | Call the tool through `call_tool`, which applies the timeout, retries, the circuit breaker and error capture. Results are merged with stable per-turn ids (`doc-1`, `web-1`). |
| `grade_context` | Claude with structured output (`messages.parse` → `Grade`) decides whether the context is sufficient. If not, it says whether the missing information is external. |
| `rewrite_query` | Rephrases the search query for a second retrieval attempt (corrective RAG). |
| `synthesize` | Sends the retrieved documents as `search_result` blocks with citations enabled, and turns Claude's `search_result_location` citations into `[doc-x]` / `[web-x]` markers and a source list. Says honestly when no context was found. |

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

## Citations

Citations use Claude's [search result content blocks](https://platform.claude.com/docs/en/build-with-claude/search-results), the API feature built for RAG. They are not prompt-based.

```json
{"type": "search_result", "source": "docs/refund-policy.md", "title": "Internal: Refund Policy - Annual plans",
 "content": [{"type": "text", "text": "..."}], "citations": {"enabled": true}}
```

- **Granularity:** each retrieved chunk becomes one `search_result`, split into one text block per paragraph. Claude cites whole blocks, so smaller blocks give more precise citations.
- **Response format:** cited text blocks come back with `search_result_location` citations (`search_result_index`, `start_block_index`, `end_block_index`, `cited_text`).
- **Mapping to sources:** `search_result_index` counts search results in request order. Only the current turn carries search results, so the index maps directly onto this turn's documents.
- **Advantages over asking the model to write `[doc-1]` itself,** per the docs:
  - Citations are guaranteed to point at real provided sources, so no hallucinated-id cleanup is needed.
  - They are more likely to point at the most relevant passage.
  - `cited_text` doesn't count toward output tokens.
- **Multi-turn:**
  - Only the plain answer text is persisted in the conversation. Sources are re-retrieved for every turn rather than resent.
  - Citation markers are scoped to a turn and are stripped from history before it is sent back.
- **Incompatible with structured outputs:** combining the two in one request is a 400. Here they are used in separate calls: planning and grading use structured outputs, and the answer call uses citations.

## Claude API usage

| Concern | Implementation |
|---|---|
| Structured outputs | `client.beta.messages.parse(output_format=Plan / Grade)`, which uses `output_config.format` with a JSON schema |
| Effort | `output_config.effort`: `low` for routing steps (plan, grade, rewrite), `medium` for the final answer |
| Stop reasons | Checked before reading content. `refusal` raises `LLMRefusalError`. `max_tokens` raises for structured steps; for the answer, the text is kept and flagged as truncated. |
| Refusals | Server-side fallback (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`) re-runs a declined request on Anthropic's recommended fallback model. If the whole chain declines, the agent returns a short refusal and skips the tools. |
| Multi-turn | Prior turns are sent as a real user/assistant `messages` array. The planner receives them as a transcript, because there the history is input data. |
| Retries / timeouts | SDK built-in retries (429/5xx/connection errors with backoff) and a per-request timeout |
| Model | `claude-sonnet-5-5`, which keeps 3–4 calls per turn fast and inexpensive. Configurable in `Settings`. |

## Error handling

- **Nodes never raise.** Every LLM and tool call is wrapped. A failure is appended to `state["errors"]` and a deterministic fallback keeps the turn moving.
- **LLM errors are classified, most specific first,** by `classify_error`:
  - Transient failures (rate limits, 5xx, network) are logged as warnings.
  - Bugs and misconfiguration (400, auth, unknown model) are logged as errors with a traceback.

  The fallbacks are:
  - Planner fails → heuristic plan: search internal docs, allow the web.
  - Grader fails → fail open and answer with the retrieved context.
  - Rewrite fails → keep the original query.
  - Answer generation fails → return the list of the most relevant sources.
- **Graceful degradation:**
  - If RAG is down, rewriting is skipped (pointless) and the agent goes to the web, if the plan allows it.
  - The answer then says it is based on external sources only.
  - If nothing is available, the agent says so instead of guessing.
- **Refusals:** `refusal` is handled explicitly instead of producing an empty answer.
- **Checkpoint safety:** state models are explicitly allow-listed for checkpoint deserialization.
- **Top-level guard:** `run_turn` catches anything unexpected, logs it with a traceback, and returns a safe message.

## State and multi-turn memory

- `AgentState` (`agent/state.py`) separates two kinds of fields:
  - `messages` is persisted across turns by the checkpointer (`InMemorySaver`, keyed by `thread_id`).
  - Everything else is **per-turn working state** and is reset by `analyze_query`, so a new question never inherits the previous turn's documents or tool budget.
- The planner sees the most recent history window. It uses it to resolve follow-ups like *"Does that also apply to Enterprise customers?"* into a standalone search query.

## Project layout

```
agent/
  config.py    Settings (loop limits, timeouts, models) + .env loading
  state.py     AgentState, Plan, Grade, Document
  llm.py       Reasoner protocol, Claude implementation (anthropic SDK), search-result citations,
               stop-reason / refusal handling, error classification, fallbacks
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

uv run pytest                      # 38 tests, no API keys or network needed
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
