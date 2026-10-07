"""Tools (internal RAG, mocked web search) and the resilience layer around them.

Every tool is a plain callable ``(query) -> list[Hit]``. Tools are free to raise;
``call_tool`` is the single choke point that adds timeouts, retries with backoff
and a per-tool circuit breaker, and converts failures into data (never raises).
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, TypedDict

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

log = logging.getLogger(__name__)


class Hit(TypedDict):
    title: str
    location: str
    content: str
    score: float | None


Tool = Callable[[str], list[Hit]]


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #
class BreakerState(str, Enum):
    CLOSED = "closed"        # normal operation
    OPEN = "open"            # failing fast, calls are skipped
    HALF_OPEN = "half_open"  # cool-down elapsed, next call is a trial


class CircuitBreaker:
    """Classic closed -> open -> half-open breaker, shared across turns and threads.

    After ``failure_threshold`` consecutive failed calls the breaker opens and the
    tool is skipped instantly. After ``recovery_timeout`` seconds one trial call is
    allowed; success closes the breaker, failure re-opens it.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock
        self._lock = threading.Lock()
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = 0.0

    @property
    def state(self) -> BreakerState:
        with self._lock:
            return self._current_state()

    def _current_state(self) -> BreakerState:
        if (
            self._state is BreakerState.OPEN
            and self._clock() - self._opened_at >= self.recovery_timeout
        ):
            self._state = BreakerState.HALF_OPEN
        return self._state

    def allow_request(self) -> bool:
        with self._lock:
            return self._current_state() is not BreakerState.OPEN

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = BreakerState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if (
                self._current_state() is BreakerState.HALF_OPEN
                or self._failures >= self.failure_threshold
            ):
                if self._state is not BreakerState.OPEN:
                    log.warning("circuit '%s' opened after %d failure(s)", self.name, self._failures)
                self._state = BreakerState.OPEN
                self._opened_at = self._clock()


# --------------------------------------------------------------------------- #
# Resilient tool invocation
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    hits: list[Hit] = field(default_factory=list)
    error: str | None = None


# Shared pool so a hung tool call can be abandoned after the timeout.
_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")


def call_tool(
    name: str,
    tool: Tool,
    query: str,
    breaker: CircuitBreaker,
    *,
    timeout_s: float,
    retries: int,
    backoff_s: float,
) -> ToolResult:
    """Run ``tool(query)`` with breaker, timeout and retries. Never raises."""
    if not breaker.allow_request():
        return ToolResult(error=f"{name}: circuit open, call skipped")

    last_error = "unknown error"
    for attempt in range(retries + 1):
        try:
            hits = _EXECUTOR.submit(tool, query).result(timeout=timeout_s)
            breaker.record_success()
            return ToolResult(hits=hits)
        except FutureTimeout:
            last_error = f"timed out after {timeout_s:.0f}s"
        except Exception as exc:  # tool boundary: any failure becomes data
            last_error = f"{type(exc).__name__}: {exc}"
        log.warning("%s attempt %d/%d failed: %s", name, attempt + 1, retries + 1, last_error)
        if attempt < retries:
            time.sleep(backoff_s * 2**attempt)

    breaker.record_failure()
    return ToolResult(error=f"{name}: {last_error}")


# --------------------------------------------------------------------------- #
# Internal docs retriever (real embeddings, in-memory vector store)
# --------------------------------------------------------------------------- #
class InternalDocsRetriever:
    """Semantic search over ``docs_dir/*.md``, chunked by markdown section.

    The index is built lazily on first use so the app starts instantly and
    embedding failures surface as tool errors (handled by ``call_tool``).
    """

    def __init__(self, docs_dir: Path, embeddings: Embeddings, k: int = 4) -> None:
        self.docs_dir = docs_dir
        self.embeddings = embeddings
        self.k = k
        self._store: InMemoryVectorStore | None = None
        self._lock = threading.Lock()

    def __call__(self, query: str) -> list[Hit]:
        store = self._index()
        results = store.similarity_search_with_score(query, k=self.k)
        return [
            Hit(
                title=doc.metadata["title"],
                location=doc.metadata["location"],
                content=doc.page_content,
                score=round(float(score), 3),
            )
            for doc, score in results
        ]

    def _index(self) -> InMemoryVectorStore:
        with self._lock:
            if self._store is None:
                texts, metadatas = [], []
                for path in sorted(self.docs_dir.glob("*.md")):
                    for title, body in split_markdown_sections(path.read_text()):
                        texts.append(f"{title}\n\n{body}")
                        metadatas.append({"title": title, "location": f"docs/{path.name}"})
                if not texts:
                    raise FileNotFoundError(f"no markdown docs found in {self.docs_dir}")
                store = InMemoryVectorStore(self.embeddings)
                store.add_texts(texts, metadatas=metadatas)
                self._store = store
                log.info("indexed %d chunks from %s", len(texts), self.docs_dir)
            return self._store


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split a markdown file into ``(title, body)`` chunks on ``##`` headings."""
    doc_title, sections, current_title, buf = "", [], None, []
    for line in text.splitlines():
        if line.startswith("# "):
            doc_title = line[2:].strip()
        elif line.startswith("## "):
            if current_title and "".join(buf).strip():
                sections.append((current_title, "\n".join(buf).strip()))
            current_title, buf = f"{doc_title} - {line[3:].strip()}", []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((current_title or doc_title, "\n".join(buf).strip()))
    return sections


# --------------------------------------------------------------------------- #
# Mock web search (orchestration is the focus; swap for Tavily/Brave/etc.)
# --------------------------------------------------------------------------- #
_WEB_CORPUS: list[dict] = [
    {
        "keywords": {"okta", "saml", "sso", "single", "sign-on", "idp"},
        "title": "Okta Docs - SAML app integration best practices",
        "url": "https://developer.okta.com/docs/concepts/saml/",
        "snippet": "Okta recommends signing both SAML assertions and responses with SHA-256, "
        "enforcing SP-initiated flows where possible, keeping assertion validity under 5 minutes, "
        "enabling SCIM for automated deprovisioning, and requiring MFA at the IdP.",
    },
    {
        "keywords": {"session", "timeout", "owasp", "idle", "security", "sso"},
        "title": "OWASP Session Management Cheat Sheet",
        "url": "https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html",
        "snippet": "OWASP suggests idle timeouts of 15-30 minutes for low-risk applications and "
        "2-5 minutes for high-value applications, plus an absolute session timeout.",
    },
    {
        "keywords": {"incident", "severity", "sev1", "sev", "on-call", "oncall", "pagerduty", "response"},
        "title": "PagerDuty Incident Response - Severity Levels",
        "url": "https://response.pagerduty.com/before/severity_levels/",
        "snippet": "SEV-1 is a critical issue affecting a large number of users. Page the incident "
        "commander immediately, open a dedicated channel, and post status updates at a fixed cadence.",
    },
    {
        "keywords": {"refund", "saas", "industry", "standard", "subscription", "chargeback"},
        "title": "SaaS refund policy benchmarks (2026)",
        "url": "https://example-research.com/saas-refund-benchmarks",
        "snippet": "Most B2B SaaS vendors offer 14-30 day refund windows on annual plans; "
        "prorated refunds after that window are uncommon.",
    },
]


class MockWebSearch:
    """Keyword-overlap search over a tiny canned corpus. ``fail=True`` simulates an outage."""

    def __init__(self, fail: bool = False, max_results: int = 3) -> None:
        self.fail = fail
        self.max_results = max_results

    def __call__(self, query: str) -> list[Hit]:
        if self.fail:
            raise ConnectionError("web search provider unavailable (simulated)")
        terms = {t.strip("?,.!:;()'\"").lower() for t in query.split()}
        scored = []
        for entry in _WEB_CORPUS:
            overlap = len(terms & entry["keywords"])
            if overlap:
                scored.append((overlap / len(entry["keywords"]), entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            Hit(title=e["title"], location=e["url"], content=e["snippet"], score=round(s, 3))
            for s, e in scored[: self.max_results]
        ]
