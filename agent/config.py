"""Runtime configuration.

All tunables live in a single frozen dataclass so they can be injected into the
graph (and overridden in tests) instead of being read from globals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The local .env uses non-standard names; map them onto what the SDKs expect.
_ENV_ALIASES = {
    "ANTHROPIC_API": "ANTHROPIC_API_KEY",
    "OPEN_AI_API": "OPENAI_API_KEY",
}


def load_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    for alias, canonical in _ENV_ALIASES.items():
        if os.getenv(alias) and not os.getenv(canonical):
            os.environ[canonical] = os.environ[alias]


@dataclass(frozen=True)
class Settings:
    # --- Loop guards -------------------------------------------------------
    max_tool_calls: int = 4          # hard budget of tool invocations per turn
    max_rewrites: int = 1            # corrective query rewrites per turn
    recursion_limit: int = 25        # LangGraph super-step backstop

    # --- Tool execution ----------------------------------------------------
    tool_timeout_s: float = 10.0
    tool_retries: int = 1            # retries after the first attempt
    tool_backoff_s: float = 0.5      # exponential backoff base
    breaker_failure_threshold: int = 3
    breaker_recovery_s: float = 30.0

    # --- Models & retrieval ------------------------------------------------
    llm_model: str = "claude-sonnet-5-5"
    llm_step_effort: str = "low"       # plan / grade / rewrite / chat: routing decisions
    llm_answer_effort: str = "medium"  # final grounded answer
    llm_max_tokens: int = 16000
    llm_timeout_s: float = 60.0
    embedding_model: str = "text-embedding-3-small"
    rag_top_k: int = 4
    history_window: int = 10         # messages passed to the LLM as context
    docs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "docs")
