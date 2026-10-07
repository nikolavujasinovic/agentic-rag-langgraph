import time

from agent.tools import BreakerState, CircuitBreaker, MockWebSearch, call_tool, split_markdown_sections


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def failing(query):
    raise RuntimeError("boom")


def test_breaker_opens_after_threshold_and_recovers_via_half_open():
    clock = FakeClock()
    breaker = CircuitBreaker("t", failure_threshold=2, recovery_timeout=30, clock=clock)

    breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert not breaker.allow_request()

    clock.now = 31
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.allow_request()
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED


def test_failed_trial_in_half_open_reopens_immediately():
    clock = FakeClock()
    breaker = CircuitBreaker("t", failure_threshold=3, recovery_timeout=10, clock=clock)
    for _ in range(3):
        breaker.record_failure()
    clock.now = 11
    assert breaker.state is BreakerState.HALF_OPEN
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN


def test_call_tool_retries_then_reports_error_without_raising():
    attempts = []

    def flaky(query):
        attempts.append(query)
        raise ConnectionError("down")

    breaker = CircuitBreaker("t", failure_threshold=5)
    result = call_tool("t", flaky, "q", breaker, timeout_s=1, retries=2, backoff_s=0)
    assert len(attempts) == 3
    assert result.hits == [] and "ConnectionError" in result.error


def test_call_tool_skips_when_circuit_open():
    breaker = CircuitBreaker("t", failure_threshold=1)
    call_tool("t", failing, "q", breaker, timeout_s=1, retries=0, backoff_s=0)
    result = call_tool("t", failing, "q", breaker, timeout_s=1, retries=0, backoff_s=0)
    assert "circuit open" in result.error


def test_call_tool_times_out_slow_tools():
    def slow(query):
        time.sleep(0.5)
        return []

    result = call_tool("t", slow, "q", CircuitBreaker("t"), timeout_s=0.05, retries=0, backoff_s=0)
    assert "timed out" in result.error


def test_mock_web_search_matches_keywords_and_can_fail():
    assert MockWebSearch()("Okta SAML best practices")[0]["location"].startswith("https://developer.okta.com")
    assert MockWebSearch()("completely unrelated") == []
    try:
        MockWebSearch(fail=True)("okta")
        raise AssertionError("expected failure")
    except ConnectionError:
        pass


def test_markdown_split_by_sections():
    chunks = split_markdown_sections("# Doc\n\n## A\nalpha\n\n## B\nbeta\n")
    assert chunks == [("Doc - A", "alpha"), ("Doc - B", "beta")]
