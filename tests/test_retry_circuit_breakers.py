from csv_sandbox_agent.orchestrator import RetryCircuitBreaker, hash_code


def test_max_retries_stops():
    breaker = RetryCircuitBreaker(max_retries=0)
    reason = breaker.record_failure(
        attempt_index=0,
        code_hash="abc",
        fingerprint="fp",
        compacted_error="ValueError: bad",
    )
    assert reason == "max retries reached"


def test_identical_code_hash_stops():
    breaker = RetryCircuitBreaker(max_retries=5)
    code_hash = hash_code("print('broken')")
    breaker.record_failure(
        attempt_index=0,
        code_hash=code_hash,
        fingerprint="fp1",
        compacted_error="ValueError: bad",
    )
    assert "identical" in (breaker.should_stop_for_code_hash(code_hash) or "")


def test_repeated_fingerprint_stops():
    breaker = RetryCircuitBreaker(max_retries=5, repeated_fingerprint_limit=3)
    assert breaker.record_failure(attempt_index=0, code_hash="a", fingerprint="same", compacted_error="err") is None
    assert breaker.record_failure(attempt_index=1, code_hash="b", fingerprint="same", compacted_error="err") is None
    reason = breaker.record_failure(attempt_index=2, code_hash="c", fingerprint="same", compacted_error="err")
    assert "fingerprint" in (reason or "")


def test_invalid_python_twice_stops():
    breaker = RetryCircuitBreaker(max_retries=5, invalid_python_limit=2)
    assert breaker.record_failure(
        attempt_index=0,
        code_hash="a",
        fingerprint="fp1",
        invalid_python=True,
        compacted_error="SyntaxError",
    ) is None
    reason = breaker.record_failure(
        attempt_index=1,
        code_hash="b",
        fingerprint="fp2",
        invalid_python=True,
        compacted_error="SyntaxError",
    )
    assert "invalid Python" in (reason or "")
