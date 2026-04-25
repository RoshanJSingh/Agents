from csv_sandbox_agent.error_compaction import compact_error


def test_huge_traceback_gets_shortened_and_preserves_signal():
    stderr = "\n".join(
        [f'  File "/usr/local/lib/python3.11/site-packages/pandas/core/frame.py", line {i}, in x' for i in range(300)]
        + [
            "Traceback (most recent call last):",
            '  File "/workspace/generated_script.py", line 42, in main',
            "    df['missing']",
            "KeyError: 'missing'",
        ]
    )
    compacted = compact_error("", stderr, max_chars=1000)
    assert len(compacted) <= 1000
    assert "KeyError: 'missing'" in compacted
    assert "generated_script.py" in compacted
    assert "line 42" in compacted


def test_stdout_tail_used_if_stderr_empty():
    stdout = "\n".join([f"line {i}" for i in range(100)] + ["ValueError: bad value"])
    compacted = compact_error(stdout, "")
    assert "ValueError: bad value" in compacted
    assert "line 99" in compacted
