"""Unit tests for isolated sandbox packaging, JUnit XML parsing, and verification outcomes."""

import io
import tarfile
from pathlib import Path

from poseidon.sandbox.docker_runner import (
    build_workspace_tar,
    evaluate_test_outcome,
    parse_junit_xml,
)

SAMPLE_JUNIT_WITH_FAILURE = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests">
  <testsuite name="pytest" errors="0" failures="1" skipped="0" tests="3">
    <testcase classname="test_ledger" name="test_ok_one" time="0.01"/>
    <testcase classname="test_ledger" name="test_ok_two" time="0.01"/>
    <testcase classname="test_ledger" name="test_sort_amount" time="0.02">
      <failure message="AssertionError: assert 'food' == 'rent'">details here</failure>
    </testcase>
  </testsuite>
</testsuites>
"""


def test_parse_junit_xml_extracts_counts_and_failure_details() -> None:
    passed, failed, errors, skipped, failures = parse_junit_xml(SAMPLE_JUNIT_WITH_FAILURE)
    assert (passed, failed, errors, skipped) == (2, 1, 0, 0)
    assert len(failures) == 1
    assert failures[0].name == "test_ledger::test_sort_amount"
    assert "AssertionError" in failures[0].message


def test_evaluate_test_outcome_verified_and_unverified_cases() -> None:
    ok_xml = (
        '<testsuite tests="2" failures="0" errors="0" skipped="0">'
        '<testcase classname="t" name="a"/><testcase classname="t" name="b"/>'
        "</testsuite>"
    )
    ok_res = evaluate_test_outcome(
        command="pytest -v --tb=short",
        timeout_seconds=15,
        exit_code=0,
        timed_out=False,
        runner_error=None,
        xml_text=ok_xml,
        output="2 passed",
        elapsed_seconds=0.4,
    )
    assert ok_res.verified
    assert ok_res.passed == 2
    assert ok_res.unverified_reason is None

    # 0 tests collected -> MUST be unverified
    zero_res = evaluate_test_outcome(
        command="pytest -v --tb=short",
        timeout_seconds=15,
        exit_code=5,
        timed_out=False,
        runner_error=None,
        xml_text=None,
        output="collected 0 items",
        elapsed_seconds=0.2,
    )
    assert not zero_res.verified
    assert "No tests ran" in (zero_res.unverified_reason or "")

    # Timeout -> MUST be unverified
    timeout_res = evaluate_test_outcome(
        command="pytest -v --tb=short",
        timeout_seconds=5,
        exit_code=None,
        timed_out=True,
        runner_error=None,
        xml_text=None,
        output="",
        elapsed_seconds=5.0,
    )
    assert not timeout_res.verified
    assert "timed out after 5s" in (timeout_res.unverified_reason or "")

    # Runner error -> MUST be unverified
    err_res = evaluate_test_outcome(
        command="pytest -v --tb=short",
        timeout_seconds=15,
        exit_code=None,
        timed_out=False,
        runner_error="Docker daemon stopped",
        xml_text=None,
        output="",
        elapsed_seconds=0.1,
    )
    assert not err_res.verified
    assert "could not run" in (err_res.unverified_reason or "")


def test_build_workspace_tar_excludes_secrets_and_caches(tmp_path: Path) -> None:
    (tmp_path / "ledger.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=secret\n", encoding="utf-8")
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "ledger.cpython-312.pyc").write_bytes(b"bytecode")

    tar_bytes = build_workspace_tar(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tf:
        names = tf.getnames()

    assert names == ["ledger.py"]
