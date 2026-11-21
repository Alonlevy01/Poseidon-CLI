"""Unit tests for Milestone 3 native tool-calling agent loop and call limits."""

import json
from pathlib import Path
from typing import Any

from poseidon.core.agent import run_agent_loop
from poseidon.llm.openrouter import ChatResponse, TokenUsage, ToolCall
from poseidon.observability.respan import PoseidonTracer
from poseidon.sandbox.docker_runner import TestFailure, TestRunResult
from poseidon.tools.builtin import register_core_tools
from poseidon.tools.registry import ToolRegistry, ToolSessionState
from poseidon.workspace import WorkspaceGuard


class ScriptedToolClient:
    """Deterministic OpenRouter mock returning scripted tool calls and responses."""

    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)
        self.received_messages: list[list[dict[str, Any]]] = []
        self.received_tools: list[list[dict[str, Any]] | None] = []

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ChatResponse:
        # Snapshot messages at the moment of the call
        self.received_messages.append([dict(m) for m in messages])
        self.received_tools.append(tools)
        return self._responses.pop(0)


def _resp(
    *,
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    out_tokens: int = 30,
    reasoning_tokens: int = 10,
) -> ChatResponse:
    return ChatResponse(
        model="nvidia/nemotron-3.5-super-vl:free",
        content=content,
        tool_calls=tool_calls or [],
        usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=out_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=100 + out_tokens,
            cost_usd=0.0,
        ),
        elapsed_seconds=0.05,
        finish_reason="tool_calls" if tool_calls else "stop",
    )


def test_agent_loop_reads_patches_and_matches_tool_call_ids(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("calc.py", "def add(a, b):\n    return a - b\n")

    registry = ToolRegistry()
    register_core_tools(registry=registry, workspace=guard, state=ToolSessionState())

    valid_diff = (
        "--- a/calc.py\n"
        "+++ b/calc.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n"
        "-    return a - b\n"
        "+    return a + b\n"
    )
    client = ScriptedToolClient(
        [
            _resp(
                tool_calls=[
                    ToolCall(
                        id="call_read_1",
                        name="read_file",
                        arguments=json.dumps({"path": "calc.py"}),
                    )
                ]
            ),
            _resp(
                tool_calls=[
                    ToolCall(
                        id="call_diff_2",
                        name="apply_diff",
                        arguments=json.dumps({"diff": valid_diff}),
                    )
                ]
            ),
            _resp(content="Fixed `add` to return `a + b`."),
        ]
    )

    result = run_agent_loop(
        client=client,
        registry=registry,
        model="nvidia/nemotron-3.5-super-vl:free",
        user_prompt="Fix the bug in calc.py.",
        max_calls=5,
        tracer=PoseidonTracer(enabled=False),
    )

    assert result.calls_made == 3
    assert result.max_calls == 5
    assert "completed" in result.stop_reason
    assert result.modified_files == ["calc.py"]
    assert "return a + b" in guard.read_text("calc.py")
    assert result.usage.completion_tokens == 90
    assert result.usage.reasoning_tokens == 30
    assert not result.traced_to_respan

    # Verify tool_call_id matching in messages sent back to the model
    second_call_msgs = client.received_messages[1]
    tool_msg_1 = [m for m in second_call_msgs if m.get("role") == "tool"][0]
    assert tool_msg_1["tool_call_id"] == "call_read_1"

    third_call_msgs = client.received_messages[2]
    tool_msg_2 = [m for m in third_call_msgs if m.get("role") == "tool"][-1]
    assert tool_msg_2["tool_call_id"] == "call_diff_2"


def test_agent_loop_enforces_max_calls_limit(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("calc.py", "x = 1\n")
    registry = ToolRegistry()
    register_core_tools(registry=registry, workspace=guard, state=ToolSessionState())

    # Model keeps requesting read_file every turn; harness must stop at max_calls=2
    client = ScriptedToolClient(
        [
            _resp(
                tool_calls=[
                    ToolCall(id="c1", name="read_file", arguments=json.dumps({"path": "calc.py"}))
                ]
            ),
            _resp(
                tool_calls=[
                    ToolCall(id="c2", name="read_file", arguments=json.dumps({"path": "calc.py"}))
                ]
            ),
            _resp(content="Should never be reached"),
        ]
    )

    result = run_agent_loop(
        client=client,
        registry=registry,
        model="nvidia/nemotron-3.5-super-vl:free",
        user_prompt="Inspect calc.py",
        max_calls=2,
        tracer=PoseidonTracer(enabled=False),
    )

    assert result.calls_made == 2
    assert result.max_calls == 2
    assert "limit" in result.stop_reason
    assert len(client.received_messages) == 2


class ScriptedTestRunner:
    """Fake isolated test runner for testing post-edit auto-verification and repair loops."""

    def __init__(self, results: list[TestRunResult]) -> None:
        self._results = list(results)
        self.runs = 0

    @property
    def timeout_seconds(self) -> int:
        return 15

    def run(
        self,
        workspace_root: Path,
        *,
        test_args: list[str] | None = None,
        extra_files: dict[str, str] | None = None,
    ) -> TestRunResult:
        self.runs += 1
        return self._results.pop(0)


def test_agent_auto_verifies_edit_and_repairs_failing_test(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("calc.py", "def add(a, b):\n    return a - b\n")
    state = ToolSessionState()
    fail_res = TestRunResult(
        command="pytest -v --tb=short",
        timeout_seconds=15,
        exit_code=1,
        passed=0,
        failed=1,
        total=1,
        failures=[TestFailure(name="test_calc::test_add", message="assert -1 == 5")],
        output="FAILED test_calc.py::test_add",
        verified=False,
        unverified_reason="1 test(s) failed, 0 error(s) (exit status 1).",
    )
    pass_res = TestRunResult(
        command="pytest -v --tb=short",
        timeout_seconds=15,
        exit_code=0,
        passed=1,
        failed=0,
        total=1,
        output="1 passed",
        verified=True,
    )
    runner = ScriptedTestRunner([fail_res, pass_res])
    registry = ToolRegistry()
    register_core_tools(registry=registry, workspace=guard, state=state, test_runner=runner)

    bad_fix = (
        "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n-    return a - b\n+    return a * b\n"
    )
    good_fix = (
        "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n-    return a * b\n+    return a + b\n"
    )
    client = ScriptedToolClient(
        [
            _resp(
                tool_calls=[
                    ToolCall(id="c1", name="read_file", arguments=json.dumps({"path": "calc.py"}))
                ]
            ),
            _resp(
                tool_calls=[
                    ToolCall(id="c2", name="apply_diff", arguments=json.dumps({"diff": bad_fix}))
                ]
            ),
            # Model tries to stop without running tests -> harness auto-runs tests & detects failure
            _resp(content="I think it is fixed."),
            # Harness feeds failure back -> model repairs and calls run_tests
            _resp(
                tool_calls=[
                    ToolCall(id="c3", name="apply_diff", arguments=json.dumps({"diff": good_fix}))
                ]
            ),
            _resp(tool_calls=[ToolCall(id="c4", name="run_tests", arguments="{}")]),
            _resp(content="Fixed add to return a + b; all tests pass."),
        ]
    )

    result = run_agent_loop(
        client=client,
        registry=registry,
        model="nvidia/nemotron-3.5-super-vl:free",
        user_prompt="Fix add in calc.py.",
        max_calls=8,
        tracer=PoseidonTracer(enabled=False),
        workspace_root=guard.root,
        tool_state=state,
        test_runner=runner,
    )

    assert runner.runs == 2
    assert result.edit_verified is True
    assert "verified (1/1 passed)" in result.stop_reason
    assert "return a + b" in guard.read_text("calc.py")


def test_agent_marks_edit_unverified_on_timeout(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("calc.py", "def add(a, b):\n    return a - b\n")
    state = ToolSessionState()
    timeout_res = TestRunResult(
        command="pytest -v --tb=short",
        timeout_seconds=5,
        exit_code=None,
        timed_out=True,
        verified=False,
        unverified_reason="Test run timed out after 5s.",
    )
    runner = ScriptedTestRunner([timeout_res])
    registry = ToolRegistry()
    register_core_tools(registry=registry, workspace=guard, state=state, test_runner=runner)

    good_fix = (
        "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n-    return a - b\n+    return a + b\n"
    )
    client = ScriptedToolClient(
        [
            _resp(
                tool_calls=[
                    ToolCall(id="c1", name="read_file", arguments=json.dumps({"path": "calc.py"}))
                ]
            ),
            _resp(
                tool_calls=[
                    ToolCall(id="c2", name="apply_diff", arguments=json.dumps({"diff": good_fix}))
                ]
            ),
            _resp(content="Done."),
        ]
    )

    result = run_agent_loop(
        client=client,
        registry=registry,
        model="nvidia/nemotron-3.5-super-vl:free",
        user_prompt="Fix calc.py",
        max_calls=6,
        tracer=PoseidonTracer(enabled=False),
        workspace_root=guard.root,
        tool_state=state,
        test_runner=runner,
    )

    assert result.edit_verified is False
    assert "timed out after 5s" in result.stop_reason
