"""Core native tool-calling agent loop with call limits, isolated test verification, and Respan."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from poseidon.context.compactor import compact_messages
from poseidon.llm.openrouter import ChatResponse, TokenUsage
from poseidon.observability.respan import PoseidonTracer
from poseidon.sandbox.docker_runner import TestRunner, TestRunResult
from poseidon.tools.registry import ToolExecutionResult, ToolRegistry, ToolSessionState

AGENT_SYSTEM_PROMPT = """You are Poseidon-CLI, a precise Python coding agent inside a sandbox.

Workflow rules:
1. Start by calling `search_text` or `semantic_search` (and `list_files` if needed) to locate
   relevant symbols and files.
2. Read targeted sections with `read_lines` or full files with `read_file` before `apply_diff`.
3. Respect modular boundaries: put computation, formatting, parsing, and CLI flags in their
   respective modules rather than dumping everything into one file.
4. Use `create_file` only for new files; use minimal Git unified diffs (`apply_diff`, 3 lines of
   exact context, no line numbers) for existing files.
5. Never edit unit test files while repairing a failing test; fix the implementation instead.
6. After editing code, run `run_tests` to verify the full suite passes; fix any failures.
7. Once verified (or no edits needed), give a concise summary and stop calling tools.
"""

PLAN_MODE_DIRECTIVE = """
PLAN MODE ACTIVE (READ-ONLY):
- Write tools (`apply_diff`, `create_file`) are disabled. Do NOT attempt to edit files.
- Explore the codebase using `list_files`, `search_text`, `semantic_search`, `read_lines`,
  `read_file`, and `run_tests`.
- Deliver a clear, numbered implementation plan listing:
  1. Exact files and functions to modify or create
  2. Concrete behavioral changes and edge cases
  3. Verification & test strategy
"""


def build_system_prompt(
    *,
    base_prompt: str = AGENT_SYSTEM_PROMPT,
    project_memory: str = "",
    plan_mode: bool = False,
) -> str:
    """Compose the static top system prompt with optional project memory and plan-mode rules."""
    parts = [base_prompt.strip()]
    if project_memory.strip():
        parts.append("## Project Memory (from repository rules)\n\n" + project_memory.strip())
    if plan_mode:
        parts.append(PLAN_MODE_DIRECTIVE.strip())
    return "\n\n".join(parts)


class ToolCapableClient(Protocol):
    """Protocol for an LLM client supporting OpenRouter native tool calling."""

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ChatResponse: ...


class ToolEvent(BaseModel):
    """Recorded trace of a single tool invocation during an agent run."""

    call_index: int
    tool_call_id: str
    tool_name: str
    ok: bool
    summary: str


class AgentRunResult(BaseModel):
    """Complete result and telemetry summary of an agent execution."""

    final_response: str
    model: str
    calls_made: int
    max_calls: int
    stop_reason: str
    tool_events: list[ToolEvent] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)
    last_test_result: TestRunResult | None = None
    edit_verified: bool | None = None
    estimated_tokens_saved: int = 0
    usage: TokenUsage
    elapsed_seconds: float
    traced_to_respan: bool = False


def _merge_usage(usages: list[TokenUsage]) -> TokenUsage:
    """Aggregate token usage and cost across all model calls in a run."""
    prompts = [u.prompt_tokens for u in usages if u.prompt_tokens is not None]
    completions = [u.completion_tokens for u in usages if u.completion_tokens is not None]
    reasonings = [u.reasoning_tokens for u in usages if u.reasoning_tokens is not None]
    totals = [u.total_tokens for u in usages if u.total_tokens is not None]
    costs = [u.cost_usd for u in usages if u.cost_usd is not None]
    return TokenUsage(
        prompt_tokens=sum(prompts) if prompts else None,
        completion_tokens=sum(completions) if completions else None,
        reasoning_tokens=sum(reasonings) if reasonings else None,
        total_tokens=sum(totals) if totals else None,
        cost_usd=sum(costs) if costs else None,
    )


def run_agent_loop(
    *,
    client: ToolCapableClient,
    registry: ToolRegistry,
    model: str,
    user_prompt: str,
    max_calls: int = 8,
    tracer: PoseidonTracer | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    on_model_call: Callable[[int, int], None] | None = None,
    on_tool_result: Callable[[ToolEvent], None] | None = None,
    workspace_root: Path | None = None,
    tool_state: ToolSessionState | None = None,
    test_runner: TestRunner | None = None,
    on_test_result: Callable[[TestRunResult], None] | None = None,
    system_prompt: str = AGENT_SYSTEM_PROMPT,
) -> AgentRunResult:
    """Run the tool-calling loop with call limits, test verification, and Respan tracing."""
    if max_calls < 1:
        raise ValueError("max_calls must be at least 1.")

    active_tracer = tracer or PoseidonTracer(enabled=False)
    state = tool_state if tool_state is not None else ToolSessionState()
    result_holder: list[AgentRunResult] = []

    def _execute_workflow(request: str, selected_model: str, call_limit: int) -> dict[str, Any]:
        started = time.perf_counter()
        messages: list[dict[str, Any]]
        if conversation_history is not None:
            messages = conversation_history
            if not messages:
                messages.append({"role": "system", "content": system_prompt})
        else:
            messages = [{"role": "system", "content": system_prompt}]

        messages.append({"role": "user", "content": request})

        tool_schemas = registry.schemas()
        calls_made = 0
        usages: list[TokenUsage] = []
        tool_events: list[ToolEvent] = []
        modified_files: list[str] = []
        resolved_model = selected_model
        final_text = ""
        stop_reason = f"reached model-call limit ({call_limit}/{call_limit})"
        latest_response: ChatResponse | None = None
        latest_tool_res: ToolExecutionResult | None = None
        latest_auto_test: TestRunResult | None = None
        total_tokens_saved = 0

        def _invoke_model(
            call_number: int,
            target_model: str,
            message_count: int,
            available_tools: list[str],
        ) -> dict[str, Any]:
            nonlocal latest_response
            resp = client.chat_completion(
                model=target_model,
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                tool_choice="auto" if tool_schemas else None,
            )
            latest_response = resp
            return {
                "call_number": call_number,
                "model": resp.model,
                "finish_reason": resp.finish_reason,
                "requested_tools": [tc.name for tc in resp.tool_calls],
                "content_preview": resp.content[:500],
                "usage": resp.usage.model_dump(),
                "elapsed_seconds": round(resp.elapsed_seconds, 3),
            }

        def _invoke_tool(
            tool_call_id: str,
            tool_name: str,
            arguments: str,
        ) -> dict[str, Any]:
            nonlocal latest_tool_res
            exec_res = registry.execute(tool_name, arguments)
            latest_tool_res = exec_res
            return {
                "tool_call_id": tool_call_id,
                "tool_name": exec_res.tool_name,
                "ok": exec_res.ok,
                "summary": exec_res.summary,
                "output_preview": exec_res.output[:2000],
            }

        def _invoke_auto_test(reason: str) -> dict[str, Any]:
            nonlocal latest_auto_test
            assert test_runner is not None and workspace_root is not None
            res = test_runner.run(workspace_root)
            latest_auto_test = res
            state.last_test_result = res
            state.dirty_since_last_test = False
            if on_test_result is not None:
                on_test_result(res)
            return {
                "reason": reason,
                "command": res.command,
                "verified": res.verified,
                "passed": res.passed,
                "failed": res.failed,
                "errors": res.errors,
                "total": res.total,
                "exit_code": res.exit_code,
                "unverified_reason": res.unverified_reason,
            }

        while calls_made < call_limit:
            calls_made += 1
            if on_model_call is not None:
                on_model_call(calls_made, call_limit)

            # Automatically mask stale bulky tool outputs while keeping recent 4 verbatim
            _, compact_stats = compact_messages(
                messages,
                keep_recent_tool_outputs=4,
                min_chars_to_mask=600,
                in_place=True,
            )
            total_tokens_saved += compact_stats.estimated_tokens_saved

            active_tracer.run_model_call(
                "openrouter_chat_completion",
                _invoke_model,
                calls_made,
                selected_model,
                len(messages),
                registry.names(),
            )
            assert latest_response is not None
            response = latest_response
            resolved_model = response.model
            usages.append(response.usage)
            messages.append(response.to_assistant_message())

            if not response.tool_calls:
                # Enforce automatic full-suite verification after edits before reporting success
                if (
                    test_runner is not None
                    and workspace_root is not None
                    and state.dirty_since_last_test
                ):
                    active_tracer.run_tool_call(
                        "auto_verify_tests",
                        _invoke_auto_test,
                        "post-edit verification",
                    )
                    assert latest_auto_test is not None
                    test_res = latest_auto_test
                    event = ToolEvent(
                        call_index=calls_made,
                        tool_call_id="auto_verify",
                        tool_name="run_tests",
                        ok=test_res.verified,
                        summary=(
                            f"PASS ({test_res.passed}/{test_res.total} passed)"
                            if test_res.verified
                            else f"UNVERIFIED ({test_res.unverified_reason})"
                        ),
                    )
                    tool_events.append(event)
                    if on_tool_result is not None:
                        on_tool_result(event)

                    if test_res.verified:
                        final_text = response.content.strip()
                        stop_reason = (
                            f"completed and verified ({test_res.passed}/{test_res.total} passed)"
                        )
                        break

                    # If no tests ran, timed out, or runner failed -> leave edit unverified
                    if test_res.timed_out or test_res.runner_error or test_res.total == 0:
                        final_text = response.content.strip()
                        stop_reason = f"edit unverified ({test_res.unverified_reason})"
                        break

                    # Real test failure -> hand failure back to the model to repair if calls remain
                    if calls_made < call_limit:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Automatic post-edit test verification FAILED. "
                                    "Do NOT modify test files; repair the implementation and "
                                    "verify with run_tests.\n\n"
                                    f"{test_res.format_for_model()}"
                                ),
                            }
                        )
                        continue

                    final_text = response.content.strip()
                    stop_reason = (
                        f"reached model-call limit ({calls_made}/{call_limit}) "
                        f"with failing tests ({test_res.unverified_reason})"
                    )
                    break

                final_text = response.content.strip()
                if modified_files and state.last_test_result is not None:
                    if state.last_test_result.verified:
                        tr = state.last_test_result
                        stop_reason = f"completed and verified ({tr.passed}/{tr.total} passed)"
                    else:
                        stop_reason = (
                            f"edit unverified ({state.last_test_result.unverified_reason})"
                        )
                else:
                    stop_reason = "completed (model finished without further tool calls)"
                break

            for tc in response.tool_calls:
                active_tracer.run_tool_call(
                    tc.name,
                    _invoke_tool,
                    tc.id,
                    tc.name,
                    tc.arguments,
                )
                assert latest_tool_res is not None
                tool_res = latest_tool_res

                for mod_file in tool_res.modified_files:
                    if mod_file not in modified_files:
                        modified_files.append(mod_file)

                event = ToolEvent(
                    call_index=calls_made,
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    ok=tool_res.ok,
                    summary=tool_res.summary,
                )
                tool_events.append(event)
                if on_tool_result is not None:
                    on_tool_result(event)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": tool_res.output,
                    }
                )
        else:
            # If the final tool call was an edit and we hit max_calls, run verification once
            if (
                test_runner is not None
                and workspace_root is not None
                and state.dirty_since_last_test
            ):
                active_tracer.run_tool_call(
                    "auto_verify_tests",
                    _invoke_auto_test,
                    "post-edit verification at call limit",
                )
            stop_reason = f"reached model-call limit ({calls_made}/{call_limit})"
            if not final_text:
                final_text = (
                    f"Stopped after {calls_made} model call(s) because the model-call limit "
                    f"({call_limit}) was reached."
                )

        edit_verified: bool | None = None
        if modified_files:
            if state.last_test_result is not None and not state.dirty_since_last_test:
                edit_verified = state.last_test_result.verified
            else:
                edit_verified = False

        run_result = AgentRunResult(
            final_response=final_text,
            model=resolved_model,
            calls_made=calls_made,
            max_calls=call_limit,
            stop_reason=stop_reason,
            tool_events=tool_events,
            modified_files=modified_files,
            last_test_result=state.last_test_result,
            edit_verified=edit_verified,
            estimated_tokens_saved=total_tokens_saved,
            usage=_merge_usage(usages),
            elapsed_seconds=time.perf_counter() - started,
            traced_to_respan=active_tracer.enabled,
        )
        result_holder.append(run_result)
        return run_result.model_dump()

    try:
        active_tracer.run_workflow(
            "poseidon_agent_run",
            _execute_workflow,
            user_prompt,
            model,
            max_calls,
        )
        return result_holder[0]
    finally:
        active_tracer.flush()
