"""Autonomous Plan -> Execute -> Verify -> Reflect `/goal` loop shared by CLI and Web UI."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from poseidon.core.agent import (
    AgentRunResult,
    ToolEvent,
    build_system_prompt,
    run_agent_loop,
)
from poseidon.llm.openrouter import OpenRouterClient
from poseidon.observability.respan import PoseidonTracer
from poseidon.sandbox.docker_runner import TestRunner, TestRunResult
from poseidon.tools.registry import ToolRegistry, ToolSessionState

_GOAL_STATUS_RE = re.compile(r"<goal_status>\s*(COMPLETE|CONTINUE|BLOCKED)\s*</goal_status>", re.I)


class GoalIterationRecord(BaseModel):
    """Outcome summary for one execute-and-verify iteration inside `/goal`."""

    iteration: int
    calls_made: int
    status_signal: str
    modified_files: list[str] = Field(default_factory=list)
    edit_verified: bool = True
    summary: str = ""
    elapsed_seconds: float = 0.0
    total_tokens: int = 0
    cost_usd: float = 0.0


class GoalRunResult(BaseModel):
    """End-to-end report returned after `/goal` completes, blocks, cancels, or hits max steps."""

    objective: str
    model: str
    plan_text: str = ""
    completed: bool = False
    stop_reason: str = ""
    iterations_run: int = 0
    max_iterations: int = 5
    modified_files: list[str] = Field(default_factory=list)
    all_edits_verified: bool = True
    iterations: list[GoalIterationRecord] = Field(default_factory=list)
    total_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    final_summary: str = ""


def parse_goal_command(raw_arg: str, *, default_steps: int = 5) -> tuple[str, int]:
    """Parse `/goal [--steps N] <objective>` into `(objective, max_iterations)`."""
    text = raw_arg.strip()
    steps = max(1, min(default_steps, 15))
    match = re.match(r"^--steps(?:=|\s+)(\d+)\s+(.+)$", text, flags=re.S)
    if match:
        steps = max(1, min(int(match.group(1)), 15))
        text = match.group(2).strip()
    if not text:
        raise ValueError("Usage: /goal [--steps N] <clear objective to accomplish>")
    return text, steps


def extract_goal_status(response_text: str) -> str:
    """Extract `<goal_status>COMPLETE|CONTINUE|BLOCKED</goal_status>` from model output."""
    match = _GOAL_STATUS_RE.search(response_text or "")
    if not match:
        return "CONTINUE"
    return match.group(1).upper()


def run_goal_loop(
    *,
    client: OpenRouterClient,
    read_only_registry: ToolRegistry,
    write_registry: ToolRegistry,
    model: str,
    objective: str,
    max_iterations: int = 5,
    max_calls_per_iteration: int = 8,
    project_memory: str = "",
    tracer: PoseidonTracer | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    workspace_root: Path | None = None,
    tool_state: ToolSessionState | None = None,
    test_runner: TestRunner | None = None,
    on_plan_ready: Callable[[str], None] | None = None,
    on_iteration_start: Callable[[int, int], None] | None = None,
    on_iteration_complete: Callable[[GoalIterationRecord], None] | None = None,
    on_model_call: Callable[[int, int], None] | None = None,
    on_tool_result: Callable[[ToolEvent], None] | None = None,
    on_test_result: Callable[[TestRunResult], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> GoalRunResult:
    """Execute a disciplined Plan -> Execute -> Verify -> Reflect autonomous goal loop."""
    started = time.perf_counter()
    bounded_iters = max(1, min(int(max_iterations), 15))
    state = tool_state if tool_state is not None else ToolSessionState()
    history = conversation_history if conversation_history is not None else []

    # ── Phase 1: Read-Only Planning Pass ─────────────────────────────────────
    plan_prompt = (
        f"GOAL OBJECTIVE:\n{objective}\n\n"
        "You are in the READ-ONLY PLANNING phase of `/goal`. Inspect the workspace first "
        "(using list_files, semantic_search, read_file, etc.) and produce a concise, concrete "
        "numbered checklist of steps + verification criteria needed to fully achieve this goal. "
        "Do NOT attempt to edit files yet."
    )
    plan_sys = build_system_prompt(project_memory=project_memory, plan_mode=True)
    plan_run: AgentRunResult = run_agent_loop(
        client=client,
        registry=read_only_registry,
        model=model,
        user_prompt=plan_prompt,
        max_calls=min(max_calls_per_iteration, 6),
        tracer=tracer,
        conversation_history=history,
        on_model_call=on_model_call,
        on_tool_result=on_tool_result,
        workspace_root=workspace_root,
        tool_state=state,
        test_runner=None,
        system_prompt=plan_sys,
    )
    plan_text = plan_run.final_response.strip() or "(No explicit plan returned.)"
    if on_plan_ready is not None:
        on_plan_ready(plan_text)

    total_calls = plan_run.calls_made
    total_tokens = plan_run.usage.total_tokens or 0
    total_cost = plan_run.usage.cost_usd or 0.0
    all_modified: list[str] = []
    iterations: list[GoalIterationRecord] = []
    completed = False
    stop_reason = f"reached max goal iterations ({bounded_iters})"
    final_summary = plan_text
    stagnant_streak = 0
    last_feedback = ""

    exec_sys = build_system_prompt(project_memory=project_memory, plan_mode=False)

    # ── Phase 2: Bounded Execute & Verify Iterations ─────────────────────────
    for idx in range(1, bounded_iters + 1):
        if cancel_event is not None and cancel_event.is_set():
            stop_reason = "cancelled by user"
            break

        if on_iteration_start is not None:
            on_iteration_start(idx, bounded_iters)

        iter_prompt = (
            f"GOAL ITERATION {idx}/{bounded_iters}\n"
            f"Objective: {objective}\n\n"
            f"Current Plan:\n{plan_text}\n\n"
            + (f"Previous Iteration Feedback:\n{last_feedback}\n\n" if last_feedback else "")
            + "Execute the next concrete step(s), make minimal clean edits, and verify with "
            "`run_tests`. At the very end of your reply, include an updated checklist and EXACTLY "
            "one status tag on its own line:\n"
            "- `<goal_status>COMPLETE</goal_status>` ONLY if ALL checklist items are done and "
            "tests are verified passing.\n"
            "- `<goal_status>CONTINUE</goal_status>` if more work remains for the next iteration.\n"
            "- `<goal_status>BLOCKED</goal_status>` if you hit an external blocker requiring user "
            "input."
        )

        # Ensure system prompt switches from read-only plan prompt to full execution prompt
        if history and history[0].get("role") == "system":
            history[0]["content"] = exec_sys

        iter_run = run_agent_loop(
            client=client,
            registry=write_registry,
            model=model,
            user_prompt=iter_prompt,
            max_calls=max_calls_per_iteration,
            tracer=tracer,
            conversation_history=history,
            on_model_call=on_model_call,
            on_tool_result=on_tool_result,
            workspace_root=workspace_root,
            tool_state=state,
            test_runner=test_runner,
            on_test_result=on_test_result,
            system_prompt=exec_sys,
        )

        total_calls += iter_run.calls_made
        total_tokens += iter_run.usage.total_tokens or 0
        total_cost += iter_run.usage.cost_usd or 0.0
        for path in iter_run.modified_files:
            if path not in all_modified:
                all_modified.append(path)

        signal = extract_goal_status(iter_run.final_response)
        # Strict completion gate: refuse COMPLETE if modified files are unverified!
        if signal == "COMPLETE" and iter_run.modified_files and not iter_run.edit_verified:
            signal = "CONTINUE"
            last_feedback = (
                "Completion rejected by verification gate: edits in the last iteration are "
                f"UNVERIFIED ({iter_run.stop_reason}). Fix all failing tests before marking "
                "COMPLETE."
            )
        else:
            last_feedback = (
                f"Iteration {idx} modified: {', '.join(iter_run.modified_files) or 'none'} | "
                f"Stop: {iter_run.stop_reason}."
            )

        rec = GoalIterationRecord(
            iteration=idx,
            calls_made=iter_run.calls_made,
            status_signal=signal,
            modified_files=list(iter_run.modified_files),
            edit_verified=iter_run.edit_verified,
            summary=iter_run.final_response.strip(),
            elapsed_seconds=iter_run.elapsed_seconds,
            total_tokens=iter_run.usage.total_tokens or 0,
            cost_usd=iter_run.usage.cost_usd or 0.0,
        )
        iterations.append(rec)
        final_summary = rec.summary or final_summary
        if on_iteration_complete is not None:
            on_iteration_complete(rec)

        if signal == "COMPLETE":
            completed = True
            stop_reason = f"goal achieved and verified in {idx}/{bounded_iters} iteration(s)"
            break

        if signal == "BLOCKED":
            stop_reason = f"blocked at iteration {idx}/{bounded_iters}"
            break

        if not iter_run.modified_files:
            stagnant_streak += 1
            if stagnant_streak >= 2:
                stop_reason = (
                    f"stopped after {stagnant_streak} consecutive iterations with no workspace "
                    "progress (anti-stuck guard)"
                )
                break
        else:
            stagnant_streak = 0

    all_verified = all(it.edit_verified for it in iterations if it.modified_files)
    return GoalRunResult(
        objective=objective,
        model=model,
        plan_text=plan_text,
        completed=completed,
        stop_reason=stop_reason,
        iterations_run=len(iterations),
        max_iterations=bounded_iters,
        modified_files=all_modified,
        all_edits_verified=all_verified,
        iterations=iterations,
        total_calls=total_calls,
        total_tokens=total_tokens,
        total_cost_usd=total_cost,
        elapsed_seconds=time.perf_counter() - started,
        final_summary=final_summary,
    )
