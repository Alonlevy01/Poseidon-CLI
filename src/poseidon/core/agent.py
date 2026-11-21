"""Core native tool-calling agent loop with call-limit enforcement and Respan tracing."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field

from poseidon.llm.openrouter import ChatResponse, TokenUsage
from poseidon.observability.respan import PoseidonTracer
from poseidon.tools.registry import ToolExecutionResult, ToolRegistry

AGENT_SYSTEM_PROMPT = """You are Poseidon-CLI, a precise Python coding agent inside a sandbox.

Workflow rules:
1. Use the provided tools to inspect files before making edits.
2. Always call `read_file` on a file before calling `apply_diff` on it.
3. Send clean, minimal Git unified diffs (with 3 lines of exact context, no line numbers).
4. If a tool returns an error, inspect the error and adjust your next call.
5. Once your edits are complete, summarize clearly what you changed and stop calling tools.
"""


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
    system_prompt: str = AGENT_SYSTEM_PROMPT,
) -> AgentRunResult:
    """Run the native tool-calling loop with strict call limits and Respan tracing."""
    if max_calls < 1:
        raise ValueError("max_calls must be at least 1.")

    active_tracer = tracer or PoseidonTracer(enabled=False)
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

        while calls_made < call_limit:
            calls_made += 1
            if on_model_call is not None:
                on_model_call(calls_made, call_limit)

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
                final_text = response.content.strip()
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
            stop_reason = f"reached model-call limit ({calls_made}/{call_limit})"
            if not final_text:
                final_text = (
                    f"Stopped after {calls_made} model call(s) because the model-call limit "
                    f"({call_limit}) was reached."
                )

        run_result = AgentRunResult(
            final_response=final_text,
            model=resolved_model,
            calls_made=calls_made,
            max_calls=call_limit,
            stop_reason=stop_reason,
            tool_events=tool_events,
            modified_files=modified_files,
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
