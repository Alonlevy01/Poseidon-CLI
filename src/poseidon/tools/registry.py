"""Extensible ToolRegistry for native OpenRouter tool calling."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from poseidon.patching.unified_diff import PatchError
from poseidon.sandbox.docker_runner import TestRunResult
from poseidon.workspace import WorkspaceSecurityError


class ToolExecutionResult(BaseModel):
    """Normalized outcome returned by a tool execution."""

    tool_name: str
    ok: bool
    output: str
    summary: str
    modified_files: list[str] = []


@dataclass(frozen=True)
class ToolDefinition:
    """Metadata, JSON Schema, and callable handler for one agent tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], ToolExecutionResult]

    def to_openrouter_schema(self) -> dict[str, Any]:
        """Serialize into the OpenRouter/OpenAI function tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolSessionState:
    """Shared per-run state for workspace tools (read hashes and test verification)."""

    read_hashes: dict[str, str] = field(default_factory=dict)
    dirty_since_last_test: bool = False
    last_test_result: TestRunResult | None = None


class ToolRegistry:
    """Central registry for built-in (and future plugin/MCP) agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition by unique name."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        """Return registered tool names in registration order."""
        return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        """Return OpenRouter-compatible JSON schemas for all registered tools."""
        return [tool.to_openrouter_schema() for tool in self._tools.values()]

    def parse_arguments(self, raw_arguments: str) -> dict[str, Any]:
        """Parse a JSON tool-argument string into a dictionary."""
        if not raw_arguments or not raw_arguments.strip():
            return {}
        parsed = json.loads(raw_arguments)
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must decode to a JSON object.")
        return parsed

    def execute(self, name: str, raw_arguments: str) -> ToolExecutionResult:
        """Validate, dispatch, and safely execute a tool call."""
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self.names()) or "(none)"
            return ToolExecutionResult(
                tool_name=name,
                ok=False,
                output=f"Error: unknown tool '{name}'. Available tools: {available}.",
                summary=f"unknown tool '{name}'",
            )

        try:
            args = self.parse_arguments(raw_arguments)
        except (json.JSONDecodeError, ValueError) as exc:
            return ToolExecutionResult(
                tool_name=name,
                ok=False,
                output=f"Error: invalid JSON arguments for '{name}': {exc}",
                summary="invalid JSON arguments",
            )

        required = tool.parameters.get("required") or []
        missing = [key for key in required if key not in args]
        if missing:
            return ToolExecutionResult(
                tool_name=name,
                ok=False,
                output=f"Error: missing required argument(s) for '{name}': {', '.join(missing)}.",
                summary=f"missing args: {', '.join(missing)}",
            )

        try:
            return tool.handler(args)
        except (WorkspaceSecurityError, PatchError, FileNotFoundError, OSError, ValueError) as exc:
            return ToolExecutionResult(
                tool_name=name,
                ok=False,
                output=f"Error: {exc}",
                summary=str(exc)[:100],
            )
