"""Opt-in MCP stdio client bridging external MCP server tools into Poseidon's ToolRegistry.

Reads standard `mcpServers` config from `<workspace>/.poseidon/mcp.json` or
`<workspace>/mcp.json`, connects over stdio via the official Python `mcp` SDK in a
dedicated background asyncio loop thread, and registers each discovered tool as
`mcp_<server>_<tool>` with guaranteed clean shutdown.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from poseidon.tools.registry import ToolDefinition, ToolExecutionResult, ToolRegistry

MCP_CONFIG_CANDIDATES: tuple[str, ...] = (".poseidon/mcp.json", "mcp.json")
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_]+")


class MCPServerConfig(BaseModel):
    """Configuration for one stdio MCP server entry."""

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


@dataclass
class MCPLoadReport:
    """Summary of connected MCP servers, registered tools, and any warnings."""

    config_path: Path | None = None
    connected_servers: list[str] = field(default_factory=list)
    registered_tools: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_mcp_config(workspace_root: Path) -> tuple[Path | None, dict[str, MCPServerConfig]]:
    """Locate and parse an opt-in mcp.json configuration if present."""
    for rel in MCP_CONFIG_CANDIDATES:
        candidate = workspace_root / rel
        if not candidate.is_file():
            continue
        raw = json.loads(candidate.read_text(encoding="utf-8"))
        servers_raw = raw.get("mcpServers") or {}
        if not isinstance(servers_raw, dict):
            return candidate, {}
        parsed: dict[str, MCPServerConfig] = {}
        for name, cfg in servers_raw.items():
            if isinstance(cfg, dict) and cfg.get("command"):
                parsed[str(name)] = MCPServerConfig(**cfg)
        return candidate, parsed
    return None, {}


def _sanitize_identifier(name: str) -> str:
    return _SAFE_NAME_RE.sub("_", name).strip("_") or "tool"


def _extract_mcp_result_text(result: Any) -> str:
    """Format MCP CallToolResult content blocks into clean text for the LLM."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
        else:
            parts.append(str(block))
    return "\n".join(parts) if parts else "(empty MCP tool output)"


class MCPClientManager:
    """Lifecycle-managed sync bridge over official `mcp` stdio ClientSession instances."""

    def __init__(self, *, connect_timeout: float = 15.0, call_timeout: float = 30.0) -> None:
        self._connect_timeout = connect_timeout
        self._call_timeout = call_timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._sessions: dict[str, Any] = {}

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(target=_runner, name="poseidon-mcp-loop", daemon=True)
        thread.start()
        ready.wait(timeout=5.0)
        self._loop = loop
        self._thread = thread
        self._exit_stack = AsyncExitStack()
        return loop

    def _run_coro(self, coro: Any, *, timeout: float) -> Any:
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    def connect_and_register(
        self,
        *,
        workspace_root: Path,
        registry: ToolRegistry,
    ) -> MCPLoadReport:
        """Connect to all configured opt-in MCP servers and register their tools."""
        report = MCPLoadReport()
        try:
            config_path, servers = load_mcp_config(workspace_root)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            report.warnings.append(f"Failed to parse MCP config: {exc}")
            return report

        report.config_path = config_path
        if not servers:
            return report

        for server_name, cfg in servers.items():
            try:
                discovered = self._run_coro(
                    self._connect_server(server_name, cfg),
                    timeout=self._connect_timeout,
                )
            except Exception as exc:
                report.warnings.append(f"MCP server '{server_name}' failed to connect: {exc}")
                continue

            report.connected_servers.append(server_name)
            safe_server = _sanitize_identifier(server_name)
            for tool in discovered:
                raw_tool_name = str(tool.name)
                safe_tool = _sanitize_identifier(raw_tool_name)
                registered_name = f"mcp_{safe_server}_{safe_tool}"
                schema = dict(getattr(tool, "inputSchema", None) or {"type": "object"})
                if "type" not in schema:
                    schema["type"] = "object"
                description = str(getattr(tool, "description", "") or f"MCP tool {raw_tool_name}")

                handler = self._make_tool_handler(
                    server_name=server_name,
                    remote_tool_name=raw_tool_name,
                    registered_name=registered_name,
                )
                try:
                    registry.register(
                        ToolDefinition(
                            name=registered_name,
                            description=f"[MCP:{server_name}] {description}",
                            parameters=schema,
                            handler=handler,
                        )
                    )
                    report.registered_tools.append(registered_name)
                except ValueError as exc:
                    report.warnings.append(str(exc))

        return report

    async def _connect_server(self, server_name: str, cfg: MCPServerConfig) -> list[Any]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        assert self._exit_stack is not None
        merged_env = dict(os.environ)
        merged_env.update(cfg.env)
        params = StdioServerParameters(
            command=cfg.command,
            args=cfg.args,
            env=merged_env,
        )
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        self._sessions[server_name] = session
        tools_resp = await session.list_tools()
        return list(getattr(tools_resp, "tools", None) or [])

    def _make_tool_handler(
        self,
        *,
        server_name: str,
        remote_tool_name: str,
        registered_name: str,
    ) -> Any:
        def _handler(args: dict[str, Any]) -> ToolExecutionResult:
            session = self._sessions.get(server_name)
            if session is None:
                return ToolExecutionResult(
                    tool_name=registered_name,
                    ok=False,
                    output=f"Error: MCP server '{server_name}' is not connected.",
                    summary="MCP server disconnected",
                )
            try:
                res = self._run_coro(
                    session.call_tool(remote_tool_name, arguments=args),
                    timeout=self._call_timeout,
                )
                text = _extract_mcp_result_text(res)
                is_error = bool(
                    getattr(res, "isError", False) or getattr(res, "is_error", False)
                )
                return ToolExecutionResult(
                    tool_name=registered_name,
                    ok=not is_error,
                    output=text,
                    summary=text.splitlines()[0][:80] if text else "completed",
                )
            except Exception as exc:
                return ToolExecutionResult(
                    tool_name=registered_name,
                    ok=False,
                    output=f"Error calling MCP tool '{registered_name}': {exc}",
                    summary=f"MCP error: {str(exc)[:60]}",
                )

        return _handler

    def close(self) -> None:
        """Cleanly close all MCP stdio subprocesses and stop the background asyncio loop."""
        if self._loop is None:
            return
        if self._exit_stack is not None:
            with contextlib.suppress(Exception):
                future = asyncio.run_coroutine_threadsafe(
                    self._exit_stack.aclose(),
                    self._loop,
                )
                future.result(timeout=5.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        with contextlib.suppress(Exception):
            self._loop.close()
        self._loop = None
        self._thread = None
        self._exit_stack = None
        self._sessions.clear()

    def __enter__(self) -> MCPClientManager:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
