"""Starlette + WebSocket backend powering `poseidon ui` on http://localhost:2701.

Reuses the exact same Poseidon-CLI core engine, WorkspaceGuard, DockerTestRunner,
SemanticCodeIndex, SessionStore, Observation Masking, Undo Checkpoints, Plugins,
MCP client, and full terminal-parity slash commands.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from poseidon import APP_NAME, __version__
from poseidon.config import RECOMMENDED_MODELS, Settings
from poseidon.context.checkpoints import WorkspaceCheckpointManager
from poseidon.context.compactor import compact_messages
from poseidon.context.custom_commands import CustomCommandStore
from poseidon.context.memory import load_project_memory, starter_poseidon_md
from poseidon.context.mentions import expand_file_mentions
from poseidon.context.secrets_store import ProviderSecretsStore
from poseidon.context.sessions import SessionStore, generate_session_id
from poseidon.context.user_settings import (
    UserSettings,
    UserSettingsStore,
    add_mcp_server,
    list_mcp_configs,
    remove_mcp_server,
)
from poseidon.core.agent import ToolEvent, build_system_prompt, run_agent_loop
from poseidon.core.diff_edit import run_diff_edit_request
from poseidon.core.doctor import run_doctor_checks
from poseidon.core.goal import GoalIterationRecord, parse_goal_command, run_goal_loop
from poseidon.core.single_file import FileExtractionError, run_single_file_request
from poseidon.indexing.semantic_index import FastEmbedProvider, SemanticCodeIndex
from poseidon.llm.openrouter import (
    DIRECT_PROVIDER_PRESETS,
    ModelToolSupportError,
    OpenRouterClient,
    OpenRouterError,
    discover_local_models,
    normalize_local_url,
)
from poseidon.mcp.client import MCPClientManager
from poseidon.observability.respan import PoseidonTracer
from poseidon.plugins.hooks import HookManager
from poseidon.plugins.loader import load_workspace_plugins
from poseidon.sandbox.docker_runner import DockerTestRunner, TestRunResult
from poseidon.search.lexical import search_workspace_text
from poseidon.tools.builtin import register_core_tools
from poseidon.tools.registry import ToolRegistry, ToolSessionState
from poseidon.workspace import WorkspaceGuard, WorkspaceSecurityError

STATIC_DIR = Path(__file__).parent / "static"

SLASH_COMMANDS: list[dict[str, str]] = [
    {"name": "/help", "usage": "/help", "desc": "Show all available slash commands"},
    {
        "name": "/goal",
        "usage": "/goal [--steps N] <objective>",
        "desc": "Autonomous Plan -> Execute -> Verify loop until goal is achieved",
    },
    {"name": "/plan", "usage": "/plan", "desc": "Toggle read-only Plan Mode (disables writes)"},
    {"name": "/test", "usage": "/test", "desc": "Run isolated Docker sandbox pytest suite"},
    {"name": "/tools", "usage": "/tools", "desc": "Inspect active Built-in, Plugin, and MCP tools"},
    {
        "name": "/mcp",
        "usage": "/mcp [add|remove]",
        "desc": "Inspect, add, or remove stdio MCP servers (.poseidon/mcp.json)",
    },
    {
        "name": "/settings",
        "usage": "/settings",
        "desc": "Inspect persisted workspace settings (.poseidon/settings.json)",
    },
    {
        "name": "/doctor",
        "usage": "/doctor",
        "desc": "Run 1-second health check (Docker, sandbox, keys, Ollama/LM Studio, tools)",
    },
    {
        "name": "/cost",
        "usage": "/cost",
        "desc": "Show cumulative session tokens, masked savings, and total USD spend",
    },
    {
        "name": "/cmd",
        "usage": "/cmd [list|add|remove]",
        "desc": "Manage persistent Global & Workspace custom /command templates",
    },
    {
        "name": "/providers",
        "usage": "/providers",
        "desc": "Check Cloud API keys (Claude, Gemini, GPT, OpenRouter) & Local AI",
    },
    {
        "name": "/local",
        "usage": "/local",
        "desc": "Discover running local models in Ollama and LM Studio",
    },
    {"name": "/files", "usage": "/files", "desc": "List accessible workspace files"},
    {
        "name": "/search",
        "usage": "/search <query>",
        "desc": "Local semantic code search (FastEmbed)",
    },
    {"name": "/grep", "usage": "/grep <text>", "desc": "Fast lexical code search (ripgrep)"},
    {"name": "/web", "usage": "/web <on|off>", "desc": "Enable/disable web_search and fetch_url"},
    {"name": "/compact", "usage": "/compact", "desc": "Mask old tool outputs to save context"},
    {"name": "/undo", "usage": "/undo", "desc": "Undo latest file edit checkpoint"},
    {"name": "/checkpoints", "usage": "/checkpoints", "desc": "List captured pre-edit checkpoints"},
    {"name": "/memory", "usage": "/memory", "desc": "Inspect loaded POSEIDON.md / AGENTS.md"},
    {"name": "/init", "usage": "/init", "desc": "Scaffold starter POSEIDON.md in workspace"},
    {"name": "/save", "usage": "/save [name]", "desc": "Save current session to JSONL"},
    {"name": "/resume", "usage": "/resume [id]", "desc": "Resume latest or named saved session"},
    {"name": "/sessions", "usage": "/sessions", "desc": "List saved JSONL sessions in workspace"},
    {
        "name": "/model",
        "usage": "/model [add|remove|<id>]",
        "desc": "List, switch, add, or remove OpenRouter models",
    },
    {"name": "/limit", "usage": "/limit <N>", "desc": "Set max model calls per turn (1-30)"},
    {"name": "/timeout", "usage": "/timeout <sec>", "desc": "Set Docker test timeout in seconds"},
    {"name": "/auto", "usage": "/auto [on|off]", "desc": "Toggle or set auto-approve edits"},
    {"name": "/mode", "usage": "/mode <agent|diff|whole>", "desc": "Switch agent execution mode"},
    {"name": "/clear", "usage": "/clear", "desc": "Clear chat history and start a fresh session"},
    {
        "name": "/exit",
        "usage": "/exit",
        "desc": "Save session, stop Web UI, and close the Docker container",
    },
]


@dataclass
class WebSessionContext:
    """Mutable workspace state shared across REST and WebSocket endpoints for one UI server."""

    workspace: WorkspaceGuard
    settings: Settings
    model: str
    max_calls: int
    test_timeout: int
    sandbox_image: str = "poseidon-sandbox:dev"
    web_tools_enabled: bool = True
    goal_max_iterations: int = 5
    auto_close_seconds: int = 8
    ollama_base_url: str = "http://localhost:11434/v1"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    local_ai_base_url: str = ""
    mode: str = "agent"
    plan_mode: bool = False
    auto_approve: bool = False
    session_id: str = field(default_factory=generate_session_id)
    history: list[dict[str, Any]] = field(default_factory=list)
    tool_state: ToolSessionState = field(default_factory=ToolSessionState)
    checkpoints: WorkspaceCheckpointManager = field(init=False)
    session_store: SessionStore = field(init=False)
    user_settings_store: UserSettingsStore = field(init=False)
    custom_commands: CustomCommandStore = field(init=False)
    secrets_store: ProviderSecretsStore = field(default_factory=ProviderSecretsStore)
    run_lock: threading.Lock = field(default_factory=threading.Lock)
    goal_cancel_event: threading.Event = field(default_factory=threading.Event)
    session_turns: int = 0
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    session_reasoning_tokens: int = 0
    session_total_tokens: int = 0
    session_cost_usd: float = 0.0
    session_tokens_masked: int = 0

    def __post_init__(self) -> None:
        self.checkpoints = WorkspaceCheckpointManager(self.workspace)
        self.session_store = SessionStore(self.workspace.root)
        self.user_settings_store = UserSettingsStore(self.workspace.root)
        self.custom_commands = CustomCommandStore(self.workspace.root)

    def record_usage(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
        tokens_masked: int = 0,
    ) -> None:
        """Accumulate session-level token and cost telemetry."""
        self.session_turns += 1
        self.session_prompt_tokens += prompt_tokens or 0
        self.session_completion_tokens += completion_tokens or 0
        self.session_reasoning_tokens += reasoning_tokens or 0
        self.session_total_tokens += total_tokens or 0
        self.session_cost_usd += cost_usd or 0.0
        self.session_tokens_masked += tokens_masked or 0

    def persist_user_settings(self) -> UserSettings:
        """Persist mutable workspace preferences to `.poseidon/settings.json`."""
        existing = self.user_settings_store.load()
        updated = UserSettings(
            default_model=self.model,
            custom_models=existing.custom_models,
            max_calls=self.max_calls,
            test_timeout=self.test_timeout,
            sandbox_image=self.sandbox_image,
            web_tools_enabled=self.web_tools_enabled,
            goal_max_iterations=self.goal_max_iterations,
            auto_close_seconds=self.auto_close_seconds,
            ollama_base_url=self.ollama_base_url,
            lmstudio_base_url=self.lmstudio_base_url,
        )
        return self.user_settings_store.save(updated)

    def snapshot_state(self) -> dict[str, Any]:
        """Build JSON-serializable workspace & session metadata for the Web UI."""
        memory = load_project_memory(self.workspace.root)
        user_cfg = self.user_settings_store.load()
        context_chars = sum(len(str(m.get("content") or "")) for m in self.history)
        host_ws = os.environ.get("POSEIDON_HOST_WORKSPACE") or str(self.workspace.root)
        folder_name = host_ws.rstrip("\\/").replace("\\", "/").split("/")[-1] or host_ws
        recommended = [
            {"id": mid, "notes": desc, "custom": False} for mid, desc in RECOMMENDED_MODELS
        ]
        seen_ids = {m["id"] for m in recommended}
        for custom_id in user_cfg.custom_models:
            if custom_id not in seen_ids:
                recommended.append(
                    {"id": custom_id, "notes": "Custom OpenRouter model", "custom": True}
                )
                seen_ids.add(custom_id)
        mcp_configs = list_mcp_configs(self.workspace.root)
        saved_custom_cmds = self.custom_commands.list_commands()
        merged_slash = [
            *SLASH_COMMANDS,
            *(
                {
                    "name": c.slash_name,
                    "usage": f"{c.slash_name} [args]",
                    "desc": f"[{c.scope}] {c.description}",
                }
                for c in saved_custom_cmds
            ),
        ]
        return {
            "app_name": APP_NAME,
            "version": __version__,
            "workspace": host_ws,
            "workspace_folder": folder_name,
            "model": self.model,
            "mode": self.mode,
            "recommended_models": recommended,
            "custom_models": user_cfg.custom_models,
            "plan_mode": self.plan_mode,
            "auto_approve": self.auto_approve,
            "max_calls": self.max_calls,
            "test_timeout": self.test_timeout,
            "sandbox_image": self.sandbox_image,
            "web_tools_enabled": self.web_tools_enabled,
            "goal_max_iterations": self.goal_max_iterations,
            "auto_close_seconds": self.auto_close_seconds,
            "ollama_base_url": self.ollama_base_url,
            "lmstudio_base_url": self.lmstudio_base_url,
            "mcp_server_count": len(mcp_configs),
            "session_id": self.session_id,
            "memory_sources": memory.sources,
            "files": self.workspace.list_files(),
            "sessions": [s.model_dump() for s in self.session_store.list_sessions()[:15]],
            "checkpoints": [
                {"id": cp.id, "label": cp.label, "created_at": cp.created_at}
                for cp in self.checkpoints.list_checkpoints()
            ],
            "context_chars": context_chars,
            "message_count": sum(1 for m in self.history if m.get("role") != "system"),
            "session_stats": {
                "turns": self.session_turns,
                "prompt_tokens": self.session_prompt_tokens,
                "completion_tokens": self.session_completion_tokens,
                "reasoning_tokens": self.session_reasoning_tokens,
                "total_tokens": self.session_total_tokens,
                "cost_usd": round(self.session_cost_usd, 6),
                "tokens_masked": self.session_tokens_masked,
            },
            "custom_commands": [c.model_dump() for c in saved_custom_cmds],
            "slash_commands": merged_slash,
        }

    def collect_tools_inventory(self) -> dict[str, Any]:
        """Collect all currently registered Built-in, Plugin, and MCP tools."""
        registry = ToolRegistry()
        hooks = HookManager()
        register_core_tools(
            registry=registry,
            workspace=self.workspace,
            state=ToolSessionState(),
            confirm_patch=lambda _: False,
            test_runner=DockerTestRunner(
                image=self.sandbox_image,
                timeout_seconds=self.test_timeout,
            ),
            semantic_index=SemanticCodeIndex(
                self.workspace,
                FastEmbedProvider(
                    model_name=self.settings.embedding_model,
                    cache_dir=self.settings.embedding_cache_dir,
                ),
            ),
            hooks=hooks,
            read_only=self.plan_mode,
            enable_web_tools=self.web_tools_enabled,
            tavily_api_key=self.settings.tavily_api_key.get_secret_value(),
            brave_api_key=self.settings.brave_api_key.get_secret_value(),
        )
        plugin_report = load_workspace_plugins(
            workspace=self.workspace,
            registry=registry,
            hooks=hooks,
        )
        plugin_names = set(plugin_report.registered_tools)
        mcp_manager = MCPClientManager()
        try:
            mcp_report = mcp_manager.connect_and_register(
                workspace_root=self.workspace.root,
                registry=registry,
            )
            mcp_names = set(mcp_report.registered_tools)
            write_tools = {"apply_diff", "create_file"}
            tools = []
            for schema in registry.schemas():
                fn = schema.get("function", {})
                name = str(fn.get("name") or "")
                source = (
                    "mcp"
                    if name in mcp_names
                    else ("plugin" if name in plugin_names else "built-in")
                )
                tools.append(
                    {
                        "name": name,
                        "source": source,
                        "read_only": name not in write_tools,
                        "description": str(fn.get("description") or ""),
                    }
                )
        finally:
            mcp_manager.close()

        return {
            "plan_mode": self.plan_mode,
            "tools": tools,
            "plugin_files": plugin_report.loaded_plugins,
            "mcp_servers": mcp_report.connected_servers,
            "warnings": [*plugin_report.warnings, *mcp_report.warnings],
        }


class _WebSocketBridge:
    """Thread-safe bridge pushing agent progress events and awaiting diff approvals via WS."""

    def __init__(self, loop: asyncio.AbstractEventLoop, websocket: WebSocket) -> None:
        self._loop = loop
        self._ws = websocket
        self._pending: dict[str, asyncio.Future[bool]] = {}

    def emit(self, payload: dict[str, Any]) -> None:
        """Fire-and-forget JSON event from background agent thread to browser."""
        try:
            fut = asyncio.run_coroutine_threadsafe(self._ws.send_json(payload), self._loop)
            fut.result(timeout=5.0)
        except Exception:
            pass

    def request_diff_approval(self, diff_text: str) -> bool:
        """Send diff preview to UI and block agent thread until user clicks Approve or Reject."""
        approval_id = uuid.uuid4().hex[:10]

        async def _ask() -> bool:
            fut: asyncio.Future[bool] = self._loop.create_future()
            self._pending[approval_id] = fut
            await self._ws.send_json(
                {
                    "type": "diff_approval_request",
                    "approval_id": approval_id,
                    "diff": diff_text,
                }
            )
            try:
                return await fut
            finally:
                self._pending.pop(approval_id, None)

        try:
            return asyncio.run_coroutine_threadsafe(_ask(), self._loop).result(timeout=300.0)
        except Exception:
            return False

    def resolve_approval(self, approval_id: str, approved: bool) -> None:
        """Fulfill a pending approval future from the ASGI WebSocket receive loop."""
        fut = self._pending.get(approval_id)
        if fut is not None and not fut.done():
            fut.set_result(approved)

    def cancel_all(self) -> None:
        """Unblock any waiting approval threads if the browser disconnects."""
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_result(False)
        self._pending.clear()


def _visible_transcript(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    for m in messages:
        role = str(m.get("role") or "")
        if role not in ("user", "assistant"):
            continue
        raw = m.get("content")
        if isinstance(raw, list):
            texts = [
                str(p.get("text") or "")
                for p in raw
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            imgs = [
                str((p.get("image_url") or {}).get("url") or "")
                for p in raw
                if isinstance(p, dict) and p.get("type") == "image_url"
            ]
            joined = "\n".join(t for t in texts if t.strip()).strip()
            if joined or imgs:
                visible.append({"role": role, "content": joined, "images": imgs})
        else:
            text = str(raw or "")
            if text.strip():
                visible.append({"role": role, "content": text})
    return visible


def _execute_slash_command(ctx: WebSessionContext, raw_command: str) -> dict[str, Any]:
    """Execute any terminal-parity `/command` synchronously (designed to run via to_thread)."""
    text = raw_command.strip()
    parts = text.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) == 2 else ""

    if cmd == "/help":
        return {
            "ok": True,
            "command": cmd,
            "title": "Poseidon Slash Commands (100% Terminal Parity)",
            "kind": "table",
            "columns": ["Command", "Description"],
            "rows": [[c["usage"], c["desc"]] for c in SLASH_COMMANDS],
            "state": ctx.snapshot_state(),
        }

    if cmd == "/plan":
        if arg.lower() in {"on", "true", "1"}:
            ctx.plan_mode = True
        elif arg.lower() in {"off", "false", "0"}:
            ctx.plan_mode = False
        else:
            ctx.plan_mode = not ctx.plan_mode
        msg = (
            "Plan Mode ON — read-only; write tools are physically removed."
            if ctx.plan_mode
            else "Plan Mode OFF — edit tools enabled."
        )
        return {
            "ok": True,
            "command": cmd,
            "title": "Plan Mode",
            "message": msg,
            "state": ctx.snapshot_state(),
        }

    if cmd == "/files":
        files = ctx.workspace.list_files()
        return {
            "ok": True,
            "command": cmd,
            "title": f"Workspace Files ({len(files)})",
            "kind": "files",
            "files": files,
            "message": f"{len(files)} accessible file(s) in workspace." if files else "Empty.",
            "state": ctx.snapshot_state(),
        }

    if cmd == "/tools":
        inv = ctx.collect_tools_inventory()
        rows = [
            [
                t["name"],
                t["source"],
                "read-only" if t["read_only"] else "write",
                t["description"],
            ]
            for t in inv["tools"]
        ]
        note = "Plan Mode active (write tools omitted)." if inv["plan_mode"] else ""
        return {
            "ok": True,
            "command": cmd,
            "title": f"Active Tools ({len(rows)})",
            "kind": "table",
            "columns": ["Tool", "Source", "Access", "Description"],
            "rows": rows,
            "message": note,
            "warnings": inv["warnings"],
            "state": ctx.snapshot_state(),
        }

    if cmd == "/memory":
        memory = load_project_memory(ctx.workspace.root)
        if not memory.sources:
            return {
                "ok": True,
                "command": cmd,
                "title": "Project Memory",
                "message": "No POSEIDON.md or AGENTS.md found. Run `/init` to create one.",
                "state": ctx.snapshot_state(),
            }
        return {
            "ok": True,
            "command": cmd,
            "title": f"Project Memory ({', '.join(memory.sources)})",
            "message": f"Loaded {len(memory.content)} chars from {', '.join(memory.sources)}.",
            "code": memory.content,
            "state": ctx.snapshot_state(),
        }

    if cmd == "/init":
        target = ctx.workspace.root / "POSEIDON.md"
        if target.exists():
            return {
                "ok": True,
                "command": cmd,
                "title": "Project Memory",
                "message": "POSEIDON.md already exists in this workspace.",
                "state": ctx.snapshot_state(),
            }
        ctx.workspace.write_text("POSEIDON.md", starter_poseidon_md(ctx.workspace.root.name))
        return {
            "ok": True,
            "command": cmd,
            "title": "Project Memory",
            "message": "Created POSEIDON.md in workspace.",
            "state": ctx.snapshot_state(),
        }

    if cmd == "/sessions":
        saved = ctx.session_store.list_sessions()
        if not saved:
            return {
                "ok": True,
                "command": cmd,
                "title": "Saved Sessions",
                "message": "No saved sessions in this workspace yet.",
                "state": ctx.snapshot_state(),
            }
        return {
            "ok": True,
            "command": cmd,
            "title": f"Saved Sessions ({len(saved)})",
            "kind": "table",
            "columns": ["Session ID", "Messages", "Model", "Title"],
            "rows": [
                [s.session_id, str(s.message_count), s.model, s.title]
                for s in saved[:15]
            ],
            "state": ctx.snapshot_state(),
        }

    if cmd == "/save":
        custom_id = arg or ctx.session_id
        try:
            saved_meta = ctx.session_store.save(
                session_id=custom_id,
                model=ctx.model,
                messages=ctx.history,
                plan_mode=ctx.plan_mode,
            )
            ctx.session_id = saved_meta.session_id
        except ValueError as exc:
            return {
                "ok": False,
                "command": cmd,
                "title": "Save Failed",
                "message": str(exc),
                "state": ctx.snapshot_state(),
            }
        return {
            "ok": True,
            "command": cmd,
            "title": "Session Saved",
            "message": (
                f"Saved session `{saved_meta.session_id}` ({saved_meta.message_count} messages)."
            ),
            "state": ctx.snapshot_state(),
        }

    if cmd == "/resume":
        target_id = arg or ctx.session_store.latest_session_id()
        if not target_id:
            return {
                "ok": False,
                "command": cmd,
                "title": "Resume Session",
                "message": "No saved sessions to resume.",
                "state": ctx.snapshot_state(),
            }
        try:
            loaded = ctx.session_store.load(target_id)
        except (FileNotFoundError, ValueError) as exc:
            return {
                "ok": False,
                "command": cmd,
                "title": "Resume Failed",
                "message": str(exc),
                "state": ctx.snapshot_state(),
            }
        ctx.session_id = loaded.summary.session_id
        ctx.history = loaded.messages
        ctx.plan_mode = loaded.summary.plan_mode
        if loaded.summary.model != "unknown":
            ctx.model = loaded.summary.model
        return {
            "ok": True,
            "command": cmd,
            "title": "Session Resumed",
            "message": (
                f"Resumed `{ctx.session_id}` ({loaded.summary.message_count} messages)."
            ),
            "hydrate_messages": _visible_transcript(loaded.messages),
            "state": ctx.snapshot_state(),
        }

    if cmd == "/clear":
        ctx.history.clear()
        ctx.session_id = generate_session_id()
        return {
            "ok": True,
            "command": cmd,
            "title": "Session Cleared",
            "message": f"Cleared history. New session: `{ctx.session_id}`.",
            "clear_stream": True,
            "state": ctx.snapshot_state(),
        }

    if cmd == "/compact":
        _, stats = compact_messages(
            ctx.history,
            keep_recent_tool_outputs=2,
            min_chars_to_mask=150,
            in_place=True,
        )
        return {
            "ok": True,
            "command": cmd,
            "title": "Context Compacted",
            "message": (
                f"Masked {stats.masked_count} old tool output(s), saved {stats.chars_saved} chars "
                f"(~{stats.estimated_tokens_saved} tokens; "
                f"{stats.chars_before} -> {stats.chars_after} chars)."
            ),
            "state": ctx.snapshot_state(),
        }

    if cmd == "/checkpoints":
        cps = ctx.checkpoints.list_checkpoints()
        if not cps:
            return {
                "ok": True,
                "command": cmd,
                "title": "Workspace Checkpoints",
                "message": "No edit checkpoints captured yet.",
                "state": ctx.snapshot_state(),
            }
        return {
            "ok": True,
            "command": cmd,
            "title": f"Workspace Checkpoints ({len(cps)})",
            "kind": "table",
            "columns": ["ID", "Timestamp", "Label"],
            "rows": [[f"#{cp.id}", cp.created_at, cp.label] for cp in cps],
            "state": ctx.snapshot_state(),
        }

    if cmd == "/undo":
        cp, restored = ctx.checkpoints.undo()
        if cp is None:
            return {
                "ok": True,
                "command": cmd,
                "title": "Undo",
                "message": "Nothing to undo (checkpoint stack is empty).",
                "state": ctx.snapshot_state(),
            }
        ctx.tool_state.read_hashes.clear()
        ctx.tool_state.dirty_since_last_test = True
        return {
            "ok": True,
            "command": cmd,
            "title": f"Undid Checkpoint #{cp.id}",
            "message": f"Restored ({cp.label}): {', '.join(restored)}",
            "state": ctx.snapshot_state(),
        }

    if cmd == "/test":
        runner = DockerTestRunner(
            image=ctx.settings.sandbox_image,
            timeout_seconds=ctx.test_timeout,
        )
        res = runner.run(ctx.workspace.root)
        return {
            "ok": res.verified,
            "command": cmd,
            "title": "Isolated Unit Test Report",
            "test_result": res.model_dump(),
            "state": ctx.snapshot_state(),
        }

    if cmd == "/search":
        if not arg:
            return {
                "ok": False,
                "command": cmd,
                "title": "Semantic Search",
                "message": "Usage: /search <plain-English query>",
                "state": ctx.snapshot_state(),
            }
        index = SemanticCodeIndex(
            ctx.workspace,
            FastEmbedProvider(
                model_name=ctx.settings.embedding_model,
                cache_dir=ctx.settings.embedding_cache_dir,
            ),
        )
        res = index.search(arg, top_k=5, auto_sync=True)
        return {
            "ok": True,
            "command": cmd,
            "title": f"Semantic Search: {arg}",
            "kind": "search_hits",
            "message": (
                f"{len(res.hits)} hit(s) in {res.lookup_ms:.2f}ms "
                f"({res.sync_stats.total_chunks} indexed chunks)."
            ),
            "hits": [
                {
                    "path": h.path,
                    "location": f"{h.path}:{h.start_line}-{h.end_line}",
                    "symbol": h.symbol,
                    "detail": f"{h.kind} | score {h.score:.3f}",
                }
                for h in res.hits
            ],
            "state": ctx.snapshot_state(),
        }

    if cmd == "/grep":
        if not arg:
            return {
                "ok": False,
                "command": cmd,
                "title": "Lexical Search (ripgrep)",
                "message": "Usage: /grep <text or pattern>",
                "state": ctx.snapshot_state(),
            }
        res = search_workspace_text(workspace=ctx.workspace, query=arg, max_results=25)
        return {
            "ok": True,
            "command": cmd,
            "title": f"Grep ({res.engine}): {arg}",
            "kind": "search_hits",
            "message": f"{len(res.hits)} match(es) in {res.elapsed_ms:.2f}ms.",
            "hits": [
                {
                    "path": h.path,
                    "location": f"{h.path}:{h.line_number}",
                    "symbol": h.line_text.strip()[:100],
                    "detail": res.engine,
                }
                for h in res.hits
            ],
            "state": ctx.snapshot_state(),
        }

    if cmd == "/model":
        if not arg:
            snap = ctx.snapshot_state()
            return {
                "ok": True,
                "command": cmd,
                "title": f"Active Model: {ctx.model}",
                "kind": "table",
                "columns": ["Model ID", "Notes"],
                "rows": [[m["id"], m["notes"]] for m in snap["recommended_models"]],
                "message": (
                    "Usage: `/model <id>`, `/model add <openrouter_id>`, "
                    "or `/model remove <id>`."
                ),
                "state": snap,
            }
        sub_parts = arg.split(maxsplit=1)
        sub = sub_parts[0].lower()
        if sub == "add" and len(sub_parts) == 2:
            try:
                ctx.user_settings_store.add_custom_model(sub_parts[1])
                ctx.model = sub_parts[1].strip()
                ctx.persist_user_settings()
            except ValueError as exc:
                return {
                    "ok": False,
                    "command": cmd,
                    "title": "Add Model Failed",
                    "message": str(exc),
                    "state": ctx.snapshot_state(),
                }
            return {
                "ok": True,
                "command": cmd,
                "title": "Custom Model Added",
                "message": f"Added and activated `{ctx.model}` in workspace settings.",
                "state": ctx.snapshot_state(),
            }
        if sub == "remove" and len(sub_parts) == 2:
            target_m = sub_parts[1].strip()
            ctx.user_settings_store.remove_custom_model(target_m)
            return {
                "ok": True,
                "command": cmd,
                "title": "Custom Model Removed",
                "message": f"Removed `{target_m}` from custom models.",
                "state": ctx.snapshot_state(),
            }
        ctx.model = arg
        ctx.persist_user_settings()
        return {
            "ok": True,
            "command": cmd,
            "title": "Model Switched",
            "message": f"Active model set to `{ctx.model}`.",
            "state": ctx.snapshot_state(),
        }

    if cmd == "/settings":
        snap = ctx.snapshot_state()
        return {
            "ok": True,
            "command": cmd,
            "title": "Workspace Settings (.poseidon/settings.json)",
            "kind": "table",
            "columns": ["Setting", "Value"],
            "rows": [
                ["Active Model", ctx.model],
                ["Custom Models", ", ".join(snap["custom_models"]) or "(none)"],
                ["Execution Mode", ctx.mode],
                ["Plan Mode (read-only)", "ON" if ctx.plan_mode else "OFF"],
                ["Auto-Approve Edits", "ON" if ctx.auto_approve else "OFF"],
                ["Max Model Calls / Turn", str(ctx.max_calls)],
                ["Docker Test Timeout", f"{ctx.test_timeout}s"],
                ["Sandbox Image", ctx.sandbox_image],
                ["Web Search & Fetch Tools", "ON" if ctx.web_tools_enabled else "OFF"],
                ["Goal Max Iterations", str(ctx.goal_max_iterations)],
                ["MCP Servers Configured", str(snap["mcp_server_count"])],
            ],
            "state": snap,
        }

    if cmd == "/doctor":
        checks = run_doctor_checks(
            workspace_root=ctx.workspace.root,
            settings=ctx.settings,
            secrets_store=ctx.secrets_store,
        )
        return {
            "ok": all(c.ok for c in checks if c.status not in {"OFFLINE", "NONE", "WARN"}),
            "command": cmd,
            "title": "Poseidon Doctor — Environment & Provider Health",
            "kind": "table",
            "columns": ["Check", "Status", "Details"],
            "rows": [[c.name, c.status, c.detail] for c in checks],
            "state": ctx.snapshot_state(),
        }

    if cmd == "/cost":
        snap = ctx.snapshot_state()
        st = snap["session_stats"]
        return {
            "ok": True,
            "command": cmd,
            "title": "Session Token & Cost Telemetry",
            "kind": "table",
            "columns": ["Metric", "Session Total"],
            "rows": [
                ["Completed Turns", str(st["turns"])],
                ["Prompt Tokens", f"{st['prompt_tokens']:,}"],
                ["Completion Tokens", f"{st['completion_tokens']:,}"],
                ["Reasoning Tokens", f"{st['reasoning_tokens']:,}"],
                ["Total Tokens", f"{st['total_tokens']:,}"],
                ["Tokens Saved (Observation Masking)", f"~{st['tokens_masked']:,}"],
                ["Total Cost (USD)", f"${st['cost_usd']:.6f}"],
            ],
            "state": snap,
        }

    if cmd == "/cmd":
        if not arg or arg.lower() == "list":
            cmds = ctx.custom_commands.list_commands()
            if not cmds:
                return {
                    "ok": True,
                    "command": cmd,
                    "title": "Custom Slash Commands (Global & Workspace)",
                    "message": (
                        "No custom commands saved yet. Create one in Settings -> Commands or via "
                        "`/cmd add --global <name> <prompt template>`."
                    ),
                    "state": ctx.snapshot_state(),
                }
            return {
                "ok": True,
                "command": cmd,
                "title": f"Custom Slash Commands ({len(cmds)})",
                "kind": "table",
                "columns": ["Command", "Scope", "Description"],
                "rows": [[c.slash_name, c.scope, c.description] for c in cmds],
                "message": "Global commands persist across EVERY project & session!",
                "state": ctx.snapshot_state(),
            }
        parts_cmd = arg.split()
        sub_c = parts_cmd[0].lower()
        if sub_c == "add" and len(parts_cmd) >= 3:
            scope = "global"
            idx = 1
            if parts_cmd[idx].lower() in {"--global", "--workspace"}:
                scope = parts_cmd[idx][2:].lower()
                idx += 1
            if len(parts_cmd) <= idx + 1:
                return {
                    "ok": False,
                    "command": cmd,
                    "title": "Add Custom Command",
                    "message": "Usage: `/cmd add [--global|--workspace] <name> <prompt template>`",
                    "state": ctx.snapshot_state(),
                }
            cname = parts_cmd[idx]
            template = " ".join(parts_cmd[idx + 1 :])
            try:
                saved_c = ctx.custom_commands.save(name=cname, template=template, scope=scope)
            except ValueError as exc:
                return {
                    "ok": False,
                    "command": cmd,
                    "title": "Add Custom Command Failed",
                    "message": str(exc),
                    "state": ctx.snapshot_state(),
                }
            return {
                "ok": True,
                "command": cmd,
                "title": "Custom Command Saved",
                "message": (
                    f"Saved `{saved_c.slash_name}` ({saved_c.scope}). "
                    "Type it anytime in Web UI or Terminal!"
                ),
                "state": ctx.snapshot_state(),
            }
        if sub_c == "remove" and len(parts_cmd) >= 2:
            scope = "global"
            idx = 1
            if parts_cmd[idx].lower() in {"--global", "--workspace"}:
                scope = parts_cmd[idx][2:].lower()
                idx += 1
            if idx >= len(parts_cmd):
                return {
                    "ok": False,
                    "command": cmd,
                    "title": "Remove Custom Command",
                    "message": "Usage: `/cmd remove [--global|--workspace] <name>`",
                    "state": ctx.snapshot_state(),
                }
            cname = parts_cmd[idx]
            removed_c = ctx.custom_commands.delete(name=cname, scope=scope)
            return {
                "ok": removed_c,
                "command": cmd,
                "title": "Custom Command Removed" if removed_c else "Not Found",
                "message": (
                    f"Removed `/{cname.lstrip('/')}` ({scope})."
                    if removed_c
                    else f"No {scope} command named `/{cname.lstrip('/')}` found."
                ),
                "state": ctx.snapshot_state(),
            }
        return {
            "ok": False,
            "command": cmd,
            "title": "Custom Commands Usage",
            "message": (
                "Usage: `/cmd list`, `/cmd add [--global|--workspace] <name> <template>`, "
                "or `/cmd remove [--global|--workspace] <name>`."
            ),
            "state": ctx.snapshot_state(),
        }

    if cmd == "/providers":
        key_status = ctx.secrets_store.masked_status(ctx.settings)
        local_info = discover_local_models(
            ollama_base_url=ctx.ollama_base_url,
            lmstudio_base_url=ctx.lmstudio_base_url,
        )
        rows = [
            [
                info["label"],
                "configured (" + info["source"] + ")" if info["configured"] else "not set",
                info["hint"] or info["env_var"],
            ]
            for info in key_status.values()
        ]
        for pname, label in (("ollama", "Ollama (Local)"), ("lmstudio", "LM Studio (Local)")):
            li = local_info[pname]
            rows.append(
                [
                    label,
                    f"online ({len(li['models'])} models)" if li["reachable"] else "offline",
                    li["base_url"],
                ]
            )
        return {
            "ok": True,
            "command": cmd,
            "title": "AI Providers & Local AI Status",
            "kind": "table",
            "columns": ["Provider", "Status", "Key / Endpoint"],
            "rows": rows,
            "message": "Configure keys & local endpoints in Settings -> Providers & Local AI.",
            "state": ctx.snapshot_state(),
        }

    if cmd == "/local":
        local_info = discover_local_models(
            ollama_base_url=ctx.ollama_base_url,
            lmstudio_base_url=ctx.lmstudio_base_url,
        )
        rows = []
        for pname in ("ollama", "lmstudio"):
            for mid in local_info[pname]["models"]:
                rows.append([mid, pname, local_info[pname]["base_url"]])
        if not rows:
            return {
                "ok": True,
                "command": cmd,
                "title": "Local AI Discovery (Ollama & LM Studio)",
                "message": (
                    f"No running local models detected at Ollama (`{ctx.ollama_base_url}`) "
                    f"or LM Studio (`{ctx.lmstudio_base_url}`). Start Ollama or LM Studio "
                    "server on your host machine and run `/local` again."
                ),
                "state": ctx.snapshot_state(),
            }
        return {
            "ok": True,
            "command": cmd,
            "title": f"Detected Local AI Models ({len(rows)})",
            "kind": "table",
            "columns": ["Model Slug", "Runtime", "Endpoint"],
            "rows": rows,
            "message": "Run `/model add <slug>` or click `+ Add` in Settings -> Providers.",
            "state": ctx.snapshot_state(),
        }

    if cmd == "/web":
        if arg.lower() == "on":
            ctx.web_tools_enabled = True
        elif arg.lower() == "off":
            ctx.web_tools_enabled = False
        elif not arg:
            ctx.web_tools_enabled = not ctx.web_tools_enabled
        else:
            return {
                "ok": False,
                "command": cmd,
                "title": "Web Tools",
                "message": "Usage: /web [on|off]",
                "state": ctx.snapshot_state(),
            }
        ctx.persist_user_settings()
        return {
            "ok": True,
            "command": cmd,
            "title": "Web Search & Fetch Tools",
            "message": (
                f"`web_search` & `fetch_url` are now "
                f"{'ENABLED' if ctx.web_tools_enabled else 'DISABLED'}."
            ),
            "state": ctx.snapshot_state(),
        }

    if cmd == "/mcp":
        if not arg:
            configs = list_mcp_configs(ctx.workspace.root)
            inv = ctx.collect_tools_inventory()
            if not configs:
                return {
                    "ok": True,
                    "command": cmd,
                    "title": "MCP Servers (.poseidon/mcp.json)",
                    "message": (
                        "No MCP servers configured yet. Add one via Settings → MCP Servers "
                        "or `/mcp add <name> <command> [args...]`."
                    ),
                    "state": ctx.snapshot_state(),
                }
            connected = set(inv["mcp_servers"])
            rows = [
                [
                    name,
                    "connected" if name in connected else "configured",
                    f"{entry.get('command', '')} {' '.join(entry.get('args', []))}".strip(),
                ]
                for name, entry in configs.items()
            ]
            return {
                "ok": True,
                "command": cmd,
                "title": f"MCP Servers ({len(rows)})",
                "kind": "table",
                "columns": ["Server", "Status", "Command"],
                "rows": rows,
                "warnings": inv["warnings"],
                "state": ctx.snapshot_state(),
            }
        try:
            tokens = shlex.split(arg)
        except ValueError as exc:
            return {
                "ok": False,
                "command": cmd,
                "title": "MCP Command Error",
                "message": str(exc),
                "state": ctx.snapshot_state(),
            }
        sub = tokens[0].lower()
        if sub == "add" and len(tokens) >= 3:
            try:
                add_mcp_server(
                    ctx.workspace.root,
                    name=tokens[1],
                    command=tokens[2],
                    args=tokens[3:],
                )
            except ValueError as exc:
                return {
                    "ok": False,
                    "command": cmd,
                    "title": "Add MCP Server Failed",
                    "message": str(exc),
                    "state": ctx.snapshot_state(),
                }
            return {
                "ok": True,
                "command": cmd,
                "title": "MCP Server Added",
                "message": f"Added stdio MCP server `{tokens[1]}` to `.poseidon/mcp.json`.",
                "state": ctx.snapshot_state(),
            }
        if sub == "remove" and len(tokens) == 2:
            removed, _ = remove_mcp_server(ctx.workspace.root, tokens[1])
            return {
                "ok": removed,
                "command": cmd,
                "title": "MCP Server Removed" if removed else "MCP Server Not Found",
                "message": (
                    f"Removed MCP server `{tokens[1]}`."
                    if removed
                    else f"No MCP server named `{tokens[1]}` found."
                ),
                "state": ctx.snapshot_state(),
            }
        return {
            "ok": False,
            "command": cmd,
            "title": "MCP Usage",
            "message": (
                "Usage: `/mcp`, `/mcp add <name> <command> [args...]`, "
                "or `/mcp remove <name>`."
            ),
            "state": ctx.snapshot_state(),
        }

    if cmd == "/limit":
        if not arg.isdigit() or int(arg) < 1:
            return {
                "ok": False,
                "command": cmd,
                "title": "Call Limit",
                "message": "Usage: /limit <positive_integer> (1-30)",
                "state": ctx.snapshot_state(),
            }
        ctx.max_calls = max(1, min(int(arg), 30))
        ctx.persist_user_settings()
        return {
            "ok": True,
            "command": cmd,
            "title": "Call Limit Updated",
            "message": f"Model-call limit set to {ctx.max_calls}.",
            "state": ctx.snapshot_state(),
        }

    if cmd == "/timeout":
        if not arg.isdigit() or int(arg) < 1:
            return {
                "ok": False,
                "command": cmd,
                "title": "Test Timeout",
                "message": "Usage: /timeout <seconds> (1-300)",
                "state": ctx.snapshot_state(),
            }
        ctx.test_timeout = max(1, min(int(arg), 300))
        ctx.persist_user_settings()
        return {
            "ok": True,
            "command": cmd,
            "title": "Test Timeout Updated",
            "message": f"Docker sandbox test timeout set to {ctx.test_timeout}s.",
            "state": ctx.snapshot_state(),
        }

    if cmd == "/auto":
        if arg.lower() == "on":
            ctx.auto_approve = True
        elif arg.lower() == "off":
            ctx.auto_approve = False
        elif not arg:
            ctx.auto_approve = not ctx.auto_approve
        else:
            return {
                "ok": False,
                "command": cmd,
                "title": "Auto-Approve",
                "message": "Usage: /auto [on|off]",
                "state": ctx.snapshot_state(),
            }
        return {
            "ok": True,
            "command": cmd,
            "title": "Auto-Approve Edits",
            "message": f"Auto-approve edits: {'ON' if ctx.auto_approve else 'OFF'}.",
            "state": ctx.snapshot_state(),
        }

    if cmd == "/mode":
        if arg.lower() not in {"agent", "diff", "whole"}:
            return {
                "ok": False,
                "command": cmd,
                "title": "Execution Mode",
                "message": f"Current mode: `{ctx.mode}`. Usage: /mode <agent|diff|whole>",
                "state": ctx.snapshot_state(),
            }
        ctx.mode = arg.lower()
        return {
            "ok": True,
            "command": cmd,
            "title": "Mode Switched",
            "message": f"Execution mode switched to `{ctx.mode}`.",
            "state": ctx.snapshot_state(),
        }

    if cmd in {"/exit", "/quit"}:
        if ctx.history:
            with contextlib.suppress(ValueError):
                ctx.session_store.save(
                    session_id=ctx.session_id,
                    model=ctx.model,
                    messages=ctx.history,
                    plan_mode=ctx.plan_mode,
                )
        return {
            "ok": True,
            "command": cmd,
            "title": "Shutting Down Poseidon",
            "message": "Session saved. Stopping Web UI and closing Docker container...",
            "shutdown": True,
            "state": ctx.snapshot_state(),
        }

    return {
        "ok": False,
        "command": cmd,
        "title": "Unknown Command",
        "message": f"Unknown command `{cmd}`. Type `/help` to see all available commands.",
        "state": ctx.snapshot_state(),
    }


def create_web_app(
    *,
    workspace: Path,
    settings: Settings | None = None,
    model: str | None = None,
    plan_mode: bool = False,
    auto_approve: bool = False,
    max_calls: int | None = None,
    test_timeout: int | None = None,
    resume_session_id: str | None = None,
    auto_shutdown_seconds: float | None = None,
    on_shutdown: Callable[[], None] | None = None,
) -> Starlette:
    """Create the Starlette ASGI application serving Poseidon Web UI."""
    cfg = settings or Settings()
    guard = WorkspaceGuard(workspace)
    saved_user_cfg = UserSettingsStore(guard.root).load()
    ctx = WebSessionContext(
        workspace=guard,
        settings=cfg,
        model=model or saved_user_cfg.default_model or cfg.default_model,
        max_calls=max_calls if max_calls is not None else saved_user_cfg.max_calls,
        test_timeout=test_timeout if test_timeout is not None else saved_user_cfg.test_timeout,
        sandbox_image=saved_user_cfg.sandbox_image or cfg.sandbox_image,
        web_tools_enabled=saved_user_cfg.web_tools_enabled,
        goal_max_iterations=saved_user_cfg.goal_max_iterations,
        auto_close_seconds=saved_user_cfg.auto_close_seconds,
        ollama_base_url=normalize_local_url(saved_user_cfg.ollama_base_url or cfg.ollama_base_url),
        lmstudio_base_url=normalize_local_url(
            saved_user_cfg.lmstudio_base_url or cfg.lmstudio_base_url
        ),
        local_ai_base_url=normalize_local_url(cfg.local_ai_base_url),
        plan_mode=plan_mode,
        auto_approve=auto_approve,
    )
    if resume_session_id:
        with contextlib.suppress(FileNotFoundError, ValueError):
            loaded = ctx.session_store.load(resume_session_id)
            ctx.session_id = loaded.summary.session_id
            ctx.history = loaded.messages
            if model is None and loaded.summary.model != "unknown":
                ctx.model = loaded.summary.model
            if not plan_mode:
                ctx.plan_mode = loaded.summary.plan_mode

    shutdown_lock = threading.Lock()
    shutdown_started = False
    active_ws_count = 0
    had_ws_connection = False
    idle_shutdown_task: asyncio.Task[None] | None = None

    def _trigger_shutdown(delay_seconds: float = 0.15) -> None:
        nonlocal shutdown_started
        with shutdown_lock:
            if shutdown_started:
                return
            shutdown_started = True
        if ctx.history:
            with contextlib.suppress(ValueError):
                ctx.session_store.save(
                    session_id=ctx.session_id,
                    model=ctx.model,
                    messages=ctx.history,
                    plan_mode=ctx.plan_mode,
                )

        def _stop() -> None:
            if on_shutdown is not None:
                on_shutdown()
            else:
                os.kill(os.getpid(), signal.SIGTERM)

        timer = threading.Timer(delay_seconds, _stop)
        timer.daemon = True
        timer.start()

    async def index(_request: Request) -> HTMLResponse:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    async def get_state(_request: Request) -> JSONResponse:
        return JSONResponse(ctx.snapshot_state())

    async def get_tools(_request: Request) -> JSONResponse:
        inv = await asyncio.to_thread(ctx.collect_tools_inventory)
        return JSONResponse(inv)

    async def get_file(request: Request) -> JSONResponse:
        rel_path = (request.query_params.get("path") or "").strip()
        if not rel_path:
            return JSONResponse({"error": "Missing 'path' query parameter."}, status_code=400)
        try:
            content = ctx.workspace.read_text(rel_path)
        except WorkspaceSecurityError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        return JSONResponse(
            {
                "path": rel_path,
                "lines": len(content.splitlines()),
                "content": content,
            }
        )

    async def update_settings(request: Request) -> JSONResponse:
        body = await request.json()
        should_persist = False
        if "model" in body and str(body["model"]).strip():
            ctx.model = str(body["model"]).strip()
            should_persist = True
        if "mode" in body and str(body["mode"]).strip().lower() in {"agent", "diff", "whole"}:
            ctx.mode = str(body["mode"]).strip().lower()
        if "plan_mode" in body:
            ctx.plan_mode = bool(body["plan_mode"])
        if "auto_approve" in body:
            ctx.auto_approve = bool(body["auto_approve"])
        if "max_calls" in body:
            ctx.max_calls = max(1, min(int(body["max_calls"]), 30))
            should_persist = True
        if "test_timeout" in body:
            ctx.test_timeout = max(1, min(int(body["test_timeout"]), 300))
            should_persist = True
        if "sandbox_image" in body and str(body["sandbox_image"]).strip():
            ctx.sandbox_image = str(body["sandbox_image"]).strip()
            should_persist = True
        if "web_tools_enabled" in body:
            ctx.web_tools_enabled = bool(body["web_tools_enabled"])
            should_persist = True
        if "goal_max_iterations" in body:
            ctx.goal_max_iterations = max(1, min(int(body["goal_max_iterations"]), 15))
            should_persist = True
        if "auto_close_seconds" in body:
            ctx.auto_close_seconds = max(0, min(int(body["auto_close_seconds"]), 3600))
            should_persist = True
        if should_persist:
            ctx.persist_user_settings()
        return JSONResponse(ctx.snapshot_state())

    async def browse_openrouter_models(request: Request) -> JSONResponse:
        query = (request.query_params.get("q") or "").strip().lower()
        free_only = (request.query_params.get("free_only") or "true").lower() != "false"
        tools_only = (request.query_params.get("tools_only") or "true").lower() != "false"

        def _fetch() -> list[dict[str, Any]]:
            resp = httpx.get(
                f"{ctx.settings.openrouter_base_url.rstrip('/')}/models",
                timeout=15.0,
            )
            resp.raise_for_status()
            items: list[dict[str, Any]] = []
            for m in resp.json().get("data", []):
                mid = str(m.get("id") or "")
                name = str(m.get("name") or mid)
                params = m.get("supported_parameters") or []
                is_free = mid.endswith(":free")
                has_tools = "tools" in params
                if free_only and not is_free:
                    continue
                if tools_only and not has_tools:
                    continue
                if query and query not in mid.lower() and query not in name.lower():
                    continue
                items.append(
                    {
                        "id": mid,
                        "name": name,
                        "context_length": m.get("context_length"),
                        "is_free": is_free,
                        "supports_tools": has_tools,
                    }
                )
            items.sort(key=lambda x: (not x["is_free"], x["id"]))
            return items[:60]

        try:
            models = await asyncio.to_thread(_fetch)
        except httpx.HTTPError as exc:
            return JSONResponse({"error": f"OpenRouter catalog error: {exc}"}, status_code=502)
        return JSONResponse({"models": models})

    async def add_custom_model_api(request: Request) -> JSONResponse:
        body = await request.json()
        model_id = str(body.get("model_id") or "").strip()
        try:
            ctx.user_settings_store.add_custom_model(model_id)
            if body.get("activate", True):
                ctx.model = model_id
                ctx.persist_user_settings()
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "state": ctx.snapshot_state()})

    async def remove_custom_model_api(request: Request) -> JSONResponse:
        body = await request.json()
        model_id = str(body.get("model_id") or "").strip()
        ctx.user_settings_store.remove_custom_model(model_id)
        return JSONResponse({"ok": True, "state": ctx.snapshot_state()})

    async def get_mcp_api(_request: Request) -> JSONResponse:
        def _inspect() -> dict[str, Any]:
            configs = list_mcp_configs(ctx.workspace.root)
            inv = ctx.collect_tools_inventory()
            connected = set(inv["mcp_servers"])
            servers = []
            for name, entry in configs.items():
                prefix = f"mcp_{name}_"
                srv_tools = [
                    t["name"]
                    for t in inv["tools"]
                    if t["source"] == "mcp" and t["name"].startswith(prefix)
                ]
                servers.append(
                    {
                        "name": name,
                        "command": entry.get("command", ""),
                        "args": entry.get("args", []),
                        "connected": name in connected,
                        "tools": srv_tools,
                    }
                )
            return {
                "servers": servers,
                "warnings": inv["warnings"],
                "mcp_tool_count": sum(1 for t in inv["tools"] if t["source"] == "mcp"),
            }

        data = await asyncio.to_thread(_inspect)
        return JSONResponse(data)

    async def add_mcp_api(request: Request) -> JSONResponse:
        body = await request.json()
        name = str(body.get("name") or "").strip()
        command = str(body.get("command") or "").strip()
        raw_args = body.get("args", [])
        if isinstance(raw_args, str):
            try:
                args_list = shlex.split(raw_args)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
        elif isinstance(raw_args, list):
            args_list = [str(a) for a in raw_args if str(a).strip()]
        else:
            args_list = []
        try:
            add_mcp_server(ctx.workspace.root, name=name, command=command, args=args_list)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "state": ctx.snapshot_state()})

    async def remove_mcp_api(request: Request) -> JSONResponse:
        body = await request.json()
        name = str(body.get("name") or "").strip()
        removed, _ = remove_mcp_server(ctx.workspace.root, name)
        if not removed:
            return JSONResponse({"error": f"MCP server not found: {name}"}, status_code=404)
        return JSONResponse({"ok": True, "state": ctx.snapshot_state()})

    async def cancel_goal_api(_request: Request) -> JSONResponse:
        ctx.goal_cancel_event.set()
        return JSONResponse({"ok": True, "message": "Cancellation requested for active goal."})

    async def get_providers_api(_request: Request) -> JSONResponse:
        def _inspect() -> dict[str, Any]:
            return {
                "keys": ctx.secrets_store.masked_status(ctx.settings),
                "local": discover_local_models(
                    ollama_base_url=ctx.ollama_base_url,
                    lmstudio_base_url=ctx.lmstudio_base_url,
                    local_ai_base_url=ctx.local_ai_base_url,
                ),
                "presets": [
                    {"id": mid, "notes": desc} for mid, desc in DIRECT_PROVIDER_PRESETS
                ],
                "ollama_base_url": ctx.ollama_base_url,
                "lmstudio_base_url": ctx.lmstudio_base_url,
                "local_ai_base_url": ctx.local_ai_base_url,
            }

        return JSONResponse(await asyncio.to_thread(_inspect))

    async def save_provider_key_api(request: Request) -> JSONResponse:
        body = await request.json()
        provider = str(body.get("provider") or "").strip().lower()
        action = str(body.get("action") or "set").strip().lower()
        try:
            if action == "clear":
                ctx.secrets_store.clear_key(provider)
            elif action == "use":
                target = str(body.get("target") or "auto").strip()
                ctx.secrets_store.set_preferred_source(provider, target)
            else:
                api_key = str(body.get("api_key") or "").strip()
                ctx.secrets_store.set_key(provider, api_key)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(
            {
                "ok": True,
                "keys": ctx.secrets_store.masked_status(ctx.settings),
                "state": ctx.snapshot_state(),
            }
        )

    async def save_local_urls_api(request: Request) -> JSONResponse:
        body = await request.json()
        ollama_url = str(body.get("ollama_base_url") or "").strip()
        lm_url = str(body.get("lmstudio_base_url") or "").strip()
        local_url = str(body.get("local_ai_base_url") or "").strip()
        if ollama_url:
            ctx.ollama_base_url = normalize_local_url(ollama_url)
        if lm_url:
            ctx.lmstudio_base_url = normalize_local_url(lm_url)
        ctx.local_ai_base_url = normalize_local_url(local_url)
        ctx.settings.local_ai_base_url = ctx.local_ai_base_url
        ctx.persist_user_settings()
        local = await asyncio.to_thread(
            discover_local_models,
            ollama_base_url=ctx.ollama_base_url,
            lmstudio_base_url=ctx.lmstudio_base_url,
            local_ai_base_url=ctx.local_ai_base_url,
        )
        return JSONResponse({"ok": True, "local": local, "state": ctx.snapshot_state()})

    async def doctor_api(_request: Request) -> JSONResponse:
        checks = await asyncio.to_thread(
            run_doctor_checks,
            workspace_root=ctx.workspace.root,
            settings=ctx.settings,
            secrets_store=ctx.secrets_store,
        )
        return JSONResponse({"checks": [c.model_dump() for c in checks]})

    async def list_custom_commands_api(_request: Request) -> JSONResponse:
        cmds = [c.model_dump() for c in ctx.custom_commands.list_commands()]
        return JSONResponse({"commands": cmds})

    async def save_custom_command_api(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            saved_c = ctx.custom_commands.save(
                name=str(body.get("name") or ""),
                description=str(body.get("description") or ""),
                template=str(body.get("template") or ""),
                scope=str(body.get("scope") or "global"),
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(
            {"ok": True, "command": saved_c.model_dump(), "state": ctx.snapshot_state()}
        )

    async def delete_custom_command_api(request: Request) -> JSONResponse:
        body = await request.json()
        removed_c = ctx.custom_commands.delete(
            name=str(body.get("name") or ""),
            scope=str(body.get("scope") or "global"),
        )
        if not removed_c:
            return JSONResponse({"error": "Custom command not found."}, status_code=404)
        return JSONResponse({"ok": True, "state": ctx.snapshot_state()})

    async def run_command_api(request: Request) -> JSONResponse:
        body = await request.json()
        cmd_text = str(body.get("command") or "").strip()
        if not cmd_text.startswith("/"):
            return JSONResponse({"error": "Commands must start with '/'."}, status_code=400)
        res = await asyncio.to_thread(_execute_slash_command, ctx, cmd_text)
        if res.get("shutdown"):
            _trigger_shutdown(0.15)
        return JSONResponse(res)

    async def shutdown_server(_request: Request) -> JSONResponse:
        _trigger_shutdown(0.15)
        return JSONResponse(
            {
                "ok": True,
                "message": "Poseidon Web UI shutting down and closing container...",
            }
        )

    async def run_sandbox_tests(_request: Request) -> JSONResponse:
        runner = DockerTestRunner(
            image=ctx.settings.sandbox_image,
            timeout_seconds=ctx.test_timeout,
        )
        res = await asyncio.to_thread(runner.run, ctx.workspace.root)
        return JSONResponse(res.model_dump())

    async def compact_now(_request: Request) -> JSONResponse:
        _, stats = compact_messages(
            ctx.history,
            keep_recent_tool_outputs=2,
            min_chars_to_mask=150,
            in_place=True,
        )
        return JSONResponse({"stats": stats.model_dump(), "state": ctx.snapshot_state()})

    async def undo_last(_request: Request) -> JSONResponse:
        cp, restored = ctx.checkpoints.undo()
        if cp is None:
            return JSONResponse(
                {
                    "undone": False,
                    "message": "No checkpoints to undo.",
                    "state": ctx.snapshot_state(),
                }
            )
        ctx.tool_state.read_hashes.clear()
        ctx.tool_state.dirty_since_last_test = True
        return JSONResponse(
            {
                "undone": True,
                "checkpoint_id": cp.id,
                "label": cp.label,
                "restored": restored,
                "state": ctx.snapshot_state(),
            }
        )

    async def load_session(request: Request) -> JSONResponse:
        body = await request.json()
        sid = str(body.get("session_id") or "").strip()
        if not sid:
            return JSONResponse({"error": "Missing session_id."}, status_code=400)
        try:
            loaded = ctx.session_store.load(sid)
        except (FileNotFoundError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        ctx.session_id = loaded.summary.session_id
        ctx.history = loaded.messages
        ctx.plan_mode = loaded.summary.plan_mode
        if loaded.summary.model != "unknown":
            ctx.model = loaded.summary.model
        return JSONResponse(
            {
                "state": ctx.snapshot_state(),
                "messages": _visible_transcript(loaded.messages),
            }
        )

    async def clear_session(_request: Request) -> JSONResponse:
        ctx.history.clear()
        ctx.session_id = generate_session_id()
        return JSONResponse({"state": ctx.snapshot_state()})

    async def delete_session(request: Request) -> JSONResponse:
        body = await request.json()
        sid = str(body.get("session_id") or "").strip()
        if not sid:
            return JSONResponse({"error": "Missing session_id."}, status_code=400)
        try:
            deleted = ctx.session_store.delete(sid)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if not deleted:
            return JSONResponse({"error": f"Saved session not found: {sid}"}, status_code=404)
        cleared_active = False
        if ctx.session_id == sid:
            ctx.history.clear()
            ctx.session_id = generate_session_id()
            cleared_active = True
        return JSONResponse(
            {
                "deleted": True,
                "session_id": sid,
                "cleared_active": cleared_active,
                "state": ctx.snapshot_state(),
            }
        )

    def _run_prompt_in_thread(
        prompt: str,
        bridge: _WebSocketBridge,
        attached_images: list[dict[str, str]] | None = None,
    ) -> None:
        if not ctx.run_lock.acquire(blocking=False):
            bridge.emit({"type": "error", "message": "Another run is already in progress."})
            return

        mcp_manager = MCPClientManager()
        try:
            # Handle autonomous `/goal [--steps N] <objective>` with live streaming!
            if prompt == "/goal" or prompt.startswith("/goal "):
                raw_goal_arg = prompt[5:].strip()
                try:
                    objective, goal_steps = parse_goal_command(
                        raw_goal_arg,
                        default_steps=ctx.goal_max_iterations,
                    )
                except ValueError as exc:
                    bridge.emit({"type": "error", "message": str(exc)})
                    return
                if ctx.plan_mode:
                    bridge.emit(
                        {
                            "type": "error",
                            "message": (
                                "Plan Mode is ON (read-only). Turn Plan Mode OFF to run `/goal`."
                            ),
                        }
                    )
                    return

                ctx.goal_cancel_event.clear()
                with OpenRouterClient(
                    ctx.settings,
                    ollama_base_url=ctx.ollama_base_url,
                    lmstudio_base_url=ctx.lmstudio_base_url,
                    secrets_store=ctx.secrets_store,
                ) as client:
                    try:
                        client.validate_tool_model(ctx.model)
                    except ModelToolSupportError as exc:
                        bridge.emit({"type": "error", "message": str(exc)})
                        return
                    except OpenRouterError:
                        pass

                    test_runner = DockerTestRunner(
                        image=ctx.sandbox_image,
                        timeout_seconds=ctx.test_timeout,
                    )
                    sem_index = SemanticCodeIndex(
                        ctx.workspace,
                        FastEmbedProvider(
                            model_name=ctx.settings.embedding_model,
                            cache_dir=ctx.settings.embedding_cache_dir,
                        ),
                    )
                    hooks = HookManager()

                    def _goal_on_test(res: TestRunResult) -> None:
                        bridge.emit({"type": "test_result", "result": res.model_dump()})

                    def _goal_confirm(diff_text: str) -> bool:
                        if ctx.auto_approve:
                            bridge.emit({"type": "diff_auto_applied", "diff": diff_text})
                            return True
                        return bridge.request_diff_approval(diff_text)

                    ro_reg = ToolRegistry()
                    rw_reg = ToolRegistry()
                    for reg, is_ro in ((ro_reg, True), (rw_reg, False)):
                        register_core_tools(
                            registry=reg,
                            workspace=ctx.workspace,
                            state=ctx.tool_state,
                            confirm_patch=_goal_confirm,
                            test_runner=test_runner,
                            on_test_result=_goal_on_test,
                            semantic_index=sem_index,
                            checkpoints=ctx.checkpoints,
                            hooks=hooks,
                            protect_tests=True,
                            read_only=is_ro,
                            enable_web_tools=ctx.web_tools_enabled,
                            tavily_api_key=ctx.settings.tavily_api_key.get_secret_value(),
                            brave_api_key=ctx.settings.brave_api_key.get_secret_value(),
                        )
                        load_workspace_plugins(
                            workspace=ctx.workspace,
                            registry=reg,
                            hooks=hooks,
                        )
                    mcp_manager.connect_and_register(
                        workspace_root=ctx.workspace.root,
                        registry=rw_reg,
                    )

                    memory = load_project_memory(ctx.workspace.root)
                    tracer = PoseidonTracer(ctx.settings)
                    bridge.emit(
                        {
                            "type": "goal_started",
                            "objective": objective,
                            "max_iterations": goal_steps,
                        }
                    )

                    def _on_plan(plan_text: str) -> None:
                        bridge.emit({"type": "goal_plan", "plan": plan_text})

                    def _on_iter_start(i: int, total: int) -> None:
                        bridge.emit(
                            {
                                "type": "goal_iteration_start",
                                "iteration": i,
                                "max_iterations": total,
                            }
                        )

                    def _on_iter_done(rec: GoalIterationRecord) -> None:
                        bridge.emit({"type": "goal_iteration_complete", "record": rec.model_dump()})

                    def _on_goal_call(c: int, lim: int) -> None:
                        bridge.emit(
                            {
                                "type": "model_call",
                                "call_index": c,
                                "max_calls": lim,
                                "model": ctx.model,
                                "plan_mode": False,
                            }
                        )

                    def _on_goal_tool(ev: ToolEvent) -> None:
                        bridge.emit({"type": "tool_result", "event": ev.model_dump()})

                    goal_res = run_goal_loop(
                        client=client,
                        read_only_registry=ro_reg,
                        write_registry=rw_reg,
                        model=ctx.model,
                        objective=objective,
                        max_iterations=goal_steps,
                        max_calls_per_iteration=ctx.max_calls,
                        project_memory=memory.content,
                        tracer=tracer,
                        conversation_history=ctx.history,
                        workspace_root=ctx.workspace.root,
                        tool_state=ctx.tool_state,
                        test_runner=test_runner,
                        on_plan_ready=_on_plan,
                        on_iteration_start=_on_iter_start,
                        on_iteration_complete=_on_iter_done,
                        on_model_call=_on_goal_call,
                        on_tool_result=_on_goal_tool,
                        on_test_result=_goal_on_test,
                        cancel_event=ctx.goal_cancel_event,
                    )
                    ctx.record_usage(
                        total_tokens=goal_res.total_tokens,
                        cost_usd=goal_res.total_cost_usd,
                    )
                    ctx.session_store.save(
                        session_id=ctx.session_id,
                        model=ctx.model,
                        messages=ctx.history,
                        plan_mode=ctx.plan_mode,
                    )
                    bridge.emit(
                        {
                            "type": "goal_complete",
                            "result": goal_res.model_dump(),
                            "state": ctx.snapshot_state(),
                        }
                    )
                return

            # Handle terminal-parity slash commands (and custom Global/Workspace `/commands`)!
            if prompt.startswith("/"):
                rendered = ctx.custom_commands.render(prompt)
                if rendered is not None:
                    custom_cmd, expanded_template = rendered
                    bridge.emit(
                        {
                            "type": "tool_result",
                            "event": {
                                "call_index": 0,
                                "tool_call_id": "custom_cmd",
                                "tool_name": custom_cmd.slash_name,
                                "ok": True,
                                "summary": f"expanded {custom_cmd.scope} template",
                            },
                        }
                    )
                    prompt = expanded_template
                else:
                    cmd_res = _execute_slash_command(ctx, prompt)
                    if "test_result" in cmd_res:
                        bridge.emit({"type": "test_result", "result": cmd_res["test_result"]})
                        bridge.emit({"type": "state_update", "state": cmd_res["state"]})
                    else:
                        bridge.emit({"type": "command_result", "result": cmd_res})
                    if cmd_res.get("shutdown"):
                        _trigger_shutdown(0.15)
                    return

            # Expand safe `@file` (text + image) mentions and merge attached screenshots
            mention_res = expand_file_mentions(prompt, ctx.workspace, ctx.tool_state)
            if mention_res.loaded_files:
                prompt = mention_res.expanded_prompt
                bridge.emit(
                    {
                        "type": "tool_result",
                        "event": {
                            "call_index": 0,
                            "tool_call_id": "mentions",
                            "tool_name": "@file",
                            "ok": True,
                            "summary": f"attached {', '.join(mention_res.loaded_files)}",
                        },
                    }
                )
            all_images: list[dict[str, str]] = [
                *(attached_images or []),
                *mention_res.images,
            ][:4]

            with OpenRouterClient(
                ctx.settings,
                ollama_base_url=ctx.ollama_base_url,
                lmstudio_base_url=ctx.lmstudio_base_url,
                secrets_store=ctx.secrets_store,
            ) as client:
                if ctx.mode == "agent":
                    try:
                        client.validate_tool_model(ctx.model)
                    except ModelToolSupportError as exc:
                        bridge.emit({"type": "error", "message": str(exc)})
                        return
                    except OpenRouterError:
                        pass

                    registry = ToolRegistry()
                    hooks = HookManager()
                    test_runner = DockerTestRunner(
                        image=ctx.sandbox_image,
                        timeout_seconds=ctx.test_timeout,
                    )
                    sem_index = SemanticCodeIndex(
                        ctx.workspace,
                        FastEmbedProvider(
                            model_name=ctx.settings.embedding_model,
                            cache_dir=ctx.settings.embedding_cache_dir,
                        ),
                    )

                    def _on_test(res: TestRunResult) -> None:
                        bridge.emit({"type": "test_result", "result": res.model_dump()})

                    def _confirm(diff_text: str) -> bool:
                        if ctx.auto_approve:
                            bridge.emit({"type": "diff_auto_applied", "diff": diff_text})
                            return True
                        return bridge.request_diff_approval(diff_text)

                    register_core_tools(
                        registry=registry,
                        workspace=ctx.workspace,
                        state=ctx.tool_state,
                        confirm_patch=_confirm,
                        test_runner=test_runner,
                        on_test_result=_on_test,
                        semantic_index=sem_index,
                        checkpoints=ctx.checkpoints,
                        hooks=hooks,
                        protect_tests=True,
                        read_only=ctx.plan_mode,
                        enable_web_tools=ctx.web_tools_enabled,
                        tavily_api_key=ctx.settings.tavily_api_key.get_secret_value(),
                        brave_api_key=ctx.settings.brave_api_key.get_secret_value(),
                    )
                    load_workspace_plugins(
                        workspace=ctx.workspace,
                        registry=registry,
                        hooks=hooks,
                    )
                    mcp_manager.connect_and_register(
                        workspace_root=ctx.workspace.root,
                        registry=registry,
                    )

                    memory = load_project_memory(ctx.workspace.root)
                    sys_prompt = build_system_prompt(
                        project_memory=memory.content,
                        plan_mode=ctx.plan_mode,
                    )
                    if ctx.history and ctx.history[0].get("role") == "system":
                        ctx.history[0]["content"] = sys_prompt

                    def _on_model_call(call_num: int, limit: int) -> None:
                        bridge.emit(
                            {
                                "type": "model_call",
                                "call_index": call_num,
                                "max_calls": limit,
                                "model": ctx.model,
                                "plan_mode": ctx.plan_mode,
                            }
                        )

                    def _on_tool_result(event: ToolEvent) -> None:
                        bridge.emit({"type": "tool_result", "event": event.model_dump()})

                    tracer = PoseidonTracer(ctx.settings)
                    res = run_agent_loop(
                        client=client,
                        registry=registry,
                        model=ctx.model,
                        user_prompt=prompt,
                        max_calls=ctx.max_calls,
                        tracer=tracer,
                        conversation_history=ctx.history,
                        on_model_call=_on_model_call,
                        on_tool_result=_on_tool_result,
                        workspace_root=ctx.workspace.root,
                        tool_state=ctx.tool_state,
                        test_runner=test_runner,
                        on_test_result=_on_test,
                        system_prompt=sys_prompt,
                        user_images=all_images or None,
                    )
                    ctx.record_usage(
                        prompt_tokens=res.usage.prompt_tokens,
                        completion_tokens=res.usage.completion_tokens,
                        reasoning_tokens=res.usage.reasoning_tokens,
                        total_tokens=res.usage.total_tokens,
                        cost_usd=res.usage.cost_usd,
                        tokens_masked=res.estimated_tokens_saved,
                    )
                    ctx.session_store.save(
                        session_id=ctx.session_id,
                        model=ctx.model,
                        messages=ctx.history,
                        plan_mode=ctx.plan_mode,
                    )
                    bridge.emit(
                        {
                            "type": "run_complete",
                            "result": res.model_dump(),
                            "state": ctx.snapshot_state(),
                        }
                    )
                elif ctx.mode == "diff":
                    files = ctx.workspace.list_files()
                    if not files:
                        bridge.emit(
                            {
                                "type": "error",
                                "message": "Workspace is empty; switch to `/mode whole` first.",
                            }
                        )
                        return
                    bridge.emit(
                        {
                            "type": "model_call",
                            "call_index": 1,
                            "max_calls": 1,
                            "model": ctx.model,
                            "plan_mode": False,
                        }
                    )

                    def _confirm_diff(diff_text: str) -> bool:
                        if ctx.auto_approve:
                            bridge.emit({"type": "diff_auto_applied", "diff": diff_text})
                            return True
                        return bridge.request_diff_approval(diff_text)

                    diff_res = run_diff_edit_request(
                        client=client,
                        workspace=ctx.workspace,
                        model=ctx.model,
                        prompt=prompt,
                        target_file=files[0],
                        preview_and_confirm=_confirm_diff,
                    )
                    summary = (
                        f"Patched {', '.join(diff_res.modified_files)} "
                        f"(+{diff_res.added_lines}/-{diff_res.removed_lines})."
                        if diff_res.applied
                        else (
                            "Patch skipped by user (files unchanged)."
                            if diff_res.rejected_by_user
                            else f"Patch failed: {diff_res.reason}"
                        )
                    )
                    bridge.emit(
                        {
                            "type": "run_complete",
                            "result": {
                                "final_response": summary,
                                "calls_made": diff_res.attempts,
                                "max_calls": diff_res.attempts,
                                "elapsed_seconds": diff_res.elapsed_seconds,
                                "usage": {
                                    "total_tokens": diff_res.total_tokens,
                                    "cost_usd": diff_res.total_cost_usd,
                                },
                                "estimated_tokens_saved": 0,
                            },
                            "state": ctx.snapshot_state(),
                        }
                    )
                else:
                    bridge.emit(
                        {
                            "type": "model_call",
                            "call_index": 1,
                            "max_calls": 1,
                            "model": ctx.model,
                            "plan_mode": False,
                        }
                    )
                    sf_res = run_single_file_request(
                        client=client,
                        workspace=ctx.workspace,
                        model=ctx.model,
                        prompt=prompt,
                    )
                    action = "Updated" if sf_res.is_update else "Created"
                    bridge.emit(
                        {
                            "type": "run_complete",
                            "result": {
                                "final_response": f"{action} `{sf_res.relative_path}`.",
                                "calls_made": 1,
                                "max_calls": 1,
                                "elapsed_seconds": sf_res.elapsed_seconds,
                                "usage": sf_res.usage.model_dump(),
                                "estimated_tokens_saved": 0,
                            },
                            "state": ctx.snapshot_state(),
                        }
                    )
        except (
            OpenRouterError,
            FileExtractionError,
            WorkspaceSecurityError,
            ValueError,
        ) as exc:
            bridge.emit({"type": "error", "message": str(exc)})
        except Exception as exc:
            bridge.emit({"type": "error", "message": f"Unexpected harness error: {exc}"})
        finally:
            mcp_manager.close()
            ctx.run_lock.release()

    async def ws_endpoint(websocket: WebSocket) -> None:
        nonlocal active_ws_count, had_ws_connection, idle_shutdown_task
        await websocket.accept()
        active_ws_count += 1
        had_ws_connection = True
        if idle_shutdown_task is not None and not idle_shutdown_task.done():
            idle_shutdown_task.cancel()
            idle_shutdown_task = None

        loop = asyncio.get_running_loop()
        bridge = _WebSocketBridge(loop, websocket)
        await websocket.send_json(
            {
                "type": "connected",
                "state": ctx.snapshot_state(),
                "messages": _visible_transcript(ctx.history),
            }
        )

        try:
            while True:
                msg = await websocket.receive_json()
                msg_type = msg.get("type")
                if msg_type == "prompt":
                    prompt_text = str(msg.get("prompt") or "").strip()
                    raw_imgs = msg.get("images") or []
                    clean_imgs: list[dict[str, str]] = [
                        {
                            "name": str(im.get("name") or "screenshot.png")[:64],
                            "data_url": str(im.get("data_url") or ""),
                        }
                        for im in raw_imgs[:4]
                        if isinstance(im, dict)
                        and str(im.get("data_url") or "").startswith("data:image/")
                    ]
                    if not prompt_text and not clean_imgs:
                        continue
                    if not prompt_text and clean_imgs:
                        prompt_text = "Describe and analyze the attached screenshot(s)."
                    worker = threading.Thread(
                        target=_run_prompt_in_thread,
                        args=(prompt_text, bridge, clean_imgs),
                        daemon=True,
                    )
                    worker.start()
                elif msg_type == "diff_approval_response":
                    approval_id = str(msg.get("approval_id") or "")
                    approved = bool(msg.get("approved", False))
                    bridge.resolve_approval(approval_id, approved)
        except WebSocketDisconnect:
            bridge.cancel_all()
        finally:
            active_ws_count = max(0, active_ws_count - 1)
            if (
                auto_shutdown_seconds is not None
                and ctx.auto_close_seconds > 0
                and had_ws_connection
                and active_ws_count == 0
                and not shutdown_started
            ):
                delay = float(ctx.auto_close_seconds)

                async def _auto_close_after_grace() -> None:
                    await asyncio.sleep(delay)
                    if active_ws_count == 0:
                        _trigger_shutdown(0.05)

                idle_shutdown_task = asyncio.create_task(_auto_close_after_grace())

    routes = [
        Route("/", index, methods=["GET"]),
        Route("/api/state", get_state, methods=["GET"]),
        Route("/api/tools", get_tools, methods=["GET"]),
        Route("/api/file", get_file, methods=["GET"]),
        Route("/api/settings", update_settings, methods=["POST"]),
        Route("/api/openrouter-models", browse_openrouter_models, methods=["GET"]),
        Route("/api/models/add", add_custom_model_api, methods=["POST"]),
        Route("/api/models/remove", remove_custom_model_api, methods=["POST"]),
        Route("/api/mcp", get_mcp_api, methods=["GET"]),
        Route("/api/mcp/add", add_mcp_api, methods=["POST"]),
        Route("/api/mcp/remove", remove_mcp_api, methods=["POST"]),
        Route("/api/providers", get_providers_api, methods=["GET"]),
        Route("/api/providers/key", save_provider_key_api, methods=["POST"]),
        Route("/api/providers/local-urls", save_local_urls_api, methods=["POST"]),
        Route("/api/doctor", doctor_api, methods=["GET"]),
        Route("/api/custom-commands", list_custom_commands_api, methods=["GET"]),
        Route("/api/custom-commands/save", save_custom_command_api, methods=["POST"]),
        Route("/api/custom-commands/delete", delete_custom_command_api, methods=["POST"]),
        Route("/api/goal/cancel", cancel_goal_api, methods=["POST"]),
        Route("/api/command", run_command_api, methods=["POST"]),
        Route("/api/shutdown", shutdown_server, methods=["POST"]),
        Route("/api/test", run_sandbox_tests, methods=["POST"]),
        Route("/api/compact", compact_now, methods=["POST"]),
        Route("/api/undo", undo_last, methods=["POST"]),
        Route("/api/sessions/load", load_session, methods=["POST"]),
        Route("/api/sessions/delete", delete_session, methods=["POST"]),
        Route("/api/sessions/clear", clear_session, methods=["POST"]),
        WebSocketRoute("/ws", ws_endpoint),
    ]
    return Starlette(debug=False, routes=routes)
