"""Tests for Phase B: workspace plugins, lifecycle hooks, and real stdio MCP server integration."""

import json
import sys
from pathlib import Path

from poseidon.mcp.client import MCPClientManager
from poseidon.plugins.hooks import HookManager
from poseidon.plugins.loader import load_workspace_plugins
from poseidon.tools.builtin import register_core_tools
from poseidon.tools.registry import ToolRegistry, ToolSessionState
from poseidon.workspace import WorkspaceGuard

SAMPLE_PLUGIN_CODE = '''"""Sample Poseidon plugin with a custom tool and lifecycle hooks."""

from poseidon.tools.registry import ToolDefinition, ToolExecutionResult


def register(api):
    def handle_word_count(args):
        text = api.workspace.read_text(args["path"])
        count = len(text.split())
        return ToolExecutionResult(
            tool_name="word_count",
            ok=True,
            output=f"{args['path']} has {count} words.",
            summary=f"{count} words",
        )

    api.register_tool(
        ToolDefinition(
            name="word_count",
            description="Count whitespace-separated words in a workspace file.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            handler=handle_word_count,
        )
    )

    def block_forbidden_token(rel_path, payload):
        if "FORBIDDEN_SECRET" in payload:
            return "FORBIDDEN_SECRET is not allowed in workspace code."
        return None

    api.on_pre_edit(block_forbidden_token, name="secret_guard")
    api.on_post_edit(lambda files: f"audited {len(files)} file(s)", name="audit_trail")
'''

SAMPLE_MCP_SERVER_CODE = '''"""Minimal MCP v2 stdio server for real MCP tool testing."""

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("PoseidonTestMCP")


@mcp.tool()
def multiply_numbers(a: int, b: int) -> str:
    """Multiply two integers and return the product."""
    return f"product={a * b}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
'''


def test_workspace_plugin_registers_tool_and_lifecycle_hooks(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("notes.py", "alpha beta gamma\n")

    plugin_dir = tmp_path / ".poseidon" / "plugins"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "demo_plugin.py").write_text(SAMPLE_PLUGIN_CODE, encoding="utf-8")

    registry = ToolRegistry()
    hooks = HookManager()
    state = ToolSessionState()
    register_core_tools(registry=registry, workspace=guard, state=state, hooks=hooks)
    report = load_workspace_plugins(workspace=guard, registry=registry, hooks=hooks)

    assert report.loaded_plugins == ["demo_plugin"]
    assert report.registered_tools == ["word_count"]
    assert report.warnings == []

    # 1. Execute custom plugin tool
    wc = registry.execute("word_count", json.dumps({"path": "notes.py"}))
    assert wc.ok
    assert "3 words" in wc.output

    # 2. Verify pre_edit hook blocks forbidden content
    blocked = registry.execute(
        "create_file",
        json.dumps({"path": "bad.py", "content": "TOKEN = 'FORBIDDEN_SECRET'\n"}),
    )
    assert not blocked.ok
    assert "Blocked by pre_edit hook 'secret_guard'" in blocked.output
    assert not (tmp_path / "bad.py").exists()

    # 3. Verify post_edit hook note appears on allowed edit
    allowed = registry.execute(
        "create_file",
        json.dumps({"path": "good.py", "content": "x = 1\n"}),
    )
    assert allowed.ok
    assert "[hook:audit_trail] audited 1 file(s)" in allowed.output


def test_real_mcp_stdio_server_discovery_and_tool_call(tmp_path: Path) -> None:
    server_script = tmp_path / "mcp_math_server.py"
    server_script.write_text(SAMPLE_MCP_SERVER_CODE, encoding="utf-8")

    mcp_dir = tmp_path / ".poseidon"
    mcp_dir.mkdir(parents=True)
    mcp_cfg = {
        "mcpServers": {
            "math": {
                "command": sys.executable,
                "args": [str(server_script)],
            }
        }
    }
    (mcp_dir / "mcp.json").write_text(json.dumps(mcp_cfg), encoding="utf-8")

    registry = ToolRegistry()
    with MCPClientManager(connect_timeout=15.0, call_timeout=15.0) as manager:
        report = manager.connect_and_register(workspace_root=tmp_path, registry=registry)
        assert report.warnings == []
        assert report.connected_servers == ["math"]
        assert "mcp_math_multiply_numbers" in report.registered_tools

        res = registry.execute(
            "mcp_math_multiply_numbers",
            json.dumps({"a": 6, "b": 7}),
        )
        assert res.ok
        assert "product=42" in res.output
