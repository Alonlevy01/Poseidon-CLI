"""Persistent workspace user settings (`.poseidon/settings.json`) and MCP server config manager."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SETTINGS_REL_PATH = ".poseidon/settings.json"
MCP_REL_PATH = ".poseidon/mcp.json"
_SAFE_SERVER_RE = re.compile(r"^[a-zA-Z0-9_-]{1,48}$")


class UserSettings(BaseModel):
    """Persisted per-workspace preferences shared seamlessly between CLI and Web UI."""

    default_model: str | None = None
    custom_models: list[str] = Field(default_factory=list)
    max_calls: int = Field(default=8, ge=1, le=30)
    test_timeout: int = Field(default=15, ge=1, le=300)
    sandbox_image: str = "poseidon-sandbox:dev"
    web_tools_enabled: bool = True
    goal_max_iterations: int = Field(default=5, ge=1, le=15)
    auto_close_seconds: int = Field(default=8, ge=0, le=3600)
    ollama_base_url: str = "http://host.docker.internal:11434/v1"
    lmstudio_base_url: str = "http://host.docker.internal:1234/v1"


class UserSettingsStore:
    """Read/write `.poseidon/settings.json` atomically without touching Git-tracked files."""

    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root
        self._path = workspace_root / SETTINGS_REL_PATH

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> UserSettings:
        """Load saved settings or return defaults if none exist or JSON is malformed."""
        if not self._path.is_file():
            return UserSettings()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return UserSettings(**raw)
        except (OSError, json.JSONDecodeError, ValueError):
            return UserSettings()
        return UserSettings()

    def save(self, settings: UserSettings) -> UserSettings:
        """Persist settings atomically to `.poseidon/settings.json`."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(settings.model_dump(), indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self._path)
        return settings

    def add_custom_model(self, model_id: str) -> UserSettings:
        """Add an OpenRouter or provider-prefixed model ID to `custom_models`."""
        cleaned = model_id.strip()
        valid_prefixes = ("ollama:", "lmstudio:", "openai:", "gemini:", "anthropic:")
        if not cleaned or ("/" not in cleaned and not cleaned.startswith(valid_prefixes)):
            raise ValueError(
                "Model ID must be an OpenRouter slug (e.g. 'nvidia/nemotron-3.5-super-vl:free') "
                "or prefixed with ollama:, lmstudio:, openai:, gemini:, or anthropic:."
            )
        current = self.load()
        if cleaned not in current.custom_models:
            current.custom_models.append(cleaned)
            self.save(current)
        return current

    def remove_custom_model(self, model_id: str) -> UserSettings:
        """Remove a model ID from `custom_models`."""
        cleaned = model_id.strip()
        current = self.load()
        if cleaned in current.custom_models:
            current.custom_models = [m for m in current.custom_models if m != cleaned]
            self.save(current)
        return current


def list_mcp_configs(workspace_root: Path) -> dict[str, dict[str, Any]]:
    """Return configured `mcpServers` map from `.poseidon/mcp.json` (or `mcp.json`)."""
    for rel in (MCP_REL_PATH, "mcp.json"):
        path = workspace_root / rel
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                servers = raw.get("mcpServers", {})
                if isinstance(servers, dict):
                    return {
                        str(k): v for k, v in servers.items() if isinstance(v, dict)
                    }
            except (OSError, json.JSONDecodeError):
                return {}
    return {}


def add_mcp_server(
    workspace_root: Path,
    *,
    name: str,
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Add or update a stdio MCP server in `.poseidon/mcp.json`."""
    cleaned_name = name.strip()
    if not _SAFE_SERVER_RE.match(cleaned_name):
        raise ValueError("MCP server name must be 1-48 chars using letters, digits, '-' or '_'.")
    cleaned_cmd = command.strip()
    if not cleaned_cmd:
        raise ValueError("MCP server command cannot be empty.")

    path = workspace_root / MCP_REL_PATH
    servers = list_mcp_configs(workspace_root)
    entry: dict[str, Any] = {"command": cleaned_cmd, "args": list(args or [])}
    if env:
        entry["env"] = dict(env)
    servers[cleaned_name] = entry

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"mcpServers": servers}, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return servers


def remove_mcp_server(workspace_root: Path, name: str) -> tuple[bool, dict[str, dict[str, Any]]]:
    """Remove a configured MCP server from `.poseidon/mcp.json`."""
    cleaned = name.strip()
    path = workspace_root / MCP_REL_PATH
    servers = list_mcp_configs(workspace_root)
    if cleaned not in servers:
        return False, servers
    del servers[cleaned]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"mcpServers": servers}, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return True, servers
