"""Persistent Global & Workspace custom slash command templates (`*.md`).

Global commands live in `~/.poseidon_secrets/commands/*.md` (persisted across every project
and session via the `poseidon_secrets` Docker volume). Workspace commands live in
`<workspace>/.poseidon/commands/*.md` and override global commands of the same name.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import BaseModel

_SAFE_CMD_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
WORKSPACE_COMMANDS_SUBDIR = ".poseidon/commands"


class CustomCommand(BaseModel):
    """One reusable user-defined slash command template."""

    name: str  # without leading slash, e.g. "review"
    scope: str  # "global" or "workspace"
    description: str
    template: str

    @property
    def slash_name(self) -> str:
        return f"/{self.name}"


def default_global_commands_dir() -> Path:
    """Resolve persistent global commands directory outside any workspace repository."""
    override = os.environ.get("POSEIDON_GLOBAL_COMMANDS_DIR", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".poseidon_secrets" / "commands"


def _parse_command_file(path: Path, scope: str) -> CustomCommand | None:
    stem = path.stem.lower().strip()
    if not _SAFE_CMD_RE.match(stem):
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None

    description = f"Custom {scope} command"
    body = raw
    lines = raw.splitlines()
    if lines and lines[0].lower().startswith("description:"):
        description = lines[0].split(":", 1)[1].strip() or description
        body = "\n".join(lines[1:]).strip()
    elif lines:
        first = lines[0].lstrip("#").strip()
        if first:
            description = first[:72]

    return CustomCommand(
        name=stem,
        scope=scope,
        description=description,
        template=body or raw,
    )


class CustomCommandStore:
    """Manage both Global (cross-session/cross-project) and Workspace custom `/commands`."""

    def __init__(
        self,
        workspace_root: Path,
        global_dir: Path | None = None,
    ) -> None:
        self._ws_dir = workspace_root / WORKSPACE_COMMANDS_SUBDIR
        self._global_dir = global_dir or default_global_commands_dir()

    def _dir_for(self, scope: str) -> Path:
        norm = scope.lower().strip()
        if norm == "global":
            return self._global_dir
        if norm == "workspace":
            return self._ws_dir
        raise ValueError("Scope must be 'global' or 'workspace'.")

    def list_commands(self) -> list[CustomCommand]:
        """List all available custom commands (workspace overrides global on name collision)."""
        merged: dict[str, CustomCommand] = {}
        for scope, directory in (("global", self._global_dir), ("workspace", self._ws_dir)):
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.md")):
                cmd = _parse_command_file(path, scope=scope)
                if cmd is not None:
                    merged[cmd.name] = cmd
        return sorted(merged.values(), key=lambda c: c.name)

    def get(self, raw_name: str) -> CustomCommand | None:
        """Look up a custom command by name (with or without leading '/')."""
        cleaned = raw_name.lstrip("/").lower().strip()
        for cmd in self.list_commands():
            if cmd.name == cleaned:
                return cmd
        return None

    def save(
        self,
        *,
        name: str,
        template: str,
        description: str = "",
        scope: str = "global",
    ) -> CustomCommand:
        """Create or update a persistent Global or Workspace custom `/command`."""
        cleaned = name.lstrip("/").lower().strip()
        if not _SAFE_CMD_RE.match(cleaned):
            raise ValueError(
                "Command name must start with a letter and use 1-32 lowercase letters, "
                "digits, '-' or '_'."
            )
        body = template.strip()
        if not body:
            raise ValueError("Command prompt template cannot be empty.")

        target_dir = self._dir_for(scope)
        target_dir.mkdir(parents=True, exist_ok=True)
        desc = description.strip() or body.splitlines()[0][:72]
        content = f"Description: {desc}\n{body}\n"
        path = target_dir / f"{cleaned}.md"
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
        return CustomCommand(
            name=cleaned,
            scope=scope.lower().strip(),
            description=desc,
            template=body,
        )

    def delete(self, *, name: str, scope: str = "global") -> bool:
        """Delete a custom `/command` from the specified scope."""
        cleaned = name.lstrip("/").lower().strip()
        if not _SAFE_CMD_RE.match(cleaned):
            return False
        path = self._dir_for(scope) / f"{cleaned}.md"
        if not path.is_file():
            return False
        path.unlink()
        return True

    def render(self, raw_input: str) -> tuple[CustomCommand, str] | None:
        """If `raw_input` matches a custom `/command`, return `(command, expanded_prompt)`."""
        text = raw_input.strip()
        if not text.startswith("/"):
            return None
        parts = text.split(maxsplit=1)
        cmd_name = parts[0][1:].lower()
        arg = parts[1].strip() if len(parts) == 2 else ""
        cmd = self.get(cmd_name)
        if cmd is None:
            return None

        if "$ARGUMENTS" in cmd.template:
            expanded = cmd.template.replace("$ARGUMENTS", arg)
        elif arg:
            expanded = f"{cmd.template}\n\nAdditional context: {arg}"
        else:
            expanded = cmd.template
        return cmd, expanded.strip()
