"""JSONL session persistence for `/save`, `/resume`, `/sessions`, and `--continue`."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

SESSIONS_SUBDIR = ".poseidon_cache/sessions"
_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")


class SessionSummary(BaseModel):
    """Metadata summary for a saved JSONL chat session."""

    session_id: str
    title: str
    model: str
    plan_mode: bool = False
    message_count: int = 0
    created_at: str
    updated_at: str


class LoadedSession(BaseModel):
    """Full restored session metadata + conversation message list."""

    summary: SessionSummary
    messages: list[dict[str, Any]]


def sanitize_session_id(raw_id: str) -> str:
    """Sanitize a user-supplied session name into a safe filename stem."""
    cleaned = _SAFE_ID_RE.sub("-", raw_id.strip()).strip("-")
    if not cleaned:
        raise ValueError("Session ID must contain at least one letter or digit.")
    return cleaned[:64]


def generate_session_id() -> str:
    """Generate a sortable UTC timestamp session ID."""
    return datetime.now(UTC).strftime("session-%Y%m%d-%H%M%S")


class SessionStore:
    """Workspace-scoped JSONL transcript store (excluded from Git and agent tools)."""

    def __init__(self, workspace_root: Path) -> None:
        self._dir = workspace_root / SESSIONS_SUBDIR

    @property
    def sessions_dir(self) -> Path:
        return self._dir

    def _path_for(self, session_id: str) -> Path:
        safe = sanitize_session_id(session_id)
        return self._dir / f"{safe}.jsonl"

    def save(
        self,
        *,
        session_id: str,
        model: str,
        messages: list[dict[str, Any]],
        plan_mode: bool = False,
        title: str | None = None,
    ) -> SessionSummary:
        """Persist a conversation cleanly as line-delimited JSON (JSONL)."""
        safe_id = sanitize_session_id(session_id)
        path = self._path_for(safe_id)
        self._dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        created_at = now
        if path.is_file():
            existing = self._read_meta(path)
            if existing is not None:
                created_at = existing.created_at

        derived_title = title or _derive_title(messages) or safe_id
        non_system_count = sum(1 for m in messages if m.get("role") != "system")
        summary = SessionSummary(
            session_id=safe_id,
            title=derived_title,
            model=model,
            plan_mode=plan_mode,
            message_count=non_system_count,
            created_at=created_at,
            updated_at=now,
        )

        lines = [json.dumps({"type": "meta", **summary.model_dump()})]
        for msg in messages:
            lines.append(json.dumps({"type": "message", "payload": msg}))

        tmp_path = path.with_suffix(".jsonl.tmp")
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp_path.replace(path)
        return summary

    def load(self, session_id: str) -> LoadedSession:
        """Load a saved JSONL session by ID."""
        path = self._path_for(session_id)
        if not path.is_file():
            raise FileNotFoundError(f"Saved session not found: {session_id}")

        summary: SessionSummary | None = None
        messages: list[dict[str, Any]] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            rec_type = record.get("type")
            if rec_type == "meta":
                meta_fields = {k: v for k, v in record.items() if k != "type"}
                summary = SessionSummary(**meta_fields)
            elif rec_type == "message" and isinstance(record.get("payload"), dict):
                messages.append(record["payload"])

        if summary is None:
            now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            summary = SessionSummary(
                session_id=sanitize_session_id(session_id),
                title=session_id,
                model="unknown",
                message_count=len(messages),
                created_at=now,
                updated_at=now,
            )

        return LoadedSession(summary=summary, messages=messages)

    def list_sessions(self) -> list[SessionSummary]:
        """List saved sessions sorted newest first."""
        if not self._dir.is_dir():
            return []
        items: list[SessionSummary] = []
        for path in self._dir.glob("*.jsonl"):
            meta = self._read_meta(path)
            if meta is not None:
                items.append(meta)
        items.sort(key=lambda s: s.updated_at, reverse=True)
        return items

    def latest_session_id(self) -> str | None:
        """Return the most recently updated session ID, or None if no sessions exist."""
        sessions = self.list_sessions()
        return sessions[0].session_id if sessions else None

    def delete(self, session_id: str) -> bool:
        """Safely delete a saved JSONL session by ID. Returns True if removed."""
        path = self._path_for(session_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    @staticmethod
    def _read_meta(path: Path) -> SessionSummary | None:
        try:
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            record = json.loads(first_line)
            if record.get("type") == "meta":
                return SessionSummary(**{k: v for k, v in record.items() if k != "type"})
        except (OSError, IndexError, json.JSONDecodeError, ValueError):
            return None
        return None


def _derive_title(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            text = str(msg.get("content") or "").strip().splitlines()[0]
            if text:
                return text[:60] + ("..." if len(text) > 60 else "")
    return ""
