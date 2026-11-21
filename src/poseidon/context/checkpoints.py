"""Lightweight workspace edit checkpoints enabling instant /undo rollback."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from poseidon.workspace import WorkspaceGuard


@dataclass
class WorkspaceCheckpoint:
    """Snapshot of file contents captured right before a workspace mutation."""

    id: int
    label: str
    created_at: str
    # Maps relative path -> previous UTF-8 text, or None if the file did not exist before the edit
    before_state: dict[str, str | None] = field(default_factory=dict)


class WorkspaceCheckpointManager:
    """Maintains an in-memory rollback stack for safe /undo inside a workspace session."""

    def __init__(self, workspace: WorkspaceGuard) -> None:
        self._workspace = workspace
        self._stack: list[WorkspaceCheckpoint] = []
        self._next_id = 1

    def capture(self, *, label: str, target_paths: list[str]) -> WorkspaceCheckpoint:
        """Snapshot current contents (or non-existence) of target_paths prior to editing."""
        before: dict[str, str | None] = {}
        for rel_path in target_paths:
            resolved = self._workspace.resolve_path(rel_path)
            if resolved.is_file():
                before[rel_path] = self._workspace.read_text(rel_path)
            else:
                before[rel_path] = None

        checkpoint = WorkspaceCheckpoint(
            id=self._next_id,
            label=label,
            created_at=datetime.now(UTC).strftime("%H:%M:%S UTC"),
            before_state=before,
        )
        self._next_id += 1
        self._stack.append(checkpoint)
        return checkpoint

    def list_checkpoints(self) -> list[WorkspaceCheckpoint]:
        """Return checkpoints in chronological order."""
        return list(self._stack)

    def undo(self) -> tuple[WorkspaceCheckpoint | None, list[str]]:
        """Revert the most recent checkpoint and return (checkpoint, affected_files)."""
        if not self._stack:
            return None, []

        checkpoint = self._stack.pop()
        restored: list[str] = []
        for rel_path, previous_content in checkpoint.before_state.items():
            resolved = self._workspace.resolve_path(rel_path)
            if previous_content is None:
                if resolved.exists():
                    resolved.unlink()
                restored.append(f"deleted {rel_path}")
            else:
                self._workspace.write_text(rel_path, previous_content)
                restored.append(f"restored {rel_path}")
        return checkpoint, restored
