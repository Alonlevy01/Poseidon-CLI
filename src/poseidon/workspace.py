"""Workspace containment guard ensuring all file operations stay inside the target folder."""

from pathlib import Path


class WorkspaceSecurityError(ValueError):
    """Raised when an operation attempts to access a path outside the workspace."""


class WorkspaceGuard:
    """Enforces strict path containment within a single workspace root directory."""

    def __init__(self, root: Path | str) -> None:
        resolved_root = Path(root).expanduser().resolve()
        if resolved_root.exists() and not resolved_root.is_dir():
            raise WorkspaceSecurityError(f"Workspace path is not a directory: {resolved_root}")
        resolved_root.mkdir(parents=True, exist_ok=True)
        self.root: Path = resolved_root

    def resolve_path(self, relative_or_target: str | Path) -> Path:
        """Resolve a user/model-supplied path and verify it stays strictly inside the workspace."""
        raw_path = Path(relative_or_target)
        if raw_path.is_absolute():
            raise WorkspaceSecurityError(
                f"Absolute paths are not allowed in workspace operations: {relative_or_target}"
            )

        candidate = (self.root / raw_path).resolve()
        if not candidate.is_relative_to(self.root):
            raise WorkspaceSecurityError(
                f"Path escapes workspace root ({self.root}): {relative_or_target}"
            )
        if candidate == self.root:
            raise WorkspaceSecurityError(
                "Target path must point to a file, not the workspace root."
            )
        return candidate

    def list_files(self) -> list[str]:
        """List relative POSIX paths of non-hidden files inside the workspace."""
        files: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            files.append(rel.as_posix())
        return files

    def read_text(self, relative_path: str | Path) -> str:
        """Read UTF-8 text from a file inside the workspace."""
        target = self.resolve_path(relative_path)
        if not target.exists():
            raise FileNotFoundError(f"File not found in workspace: {relative_path}")
        return target.read_text(encoding="utf-8")

    def write_text(self, relative_path: str | Path, content: str) -> Path:
        """Write UTF-8 text to a file inside the workspace, creating parent dirs if needed."""
        target = self.resolve_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target
