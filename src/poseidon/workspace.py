"""Workspace containment guard ensuring all file operations stay inside the target folder."""

from __future__ import annotations

import fnmatch
from pathlib import Path

EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "vendor",
        "site-packages",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".poseidon",
        ".poseidon_cache",
        "dist",
        "build",
        "coverage",
        "htmlcov",
        ".eggs",
    }
)

SECRET_FILE_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*secret*",
    "*credentials*",
    ".netrc",
    "id_rsa*",
    "id_ed25519*",
)

GENERATED_FILE_PATTERNS: tuple[str, ...] = (
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dll",
    "*.exe",
    "*.egg-info",
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.lock",
    "*.onnx",
    "*.index",
    "*.faiss",
)


class WorkspaceSecurityError(ValueError):
    """Raised when an operation attempts to access a path outside or forbidden in the workspace."""


class WorkspaceGuard:
    """Enforces strict path containment and secret/noise exclusion within one workspace root."""

    def __init__(self, root: Path | str) -> None:
        resolved_root = Path(root).expanduser().resolve()
        if resolved_root.exists() and not resolved_root.is_dir():
            raise WorkspaceSecurityError(f"Workspace path is not a directory: {resolved_root}")
        resolved_root.mkdir(parents=True, exist_ok=True)
        self.root: Path = resolved_root

    @staticmethod
    def is_secret_path(rel_path: Path) -> bool:
        """Return True if a relative path matches a secret/credential file pattern."""
        lower_name = rel_path.name.lower()
        return any(fnmatch.fnmatch(lower_name, pat) for pat in SECRET_FILE_PATTERNS)

    @classmethod
    def is_excluded(cls, rel_path: Path) -> bool:
        """Return True if a relative path is a secret, Git metadata, dependency, or generated."""
        if cls.is_secret_path(rel_path):
            return True
        for part in rel_path.parts[:-1]:
            if part in EXCLUDED_DIRS or part.startswith(".") or part.endswith(".egg-info"):
                return True
        name = rel_path.name
        if name.startswith("."):
            return True
        lower_name = name.lower()
        return any(fnmatch.fnmatch(lower_name, pat) for pat in GENERATED_FILE_PATTERNS)

    def resolve_dir(self, relative_dir: str | Path | None = None) -> Path:
        """Resolve a subdirectory inside the workspace (or workspace root if None/'.')."""
        if relative_dir is None or str(relative_dir).strip() in ("", "."):
            return self.root
        raw_path = Path(relative_dir)
        if raw_path.is_absolute():
            raise WorkspaceSecurityError(
                f"Absolute paths are not allowed in workspace operations: {relative_dir}"
            )
        candidate = (self.root / raw_path).resolve()
        if not candidate.is_relative_to(self.root):
            raise WorkspaceSecurityError(
                f"Directory escapes workspace root ({self.root}): {relative_dir}"
            )
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError(f"Directory not found in workspace: {relative_dir}")
        return candidate

    def resolve_path(self, relative_or_target: str | Path) -> Path:
        """Resolve a user/model-supplied file path and verify it stays strictly in the workspace."""
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
        rel = candidate.relative_to(self.root)
        if self.is_secret_path(rel):
            raise WorkspaceSecurityError(
                f"Access to secret/credential file is forbidden: {rel.as_posix()}"
            )
        return candidate

    def list_files(self, subdir: str | Path | None = None) -> list[str]:
        """List relative POSIX paths of non-excluded files inside the workspace."""
        search_root = self.resolve_dir(subdir)
        files: list[str] = []
        for path in sorted(search_root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root)
            if self.is_excluded(rel):
                continue
            files.append(rel.as_posix())
        return files

    def read_text(self, relative_path: str | Path) -> str:
        """Read UTF-8 text from a file inside the workspace."""
        target = self.resolve_path(relative_path)
        if not target.exists():
            raise FileNotFoundError(f"File not found in workspace: {relative_path}")
        return target.read_text(encoding="utf-8")

    def read_lines(
        self,
        relative_path: str | Path,
        *,
        start_line: int,
        end_line: int,
    ) -> tuple[list[tuple[int, str]], int, str]:
        """Read a 1-based inclusive line range; returns (lines, total_lines, full_text)."""
        if start_line < 1 or end_line < start_line:
            raise ValueError(
                f"Invalid line range {start_line}-{end_line}: must satisfy 1 <= start <= end."
            )
        full_text = self.read_text(relative_path)
        all_lines = full_text.splitlines()
        total = len(all_lines)
        clamped_end = min(end_line, total)
        sliced = [
            (line_no, all_lines[line_no - 1])
            for line_no in range(start_line, clamped_end + 1)
            if line_no <= total
        ]
        return sliced, total, full_text

    def write_text(self, relative_path: str | Path, content: str) -> Path:
        """Write UTF-8 text to a file inside the workspace, creating parent dirs if needed."""
        target = self.resolve_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target
