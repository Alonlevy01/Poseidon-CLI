"""Fast lexical code search using ripgrep (--json) with strict secret/noise exclusions."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, Field

from poseidon.workspace import (
    EXCLUDED_DIRS,
    GENERATED_FILE_PATTERNS,
    SECRET_FILE_PATTERNS,
    WorkspaceGuard,
)


class TextSearchHit(BaseModel):
    """Single lexical search hit with path, 1-based line number, and matching line."""

    path: str
    line_number: int
    line_text: str


class TextSearchResult(BaseModel):
    """Structured result of a workspace text search."""

    query: str
    engine: str
    hits: list[TextSearchHit] = Field(default_factory=list)
    truncated: bool = False
    elapsed_ms: float = 0.0

    def format_for_model(self) -> str:
        """Format hits cleanly as `path:line: text` for the LLM and Respan trace."""
        if not self.hits:
            return f"No matches found for query: {self.query!r}"
        lines = [f"{h.path}:{h.line_number}: {h.line_text}" for h in self.hits]
        header = f"Found {len(self.hits)} match(es) for {self.query!r} ({self.engine}"
        if self.truncated:
            header += ", truncated"
        header += "):"
        return "\n".join([header, *lines])


def _build_rg_exclude_globs() -> list[str]:
    """Build explicit --glob exclusion flags for secrets, Git data, deps, and generated files."""
    globs: list[str] = []
    for d in sorted(EXCLUDED_DIRS):
        globs.extend(["--glob", f"!{d}", "--glob", f"!**/{d}/**"])
    for pat in (*SECRET_FILE_PATTERNS, *GENERATED_FILE_PATTERNS):
        globs.extend(["--glob", f"!{pat}", "--glob", f"!**/{pat}"])
    return globs


def _search_with_ripgrep(
    *,
    workspace: WorkspaceGuard,
    search_root: Path,
    query: str,
    regex: bool,
    max_results: int,
) -> tuple[list[TextSearchHit], bool]:
    cmd = [
        "rg",
        "--json",
        "--smart-case",
        "--no-messages",
        *_build_rg_exclude_globs(),
    ]
    if not regex:
        cmd.append("--fixed-strings")
    cmd.extend(["--", query, str(search_root)])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if proc.returncode not in (0, 1):
        raise ValueError(f"ripgrep search error: {proc.stderr.strip() or 'invalid pattern'}")

    hits: list[TextSearchHit] = []
    truncated = False
    for raw_line in proc.stdout.splitlines():
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue

        data = event.get("data") or {}
        path_text = (data.get("path") or {}).get("text")
        line_number = data.get("line_number")
        line_text = (data.get("lines") or {}).get("text", "").rstrip("\r\n")
        if not path_text or line_number is None:
            continue

        abs_path = Path(path_text).resolve()
        if not abs_path.is_relative_to(workspace.root):
            continue
        rel = abs_path.relative_to(workspace.root)
        # Defense-in-depth exclusion check
        if workspace.is_excluded(rel):
            continue

        if len(hits) >= max_results:
            truncated = True
            break
        hits.append(
            TextSearchHit(
                path=rel.as_posix(),
                line_number=int(line_number),
                line_text=line_text,
            )
        )

    return hits, truncated


def _search_with_python_fallback(
    *,
    workspace: WorkspaceGuard,
    search_root: Path,
    query: str,
    regex: bool,
    max_results: int,
) -> tuple[list[TextSearchHit], bool]:
    """Pure-Python fallback when ripgrep is not installed outside Docker."""
    flags = re.IGNORECASE if query.islower() else 0
    pattern = re.compile(query if regex else re.escape(query), flags)

    hits: list[TextSearchHit] = []
    truncated = False
    subdir_rel = (
        None if search_root == workspace.root else search_root.relative_to(workspace.root)
    )
    for rel_str in workspace.list_files(subdir=subdir_rel):
        try:
            content = workspace.read_text(rel_str)
        except (OSError, UnicodeDecodeError):
            continue
        for idx, line in enumerate(content.splitlines(), start=1):
            if pattern.search(line):
                if len(hits) >= max_results:
                    truncated = True
                    return hits, truncated
                hits.append(TextSearchHit(path=rel_str, line_number=idx, line_text=line))
    return hits, truncated


def search_workspace_text(
    *,
    workspace: WorkspaceGuard,
    query: str,
    regex: bool = False,
    max_results: int = 50,
    subdir: str | None = None,
) -> TextSearchResult:
    """Search non-excluded workspace files for text and return structured hits."""
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("Search query cannot be empty.")

    search_root = workspace.resolve_dir(subdir)
    started = time.perf_counter()

    if shutil.which("rg") is not None:
        hits, truncated = _search_with_ripgrep(
            workspace=workspace,
            search_root=search_root,
            query=cleaned_query,
            regex=regex,
            max_results=max_results,
        )
        engine = "ripgrep"
    else:
        hits, truncated = _search_with_python_fallback(
            workspace=workspace,
            search_root=search_root,
            query=cleaned_query,
            regex=regex,
            max_results=max_results,
        )
        engine = "python"

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return TextSearchResult(
        query=cleaned_query,
        engine=engine,
        hits=hits,
        truncated=truncated,
        elapsed_ms=elapsed_ms,
    )
