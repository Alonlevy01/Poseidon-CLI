"""Project memory loader supporting POSEIDON.md and AGENTS.md conventions.

Places project rules in the static top system prompt for optimal prompt-cache hit rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MEMORY_FILENAMES: tuple[str, ...] = ("POSEIDON.md", "AGENTS.md")
MAX_MEMORY_CHARS = 12_000


@dataclass(frozen=True)
class LoadedProjectMemory:
    """Loaded repository memory files and their combined prompt block."""

    sources: list[str]
    content: str


def load_project_memory(workspace_root: Path) -> LoadedProjectMemory:
    """Load POSEIDON.md and/or AGENTS.md from workspace_root if present."""
    sources: list[str] = []
    sections: list[str] = []
    remaining = MAX_MEMORY_CHARS

    for filename in MEMORY_FILENAMES:
        candidate = workspace_root / filename
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not text:
            continue
        if len(text) > remaining:
            text = text[:remaining].rstrip() + "\n... [truncated]"
        sources.append(filename)
        sections.append(f"### {filename}\n{text}")
        remaining -= len(text)
        if remaining <= 0:
            break

    return LoadedProjectMemory(
        sources=sources,
        content="\n\n".join(sections),
    )


def starter_poseidon_md(project_name: str) -> str:
    """Generate a clean starter POSEIDON.md template for `/init`."""
    return (
        f"# {project_name} — Poseidon Project Memory\n\n"
        "## Architecture & Conventions\n"
        "- Keep functions small, typed, and well-documented in English.\n"
        "- Preserve clean module boundaries (separate parsing, logic, formatting, and CLI).\n\n"
        "## Testing & Verification\n"
        "- Run unit tests (`pytest`) inside the isolated Docker sandbox after edits.\n"
        "- Never modify existing unit tests to mask an implementation bug.\n"
    )
