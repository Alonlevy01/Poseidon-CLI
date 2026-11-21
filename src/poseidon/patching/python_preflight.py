"""Fast Python AST + static undefined-name preflight check for apply_diff and create_file.

Prevents broken Python syntax from ever landing on disk and immediately surfaces missing
imports or undefined names (ruff F821/F822/F823/E999) in the tool response so the model
can self-correct before wasting a full test-sandbox cycle.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreflightReport:
    """Result of static Python preflight checks on a candidate file."""

    path: str
    syntax_error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.syntax_error is None


def check_python_source(rel_path: str, source: str) -> PreflightReport:
    """Run AST syntax validation and fast undefined-name checks on a Python source file."""
    if not rel_path.endswith(".py"):
        return PreflightReport(path=rel_path)

    # 1. Hard blocking syntax check via stdlib ast.parse
    try:
        ast.parse(source, filename=rel_path)
    except SyntaxError as exc:
        line_info = f" line {exc.lineno}" if exc.lineno else ""
        return PreflightReport(
            path=rel_path,
            syntax_error=f"SyntaxError in '{rel_path}'{line_info}: {exc.msg}",
        )

    # 2. Fast undefined-name / broken-binding check via ruff (if installed)
    warnings = _run_ruff_undefined_checks(rel_path, source)
    return PreflightReport(path=rel_path, syntax_error=None, warnings=warnings)


def _run_ruff_undefined_checks(rel_path: str, source: str) -> list[str]:
    if shutil.which("ruff") is None:
        return []
    try:
        proc = subprocess.run(
            [
                "ruff",
                "check",
                "--isolated",
                "--no-cache",
                "--select",
                "F821,F822,F823",
                "--output-format",
                "json",
                "--stdin-filename",
                rel_path,
                "-",
            ],
            input=source,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    if not proc.stdout.strip():
        return []

    try:
        diagnostics = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []

    warnings: list[str] = []
    for diag in diagnostics:
        code = diag.get("code", "")
        msg = diag.get("message", "")
        loc = diag.get("location") or {}
        row = loc.get("row", "?")
        warnings.append(f"{rel_path}:{row} [{code}] {msg}")
    return warnings
