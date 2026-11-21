"""Lifecycle hook registry supporting pre_edit, post_edit, and post_test callbacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from poseidon.sandbox.docker_runner import TestRunResult

PreEditHook = Callable[[str, str], str | None]
PostEditHook = Callable[[list[str]], str | None]
PostTestHook = Callable[[TestRunResult], str | None]


@dataclass
class HookManager:
    """Runs registered plugin lifecycle hooks safely with error isolation."""

    pre_edit_hooks: list[tuple[str, PreEditHook]] = field(default_factory=list)
    post_edit_hooks: list[tuple[str, PostEditHook]] = field(default_factory=list)
    post_test_hooks: list[tuple[str, PostTestHook]] = field(default_factory=list)

    def add_pre_edit(self, name: str, fn: PreEditHook) -> None:
        self.pre_edit_hooks.append((name, fn))

    def add_post_edit(self, name: str, fn: PostEditHook) -> None:
        self.post_edit_hooks.append((name, fn))

    def add_post_test(self, name: str, fn: PostTestHook) -> None:
        self.post_test_hooks.append((name, fn))

    def run_pre_edit(self, rel_path: str, payload: str) -> str | None:
        """Run pre_edit hooks; return first blocking error message or None if allowed."""
        for name, fn in self.pre_edit_hooks:
            try:
                rejection = fn(rel_path, payload)
                if rejection:
                    return f"Blocked by pre_edit hook '{name}': {rejection}"
            except Exception as exc:
                return f"Blocked by failing pre_edit hook '{name}': {exc}"
        return None

    def run_post_edit(self, modified_files: list[str]) -> list[str]:
        """Run post_edit hooks and collect any informational notes."""
        notes: list[str] = []
        for name, fn in self.post_edit_hooks:
            try:
                note = fn(modified_files)
                if note:
                    notes.append(f"[hook:{name}] {note}")
            except Exception as exc:
                notes.append(f"[hook:{name} error] {exc}")
        return notes

    def run_post_test(self, result: TestRunResult) -> list[str]:
        """Run post_test hooks and collect any informational notes."""
        notes: list[str] = []
        for name, fn in self.post_test_hooks:
            try:
                note = fn(result)
                if note:
                    notes.append(f"[hook:{name}] {note}")
            except Exception as exc:
                notes.append(f"[hook:{name} error] {exc}")
        return notes

    def summary(self) -> dict[str, list[str]]:
        return {
            "pre_edit": [name for name, _ in self.pre_edit_hooks],
            "post_edit": [name for name, _ in self.post_edit_hooks],
            "post_test": [name for name, _ in self.post_test_hooks],
        }
