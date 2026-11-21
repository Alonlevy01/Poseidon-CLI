"""Unit tests for unified diff parsing, stale-file checks, and atomic patching."""

from pathlib import Path

import pytest

from poseidon.patching.unified_diff import (
    PatchApplyError,
    PatchError,
    StaleFileError,
    apply_unified_diff,
    extract_diff_payload,
    sha256_text,
)
from poseidon.workspace import WorkspaceGuard, WorkspaceSecurityError

SAMPLE_SOURCE = """def greet(name: str) -> str:
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    return a - b
"""


def test_apply_valid_unified_diff(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("sample.py", SAMPLE_SOURCE)
    base_hash = sha256_text(SAMPLE_SOURCE)

    diff = """--- a/sample.py
+++ b/sample.py
@@ -4,3 +4,3 @@
 def add(a: int, b: int) -> int:
-    return a - b
+    return a + b
"""
    res = apply_unified_diff(
        workspace=guard,
        diff_text=diff,
        expected_hashes={"sample.py": base_hash},
    )
    assert res.modified_files == ["sample.py"]
    assert res.hunks_applied == 1
    assert res.added_lines == 1
    assert res.removed_lines == 1
    assert "return a + b" in guard.read_text("sample.py")


def test_stale_file_check_rejects_and_leaves_file_intact(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("sample.py", SAMPLE_SOURCE)
    old_hash = sha256_text(SAMPLE_SOURCE)

    # Simulate external edit after the model read the file
    modified = SAMPLE_SOURCE.replace("Hello", "Hi")
    guard.write_text("sample.py", modified)

    diff = """--- a/sample.py
+++ b/sample.py
@@ -4,3 +4,3 @@
 def add(a: int, b: int) -> int:
-    return a - b
+    return a + b
"""
    with pytest.raises(StaleFileError, match="Stale file check failed"):
        apply_unified_diff(
            workspace=guard,
            diff_text=diff,
            expected_hashes={"sample.py": old_hash},
        )
    # File must remain completely untouched
    assert guard.read_text("sample.py") == modified


def test_bad_patch_is_atomic_and_leaves_file_intact(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("sample.py", SAMPLE_SOURCE)

    # Hunk 1 is valid, Hunk 2 has wrong context -> entire patch must abort
    bad_diff = """--- a/sample.py
+++ b/sample.py
@@ -1,2 +1,2 @@
 def greet(name: str) -> str:
-    return f"Hello, {name}!"
+    return f"Ahoy, {name}!"
@@ -5,2 +5,2 @@
 def add(a: int, b: int) -> int:
-    return a * b
+    return a + b
"""
    with pytest.raises(PatchApplyError, match="Hunk #2") as exc_info:
        apply_unified_diff(workspace=guard, diff_text=bad_diff)
    assert "Actual lines around line" in str(exc_info.value)
    assert "return a - b" in str(exc_info.value)

    # Hunk 1 must NOT have been written
    assert guard.read_text("sample.py") == SAMPLE_SOURCE


def test_diff_cannot_escape_workspace(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    evil_diff = """--- a/../outside.py
+++ b/../outside.py
@@ -1,1 +1,1 @@
-x = 1
+x = 2
"""
    with pytest.raises(WorkspaceSecurityError):
        apply_unified_diff(workspace=guard, diff_text=evil_diff)


def test_extract_diff_payload_variants() -> None:
    tagged = "Notes\n<diff>\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n</diff>"
    assert extract_diff_payload(tagged).startswith("--- a/x.py")

    fenced = "```diff\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n```"
    assert extract_diff_payload(fenced).startswith("--- a/x.py")

    with pytest.raises(PatchError):
        extract_diff_payload("No diff here.")


def test_noop_diff_is_rejected(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("sample.py", SAMPLE_SOURCE)
    noop_diff = """--- a/sample.py
+++ b/sample.py
@@ -4,3 +4,3 @@
 def add(a: int, b: int) -> int:
     return a - b
"""
    with pytest.raises(PatchApplyError, match="No-op patch rejected"):
        apply_unified_diff(workspace=guard, diff_text=noop_diff)
