"""Strict, atomic unified diff parser and applier with stale-file verification."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from poseidon.workspace import WorkspaceGuard

_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DIFF_TAG_RE = re.compile(r"<diff>\s*\n?(.*?)\n?</diff>", re.DOTALL | re.IGNORECASE)
_DIFF_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)\n```", re.DOTALL)


class PatchError(ValueError):
    """Base error raised when a unified diff cannot be parsed or applied safely."""


class StaleFileError(PatchError):
    """Raised when a target file changed on disk since the model read it."""


class PatchApplyError(PatchError):
    """Raised when one or more hunks fail to match the target file."""


@dataclass(frozen=True)
class DiffHunk:
    """A single unified diff hunk."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


@dataclass(frozen=True)
class FilePatch:
    """Parsed unified diff targeting a single workspace file."""

    old_path: str
    new_path: str
    hunks: list[DiffHunk]

    @property
    def target_path(self) -> str:
        return self.new_path if self.new_path != "/dev/null" else self.old_path


@dataclass(frozen=True)
class PatchResult:
    """Summary of an atomically applied unified diff."""

    modified_files: list[str]
    hunks_applied: int
    added_lines: int
    removed_lines: int


def sha256_text(content: str) -> str:
    """Compute a deterministic SHA-256 hex digest for UTF-8 text."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def extract_diff_payload(raw_response: str) -> str:
    """Extract a unified diff from <diff>...</diff>, ```diff``` fence, or raw unified diff text."""
    tag_match = _DIFF_TAG_RE.search(raw_response)
    if tag_match:
        payload = tag_match.group(1).strip("\r\n")
        if payload.strip():
            return payload

    fence_match = _DIFF_FENCE_RE.search(raw_response)
    if fence_match:
        payload = fence_match.group(1).strip("\r\n")
        if payload.strip():
            return payload

    # Fallback: raw response that already contains unified diff headers
    if "--- " in raw_response and "+++ " in raw_response and "@@ " in raw_response:
        lines = raw_response.splitlines()
        for idx, line in enumerate(lines):
            if line.startswith("--- ") or line.startswith("diff --git "):
                return "\n".join(lines[idx:]).strip("\r\n")

    raise PatchError(
        "Could not extract a unified diff (<diff>...</diff> or ```diff``` block) from response."
    )


def _strip_diff_prefix(path_token: str) -> str:
    """Normalize git-style a/ and b/ path prefixes and strip trailing timestamps."""
    token = path_token.split("\t", 1)[0].strip()
    if token.startswith(("a/", "b/")):
        return token[2:]
    return token


def parse_unified_diff(diff_text: str) -> list[FilePatch]:
    """Parse a unified diff string into structured FilePatch objects."""
    raw_lines = diff_text.splitlines()
    patches: list[FilePatch] = []
    i = 0
    n = len(raw_lines)

    while i < n:
        line = raw_lines[i]
        if line.startswith("diff --git ") or line.startswith(("index ", "new file ", "deleted ")):
            i += 1
            continue

        if not line.startswith("--- "):
            i += 1
            continue

        old_path = _strip_diff_prefix(line[4:])
        i += 1
        if i >= n or not raw_lines[i].startswith("+++ "):
            raise PatchError(f"Malformed diff header: missing '+++ ' line after '--- {old_path}'.")
        new_path = _strip_diff_prefix(raw_lines[i][4:])
        i += 1

        hunks: list[DiffHunk] = []
        while i < n:
            cur = raw_lines[i]
            if cur.startswith("--- ") or cur.startswith("diff --git "):
                break
            header_match = _HUNK_HEADER_RE.match(cur)
            if not header_match:
                if not cur.strip():
                    i += 1
                    continue
                raise PatchError(f"Unexpected line while expecting hunk header: {cur!r}")

            old_start = int(header_match.group(1))
            old_count = int(header_match.group(2)) if header_match.group(2) is not None else 1
            new_start = int(header_match.group(3))
            new_count = int(header_match.group(4)) if header_match.group(4) is not None else 1
            i += 1

            hunk_lines: list[str] = []
            while i < n:
                hline = raw_lines[i]
                if (
                    hline.startswith("@@ ")
                    or hline.startswith("--- ")
                    or hline.startswith("diff --git ")
                ):
                    break
                if hline == r"\ No newline at end of file":
                    i += 1
                    continue
                if hline == "":
                    # Tolerate blank context line whose leading space was stripped
                    hunk_lines.append(" ")
                    i += 1
                    continue
                if hline[0] not in (" ", "+", "-"):
                    raise PatchError(
                        f"Invalid diff line prefix {hline[0]!r} in hunk at line {old_start}."
                    )
                hunk_lines.append(hline)
                i += 1

            if not hunk_lines:
                raise PatchError(
                    f"Empty hunk at -{old_start},{old_count} +{new_start},{new_count}."
                )

            hunks.append(
                DiffHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    lines=hunk_lines,
                )
            )

        if not hunks:
            raise PatchError(f"Diff for '{new_path}' contains no hunks.")

        patches.append(FilePatch(old_path=old_path, new_path=new_path, hunks=hunks))

    if not patches:
        raise PatchError("No valid unified diff headers ('--- ' / '+++ ') found in patch.")

    return patches


def _apply_hunks_to_text(original_text: str, file_patch: FilePatch) -> tuple[str, int, int]:
    """Apply all hunks in memory and return (new_text, added, removed)."""
    had_trailing_newline = original_text.endswith("\n")
    lines = original_text.splitlines()
    offset = 0
    total_added = 0
    total_removed = 0

    for hunk_idx, hunk in enumerate(file_patch.hunks, start=1):
        expected_old = [hl[1:] for hl in hunk.lines if hl[0] in (" ", "-")]
        replacement_new = [hl[1:] for hl in hunk.lines if hl[0] in (" ", "+")]
        total_added += sum(1 for hl in hunk.lines if hl[0] == "+")
        total_removed += sum(1 for hl in hunk.lines if hl[0] == "-")

        expected_idx = max(0, (hunk.old_start - 1) + offset)
        match_idx = _find_hunk_position(lines, expected_old, expected_idx)

        if match_idx is None:
            lo = max(0, expected_idx - 3)
            hi = min(len(lines), expected_idx + max(len(expected_old), 6) + 3)
            snippet = "\n".join(
                f"{ln:4d} | {lines[ln - 1]}" for ln in range(lo + 1, hi + 1)
            )
            raise PatchApplyError(
                f"Hunk #{hunk_idx} for '{file_patch.target_path}' (around line {hunk.old_start}) "
                "did not match the file contents; patch aborted and file left unchanged.\n"
                f"Actual lines around line {hunk.old_start}:\n{snippet}"
            )

        lines[match_idx : match_idx + len(expected_old)] = replacement_new
        offset += len(replacement_new) - len(expected_old)

    new_text = "\n".join(lines)
    if had_trailing_newline or (new_text and not new_text.endswith("\n")):
        new_text += "\n"
    return new_text, total_added, total_removed


def _find_hunk_position(
    file_lines: list[str],
    expected_old: list[str],
    preferred_idx: int,
    max_drift: int = 8,
) -> int | None:
    """Locate the exact matching slice near preferred_idx (exact match required, small drift ok)."""
    length = len(expected_old)
    if length == 0:
        return min(preferred_idx, len(file_lines))

    def matches_at(idx: int) -> bool:
        return (
            idx >= 0
            and idx + length <= len(file_lines)
            and file_lines[idx : idx + length] == expected_old
        )

    if matches_at(preferred_idx):
        return preferred_idx

    for delta in range(1, max_drift + 1):
        if matches_at(preferred_idx - delta):
            return preferred_idx - delta
        if matches_at(preferred_idx + delta):
            return preferred_idx + delta

    return None


def apply_unified_diff(
    *,
    workspace: WorkspaceGuard,
    diff_text: str,
    expected_hashes: dict[str, str] | None = None,
    dry_run: bool = False,
) -> PatchResult:
    """Validate base hashes and apply a unified diff atomically inside the workspace.

    If ANY stale-file check or hunk validation fails, raises PatchError and leaves
    all workspace files completely untouched.
    """
    file_patches = parse_unified_diff(diff_text)
    staged_writes: dict[str, str] = {}
    total_hunks = 0
    total_added = 0
    total_removed = 0

    for fp in file_patches:
        target = fp.target_path
        # Validate path containment before touching disk
        workspace.resolve_path(target)

        try:
            current_text = workspace.read_text(target)
        except FileNotFoundError as exc:
            raise PatchApplyError(
                f"Target file '{target}' does not exist in workspace; patch aborted."
            ) from exc

        if expected_hashes is not None and target in expected_hashes:
            actual_hash = sha256_text(current_text)
            if actual_hash != expected_hashes[target]:
                raise StaleFileError(
                    f"Stale file check failed for '{target}': file changed on disk since the "
                    "model read it; patch aborted and file left unchanged."
                )

        new_text, added, removed = _apply_hunks_to_text(current_text, fp)
        staged_writes[target] = new_text
        total_hunks += len(fp.hunks)
        total_added += added
        total_removed += removed

    if total_added == 0 and total_removed == 0:
        raise PatchApplyError(
            "No-op patch rejected: diff contains no added ('+') or removed ('-') lines."
        )

    if not dry_run:
        for rel_path, new_content in staged_writes.items():
            workspace.write_text(rel_path, new_content)

    return PatchResult(
        modified_files=list(staged_writes.keys()),
        hunks_applied=total_hunks,
        added_lines=total_added,
        removed_lines=total_removed,
    )
