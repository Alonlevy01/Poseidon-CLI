"""Outbound privacy & secret redaction guard (gitleaks-style patterns).

Scrubs accidental hardcoded credentials/API keys from tool outputs before they leave
the local machine for cloud LLM providers or observability traces.
"""

from __future__ import annotations

import re

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "PRIVATE_KEY",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
        ),
    ),
    ("OPENROUTER_KEY", re.compile(r"\bsk-or-v1-[A-Za-z0-9]{20,}\b")),
    ("ANTHROPIC_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("OPENAI_KEY", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{24,}\b")),
    ("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b")),
    ("GITHUB_TOKEN", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("GITHUB_PAT", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b")),
    ("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{16,}\b")),
    ("BEARER_TOKEN", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{24,}\b")),
    (
        "SECRET_ASSIGNMENT",
        re.compile(
            r"""(?i)\b((?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|private[_-]?key)\s*[:=]\s*['"])([^'"\n]{8,})(['"])"""
        ),
    ),
)


def redact_secrets(text: str) -> tuple[str, int]:
    """Scrub known credential/API-key patterns from `text`, returning `(redacted, count)`."""
    if not text:
        return text, 0

    redacted = text
    total_hits = 0
    for label, pattern in _SECRET_PATTERNS:
        if label == "SECRET_ASSIGNMENT":
            def _repl_assign(match: re.Match[str], tag: str = label) -> str:
                nonlocal total_hits
                val = match.group(2)
                if val.startswith("[REDACTED"):
                    return match.group(0)
                total_hits += 1
                return f"{match.group(1)}[REDACTED:{tag}]{match.group(3)}"

            redacted = pattern.sub(_repl_assign, redacted)
        else:
            def _repl_token(match: re.Match[str], tag: str = label) -> str:
                nonlocal total_hits
                total_hits += 1
                return f"[REDACTED:{tag}]"

            redacted = pattern.sub(_repl_token, redacted)

    return redacted, total_hits
