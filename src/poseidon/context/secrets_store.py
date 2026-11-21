"""Secure user-level API key store kept strictly outside the project workspace and Git.

Keys are stored with 0600 permissions inside the user home directory (backed by a
dedicated Docker volume `poseidon_secrets`), never inside `/workspace`, never logged,
and never returned in plaintext by any API or UI endpoint.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

from poseidon.config import Settings

SUPPORTED_KEY_PROVIDERS: dict[str, tuple[str, str]] = {
    "openrouter": ("OpenRouter", "OPENROUTER_API_KEY"),
    "anthropic": ("Anthropic Claude", "ANTHROPIC_API_KEY"),
    "gemini": ("Google Gemini", "GEMINI_API_KEY"),
    "openai": ("OpenAI GPT", "OPENAI_API_KEY"),
}


def default_secrets_path() -> Path:
    """Resolve user-level secrets file outside any workspace repository."""
    override = os.environ.get("POSEIDON_SECRETS_FILE", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".poseidon_secrets" / "keys.json"


def _mask_key(raw: str) -> str:
    cleaned = raw.strip()
    if len(cleaned) <= 8:
        return "••••"
    return f"{cleaned[:6]}…{cleaned[-4:]}"


class ProviderSecretsStore:
    """Read/write user-level provider API keys without exposing plaintext secrets."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or default_secrets_path()

    @property
    def path(self) -> Path:
        return self._path

    def _read_map(self) -> dict[str, str]:
        if not self._path.is_file():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return {
                    str(k): str(v)
                    for k, v in raw.items()
                    if str(k) in SUPPORTED_KEY_PROVIDERS and str(v).strip()
                }
        except (OSError, json.JSONDecodeError):
            return {}
        return {}

    def _write_map(self, data: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(self._path.parent, 0o700)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        with contextlib.suppress(OSError):
            os.chmod(tmp, 0o600)
        tmp.replace(self._path)

    def set_key(self, provider: str, api_key: str) -> None:
        """Store an API key for one supported cloud provider."""
        prov = provider.lower().strip()
        if prov not in SUPPORTED_KEY_PROVIDERS:
            valid = ", ".join(SUPPORTED_KEY_PROVIDERS)
            raise ValueError(f"Unknown provider '{provider}'. Supported: {valid}.")
        cleaned = api_key.strip()
        if not cleaned:
            raise ValueError("API key cannot be empty (use clear_key to remove).")
        data = self._read_map()
        data[prov] = cleaned
        self._write_map(data)

    def clear_key(self, provider: str) -> bool:
        """Remove a saved API key for one provider."""
        prov = provider.lower().strip()
        if prov not in SUPPORTED_KEY_PROVIDERS:
            valid = ", ".join(SUPPORTED_KEY_PROVIDERS)
            raise ValueError(f"Unknown provider '{provider}'. Supported: {valid}.")
        data = self._read_map()
        if prov not in data:
            return False
        del data[prov]
        self._write_map(data)
        return True

    def resolve_key(self, provider: str, settings: Settings | None = None) -> str:
        """Resolve effective API key: saved user key first, then Settings / env var."""
        prov = provider.lower().strip()
        saved = self._read_map().get(prov, "").strip()
        if saved:
            return saved
        cfg = settings or Settings()
        env_map = {
            "openrouter": cfg.openrouter_api_key.get_secret_value(),
            "anthropic": cfg.anthropic_api_key.get_secret_value(),
            "gemini": cfg.gemini_api_key.get_secret_value(),
            "openai": cfg.openai_api_key.get_secret_value(),
        }
        val = env_map.get(prov, "").strip()
        if val:
            return val
        _, env_name = SUPPORTED_KEY_PROVIDERS.get(prov, ("", ""))
        return os.environ.get(env_name, "").strip()

    def masked_status(self, settings: Settings | None = None) -> dict[str, dict[str, Any]]:
        """Return non-secret key configuration status for UI and `/providers` display."""
        saved_map = self._read_map()
        status: dict[str, dict[str, Any]] = {}
        for prov, (label, env_var) in SUPPORTED_KEY_PROVIDERS.items():
            if prov in saved_map:
                status[prov] = {
                    "label": label,
                    "env_var": env_var,
                    "configured": True,
                    "source": "saved",
                    "hint": _mask_key(saved_map[prov]),
                }
            else:
                env_val = self.resolve_key(prov, settings=settings)
                status[prov] = {
                    "label": label,
                    "env_var": env_var,
                    "configured": bool(env_val),
                    "source": "env" if env_val else "none",
                    "hint": _mask_key(env_val) if env_val else "",
                }
        return status
