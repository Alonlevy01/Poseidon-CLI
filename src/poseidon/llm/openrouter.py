"""Direct OpenRouter HTTP client."""

from __future__ import annotations

import time
from typing import Any

import httpx
from pydantic import BaseModel

from poseidon.config import Settings


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter API request fails."""


class MissingApiKeyError(OpenRouterError):
    """Raised when OPENROUTER_API_KEY is not configured."""


class TokenUsage(BaseModel):
    """Token and cost metadata returned by OpenRouter."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None


class ChatResponse(BaseModel):
    """Normalized response from a chat completion call."""

    model: str
    content: str
    usage: TokenUsage
    elapsed_seconds: float
    finish_reason: str | None = None


class OpenRouterClient:
    """Direct REST client for the OpenRouter Chat Completions API."""

    def __init__(self, settings: Settings, http_client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._client = http_client or httpx.Client(
            base_url=settings.openrouter_base_url.rstrip("/"),
            timeout=settings.request_timeout_seconds,
        )
        self._owns_client = http_client is None

    def close(self) -> None:
        """Close the underlying HTTP client if owned by this instance."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
    ) -> ChatResponse:
        """Send a chat completion request directly to OpenRouter."""
        api_key = self._settings.openrouter_api_key.get_secret_value().strip()
        if not api_key:
            raise MissingApiKeyError(
                "OPENROUTER_API_KEY is not set. Add it to your environment or .env file."
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._settings.http_referer,
            "X-Title": self._settings.app_title,
        }
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        started = time.perf_counter()
        try:
            response = self._client.post("/chat/completions", headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"Network error contacting OpenRouter: {exc}") from exc
        elapsed = time.perf_counter() - started

        if response.status_code >= 400:
            detail = _extract_error_detail(response)
            raise OpenRouterError(
                f"OpenRouter request failed (HTTP {response.status_code}): {detail}"
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise OpenRouterError("OpenRouter returned no completion choices.")

        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        finish_reason = choices[0].get("finish_reason")
        raw_usage = data.get("usage") or {}

        usage = TokenUsage(
            prompt_tokens=raw_usage.get("prompt_tokens"),
            completion_tokens=raw_usage.get("completion_tokens"),
            total_tokens=raw_usage.get("total_tokens"),
            cost_usd=raw_usage.get("cost"),
        )

        return ChatResponse(
            model=data.get("model") or model,
            content=content,
            usage=usage,
            elapsed_seconds=elapsed,
            finish_reason=finish_reason,
        )


def _extract_error_detail(response: httpx.Response) -> str:
    """Safely extract an error message from an OpenRouter error payload."""
    try:
        body = response.json()
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if err:
            return str(err)
    except ValueError:
        pass
    return response.text[:300].strip() or "Unknown error"
