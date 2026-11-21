"""Direct OpenRouter HTTP client with native tool calling and usage metrics."""

from __future__ import annotations

import time
from typing import Any

import httpx
from pydantic import BaseModel, Field

from poseidon.config import Settings


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter API request fails."""


class MissingApiKeyError(OpenRouterError):
    """Raised when OPENROUTER_API_KEY is not configured."""


class ModelToolSupportError(OpenRouterError):
    """Raised when the selected model does not support tool calling or is unavailable."""


class ToolCall(BaseModel):
    """A single native tool call requested by the model."""

    id: str
    name: str
    arguments: str


class TokenUsage(BaseModel):
    """Token and cost metadata returned by OpenRouter."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None


class ChatResponse(BaseModel):
    """Normalized response from a chat completion call."""

    model: str
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage
    elapsed_seconds: float
    finish_reason: str | None = None

    def to_assistant_message(self) -> dict[str, Any]:
        """Build the OpenRouter/OpenAI-compliant assistant message for conversation history."""
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": self.content if self.content else None,
        }
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


class OpenRouterClient:
    """Direct REST client for the OpenRouter Chat Completions API."""

    def __init__(self, settings: Settings, http_client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._client = http_client or httpx.Client(
            base_url=settings.openrouter_base_url.rstrip("/"),
            timeout=settings.request_timeout_seconds,
        )
        self._owns_client = http_client is None
        self._models_catalog: dict[str, dict[str, Any]] | None = None

    def fetch_models_catalog(self) -> dict[str, dict[str, Any]]:
        """Fetch and cache the public OpenRouter models catalog (zero API spend)."""
        if self._models_catalog is not None:
            return self._models_catalog
        try:
            response = self._client.get("/models")
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"Could not fetch OpenRouter model catalog: {exc}") from exc
        if response.status_code >= 400:
            raise OpenRouterError(
                f"Failed to fetch OpenRouter model catalog (HTTP {response.status_code})."
            )
        items = response.json().get("data") or []
        self._models_catalog = {
            str(item["id"]): item for item in items if isinstance(item, dict) and item.get("id")
        }
        return self._models_catalog

    def validate_tool_model(self, model_id: str) -> dict[str, Any]:
        """Verify that model_id exists on OpenRouter and supports native tool calling."""
        catalog = self.fetch_models_catalog()
        info = catalog.get(model_id)
        if info is None:
            raise ModelToolSupportError(
                f"Model '{model_id}' is unavailable or not found on OpenRouter."
            )
        supported = info.get("supported_parameters") or []
        if "tools" not in supported:
            raise ModelToolSupportError(
                f"Model '{model_id}' does not support native tool calling "
                f"(supported_parameters={supported})."
            )
        return info

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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
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
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        started = time.perf_counter()
        try:
            response = self._client.post("/chat/completions", headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"Network error contacting OpenRouter: {exc}") from exc
        elapsed = time.perf_counter() - started

        if response.status_code >= 400:
            detail = _extract_error_detail(response)
            lower_detail = detail.lower()
            if tools and (
                "tool" in lower_detail
                or "function" in lower_detail
                or response.status_code == 404
            ):
                raise ModelToolSupportError(
                    f"Model '{model}' is unavailable or does not support tool calling "
                    f"(HTTP {response.status_code}): {detail}"
                )
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
        parsed_tool_calls = _parse_tool_calls(message.get("tool_calls"))
        raw_usage = data.get("usage") or {}

        usage = TokenUsage(
            prompt_tokens=raw_usage.get("prompt_tokens"),
            completion_tokens=raw_usage.get("completion_tokens"),
            reasoning_tokens=_extract_reasoning_tokens(raw_usage),
            total_tokens=raw_usage.get("total_tokens"),
            cost_usd=raw_usage.get("cost"),
        )

        return ChatResponse(
            model=data.get("model") or model,
            content=content,
            tool_calls=parsed_tool_calls,
            usage=usage,
            elapsed_seconds=elapsed,
            finish_reason=finish_reason,
        )


def _parse_tool_calls(raw_calls: Any) -> list[ToolCall]:
    """Normalize OpenRouter tool_calls entries into typed ToolCall models."""
    if not isinstance(raw_calls, list):
        return []
    parsed: list[ToolCall] = []
    for idx, item in enumerate(raw_calls):
        if not isinstance(item, dict):
            continue
        fn = item.get("function") or {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        call_id = str(item.get("id") or f"call_{idx}")
        arguments = fn.get("arguments")
        if not isinstance(arguments, str):
            arguments = "{}" if arguments is None else str(arguments)
        parsed.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return parsed


def _extract_reasoning_tokens(raw_usage: dict[str, Any]) -> int | None:
    """Extract hidden reasoning tokens when reported by OpenRouter."""
    details = raw_usage.get("completion_tokens_details")
    if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
        return int(details["reasoning_tokens"])
    if raw_usage.get("reasoning_tokens") is not None:
        return int(raw_usage["reasoning_tokens"])
    return None


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
