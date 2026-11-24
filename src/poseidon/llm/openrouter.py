"""Direct multi-provider LLM client (OpenRouter, Anthropic, Gemini, OpenAI, Ollama, LM Studio)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel, Field

from poseidon.config import Settings
from poseidon.context.secrets_store import ProviderSecretsStore

OPENAI_BASE_URL = "https://api.openai.com/v1"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

DIRECT_PROVIDER_PRESETS: list[tuple[str, str]] = [
    ("anthropic:claude-sonnet-4-5", "Anthropic Claude Sonnet 4.5 (requires ANTHROPIC_API_KEY)"),
    ("anthropic:claude-haiku-4-5", "Anthropic Claude Haiku 4.5 (fast, requires ANTHROPIC_API_KEY)"),
    ("gemini:gemini-2.5-pro", "Google Gemini 2.5 Pro (requires GEMINI_API_KEY)"),
    ("gemini:gemini-2.5-flash", "Google Gemini 2.5 Flash (fast, requires GEMINI_API_KEY)"),
    ("openai:gpt-4.1", "OpenAI GPT-4.1 (requires OPENAI_API_KEY)"),
    ("openai:gpt-4.1-mini", "OpenAI GPT-4.1 Mini (fast, requires OPENAI_API_KEY)"),
]


class OpenRouterError(RuntimeError):
    """Raised when an LLM provider API request fails."""


class MissingApiKeyError(OpenRouterError):
    """Raised when the required provider API key is not configured."""


class ModelToolSupportError(OpenRouterError):
    """Raised when the selected model does not support tool calling or is unavailable."""


class ToolCall(BaseModel):
    """A single native tool call requested by the model."""

    id: str
    name: str
    arguments: str


class TokenUsage(BaseModel):
    """Token and cost metadata returned by the provider."""

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
        """Build the OpenAI-compliant assistant message for conversation history."""
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


def consume_openai_sse_lines(
    lines: Any,
    *,
    requested_model: str,
    elapsed: float,
    on_token: Callable[[str], None] | None = None,
) -> ChatResponse:
    """Assemble a ChatResponse from OpenAI/OpenRouter/Ollama SSE `data: ...` lines."""
    content_parts: list[str] = []
    tool_acc: dict[int, dict[str, str]] = {}
    resolved_model = requested_model
    finish_reason: str | None = None
    usage = TokenUsage()

    for raw_line in lines:
        line = str(raw_line).strip()
        if not line or not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if chunk.get("model"):
            resolved_model = str(chunk["model"])
        raw_usage = chunk.get("usage")
        if isinstance(raw_usage, dict) and raw_usage:
            usage = TokenUsage(
                prompt_tokens=raw_usage.get("prompt_tokens"),
                completion_tokens=raw_usage.get("completion_tokens"),
                reasoning_tokens=(raw_usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                ),
                total_tokens=raw_usage.get("total_tokens"),
                cost_usd=raw_usage.get("cost"),
            )

        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason"):
            finish_reason = str(choice["finish_reason"])
        delta = choice.get("delta") or {}

        piece = delta.get("content")
        if isinstance(piece, str) and piece:
            content_parts.append(piece)
            if on_token is not None:
                on_token(piece)

        for tc_delta in delta.get("tool_calls") or []:
            idx = int(tc_delta.get("index", 0))
            slot = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc_delta.get("id"):
                slot["id"] = str(tc_delta["id"])
            fn = tc_delta.get("function") or {}
            if fn.get("name"):
                slot["name"] += str(fn["name"])
            if fn.get("arguments"):
                slot["arguments"] += str(fn["arguments"])

    tool_calls = [
        ToolCall(
            id=slot["id"] or f"call_{idx}",
            name=slot["name"],
            arguments=slot["arguments"] or "{}",
        )
        for idx, slot in sorted(tool_acc.items())
        if slot["name"]
    ]
    return ChatResponse(
        model=resolved_model,
        content="".join(content_parts),
        tool_calls=tool_calls,
        usage=usage,
        elapsed_seconds=elapsed,
        finish_reason=finish_reason,
    )


def parse_provider_model(model_id: str) -> tuple[str, str]:
    """Parse provider prefix (`ollama:`, `lmstudio:`, `openai:`, `gemini:`, `anthropic:`)."""
    cleaned = model_id.strip()
    for prefix in ("ollama:", "lmstudio:", "openai:", "gemini:", "anthropic:"):
        if cleaned.lower().startswith(prefix):
            provider = prefix[:-1]
            target = cleaned[len(prefix) :].strip()
            return provider, target
    return "openrouter", cleaned


def discover_local_models(
    *,
    ollama_base_url: str = "http://host.docker.internal:11434/v1",
    lmstudio_base_url: str = "http://host.docker.internal:1234/v1",
    timeout_seconds: float = 2.5,
) -> dict[str, dict[str, Any]]:
    """Probe local Ollama and LM Studio servers for currently available models."""
    out: dict[str, dict[str, Any]] = {}
    targets = (
        ("ollama", ollama_base_url.rstrip("/")),
        ("lmstudio", lmstudio_base_url.rstrip("/")),
    )
    for provider, base_url in targets:
        models: list[str] = []
        reachable = False
        error: str | None = None
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                resp = client.get(f"{base_url}/models")
                if resp.status_code == 200:
                    reachable = True
                    payload = resp.json()
                    for item in payload.get("data", []):
                        mid = str(item.get("id") or "").strip()
                        if mid:
                            models.append(f"{provider}:{mid}")
                elif provider == "ollama":
                    # Fallback to Ollama native /api/tags endpoint
                    root = base_url[:-3] if base_url.endswith("/v1") else base_url
                    r2 = client.get(f"{root}/api/tags")
                    if r2.status_code == 200:
                        reachable = True
                        for item in r2.json().get("models", []):
                            name = str(item.get("name") or "").strip()
                            if name:
                                models.append(f"ollama:{name}")
                else:
                    error = f"HTTP {resp.status_code}"
        except httpx.HTTPError as exc:
            error = str(exc)

        out[provider] = {
            "base_url": base_url,
            "reachable": reachable,
            "models": sorted(set(models)),
            "error": error,
        }
    return out


class OpenRouterClient:
    """Direct REST client for OpenRouter, Anthropic Claude, Gemini, OpenAI, Ollama & LM Studio."""

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.Client | None = None,
        *,
        ollama_base_url: str | None = None,
        lmstudio_base_url: str | None = None,
        secrets_store: ProviderSecretsStore | None = None,
    ) -> None:
        self._settings = settings
        self._client = http_client or httpx.Client(
            base_url=settings.openrouter_base_url.rstrip("/"),
            timeout=settings.request_timeout_seconds,
        )
        self._owns_client = http_client is None
        self._models_catalog: dict[str, dict[str, Any]] | None = None
        self._ollama_base_url = (ollama_base_url or settings.ollama_base_url).rstrip("/")
        self._lmstudio_base_url = (lmstudio_base_url or settings.lmstudio_base_url).rstrip("/")
        self._secrets = secrets_store or ProviderSecretsStore()

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
        """Verify that model_id is configured and supports native tool calling."""
        provider, target_model = parse_provider_model(model_id)
        if not target_model:
            raise ModelToolSupportError(f"Invalid empty model ID: '{model_id}'.")

        if provider in {"ollama", "lmstudio"}:
            return {"id": model_id, "provider": provider, "local": True}

        if provider in {"openai", "gemini", "anthropic"}:
            key = self._secrets.resolve_key(provider, self._settings)
            if not key:
                raise MissingApiKeyError(
                    f"API key for '{provider}' is not set. Add it in Settings -> Providers "
                    f"or via `/key {provider} set`."
                )
            return {"id": model_id, "provider": provider, "local": False}

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
        on_token: Callable[[str], None] | None = None,
    ) -> ChatResponse:
        """Route a chat completion to OpenRouter, Claude, Gemini, GPT, Ollama, or LM Studio."""
        provider, target_model = parse_provider_model(model)

        if provider == "anthropic":
            return self._anthropic_completion(
                full_model=model,
                target_model=target_model,
                messages=messages,
                temperature=temperature,
                tools=tools,
            )

        if provider in {"ollama", "lmstudio", "openai", "gemini"}:
            return self._openai_compatible_completion(
                provider=provider,
                full_model=model,
                target_model=target_model,
                messages=messages,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                on_token=on_token,
            )

        # Default: OpenRouter
        api_key = self._secrets.resolve_key("openrouter", self._settings)
        if not api_key:
            raise MissingApiKeyError(
                "OPENROUTER_API_KEY is not set. Add it in Settings -> Providers or your .env file."
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._settings.http_referer,
            "X-Title": self._settings.app_title,
        }
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        started = time.perf_counter()
        if on_token is not None:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            try:
                with self._client.stream(
                    "POST", "/chat/completions", headers=headers, json=payload
                ) as resp:
                    if resp.status_code >= 400:
                        resp.read()
                        return _parse_openai_response(
                            response=resp,
                            provider_label="OpenRouter",
                            requested_model=model,
                            tools=tools,
                            elapsed=time.perf_counter() - started,
                        )
                    return consume_openai_sse_lines(
                        resp.iter_lines(),
                        requested_model=model,
                        elapsed=time.perf_counter() - started,
                        on_token=on_token,
                    )
            except httpx.HTTPError as exc:
                raise OpenRouterError(f"Network error contacting OpenRouter: {exc}") from exc

        try:
            response = self._client.post("/chat/completions", headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"Network error contacting OpenRouter: {exc}") from exc
        elapsed = time.perf_counter() - started
        return _parse_openai_response(
            response=response,
            provider_label="OpenRouter",
            requested_model=model,
            tools=tools,
            elapsed=elapsed,
        )

    def _openai_compatible_completion(
        self,
        *,
        provider: str,
        full_model: str,
        target_model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
        on_token: Callable[[str], None] | None = None,
    ) -> ChatResponse:
        if provider == "ollama":
            base_url = self._ollama_base_url
            api_key = "ollama"
            label = "Ollama"
        elif provider == "lmstudio":
            base_url = self._lmstudio_base_url
            api_key = "lm-studio"
            label = "LM Studio"
        elif provider == "openai":
            base_url = OPENAI_BASE_URL
            api_key = self._secrets.resolve_key("openai", self._settings)
            label = "OpenAI"
        else:
            base_url = GEMINI_OPENAI_BASE_URL
            api_key = self._secrets.resolve_key("gemini", self._settings)
            label = "Google Gemini"

        if provider in {"openai", "gemini"} and not api_key:
            raise MissingApiKeyError(
                f"{label} API key is not set. Add it in Settings -> Providers "
                f"or `/key {provider} set`."
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        url = f"{base_url.rstrip('/')}/chat/completions"
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self._settings.request_timeout_seconds) as client:
                if on_token is not None:
                    payload["stream"] = True
                    with client.stream("POST", url, headers=headers, json=payload) as resp:
                        if resp.status_code >= 400:
                            resp.read()
                            return _parse_openai_response(
                                response=resp,
                                provider_label=label,
                                requested_model=full_model,
                                tools=tools,
                                elapsed=time.perf_counter() - started,
                            )
                        return consume_openai_sse_lines(
                            resp.iter_lines(),
                            requested_model=full_model,
                            elapsed=time.perf_counter() - started,
                            on_token=on_token,
                        )
                response = client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            hint = (
                " Is your local server running with network access enabled?"
                if provider in {"ollama", "lmstudio"}
                else ""
            )
            raise OpenRouterError(
                f"Network error contacting {label} ({base_url}): {exc}{hint}"
            ) from exc
        elapsed = time.perf_counter() - started
        return _parse_openai_response(
            response=response,
            provider_label=label,
            requested_model=full_model,
            tools=tools,
            elapsed=elapsed,
            is_local=provider in {"ollama", "lmstudio"},
        )

    def _anthropic_completion(
        self,
        *,
        full_model: str,
        target_model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]] | None,
    ) -> ChatResponse:
        api_key = self._secrets.resolve_key("anthropic", self._settings)
        if not api_key:
            raise MissingApiKeyError(
                "Anthropic API key is not set. Add it in Settings -> Providers "
                "or via `/key anthropic set`."
            )

        system_parts, anthropic_msgs = _convert_messages_to_anthropic(messages)
        payload: dict[str, Any] = {
            "model": target_model,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": anthropic_msgs,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if tools:
            payload["tools"] = _convert_tools_to_anthropic(tools)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self._settings.request_timeout_seconds) as client:
                response = client.post(ANTHROPIC_MESSAGES_URL, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"Network error contacting Anthropic: {exc}") from exc
        elapsed = time.perf_counter() - started

        if response.status_code >= 400:
            detail = _extract_error_detail(response)
            raise OpenRouterError(
                f"Anthropic request failed (HTTP {response.status_code}): {detail}"
            )

        data = response.json()
        text_chunks: list[str] = []
        parsed_tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_chunks.append(str(block.get("text") or ""))
            elif btype == "tool_use":
                parsed_tool_calls.append(
                    ToolCall(
                        id=str(block.get("id") or f"call_{len(parsed_tool_calls)}"),
                        name=str(block.get("name") or ""),
                        arguments=json.dumps(block.get("input") or {}),
                    )
                )

        raw_usage = data.get("usage") or {}
        in_tok = raw_usage.get("input_tokens")
        out_tok = raw_usage.get("output_tokens")
        total_tok = (in_tok + out_tok) if (in_tok is not None and out_tok is not None) else None
        return ChatResponse(
            model=full_model,
            content="\n".join(text_chunks).strip(),
            tool_calls=parsed_tool_calls,
            usage=TokenUsage(
                prompt_tokens=in_tok,
                completion_tokens=out_tok,
                total_tokens=total_tok,
                cost_usd=None,
            ),
            elapsed_seconds=elapsed,
            finish_reason=data.get("stop_reason"),
        )


def _convert_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in tools:
        fn = item.get("function") or {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        converted.append(
            {
                "name": name,
                "description": str(fn.get("description") or ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return converted


def _convert_messages_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            text = str(msg.get("content") or "").strip()
            if text:
                system_parts.append(text)
            continue
        if role == "user":
            raw_content = msg.get("content")
            if isinstance(raw_content, list):
                user_blocks: list[dict[str, Any]] = []
                for part in raw_content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type")
                    if ptype == "text":
                        user_blocks.append({"type": "text", "text": str(part.get("text") or "")})
                    elif ptype == "image_url":
                        url_str = str((part.get("image_url") or {}).get("url") or "")
                        if url_str.startswith("data:") and ";base64," in url_str:
                            header, b64_data = url_str.split(";base64,", 1)
                            media_type = header[5:] or "image/png"
                            user_blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64_data,
                                    },
                                }
                            )
                out.append({"role": "user", "content": user_blocks or ""})
            else:
                out.append({"role": "user", "content": str(raw_content or "")})
            continue
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            text = str(msg.get("content") or "").strip()
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments") or "{}"
                try:
                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    parsed_args = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(tc.get("id") or "call_0"),
                        "name": str(fn.get("name") or ""),
                        "input": parsed_args if isinstance(parsed_args, dict) else {},
                    }
                )
            if blocks:
                out.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            tool_block = {
                "type": "tool_result",
                "tool_use_id": str(msg.get("tool_call_id") or "call_0"),
                "content": str(msg.get("content") or ""),
            }
            if out and out[-1].get("role") == "user" and isinstance(out[-1].get("content"), list):
                out[-1]["content"].append(tool_block)
            else:
                out.append({"role": "user", "content": [tool_block]})
    return system_parts, out


def _parse_openai_response(
    *,
    response: httpx.Response,
    provider_label: str,
    requested_model: str,
    tools: list[dict[str, Any]] | None,
    elapsed: float,
    is_local: bool = False,
) -> ChatResponse:
    if response.status_code >= 400:
        detail = _extract_error_detail(response)
        lower_detail = detail.lower()
        if tools and (
            "tool" in lower_detail or "function" in lower_detail or response.status_code == 404
        ):
            raise ModelToolSupportError(
                f"Model '{requested_model}' is unavailable or does not support tool calling "
                f"(HTTP {response.status_code}): {detail}"
            )
        raise OpenRouterError(
            f"{provider_label} request failed (HTTP {response.status_code}): {detail}"
        )

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise OpenRouterError(f"{provider_label} returned no completion choices.")

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
        cost_usd=0.0 if is_local else raw_usage.get("cost"),
    )

    return ChatResponse(
        model=requested_model if is_local else (data.get("model") or requested_model),
        content=content,
        tool_calls=parsed_tool_calls,
        usage=usage,
        elapsed_seconds=elapsed,
        finish_reason=finish_reason,
    )


def _parse_tool_calls(raw_calls: Any) -> list[ToolCall]:
    """Normalize OpenAI/OpenRouter tool_calls entries into typed ToolCall models."""
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
            arguments = "{}" if arguments is None else json.dumps(arguments)
        parsed.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return parsed


def _extract_reasoning_tokens(raw_usage: dict[str, Any]) -> int | None:
    """Extract hidden reasoning tokens when reported by the provider."""
    details = raw_usage.get("completion_tokens_details")
    if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
        return int(details["reasoning_tokens"])
    if raw_usage.get("reasoning_tokens") is not None:
        return int(raw_usage["reasoning_tokens"])
    return None


def _extract_error_detail(response: httpx.Response) -> str:
    """Safely extract an error message from a provider error payload."""
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
