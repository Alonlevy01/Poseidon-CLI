"""Respan tracing integration for direct OpenRouter model calls and local tool calls.

Poseidon-CLI calls OpenRouter directly and sends structured workflow/task/tool spans
to Respan using RESPAN_API_KEY. Secrets are never recorded in trace payloads.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from poseidon.config import Settings

T = TypeVar("T")


class PoseidonTracer:
    """Thin, safe wrapper over respan-tracing with clean no-op fallback when unconfigured."""

    def __init__(self, settings: Settings | None = None, *, enabled: bool | None = None) -> None:
        api_key = settings.respan_api_key.get_secret_value().strip() if settings else ""
        raw_url = settings.respan_base_url if settings else None
        base_url = raw_url.strip() if raw_url else None

        self.enabled: bool = bool(api_key) if enabled is None else (enabled and bool(api_key))
        self._telemetry: Any = None
        self._workflow_decorator: Any = None
        self._task_decorator: Any = None
        self._tool_decorator: Any = None

        if self.enabled:
            self._init_respan(api_key=api_key, base_url=base_url)

    def _init_respan(self, *, api_key: str, base_url: str | None) -> None:
        """Initialize RespanTelemetry with synchronous export so CLI traces always flush."""
        from respan_tracing import RespanTelemetry, task, tool, workflow

        kwargs: dict[str, Any] = {
            "app_name": "poseidon-cli",
            "api_key": api_key,
            "is_batching_enabled": False,
            "instruments": set(),
            "log_level": "WARNING",
            "is_enabled": True,
        }
        if base_url:
            kwargs["base_url"] = base_url

        self._telemetry = RespanTelemetry(**kwargs)
        self._workflow_decorator = workflow
        self._task_decorator = task
        self._tool_decorator = tool

    def run_workflow(self, name: str, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        """Execute a top-level agent workflow inside a Respan @workflow span."""
        if not self.enabled or self._workflow_decorator is None:
            return fn(*args, **kwargs)
        wrapped = self._workflow_decorator(name=name)(fn)
        return wrapped(*args, **kwargs)

    def run_model_call(self, name: str, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        """Execute a single OpenRouter model call inside a Respan @task span."""
        if not self.enabled or self._task_decorator is None:
            return fn(*args, **kwargs)
        wrapped = self._task_decorator(name=name)(fn)
        return wrapped(*args, **kwargs)

    def run_tool_call(
        self,
        tool_name: str,
        fn: Callable[..., T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a workspace tool call inside a Respan @tool child span."""
        if not self.enabled or self._tool_decorator is None:
            return fn(*args, **kwargs)
        wrapped = self._tool_decorator(name=f"tool_{tool_name}")(fn)
        return wrapped(*args, **kwargs)

    def flush(self) -> None:
        """Flush any buffered spans cleanly before exiting."""
        if not self.enabled or self._telemetry is None:
            return
        flush_fn = getattr(self._telemetry, "flush", None)
        if callable(flush_fn):
            flush_fn()
