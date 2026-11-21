"""Unit tests for Local AI (Ollama / LM Studio) and Direct Cloud BYOK (Claude / Gemini / GPT)."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from poseidon.config import Settings
from poseidon.context.secrets_store import ProviderSecretsStore
from poseidon.llm.openrouter import (
    MissingApiKeyError,
    OpenRouterClient,
    _convert_messages_to_anthropic,
    _convert_tools_to_anthropic,
    parse_provider_model,
)
from poseidon.web.server import create_web_app


def test_parse_provider_model_handles_all_six_providers() -> None:
    assert parse_provider_model("ollama:qwen2.5-coder:7b") == ("ollama", "qwen2.5-coder:7b")
    assert parse_provider_model("lmstudio:deepseek-coder-v2") == ("lmstudio", "deepseek-coder-v2")
    assert parse_provider_model("anthropic:claude-sonnet-4-5") == ("anthropic", "claude-sonnet-4-5")
    assert parse_provider_model("gemini:gemini-2.5-flash") == ("gemini", "gemini-2.5-flash")
    assert parse_provider_model("openai:gpt-4.1-mini") == ("openai", "gpt-4.1-mini")
    assert parse_provider_model("nvidia/nemotron-3.5-super-vl:free") == (
        "openrouter",
        "nvidia/nemotron-3.5-super-vl:free",
    )


def test_secrets_store_masks_and_never_exposes_plaintext_keys(tmp_path: Path) -> None:
    store = ProviderSecretsStore(tmp_path / "keys.json")
    raw_secret = "sk-ant-api03-super-secret-token-9999"
    store.set_key("anthropic", raw_secret)

    assert store.resolve_key("anthropic", Settings()) == raw_secret
    status = store.masked_status(Settings())
    assert status["anthropic"]["configured"] is True
    assert status["anthropic"]["source"] == "saved"
    assert raw_secret not in str(status)
    assert status["anthropic"]["hint"].endswith("9999")

    assert store.clear_key("anthropic") is True
    assert store.clear_key("anthropic") is False


def test_anthropic_message_and_tool_translation() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]
    anth_tools = _convert_tools_to_anthropic(tools)
    assert anth_tools[0]["name"] == "read_file"
    assert "input_schema" in anth_tools[0]

    sys_parts, msgs = _convert_messages_to_anthropic(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Inspect app.py"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "app.py"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "print('hi')"},
        ]
    )
    assert sys_parts == ["Be concise."]
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][0]["type"] == "tool_use"
    assert msgs[1]["content"][0]["input"] == {"path": "app.py"}
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert msgs[2]["content"][0]["tool_use_id"] == "call_1"


def test_validate_tool_model_allows_local_ai_and_checks_byok_keys(tmp_path: Path) -> None:
    store = ProviderSecretsStore(tmp_path / "keys.json")
    client = OpenRouterClient(Settings(), secrets_store=store)
    try:
        # Local AI requires zero API key
        assert client.validate_tool_model("ollama:qwen2.5-coder:7b")["local"] is True
        assert client.validate_tool_model("lmstudio:llama-3.3")["local"] is True

        # Direct cloud providers check for configured key
        with pytest.raises(MissingApiKeyError):
            client.validate_tool_model("anthropic:claude-sonnet-4-5")

        store.set_key("anthropic", "sk-ant-test-key-123456")
        assert client.validate_tool_model("anthropic:claude-sonnet-4-5")["provider"] == "anthropic"
    finally:
        client.close()


def test_web_api_providers_keys_and_local_models_work_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSEIDON_SECRETS_FILE", str(tmp_path / "secrets" / "keys.json"))
    app = create_web_app(workspace=tmp_path, settings=Settings())
    client = TestClient(app)

    # Save Gemini key -> masked response never leaks raw key
    raw_gemini = "AIzaSy-very-secret-key-4321"
    saved = client.post(
        "/api/providers/key",
        json={"provider": "gemini", "action": "set", "api_key": raw_gemini},
    )
    assert saved.status_code == 200
    assert raw_gemini not in saved.text
    assert saved.json()["keys"]["gemini"]["configured"] is True
    assert saved.json()["keys"]["gemini"]["source"] == "saved"

    # Add local Ollama model + Claude preset into custom models
    add_local = client.post(
        "/api/models/add",
        json={"model_id": "ollama:qwen2.5-coder:7b", "activate": True},
    )
    assert add_local.status_code == 200
    assert add_local.json()["state"]["model"] == "ollama:qwen2.5-coder:7b"

    add_claude = client.post(
        "/api/models/add",
        json={"model_id": "anthropic:claude-sonnet-4-5", "activate": False},
    )
    assert add_claude.status_code == 200
    assert "anthropic:claude-sonnet-4-5" in add_claude.json()["state"]["custom_models"]

    # Clear Gemini key cleanly
    cleared = client.post(
        "/api/providers/key",
        json={"provider": "gemini", "action": "clear"},
    )
    assert cleared.status_code == 200
    assert cleared.json()["keys"]["gemini"]["source"] != "saved"
