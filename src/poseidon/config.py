"""Configuration management for Poseidon-CLI."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: SecretStr = Field(default=SecretStr(""), alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    respan_api_key: SecretStr = Field(default=SecretStr(""), alias="RESPAN_API_KEY")
    respan_base_url: str | None = Field(
        default=None,
        alias="RESPAN_BASE_URL",
    )
    default_model: str = Field(
        default="nvidia/nemotron-3.5-super-vl:free",
        alias="POSEIDON_DEFAULT_MODEL",
    )
    default_max_calls: int = Field(
        default=8,
        alias="POSEIDON_MAX_CALLS",
    )
    default_test_timeout: int = Field(
        default=15,
        alias="POSEIDON_TEST_TIMEOUT",
    )
    sandbox_image: str = Field(
        default="poseidon-sandbox:dev",
        alias="POSEIDON_SANDBOX_IMAGE",
    )
    request_timeout_seconds: float = Field(
        default=120.0,
        alias="POSEIDON_REQUEST_TIMEOUT",
    )
    app_title: str = "Poseidon-CLI"
    http_referer: str = "https://github.com/Alonlevy01/Poseidon-CLI"


RECOMMENDED_MODELS: list[tuple[str, str]] = [
    (
        "nvidia/nemotron-3.5-super-vl:free",
        "Free | 1M ctx | NVIDIA agentic coding & tool-calling model",
    ),
    (
        "nvidia/nemotron-3.5-lightning:free",
        "Free | 1M ctx | Fast NVIDIA tool-calling model",
    ),
    (
        "google/gemma-4-31b-it:free",
        "Free | 262k ctx | Google Gemma 4 31B (second family, tool-capable)",
    ),
    (
        "google/gemma-4-26b-a4b-it:free",
        "Free | 262k ctx | Fast Google Gemma 4 MoE (tool-capable)",
    ),
]
