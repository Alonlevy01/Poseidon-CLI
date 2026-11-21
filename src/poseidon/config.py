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
    anthropic_api_key: SecretStr = Field(default=SecretStr(""), alias="ANTHROPIC_API_KEY")
    gemini_api_key: SecretStr = Field(default=SecretStr(""), alias="GEMINI_API_KEY")
    openai_api_key: SecretStr = Field(default=SecretStr(""), alias="OPENAI_API_KEY")
    ollama_base_url: str = Field(
        default="http://host.docker.internal:11434/v1",
        alias="OLLAMA_BASE_URL",
    )
    lmstudio_base_url: str = Field(
        default="http://host.docker.internal:1234/v1",
        alias="LMSTUDIO_BASE_URL",
    )
    respan_api_key: SecretStr = Field(default=SecretStr(""), alias="RESPAN_API_KEY")
    respan_base_url: str | None = Field(
        default=None,
        alias="RESPAN_BASE_URL",
    )
    tavily_api_key: SecretStr = Field(default=SecretStr(""), alias="TAVILY_API_KEY")
    brave_api_key: SecretStr = Field(default=SecretStr(""), alias="BRAVE_API_KEY")
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
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        alias="POSEIDON_EMBEDDING_MODEL",
    )
    embedding_cache_dir: str | None = Field(
        default=None,
        alias="FASTEMBED_CACHE_PATH",
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
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "Free | 1M ctx | NVIDIA Nemotron 3 Ultra 550B MoE (tool-capable)",
    ),
    (
        "baidu/ernie-5.5-flash-preview:free",
        "Free | 1M ctx | Baidu ERNIE 5.5 Flash (second family, tool-capable)",
    ),
    (
        "inclusionai/ling-3.2-flash:free",
        "Free | 1M ctx | inclusionAI Ling 3.2 Flash (third family, tool-capable)",
    ),
]
