"""Application configuration.

Architecture decision: all configuration flows through a single, validated
``Settings`` object (12-factor style). Values come from environment variables
(or a local ``.env`` file in development). Secrets use ``SecretStr`` so they
never leak into logs or ``repr()`` output. The settings object is cached with
``lru_cache`` and exposed through ``get_settings`` so it can be used both as a
plain function and as a FastAPI dependency (and overridden in tests).
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed runtime configuration, validated at startup (fail fast)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Enterprise Support Agent"
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    openai_api_key: SecretStr
    openai_model: str = "gpt-4o"
    openai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=45.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    database_url: str
    db_pool_min_size: int = Field(default=1, ge=1)
    db_pool_max_size: int = Field(default=10, ge=1)

    # Max LLM calls per user
    agent_max_iterations: int = Field(default=8, ge=1, le=50)
    # Hard LangGraph safety net
    agent_recursion_limit: int = Field(default=30, ge=5, le=200)
    # Sent to the LLM
    max_context_messages: int = Field(default=40, ge=4)
    # For one /chat request
    request_timeout_seconds: float = Field(default=90.0, gt=0)

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, value: str) -> str:
        """Ensure a Postgres DSN that psycopg can use is supplied."""
        if not value.startswith(("postgresql://", "postgres://")):
            raise ValueError(
                "DATABASE_URL must start with 'postgresql://' or 'postgres://'"
            )
        return value

    @field_validator("agent_recursion_limit")
    @classmethod
    def _validate_recursion_limit(cls, value: int, info) -> int:
        """The recursion limit must leave room for the iteration cap."""
        max_iter = info.data.get("agent_max_iterations", 8)
        if value < 2 * max_iter + 2:
            raise ValueError(
                "AGENT_RECURSION_LIMIT must be >= 2 * AGENT_MAX_ITERATIONS + 2"
            )
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()