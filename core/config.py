"""Application configuration.

Architecture decision: all configuration flows through a single, validated
``Settings`` object (12-factor style). Values come from environment variables
(or a local ``.env`` file in development). Secrets use ``SecretStr`` so they
never leak into logs or ``repr()`` output. The settings object is cached with
``lru_cache`` and exposed through ``get_settings`` so it can be used both as a
plain function and as a FastAPI dependency (and overridden in tests).
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed runtime configuration, validated at startup (fail fast)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- General ---------------------------------------------------------
    app_name: str = "Enterprise Support Agent"
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Browser origins allowed to call the API (JSON list in the env var).
    cors_allow_origins: list[str] = Field(default_factory=list)

    # --- LLM (any OpenAI-compatible endpoint; Groq by default) ------------
    llm_api_key: SecretStr = Field(
        validation_alias=AliasChoices("LLM_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY")
    )
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=45.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # --- Persistence -----------------------------------------------------
    # Plain libpq connection string; psycopg consumes it directly.
    database_url: str
    db_pool_min_size: int = Field(default=1, ge=1)
    db_pool_max_size: int = Field(default=10, ge=1)

    # --- RAG -------------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # MUST match the model AND the vector(N) column created by migration 0002.
    embedding_dim: int = 384
    rag_collection: str = "company_kb"
    rag_top_k: int = Field(default=4, ge=1, le=10)
    # Maximum cosine *distance* (0 = identical, 2 = opposite) to accept a hit.
    rag_max_distance: float = Field(default=0.6, gt=0, le=2)

    # --- Background work (Celery + Redis) ---------------------------------
    redis_url: str = "redis://localhost:6379/0"
    # Run tasks inline instead of via the broker (tests / local debugging).
    celery_task_always_eager: bool = False
    notification_email_to: str = "oncall@example.com"
    notification_slack_channel: str = "#support-urgent"
    notification_max_attempts: int = Field(default=5, ge=1, le=20)
    notification_retry_base_seconds: int = Field(default=30, ge=1)
    # Only paths under this directory may be ingested by the Celery task.
    kb_docs_dir: str = "kb_docs"

    # --- Observability (LangSmith) ----------------------------------------
    langsmith_tracing: bool = False
    langsmith_api_key: Optional[SecretStr] = None
    langsmith_project: str = "enterprise-support-agent"
    langsmith_endpoint: Optional[str] = None

    # --- Agent behaviour -------------------------------------------------
    # Max LLM calls per user turn before the graph forces a graceful stop.
    agent_max_iterations: int = Field(default=8, ge=1, le=50)
    # Hard LangGraph safety net (supersteps); must exceed 2x max_iterations.
    agent_recursion_limit: int = Field(default=30, ge=5, le=200)
    # Sliding window of messages sent to the LLM (checkpoint keeps everything).
    max_context_messages: int = Field(default=40, ge=4)
    # Wall-clock budget for one /chat request.
    request_timeout_seconds: float = Field(default=90.0, gt=0)
    # Keep-alive comment interval on SSE streams (proxies drop idle streams).
    sse_heartbeat_seconds: float = Field(default=15.0, gt=0)

    @property
    def sqlalchemy_url(self) -> str:
        """Async SQLAlchemy URL (psycopg3 driver) for the pgvector store."""
        url = self.database_url
        for prefix in ("postgresql://", "postgres://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix) :]
        return url

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, value: str) -> str:
        """Ensure a Postgres DSN that psycopg can use is supplied."""
        if not value.startswith(("postgresql://", "postgres://")):
            raise ValueError("DATABASE_URL must start with 'postgresql://' or 'postgres://'")
        return value

    @model_validator(mode="after")
    def _tracing_needs_key(self) -> "Settings":
        """Fail fast instead of silently running untraced."""
        if self.langsmith_tracing and self.langsmith_api_key is None:
            raise ValueError("LANGSMITH_API_KEY is required when LANGSMITH_TRACING is true")
        return self

    @field_validator("agent_recursion_limit")
    @classmethod
    def _validate_recursion_limit(cls, value: int, info) -> int:
        """The recursion limit must leave room for the iteration cap."""
        max_iter = info.data.get("agent_max_iterations", 8)
        if value < 2 * max_iter + 2:
            raise ValueError("AGENT_RECURSION_LIMIT must be >= 2 * AGENT_MAX_ITERATIONS + 2")
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()  # type: ignore[call-arg]  # values come from the environment
