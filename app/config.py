# =============================================================================
# config.py — Central configuration using Pydantic Settings
# =============================================================================

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LangSmith
    langsmith_tracing: bool = Field(default=True, alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="nyaya-setu", alias="LANGSMITH_PROJECT")

    # PostgreSQL
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/nyayasetu",
        alias="DATABASE_URL"
    )
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="nyayasetu", alias="POSTGRES_DB")

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="phi3", alias="OLLAMA_MODEL")

    # Embeddings
    embed_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBED_MODEL")
    embed_dimension: int = Field(default=384, alias="EMBED_DIMENSION")

    # App
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def async_database_url(self) -> str:
        """Return the async PostgreSQL URL for asyncpg."""
        return self.database_url

    @property
    def sync_database_url(self) -> str:
        """Return sync version for migrations (psycopg2)."""
        return self.database_url.replace("+asyncpg", "")

    @property
    def alembic_database_url(self) -> str:
        """Return URL for Alembic migrations."""
        return self.database_url.replace("+asyncpg", "+psycopg2")


# Global settings instance
settings = Settings()

# Ensure LangSmith env vars are set globally
os.environ.setdefault("LANGSMITH_TRACING", str(settings.langsmith_tracing).lower())
if settings.langsmith_api_key:
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
