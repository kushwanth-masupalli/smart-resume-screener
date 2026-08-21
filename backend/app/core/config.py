"""
Application configuration loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/resume_screener"
    sync_database_url: str = "postgresql://postgres:postgres@localhost:5432/resume_screener"

    # NVIDIA NIM API (OpenAI-compatible)
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # App
    app_name: str = "Smart Resume Screener"
    sieve_top_k_percent: float = 0.20
    sieve_min_candidates: int = 5
    sieve_max_candidates: int = 50

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # spaCy
    spacy_model: str = "en_core_web_sm"

    # LLM
    llm_model: str = "meta/llama-3.3-70b-instruct"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
