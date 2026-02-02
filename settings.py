import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parents[0]
    log_level: int = logging.INFO

    openai_api_base: str = "http://localhost:8000/v1"
    openai_api_key: str = "EMPTY"
    openai_model: str = "local-model"


    openai_emb_api_base: str = "http://localhost:8000/v1"
    openai_emb_model: str = "jinaai/jina-embeddings-v3"
    openai_embedding_dim: int = 1024
    openai_embedding_max_tokens: int = 8190

    openai_llm_max_async: int = 100
    openai_emb_max_async: int = 100

    openai_use_proxy: bool = False
    proxy_url: str = ""

    num_es: int = 10

    model_config = SettingsConfigDict(
        env_file=project_root / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
