import logging
import os
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

    openai_temperature: float = 0.5

    print_api_responses: bool = False
    openai_use_proxy: bool = False
    proxy_url: str = ""

    num_es: int = 10

    max_concurrency: int = 100
    limit_threads_by_n_cpu: bool = False

    max_retries: int = 10
    max_retries_extract_triplets: int = 3

    rag_num_evidences: int = 10
    rag_num_graph: int = 0

    model_config = SettingsConfigDict(
        env_file=project_root / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def n_threads(self):
        return (
            self.max_concurrency
            if not self.limit_threads_by_n_cpu
            else min(1, os.cpu_count() - 3)
        )


settings = Settings()
