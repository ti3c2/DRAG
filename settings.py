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

    ragas_metrics: list[str] = [
        "faithfulness",
        "nv_accuracy",
        "nv_context_relevance",
        "nv_response_groundedness",
        "factual_correctness",
        "bleu_score",
        "rouge_score",
        "exact_match",
        "string_present",
        "non_llm_string_similarity",
    ]
    eval_llm: str = openai_model
    eval_llm_api_base: str = openai_api_base
    eval_embedding_model: str = openai_emb_model
    eval_embedding_model_api_base: str = openai_emb_api_base
    ragas_max_workers: int = 64
    ragas_max_retries: int = 5
    ragas_timeout: int = 1200
    ragas_batch_size: int | None = None

    ragas_max_evals: int = 0

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
