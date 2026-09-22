from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm_provider: str = "ollama"
    llm_model: str = "llama3"
    llm_temperature: float = 0.1
    llm_timeout_seconds: float = 60.0
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_path: str = "data/processed/chroma"
    sqlite_path: str = "data/processed/conversations.db"
    mlflow_uri: str = "./data/processed/mlruns"
    retrieval_top_k: int = 5
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    evaluation_dataset: str = "data/evaluation/questions.jsonl"
    regression_baseline: str = "data/evaluation/baseline.json"
    regression_max_latency_ms: float = 30000.0
    regression_min_recall: float = 0.0
    regression_min_groundedness: float = 0.0
    regression_min_citation_correctness: float = 0.0
    regression_max_failure_rate: float = 1.0


@lru_cache
def get_settings() -> Settings:
    return Settings()