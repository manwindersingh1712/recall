from functools import lru_cache
from uuid import UUID

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RECALL_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://recall:recall@localhost:5433/recall"
    test_database_url: str = "postgresql://recall:recall@localhost:5433/recall_test"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    default_user_id: UUID = UUID("00000000-0000-0000-0000-000000000001")


@lru_cache
def get_settings() -> Settings:
    return Settings()
