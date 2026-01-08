import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from urllib.parse import quote_plus


class Settings(BaseSettings):
    db_user: str
    db_password: str
    db_host: str
    db_port: str
    db_name: str
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"  # ou "text-embedding-3-large"
    embedding_dimension: int = 1536  # 1536 pour text-embedding-3-small
    index_on_startup: bool = True
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def database_url(self) -> str:
        """Construit l'URL de la DB en encodant le mot de passe"""
        encoded_password = quote_plus(self.db_password)
        return f"postgresql://{self.db_user}:{encoded_password}@{self.db_host}:{self.db_port}/{self.db_name}"

settings = Settings()
