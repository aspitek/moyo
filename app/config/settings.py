import os
from urllib.parse import quote_plus
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database - variables séparées
    db_user: str = "postgres"
    db_password: str
    db_host: str = "89.116.38.238"
    db_port: int = 5433  # INT au lieu de str
    db_name: str = "sellart"
    
    # OpenAI
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
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
        # IMPORTANT: utiliser db_port comme int, pas string
        return f"postgresql://{self.db_user}:{encoded_password}@{self.db_host}:{self.db_port}/{self.db_name}"


settings = Settings()
