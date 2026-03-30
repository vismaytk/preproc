"""Core configuration using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "DataPrep Pro"
    environment: str = "development"
    debug: bool = False

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_workers: int = 1
    api_url: str = "http://localhost:8000"
    frontend_port: int = 8501

    # CORS — defaults to localhost only (Bug Fix #5: no more wildcard)
    cors_origins: str = "http://localhost:8501,http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    # File Upload Limits
    max_csv_size: int = 10 * 1024 * 1024  # 10MB
    max_txt_size: int = 5 * 1024 * 1024   # 5MB
    max_image_size: int = 5 * 1024 * 1024  # 5MB

    # Processing
    max_rows: int = 1_000_000
    max_columns: int = 1_000
    sample_size: int = 5
    max_text_length: int = 1_000_000
    max_image_dimension: int = 4096
    image_quality: int = 85

    # Rate Limiting
    rate_limit: str = "10/minute"

    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"


@lru_cache
def get_settings() -> AppConfig:
    """Return cached singleton settings instance."""
    return AppConfig()
