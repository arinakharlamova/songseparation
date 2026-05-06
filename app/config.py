"""Application configuration using pydantic-settings."""
from typing import Literal

import torch
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", populate_by_name=True)

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    model_name: str = Field(default="htdemucs_ft", alias="MODEL_NAME")
    device: Literal["cpu", "cuda", "auto"] = Field(default="auto", alias="DEVICE")
    max_file_size_mb: int = Field(default=100, alias="MAX_FILE_SIZE_MB")
    max_duration_seconds: int = Field(default=600, alias="MAX_DURATION_SECONDS")
    timeout_seconds: int = Field(default=3600, alias="TIMEOUT_SECONDS")
    output_dir: str = Field(default="output", alias="OUTPUT_DIR")
    cache_dir: str = Field(default="output/cache", alias="CACHE_DIR")
    temp_dir: str = Field(default="output/temp", alias="TEMP_DIR")
    allowed_extensions: str = Field(default=".wav,.mp3,.flac,.ogg,.m4a", alias="ALLOWED_EXTENSIONS")
    api_key: str = Field(default="", alias="API_KEY")
    rate_limit_per_minute: int = Field(default=10, alias="RATE_LIMIT_PER_MINUTE")
    cache_cleanup_hours: int = Field(default=168, alias="CACHE_CLEANUP_HOURS")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    @property
    def ext_set(self) -> set[str]:
        return {e.strip() for e in self.allowed_extensions.split(",") if e.strip()}

    @property
    def cors_list(self) -> list[str]:
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


settings = Settings()
