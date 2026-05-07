"""
Конфигурация приложения с использованием pydantic Settings.
Все настройки можно переопределить через переменные окружения с префиксом MUSIC_SEP_.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения с поддержкой переменных окружения."""

    model_config = SettingsConfigDict(
        env_prefix="MUSIC_SEP_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # Настройки модели
    model_name: str = Field(
        default="htdemucs",
        description="Вариант модели Demucs"
    )
    device: Optional[str] = Field(
        default=None,
        description="Устройство для инференса (cuda, cpu или None для автоопределения)"
    )

    # Настройки обработки
    chunk_size: Optional[float] = Field(
        default=30.0,
        description="Максимальная длительность чанка в секундах"
    )
    overlap: float = Field(
        default=1.0,
        description="Перекрытие между чанками в секундах"
    )
    trim_silence: bool = Field(
        default=True,
        description="Включить обрезку тишины"
    )
    silence_threshold: float = Field(
        default=-60.0,
        description="Порог тишины в dB"
    )

    # Настройки API
    max_file_size: int = Field(
        default=50 * 1024 * 1024,  # 50 MB
        description="Максимальный размер загружаемого файла в байтах"
    )
    max_duration: float = Field(
        default=600.0,  # 10 minutes
        description="Максимальная длительность аудио в секундах"
    )
    request_timeout: int = Field(
        default=60,
        description="Таймаут обработки запроса в секундах"
    )

    # Paths
    cache_dir: Path = Field(
        default=Path("./cache"),
        description="Директория для кэширования результатов"
    )
    output_dir: Path = Field(
        default=Path("./output"),
        description="Директория для выходных файлов"
    )
    temp_dir: Path = Field(
        default=Path("./temp"),
        description="Директория для временных файлов"
    )

    # Настройки сервера
    host: str = Field(default="0.0.0.0", description="Хост сервера")
    port: int = Field(default=8000, description="Порт сервера")
    debug: bool = Field(default=False, description="Режим отладки")


# Глобальный экземпляр настроек
settings = Settings()