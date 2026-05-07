"""
Вспомогательные функции для обработки аудио и валидации.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import zipfile
from io import BytesIO

import numpy as np
import torchaudio
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def validate_audio_file(file_path: Path, max_size: int, max_duration: float) -> Tuple[bool, str]:
    """
    Валидация ограничений аудиофайла.
    
    Аргументы:
        file_path: Путь к аудиофайлу
        max_size: Максимальный размер файла в байтах
        max_duration: Максимальная длительность в секундах
        
    Возвращает:
        Кортеж (валиден, сообщение_об_ошибке)
    """
    if not file_path.exists():
        return False, "Файл не найден"
    
    # Проверка размера файла
    file_size = file_path.stat().st_size
    if file_size > max_size:
        return False, f"Файл слишком большой: {file_size} байт (макс: {max_size})"
    
    if file_size == 0:
        return False, "Пустой файл"
    
    # Проверка формата
    supported_formats = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    if file_path.suffix.lower() not in supported_formats:
        return False, f"Неподдерживаемый формат: {file_path.suffix}"
    
    # Проверка длительности
    try:
        duration = get_audio_duration(file_path)
        if duration is None:
            return False, "Не удалось определить длительность аудио"
        if duration > max_duration:
            return False, f"Аудио слишком длинное: {duration:.1f}с (макс: {max_duration}с)"
    except Exception as e:
        logger.error(f"Ошибка проверки длительности: {e}")
        return False, f"Повреждённый аудиофайл: {e}"
    
    return True, ""


def get_audio_duration(file_path: Path) -> Optional[float]:
    """
    Получение длительности аудиофайла в секундах.
    
    Аргументы:
        file_path: Путь к аудиофайлу
        
    Возвращает:
        Длительность в секундах или None в случае ошибки
    """
    try:
        if file_path.suffix.lower() == '.wav':
            info = torchaudio.info(str(file_path))
            return info.num_frames / info.sample_rate
        else:
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0
    except Exception as e:
        logger.error(f"Не удалось получить длительность для {file_path}: {e}")
        return None


def create_stem_zip(stem_paths: dict) -> BytesIO:
    """
    Создание ZIP архива со всеми стемами.
    
    Аргументы:
        stem_paths: Словарь с путями к файлам стемов
        
    Возвращает:
        Буфер BytesIO с ZIP архивом
    """
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for stem_name, stem_path in stem_paths.items():
            # Используем русские названия для файлов в архиве
            stem_names_ru = {
                'vocals': 'вокал',
                'drums': 'барабаны',
                'bass': 'бас',
                'other': 'остальное'
            }
            archive_name = f"{stem_names_ru.get(stem_name, stem_name)}.wav"
            zip_file.write(stem_path, archive_name)
    
    zip_buffer.seek(0)
    return zip_buffer


def ensure_directory(path: Path) -> Path:
    """
    Создание директории если она не существует.
    
    Аргументы:
        path: Путь к директории
        
    Возвращает:
        Объект Path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_temp_files(temp_dir: Path, max_age_hours: int = 24) -> int:
    """
    Очистка старых временных файлов.
    
    Аргументы:
        temp_dir: Путь к директории с временными файлами
        max_age_hours: Максимальный возраст файлов в часах
        
    Возвращает:
        Количество удалённых файлов
    """
    import time
    
    if not temp_dir.exists():
        return 0
    
    count = 0
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for file_path in temp_dir.glob("*"):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                    count += 1
                except OSError as e:
                    logger.warning(f"Не удалось удалить {file_path}: {e}")
    
    if count > 0:
        logger.info(f"Очищено {count} временных файлов")
    
    return count