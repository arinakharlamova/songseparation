"""
Модуль для разделения музыкальных источников с использованием модели Demucs.
Поддерживает входные форматы WAV/MP3, GPU/CPU инференс, чанковую обработку и кэширование.
"""

import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from pydub import AudioSegment
from pydantic import BaseModel, ConfigDict

# Настройка логирования
logger = logging.getLogger(__name__)

# Псевдонимы типов
StemDict = Dict[str, np.ndarray]
PathLike = Union[str, Path]


class SeparationResult(BaseModel):
    """Результат разделения источников."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stems: Dict[str, np.ndarray]
    sample_rate: int
    processing_time: float
    model_name: str
    device: str


class DemucsSeparator:
    """
    Разделение музыкальных источников с использованием модели Demucs от Meta.
    
    Возможности:
        - Поддерживает модель htdemucs (лучший баланс качество/скорость)
        - Инференс на GPU (CUDA) и CPU
        - Чанковая обработка длинных треков с перекрытием
        - Конвертация MP3 в WAV через pydub
        - Обрезка тишины в начале и конце
        - Кэширование результатов по хешу файла
        - Замер времени обработки
    
    Аргументы:
        model_name: Вариант модели Demucs (по умолчанию: "htdemucs")
        device: Устройство для инференса ("cuda", "cpu" или None для автоопределения)
        cache_dir: Директория для кэширования результатов (None отключает кэш)
        chunk_size: Максимальная длительность чанка в секундах (None для обработки целиком)
        overlap: Перекрытие между чанками в секундах
        trim_silence: Включить обрезку тишины в выходных стемах
        silence_threshold: Порог тишины в dB
    """
    
    # Поддерживаемые аудио форматы
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    
    # Стандартные стемы Demucs
    DEFAULT_STEMS = ['vocals', 'drums', 'bass', 'other']
    
    def __init__(
        self,
        model_name: str = "htdemucs",
        device: Optional[str] = None,
        cache_dir: Optional[PathLike] = None,
        chunk_size: Optional[float] = None,
        overlap: float = 1.0,
        trim_silence: bool = True,
        silence_threshold: float = -60.0,  # dB
    ):
        """Инициализация разделителя Demucs."""
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.trim_silence = trim_silence
        self.silence_threshold = silence_threshold
        
        # Автоопределение устройства
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Инициализация DemucsSeparator с моделью={model_name}, устройство={self.device}")
        
        # Загрузка модели
        try:
            self.model = get_model(name=model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Модель {model_name} успешно загружена на {self.device}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_name}: {e}")
            raise RuntimeError(f"Ошибка загрузки модели: {e}")
        
        # Настройка кэширования
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Кэширование включено в {self.cache_dir}")
    
    def _compute_file_hash(self, audio_path: Path) -> str:
        """
        Вычисление SHA-256 хеша аудиофайла для кэширования.
        
        Аргументы:
            audio_path: Путь к аудиофайлу
            
        Возвращает:
            Хеш-строка в hex формате
        """
        sha256 = hashlib.sha256()
        with open(audio_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _get_cache_path(self, file_hash: str, stem: str) -> Path:
        """
        Получение пути к кэшированному файлу стема.
        
        Аргументы:
            file_hash: Хеш файла
            stem: Название стема
            
        Возвращает:
            Путь к кэшированному файлу
        """
        return self.cache_dir / f"{file_hash}_{stem}.npy"
    
    def _load_from_cache(self, file_hash: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Попытка загрузки результатов из кэша.
        
        Аргументы:
            file_hash: Хеш файла
            
        Возвращает:
            Словарь со стемами или None если нет в кэше
        """
        if not self.cache_dir:
            return None
            
        stems = {}
        for stem in self.DEFAULT_STEMS:
            cache_path = self._get_cache_path(file_hash, stem)
            if not cache_path.exists():
                return None
            stems[stem] = np.load(cache_path)
            
        logger.info(f"Результаты загружены из кэша для хеша {file_hash[:16]}...")
        return stems
    
    def _save_to_cache(self, file_hash: str, stems: Dict[str, np.ndarray]) -> None:
        """
        Сохранение результатов разделения в кэш.
        
        Аргументы:
            file_hash: Хеш файла
            stems: Словарь с массивами стемов
        """
        if not self.cache_dir:
            return
            
        for stem_name, stem_data in stems.items():
            cache_path = self._get_cache_path(file_hash, stem_name)
            np.save(cache_path, stem_data)
        
        logger.info(f"Результаты сохранены в кэш для хеша {file_hash[:16]}...")
    
    def _convert_to_wav(self, audio_path: Path) -> Path:
        """
        Конвертация аудиофайла в WAV формат при необходимости.
        
        Аргументы:
            audio_path: Путь к аудиофайлу
            
        Возвращает:
            Путь к временному WAV файлу
            
        Исключения:
            ValueError: Если формат не поддерживается
        """
        suffix = audio_path.suffix.lower()
        
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Неподдерживаемый формат: {suffix}. Поддерживаемые: {self.SUPPORTED_FORMATS}")
        
        if suffix == '.wav':
            return audio_path
        
        logger.info(f"Конвертация {suffix} в WAV: {audio_path.name}")
        
        try:
            audio = AudioSegment.from_file(audio_path, format=suffix[1:])
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            audio.export(temp_wav.name, format='wav')
            return Path(temp_wav.name)
        except Exception as e:
            logger.error(f"Ошибка конвертации {audio_path}: {e}")
            raise RuntimeError(f"Ошибка конвертации аудио: {e}")
    
    def _load_audio(self, audio_path: Path) -> Tuple[torch.Tensor, int]:
        """
        Загрузка аудиофайла и возврат тензора с частотой дискретизации.
        Demucs htdemucs требует стерео вход, поэтому моно конвертируется в стерео.
        Использует soundfile вместо torchaudio (обход проблемы с torchcodec на Windows).
        
        Аргументы:
            audio_path: Путь к WAV файлу
            
        Возвращает:
            Кортеж (аудио_тензор, частота_дискретизации)
            
        Исключения:
            RuntimeError: Если загрузка не удалась
        """
        try:
            data, sample_rate = sf.read(str(audio_path), dtype='float32')
            # soundfile returns (samples, channels)
            if data.ndim == 1:
                data = data[:, np.newaxis]
            # Convert to (channels, samples) for torch
            waveform = torch.from_numpy(data.T)
            
            # Demucs htdemucs требует 2 канала (стерео)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2, :]
            
            logger.info(f"Аудио загружено: {waveform.shape}, {sample_rate} Гц")
            return waveform, sample_rate
        except Exception as e:
            logger.error(f"Ошибка загрузки аудио {audio_path}: {e}")
            raise RuntimeError(f"Ошибка загрузки аудио: {e}")
    
    def _trim_silence(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Обрезка тишины в начале и конце аудио.
        
        Аргументы:
            waveform: Аудио тензор (channels, samples)
            sample_rate: Частота дискретизации в Гц
            
        Возвращает:
            Обрезанный аудио тензор
        """
        threshold = 10 ** (self.silence_threshold / 20)
        
        # Энергия: среднее по всем каналам для каждого сэмпла
        energy = waveform.abs().mean(dim=0)  # (channels, samples) -> (samples,)
        non_silent = energy > threshold
        
        if not non_silent.any():
            logger.warning("Аудио похоже полностью тихое")
            return waveform
        
        indices = non_silent.nonzero(as_tuple=True)[0]
        start_idx = indices[0].item()
        end_idx = indices[-1].item()
        
        # Отступ 100мс
        padding = int(0.1 * sample_rate)
        start_idx = max(0, start_idx - padding)
        end_idx = min(waveform.shape[-1], end_idx + padding)
        
        if start_idx > 0 or end_idx < waveform.shape[-1]:
            logger.info(f"Тишина обрезана: {start_idx/sample_rate:.2f}с до {end_idx/sample_rate:.2f}с")
        
        return waveform[:, start_idx:end_idx]
    
    def _process_chunk(
        self,
        chunk: torch.Tensor,
        sample_rate: int,
    ) -> Dict[str, np.ndarray]:
        """
        Обработка одного аудио чанка через модель.
        
        Аргументы:
            chunk: Аудио чанк (тензор)
            sample_rate: Частота дискретизации
            
        Возвращает:
            Словарь с разделёнными стемами
        """
        with torch.no_grad():
            # Добавление размерности батча и перемещение на устройство
            chunk_input = chunk.unsqueeze(0).to(self.device)
            
            # Применение модели
            sources = apply_model(self.model, chunk_input, device=self.device)
            
            # Возврат на CPU и удаление размерности батча
            sources = sources.squeeze(0).cpu()
        
        # Конвертация в словарь
        stems = {}
        for idx, stem_name in enumerate(self.model.sources):
            stems[stem_name] = sources[idx].numpy()
        
        return stems
    
    def _process_with_overlap(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> Dict[str, np.ndarray]:
        """
        Обработка длинного аудио с перекрывающимися чанками.
        
        Аргументы:
            waveform: Полный аудио тензор (channels, samples)
            sample_rate: Частота дискретизации
            
        Возвращает:
            Объединённые стемы из всех чанков
        """
        n_channels = waveform.shape[0]
        chunk_samples = int(self.chunk_size * sample_rate)
        overlap_samples = int(self.overlap * sample_rate)
        step = chunk_samples - overlap_samples
        
        total_samples = waveform.shape[-1]
        n_chunks = max(1, int(np.ceil((total_samples - overlap_samples) / step)))
        
        logger.info(f"Обработка {n_chunks} чанков (размер_чанка={self.chunk_size}с, перекрытие={self.overlap}с)")
        
        # Инициализация аккумуляторов (channels, samples)
        stems_accum = {}
        weights_accum = {}
        
        for stem_name in self.model.sources:
            stems_accum[stem_name] = np.zeros((n_channels, total_samples), dtype=np.float32)
            weights_accum[stem_name] = np.zeros((n_channels, total_samples), dtype=np.float32)
        
        # Обработка каждого чанка
        for i in range(n_chunks):
            start = i * step
            end = min(start + chunk_samples, total_samples)
            
            logger.debug(f"Обработка чанка {i+1}/{n_chunks}: сэмплы {start}-{end}")
            
            chunk = waveform[:, start:end]
            
            # Дополнение чанка если он короче chunk_samples
            if chunk.shape[-1] < chunk_samples:
                padding = chunk_samples - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, padding))
            
            # Обработка чанка
            chunk_stems = self._process_chunk(chunk, sample_rate)
            
            # Создание весового окна для плавного перекрытия
            window = np.ones(chunk_samples, dtype=np.float32)
            if i > 0:  # Плавное появление
                window[:overlap_samples] = np.linspace(0, 1, overlap_samples)
            if i < n_chunks - 1:  # Плавное затухание
                window[-overlap_samples:] = np.linspace(1, 0, overlap_samples)
            
            # Применение к актуальной длине чанка
            actual_length = end - start
            window = window[:actual_length]
            
            # Накопление взвешенных стемов (каждый стем: channels x samples)
            for stem_name in self.model.sources:
                stem_data = chunk_stems[stem_name][:, :actual_length]
                window_2d = window[np.newaxis, :]  # (1, samples) для broadcast
                stems_accum[stem_name][:, start:end] += stem_data * window_2d
                weights_accum[stem_name][:, start:end] += window_2d
        
        # Нормализация по весам
        result_stems = {}
        epsilon = 1e-8
        for stem_name in self.model.sources:
            result_stems[stem_name] = stems_accum[stem_name] / (weights_accum[stem_name] + epsilon)
        
        return result_stems
    
    def separate(self, audio_path: PathLike) -> SeparationResult:
        """
        Разделение аудио на отдельные стемы.
        
        Аргументы:
            audio_path: Путь к аудиофайлу (WAV, MP3, и др.)
            
        Возвращает:
            SeparationResult со стемами, частотой дискретизации и метаданными
            
        Исключения:
            FileNotFoundError: Если аудиофайл не существует
            ValueError: Если формат аудио не поддерживается
            RuntimeError: Если разделение не удалось
        """
        start_time = time.time()
        audio_path = Path(audio_path)
        
        # Валидация входных данных
        if not audio_path.exists():
            raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")
        
        if audio_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Неподдерживаемый формат: {audio_path.suffix}. Поддерживаемые: {self.SUPPORTED_FORMATS}")
        
        logger.info(f"Начало разделения для {audio_path.name}")
        
        # Конвертация в WAV при необходимости
        wav_path = self._convert_to_wav(audio_path)
        
        try:
            # Вычисление хеша для кэширования
            file_hash = self._compute_file_hash(wav_path)
            
            # Проверка кэша
            cached_stems = self._load_from_cache(file_hash)
            if cached_stems is not None:
                processing_time = time.time() - start_time
                logger.info(f"Попадание в кэш! Время обработки: {processing_time:.2f}с")
                
                # Получение частоты дискретизации для результата
                _, sample_rate = self._load_audio(wav_path)
                
                return SeparationResult(
                    stems=cached_stems,
                    sample_rate=sample_rate,
                    processing_time=processing_time,
                    model_name=self.model_name,
                    device=self.device,
                )
            
            # Загрузка аудио
            waveform, sample_rate = self._load_audio(wav_path)
            
            # Обрезка тишины если включено
            if self.trim_silence:
                waveform = self._trim_silence(waveform, sample_rate)
            
            # Проверка необходимости чанковой обработки
            duration = waveform.shape[-1] / sample_rate
            logger.info(f"Длительность аудио: {duration:.2f}с")
            
            if self.chunk_size and duration > self.chunk_size:
                stems = self._process_with_overlap(waveform, sample_rate)
            else:
                stems = self._process_chunk(waveform, sample_rate)
            
            # Кэширование результатов
            self._save_to_cache(file_hash, stems)
            
            processing_time = time.time() - start_time
            logger.info(f"Разделение завершено за {processing_time:.2f}с")
            
            return SeparationResult(
                stems=stems,
                sample_rate=sample_rate,
                processing_time=processing_time,
                model_name=self.model_name,
                device=self.device,
            )
            
        except Exception as e:
            logger.error(f"Ошибка разделения для {audio_path}: {e}")
            raise RuntimeError(f"Ошибка разделения: {e}")
        finally:
            # Очистка временного WAV если была конвертация
            if wav_path != audio_path and wav_path.exists():
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass
    
    def separate_to_files(
        self,
        audio_path: PathLike,
        output_dir: PathLike,
    ) -> Dict[str, Path]:
        """
        Разделение аудио и сохранение стемов как WAV файлов.
        
        Аргументы:
            audio_path: Путь к входному аудиофайлу
            output_dir: Директория для сохранения стемов
            
        Возвращает:
            Словарь с путями к файлам стемов
            
        Исключения:
            RuntimeError: Если сохранение не удалось
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Выполнение разделения
        result = self.separate(audio_path)
        
        # Сохранение стемов
        stem_paths = {}
        for stem_name, stem_data in result.stems.items():
            # stem_data: (channels, samples) or (samples,)
            waveform = stem_data
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]
            # Transpose to (samples, channels) for soundfile
            waveform_t = waveform.T
            output_path = output_dir / f"{stem_name}.wav"

            try:
                sf.write(str(output_path), waveform_t, result.sample_rate)
                stem_paths[stem_name] = output_path
                logger.info(f"Сохранён {stem_name} в {output_path}")
            except Exception as e:
                logger.error(f"Ошибка сохранения {stem_name}: {e}")
                raise RuntimeError(f"Ошибка сохранения стема {stem_name}: {e}")
        
        return stem_paths
    
    def clear_cache(self) -> int:
        """
        Очистка всех кэшированных результатов разделения.
        
        Возвращает:
            Количество удалённых файлов кэша
        """
        if not self.cache_dir or not self.cache_dir.exists():
            return 0
        
        count = 0
        for cache_file in self.cache_dir.glob("*.npy"):
            try:
                cache_file.unlink()
                count += 1
            except OSError as e:
                logger.warning(f"Ошибка удаления файла кэша {cache_file}: {e}")
        
        logger.info(f"Очищено {count} файлов кэша")
        return count


# Пример использования
if __name__ == "__main__":
    # Настройка логирования для тестирования
    logging.basicConfig(level=logging.INFO)
    
    # Инициализация разделителя
    separator = DemucsSeparator(
        model_name="htdemucs",
        device="cpu",  # Измените на "cuda" если есть GPU
        cache_dir="./cache",
        chunk_size=30.0,  # Обработка чанками по 30 секунд
        overlap=1.0,
    )
    
    # Тестовое разделение
    try:
        result = separator.separate("test.mp3")
        print(f"Разделение завершено за {result.processing_time:.2f}с")
        for stem_name, stem_data in result.stems.items():
            print(f"{stem_name}: shape={stem_data.shape}")
    except Exception as e:
        print(f"Тест не удался: {e}")