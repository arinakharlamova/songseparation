"""
Модуль разделения аудио с использованием Demucs.

Выполняет высококачественное разделение источников с использованием модели Demucs от Facebook.
Поддерживает разделение на 4 стема: вокал, барабаны, бас, прочее.
"""
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model
from loguru import logger

from app.utils import remove_silence, compute_file_hash

logger.add("logs/separator.log", rotation="10 MB", level="INFO")

STEM_NAMES = ["drums", "bass", "other", "vocals"]


def _load_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    """Загрузить аудио с помощью soundfile, вернуть тензор torch."""
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim == 1:
        data = data[:, np.newaxis]
    # Преобразование в torch: (каналы, сэмплы)
    waveform = torch.from_numpy(data.T)
    return waveform, sr


def _save_audio(waveform: torch.Tensor, path: str, sr: int) -> None:
    """Сохранить тензор torch как WAV с помощью soundfile (24-бит для лучшего качества)."""
    data = waveform.cpu().numpy().T
    sf.write(path, data, sr, subtype="PCM_24")


class SourceSeparator:
    """
    Разделитель музыкальных источников на базе Demucs с защитой по таймауту.
    """

    def __init__(
        self,
        model_name: str = "htdemucs_ft",
        device: Optional[str] = None,
        output_dir: str = "output",
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"SourceSeparator: model={model_name}, device={self.device}")

    def separate(
        self,
        audio_path: str,
        output_subdir: Optional[str] = None,
        save_result: bool = True,
    ) -> dict[str, Path]:
        """Разделить с использованием лениво загруженной модели (в процессе)."""
        audio_p = Path(audio_path)
        if not audio_p.exists():
            raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")

        logger.info(f"Начало разделения: {audio_path}")

        if output_subdir is None:
            output_subdir = audio_p.stem
        output_p = self.output_dir / output_subdir
        output_p.mkdir(parents=True, exist_ok=True)

        model = self._get_model()

        waveform, sr = _load_audio(str(audio_p))
        logger.info(f"Загружено: {waveform.shape}, sr={sr}")

        with torch.no_grad():
            separated = apply_model(model, waveform.unsqueeze(0).to(self.device))

        separated = separated.squeeze(0).cpu()

        stems_paths = {}
        if save_result:
            for idx, stem_name in enumerate(STEM_NAMES):
                stem_waveform = separated[idx]
                # Нормализация каждого стема для предотвращения клиппинга и уменьшения артефактов
                peak = stem_waveform.abs().max()
                if peak > 1e-6:
                    stem_waveform = stem_waveform / peak * 0.95
                stem_path = output_p / f"{stem_name}.wav"
                _save_audio(stem_waveform, str(stem_path), sr)
                stems_paths[stem_name] = stem_path
                logger.info(f"Сохранён {stem_name}: {stem_path}")

        return stems_paths

    def separate_with_timeout(
        self,
        audio_path: str,
        timeout_seconds: int = 3600,
    ) -> dict[str, Path]:
        """
        Разделить аудио (упрощённая версия без multiprocessing).
        Для таймаута используйте systemd или gunicorn настройки.
        """
        import signal
        
        def handler(signum, frame):
            raise TimeoutError(f"Превышен таймаут разделения в {timeout_seconds}с")
        
        # Установка таймаута (работает только на Unix, на Windows просто пропускаем)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout_seconds)
        
        try:
            stems = self.separate(audio_path, save_result=True)
            
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Отмена таймаута
            
            return stems
        except TimeoutError:
            raise
        except Exception as e:
            raise RuntimeError(f"Ошибка разделения: {e}")

    def _get_model(self) -> Any:
        """Ленивая загрузка модели."""
        if not hasattr(self, "_model") or self._model is None:
            logger.info(f"Загрузка модели Demucs: {self.model_name}")
            self._model = get_model(name=self.model_name)
            self._model.eval()
            if self.device == "cuda":
                self._model = self._model.cuda()
            logger.info("Модель успешно загружена")
        return self._model


def separate_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Separate audio into stems")
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-m", "--model", default="htdemucs_ft", help="Model name")
    parser.add_argument("-d", "--device", default=None, help="Device (cpu/cuda)")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")

    args = parser.parse_args()

    separator = SourceSeparator(
        model_name=args.model,
        device=args.device,
        output_dir=args.output,
    )

    try:
        stems = separator.separate_with_timeout(
            args.input,
            timeout_seconds=args.timeout,
        )
        print("Separation complete!")
        for name, path in stems.items():
            print(f"  {name}: {path}")
    except TimeoutError as e:
        logger.error(e)
        raise SystemExit(1)


if __name__ == "__main__":
    separate_cli()
