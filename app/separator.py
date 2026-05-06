"""
Source Separation Module using Demucs.

Provides high-quality source separation using Facebook's Demucs model.
Supports 4-stem separation: vocals, drums, bass, other.
"""
import multiprocessing
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
    """Load audio using soundfile, return torch tensor."""
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim == 1:
        data = data[:, np.newaxis]
    # Convert to torch: (channels, samples)
    waveform = torch.from_numpy(data.T)
    return waveform, sr


def _save_audio(waveform: torch.Tensor, path: str, sr: int) -> None:
    """Save torch tensor as WAV using soundfile (24-bit for best quality)."""
    data = waveform.cpu().numpy().T
    sf.write(path, data, sr, subtype="PCM_24")


def _separation_worker(
    model_name: str,
    device: str,
    audio_path: str,
    output_dir: str,
    result_queue: multiprocessing.Queue,
) -> None:
    """Run separation in a separate process so it can be terminated."""
    try:
        logger.info(f"[Worker] Loading model: {model_name}")
        model = get_model(name=model_name)
        model.eval()
        if device == "cuda":
            model = model.cuda()

        audio_p = Path(audio_path)
        output_p = Path(output_dir) / audio_p.stem
        output_p.mkdir(parents=True, exist_ok=True)

        waveform, sr = _load_audio(str(audio_p))
        logger.info(f"[Worker] Loaded: {waveform.shape}, sr={sr}")

        with torch.no_grad():
            separated = apply_model(model, waveform.unsqueeze(0).to(device))

        separated = separated.squeeze(0).cpu()

        stems_paths = {}
        for idx, stem_name in enumerate(STEM_NAMES):
            stem_waveform = separated[idx]
            # Normalize each stem to prevent clipping and reduce artifacts
            peak = stem_waveform.abs().max()
            if peak > 1e-6:
                stem_waveform = stem_waveform / peak * 0.95
            stem_path = output_p / f"{stem_name}.wav"
            _save_audio(stem_waveform, str(stem_path), sr)
            stems_paths[stem_name] = str(stem_path)

        result_queue.put({"status": "ok", "stems": stems_paths})

    except Exception as e:
        result_queue.put({"status": "error", "message": str(e)})


class SourceSeparator:
    """
    Music source separator using Demucs with process-based timeout.
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
        """Separate using lazy-loaded model (in-process)."""
        audio_p = Path(audio_path)
        if not audio_p.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Starting separation: {audio_path}")

        if output_subdir is None:
            output_subdir = audio_p.stem
        output_p = self.output_dir / output_subdir
        output_p.mkdir(parents=True, exist_ok=True)

        model = self._get_model()

        waveform, sr = _load_audio(str(audio_p))
        logger.info(f"Loaded: {waveform.shape}, sr={sr}")

        with torch.no_grad():
            separated = apply_model(model, waveform.unsqueeze(0).to(self.device))

        separated = separated.squeeze(0).cpu()

        stems_paths = {}
        if save_result:
            for idx, stem_name in enumerate(STEM_NAMES):
                stem_waveform = separated[idx]
                # Normalize each stem to prevent clipping and reduce artifacts
                peak = stem_waveform.abs().max()
                if peak > 1e-6:
                    stem_waveform = stem_waveform / peak * 0.95
                stem_path = output_p / f"{stem_name}.wav"
                _save_audio(stem_waveform, str(stem_path), sr)
                stems_paths[stem_name] = stem_path
                logger.info(f"Saved {stem_name}: {stem_path}")

        return stems_paths

    def separate_with_timeout(
        self,
        audio_path: str,
        timeout_seconds: int = 3600,
    ) -> dict[str, Path]:
        """
        Separate audio with timeout protection using multiprocessing.
        The process is actually terminated on timeout (no thread leak).
        """
        result_queue: multiprocessing.Queue = multiprocessing.Queue()

        process = multiprocessing.Process(
            target=_separation_worker,
            args=(self.model_name, self.device, audio_path, str(self.output_dir), result_queue),
        )
        process.start()
        process.join(timeout=timeout_seconds)

        if process.is_alive():
            logger.warning(f"Timeout after {timeout_seconds}s, terminating process")
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            raise TimeoutError(f"Separation exceeded timeout of {timeout_seconds}s")

        if result_queue.empty():
            raise RuntimeError("Separation process exited without result")

        result = result_queue.get(timeout=10)

        if result["status"] == "error":
            raise RuntimeError(result["message"])

        stems = {}
        for name, path_str in result["stems"].items():
            stems[name] = Path(path_str)
        return stems

    def _get_model(self) -> Any:
        """Lazy load model."""
        if not hasattr(self, "_model") or self._model is None:
            logger.info(f"Loading Demucs model: {self.model_name}")
            self._model = get_model(name=self.model_name)
            self._model.eval()
            if self.device == "cuda":
                self._model = self._model.cuda()
            logger.info("Model loaded successfully")
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
    multiprocessing.set_start_method("spawn", force=True)
    separate_cli()
