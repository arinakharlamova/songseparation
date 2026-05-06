"""
Utility functions for audio processing.
"""
import hashlib
import io
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """
    Compute MD5 hash of file for caching.
    """
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def remove_silence(
    waveform: torch.Tensor,
    threshold: float = 0.01,
    frame_length: int = 512,
) -> torch.Tensor:
    """
    Remove silence from beginning and end of audio.
    """
    waveform = waveform.cpu()

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    num_samples = waveform.shape[-1]
    if num_samples < frame_length:
        return waveform

    frames = waveform.unfold(-1, frame_length, frame_length // 2)
    energy = torch.sqrt(torch.mean(frames ** 2, dim=-1))
    non_silent = energy > threshold

    if not non_silent.any():
        return waveform

    first_frame = torch.argmax(non_silent.float(), dim=-1).max().item()
    last_frame = non_silent.shape[-1] - torch.argmax(
        torch.flip(non_silent.float(), dims=[-1])
    ).max().item()

    hop_length = frame_length // 2
    first = max(0, int(first_frame * hop_length))
    last = min(num_samples, int(last_frame * hop_length) + frame_length)

    buffer = frame_length // 4
    first = max(0, first - buffer)
    last = min(num_samples, last + buffer)

    return waveform[..., first:last]


def trim_to_duration(
    waveform: torch.Tensor,
    duration_seconds: float,
    sample_rate: int = 44100,
) -> torch.Tensor:
    """
    Trim audio to specified duration.
    """
    max_samples = int(duration_seconds * sample_rate)
    if waveform.shape[-1] > max_samples:
        return waveform[..., :max_samples]
    return waveform


def load_audio(
    file_path: str | Path,
    target_sr: Optional[int] = None,
    normalize: bool = True,
) -> tuple[torch.Tensor, int]:
    """
    Load audio file with optional normalization.
    """
    data, sr = sf.read(str(file_path), dtype="float32")
    if data.ndim == 1:
        data = data[:, np.newaxis]
    waveform = torch.from_numpy(data.T)

    if normalize:
        peak = waveform.abs().max()
        if peak > 1e-6:
            waveform = waveform / peak

    if target_sr is not None and target_sr != sr:
        waveform = _resample(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr


def save_audio(
    waveform: torch.Tensor,
    file_path: str | Path,
    sample_rate: int = 44100,
    bit_depth: int = 16,
    format: Optional[str] = None,
) -> None:
    """
    Save audio to file.
    """
    data = waveform.cpu().numpy().T
    subtype = "PCM_16" if bit_depth == 16 else "PCM_24"
    sf.write(str(file_path), data, sample_rate, subtype=subtype)


def _resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Simple resampling using numpy interpolation."""
    if orig_sr == target_sr:
        return waveform

    waveform = waveform.cpu()
    num_channels = waveform.shape[0]
    num_samples = waveform.shape[1]
    new_samples = int(num_samples * target_sr / orig_sr)

    resampled = torch.zeros(num_channels, new_samples)
    old_t = np.linspace(0, 1, num_samples)
    new_t = np.linspace(0, 1, new_samples)

    for ch in range(num_channels):
        resampled[ch] = torch.from_numpy(
            np.interp(new_t, old_t, waveform[ch].numpy())
        )

    return resampled


def create_stem_archive(stems_dir: Path, archive_path: Optional[Path] = None) -> Path:
    """
    Create ZIP archive from separated stems.
    """
    if archive_path is None:
        archive_path = stems_dir.parent / f"{stems_dir.stem}.zip"

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for stem_file in stems_dir.glob("*.wav"):
            zipf.write(stem_file, stem_file.name)

    return archive_path


def validate_audio_format(file_path: Path) -> tuple[bool, str]:
    """
    Validate audio file format.
    """
    valid_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    suffix = file_path.suffix.lower()

    if suffix not in valid_extensions:
        return False, f"Unsupported format. Supported: {', '.join(sorted(valid_extensions))}"

    try:
        info = sf.info(str(file_path))
        return True, f"{info.samplerate}Hz, {info.channels}ch, {info.subtype}"
    except Exception as e:
        return False, f"Cannot read audio: {e}"
