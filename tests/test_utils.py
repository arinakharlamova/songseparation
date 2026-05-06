import hashlib
import tempfile
import torch
import numpy as np
from pathlib import Path

import pytest
import soundfile as sf

from app.utils import (
    compute_file_hash,
    remove_silence,
    trim_to_duration,
    create_stem_archive,
)


def test_compute_file_hash():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"Hello, world!")
        f.flush()
        file_path = Path(f.name)

    expected_hash = hashlib.md5(b"Hello, world!").hexdigest()
    assert compute_file_hash(file_path) == expected_hash

    file_path.unlink()


def test_remove_silence_all_silent():
    # All silence - should return original
    waveform = torch.zeros(1, 1000)
    processed = remove_silence(waveform, threshold=0.01)
    assert processed.shape == waveform.shape


def test_remove_silence_has_signal():
    # Signal in the middle
    waveform = torch.zeros(1, 10000)
    waveform[0, 2000:8000] = 0.5
    processed = remove_silence(waveform, threshold=0.1)
    # Should be shorter than original
    assert processed.shape[1] < waveform.shape[1]


def test_remove_silence_no_trimming_needed():
    # Signal throughout
    waveform = torch.ones(1, 2000) * 0.5
    processed = remove_silence(waveform, threshold=0.1)
    # Should not trim much (maybe small buffer on edges)
    assert processed.shape[1] >= waveform.shape[1] - 512


def test_compute_file_hash_different_files():
    import tempfile as tmp
    dir_path = tmp.mkdtemp()
    file1_path = Path(dir_path) / "file1.txt"
    file2_path = Path(dir_path) / "file2.txt"
    file1_path.write_bytes(b"content1")
    file2_path.write_bytes(b"content2")

    hash1 = compute_file_hash(file1_path)
    hash2 = compute_file_hash(file2_path)
    assert hash1 != hash2

    file1_path.unlink()
    file2_path.unlink()
    Path(dir_path).rmdir()
