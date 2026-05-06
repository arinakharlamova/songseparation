"""Tests for SourceSeparator module."""
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.separator import SourceSeparator, STEM_NAMES


@pytest.fixture
def test_audio_file(tmp_path):
    """Create a test audio file."""
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    # Stereo sine wave
    waveform = np.column_stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)])
    file_path = tmp_path / "test_track.wav"
    sf.write(str(file_path), waveform, sr)
    return file_path


@pytest.fixture
def separator(tmp_path):
    """Create a SourceSeparator instance."""
    output_dir = tmp_path / "output"
    return SourceSeparator(
        model_name="htdemucs",
        device="cpu",
        output_dir=str(output_dir),
    )


class TestSourceSeparator:
    """Test cases for SourceSeparator."""

    def test_init(self, tmp_path):
        """Test separator initialization."""
        sep = SourceSeparator(
            model_name="htdemucs",
            device="cpu",
            output_dir=str(tmp_path / "out"),
        )
        assert sep.model_name == "htdemucs"
        assert sep.device == "cpu"
        assert sep.output_dir.exists()

    def test_separate_nonexistent_file(self, separator):
        """Test separation with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            separator.separate("nonexistent.wav")

    @patch("app.separator.get_model")
    def test_separate_mock(self, mock_get_model, test_audio_file, tmp_path):
        """Test separation with mocked model."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.cuda = MagicMock()
        mock_model.return_value = np.zeros((4, 2, 88200))  # 4 stems, 2 channels, 2 sec

        mock_get_model.return_value = mock_model

        sep = SourceSeparator(
            model_name="htdemucs",
            device="cpu",
            output_dir=str(tmp_path / "output"),
        )

        # Mock apply_model to return fake separated audio
        with patch("app.separator.apply_model") as mock_apply:
            import torch
            # Create fake separated output: (1, 4, channels, samples)
            fake_output = torch.zeros((1, 4, 2, 88200))
            for i in range(4):
                fake_output[0, i] = torch.sin(torch.linspace(0, 100, 88200))

            mock_apply.return_value = fake_output

            stems = sep.separate(str(test_audio_file), save_result=True)

            assert len(stems) == 4
            for stem_name in STEM_NAMES:
                assert stem_name in stems
                assert stems[stem_name].exists()

    def test_separate_with_timeout_mock(self, test_audio_file, tmp_path):
        """Test separate_with_timeout with mocked process."""
        sep = SourceSeparator(
            model_name="htdemucs",
            device="cpu",
            output_dir=str(tmp_path / "output"),
        )

        with patch("multiprocessing.Process") as mock_process_class:
            mock_process = MagicMock()
            mock_process_class.return_value = mock_process
            mock_process.is_alive.return_value = False
            mock_process.join = MagicMock()

            # Mock the result queue
            with patch("multiprocessing.Queue") as mock_queue_class:
                mock_queue = MagicMock()
                mock_queue_class.return_value = mock_queue
                mock_queue.empty.return_value = False
                mock_queue.get.return_value = {
                    "status": "ok",
                    "stems": {
                        "vocals": str(tmp_path / "vocals.wav"),
                        "drums": str(tmp_path / "drums.wav"),
                        "bass": str(tmp_path / "bass.wav"),
                        "other": str(tmp_path / "other.wav"),
                    },
                }

                stems = sep.separate_with_timeout(
                    str(test_audio_file),
                    timeout_seconds=10,
                )

                assert len(stems) == 4
                mock_process.start.assert_called_once()

    def test_separate_with_timeout_timeout(self, test_audio_file, tmp_path):
        """Test separate_with_timeout when timeout occurs."""
        sep = SourceSeparator(
            model_name="htdemucs",
            device="cpu",
            output_dir=str(tmp_path / "output"),
        )

        with patch("multiprocessing.Process") as mock_process_class:
            mock_process = MagicMock()
            mock_process_class.return_value = mock_process
            mock_process.is_alive.return_value = True  # Still running
            mock_process.join = MagicMock()
            mock_process.terminate = MagicMock()
            mock_process.kill = MagicMock()

            with pytest.raises(TimeoutError):
                sep.separate_with_timeout(
                    str(test_audio_file),
                    timeout_seconds=0.1,
                )

            mock_process.terminate.assert_called_once()

    def test_stem_names_constant(self):
        """Test that STEM_NAMES has correct values."""
        assert STEM_NAMES == ["drums", "bass", "other", "vocals"]
        assert len(STEM_NAMES) == 4


class TestAudioLoading:
    """Test audio loading and saving functions."""

    def test_load_and_save_roundtrip(self, tmp_path):
        """Test that audio can be loaded and saved correctly."""
        from app.separator import _load_audio, _save_audio
        import torch

        # Create test audio
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        data = np.sin(2 * np.pi * 440 * t)
        input_path = tmp_path / "input.wav"
        sf.write(str(input_path), data, sr)

        # Load
        waveform, loaded_sr = _load_audio(str(input_path))
        assert isinstance(waveform, torch.Tensor)
        assert loaded_sr == sr
        assert waveform.shape[0] == 1 or waveform.shape[0] == 2  # channels

        # Save
        output_path = tmp_path / "output.wav"
        _save_audio(waveform, str(output_path), sr)
        assert output_path.exists()

        # Verify saved file
        reloaded, reloaded_sr = sf.read(str(output_path))
        assert reloaded_sr == sr
