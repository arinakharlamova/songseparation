"""Tests for evaluation module."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.evaluation import (
    STEM_NAMES,
    calculate_simple_metrics,
    evaluate_source,
    evaluate_track,
    aggregate_results,
    resample_audio,
)


class TestResampleAudio:
    """Test audio resampling."""

    def test_no_resample_needed(self):
        """Test when sample rates match."""
        data = np.random.randn(44100, 2)
        result = resample_audio(data, 44100, 44100)
        assert result.shape == data.shape

    def test_resample_22050_to_44100(self):
        """Test upsampling."""
        data = np.random.randn(22050, 2)
        result = resample_audio(data, 22050, 44100)
        assert result.shape[0] == 44100
        assert result.shape[1] == 2

    def test_resample_mono(self):
        """Test resampling mono audio."""
        data = np.random.randn(22050)
        result = resample_audio(data, 22050, 44100)
        assert result.shape[0] == 44100


class TestSimpleMetrics:
    """Test simple metrics calculation."""

    def test_perfect_reconstruction(self):
        """When reference and estimated are identical."""
        ref = np.random.randn(44100, 2)
        est = ref.copy()

        metrics = calculate_simple_metrics(ref, est)

        assert metrics["SDR"] > 20  # Very high SDR for perfect match
        assert "SIR" in metrics
        assert "SAR" in metrics

    def test_noisy_reconstruction(self):
        """When estimated has some noise."""
        ref = np.random.randn(44100, 2) * 0.5
        est = ref + np.random.randn(44100, 2) * 0.01

        metrics = calculate_simple_metrics(ref, est)

        assert metrics["SDR"] > 0  # Should still be positive
        assert "SIR" in metrics
        assert "SAR" in metrics

    def test_different_lengths(self):
        """Test with different length arrays."""
        ref = np.random.randn(44100, 2)
        est = np.random.randn(22050, 2)

        metrics = calculate_simple_metrics(ref, est)

        assert "SDR" in metrics
        assert "SIR" in metrics
        assert "SAR" in metrics


class TestEvaluateSource:
    """Test single source evaluation."""

    def test_evaluate_source_missing_files(self, tmp_path):
        """Test with missing reference file."""
        ref_path = tmp_path / "nonexistent" / "vocals.wav"
        est_path = tmp_path / "nonexistent" / "vocals.wav"

        metrics = evaluate_source(ref_path, est_path, "vocals")
        assert metrics == {"SDR": 0.0, "SIR": 0.0, "SAR": 0.0}

    @patch("app.evaluation.HAS_MIR_EVAL", True)
    @patch("app.evaluation.mir_eval")
    def test_evaluate_source_with_mir_eval(self, mock_mir_eval, tmp_path):
        """Test using mir_eval for metrics."""
        # Create fake audio files
        ref_path = tmp_path / "vocals.wav"
        est_path = tmp_path / "vocals_est.wav"

        import soundfile as sf
        data = np.random.randn(44100)
        sf.write(str(ref_path), data, 44100)
        sf.write(str(est_path), data + 0.01 * np.random.randn(44100), 44100)

        # Mock mir_eval
        mock_mir_eval.separation.bss_eval_sources.return_value = (
            np.array([10.0]),  # SDR
            np.array([15.0]),  # SIR
            np.array([12.0]),  # SAR
            None,
        )

        metrics = evaluate_source(ref_path, est_path, "vocals", prefer_mir_eval=True)

        assert "SDR" in metrics
        mock_mir_eval.separation.bss_eval_sources.assert_called_once()

    def test_evaluate_source_fallback_to_simple(self, tmp_path):
        """Test fallback to simple metrics when libraries unavailable."""
        ref_path = tmp_path / "vocals.wav"
        est_path = tmp_path / "vocals_est.wav"

        import soundfile as sf
        data = np.random.randn(44100)
        sf.write(str(ref_path), data, 44100)
        sf.write(str(est_path), data, 44100)

        with patch("app.evaluation.HAS_MIR_EVAL", False), \
             patch("app.evaluation.HAS_MUSEVAL", False):

            metrics = evaluate_source(ref_path, est_path, "vocals")
            assert "SDR" in metrics


class TestEvaluateTrack:
    """Test full track evaluation."""

    def test_evaluate_track_all_sources(self, tmp_path):
        """Test evaluating all 4 stems."""
        # Create reference and estimated directories
        ref_dir = tmp_path / "ref"
        est_dir = tmp_path / "est"
        ref_dir.mkdir()
        est_dir.mkdir()

        import soundfile as sf
        data = np.random.randn(44100)

        for source in STEM_NAMES:
            sf.write(str(ref_dir / f"{source}.wav"), data, 44100)
            sf.write(str(est_dir / f"{source}.wav"), data, 44100)

        # Mock the evaluate_source function
        with patch("app.evaluation.evaluate_source") as mock_eval:
            mock_eval.return_value = {"SDR": 10.0, "SIR": 15.0, "SAR": 12.0}

            results = evaluate_track(ref_dir, est_dir)

            assert len(results) == 4
            for source in STEM_NAMES:
                assert source in results
                assert results[source]["SDR"] == 10.0


class TestAggregateResults:
    """Test results aggregation."""

    def test_aggregate_empty(self):
        """Test aggregation with no results."""
        results = {}
        aggregated = aggregate_results(results)

        for source in STEM_NAMES:
            assert source in aggregated
            assert aggregated[source]["SDR"]["mean"] == 0.0
            assert aggregated[source]["SDR"]["std"] == 0.0

    def test_aggregate_single_track(self):
        """Test aggregation with one track."""
        results = {
            "track1": {
                "vocals": {"SDR": 10.0, "SIR": 15.0, "SAR": 12.0},
                "drums": {"SDR": 8.0, "SIR": 14.0, "SAR": 10.0},
            }
        }

        aggregated = aggregate_results(results)

        assert aggregated["vocals"]["SDR"]["mean"] == 10.0
        assert aggregated["drums"]["SDR"]["mean"] == 8.0
        assert aggregated["bass"]["SDR"]["mean"] == 0.0  # Not in results

    def test_aggregate_multiple_tracks(self):
        """Test aggregation with multiple tracks."""
        results = {
            "track1": {
                "vocals": {"SDR": 10.0, "SIR": 15.0, "SAR": 12.0},
            },
            "track2": {
                "vocals": {"SDR": 12.0, "SIR": 16.0, "SAR": 13.0},
            },
        }

        aggregated = aggregate_results(results)

        assert aggregated["vocals"]["SDR"]["mean"] == 11.0  # Average
        assert aggregated["vocals"]["SDR"]["std"] == pytest.approx(1.0, abs=0.1)
        assert aggregated["vocals"]["SDR"]["min"] == 10.0
        assert aggregated["vocals"]["SDR"]["max"] == 12.0
