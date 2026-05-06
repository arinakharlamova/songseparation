"""Application package."""
from app.config import Settings, settings
from app.separator import SourceSeparator, STEM_NAMES
from app.evaluation import (
    evaluate_track,
    evaluate_dataset,
    aggregate_results,
    run_full_evaluation,
    separate_and_evaluate,
)
from app.utils import (
    compute_file_hash,
    remove_silence,
    create_stem_archive,
    validate_audio_format,
)

__all__ = [
    "Settings",
    "settings",
    "SourceSeparator",
    "STEM_NAMES",
    "evaluate_track",
    "evaluate_dataset",
    "aggregate_results",
    "run_full_evaluation",
    "separate_and_evaluate",
    "compute_file_hash",
    "remove_silence",
    "create_stem_archive",
    "validate_audio_format",
]
