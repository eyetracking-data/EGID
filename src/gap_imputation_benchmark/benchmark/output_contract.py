"""Portable, domain-independent names for versioned benchmark outputs."""

from __future__ import annotations

from pathlib import Path


BENCHMARK_OUTPUTS = (
    "learnable_gap_table.csv",
    "input_manifest.csv",
    "selected_recordings.csv",
    "excluded_gaps.csv",
    "dataset_summary.csv",
    "coverage_table.csv",
    "metadata.json",
)
EVALUATION_OUTPUTS = (
    "lodo_gap_predictions.csv",
    "lodo_hyperparameter_tuning.csv",
    "lodo_feature_scale_floors.csv",
    "lodo_summary.csv",
    "lodo_selected_method_counts.csv",
    "lodo_selected_method_counts_overall.csv",
    "metadata.json",
)
SELECTOR_ARTIFACT_OUTPUTS = (
    "selector.joblib",
    "metadata.json",
)


def validate_output_contract(directory: str | Path, expected_files: tuple[str, ...]) -> None:
    """Fail immediately when a workflow did not write its complete public output."""
    directory = Path(directory)
    missing = [name for name in expected_files if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"Output contract is incomplete in {directory}: {missing}")
