"""Configuration and deterministic sampling rules for the Eye-Tracking domain."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from gap_imputation_benchmark.benchmark.features import RAW_FEATURE_NUMERATOR_COLUMNS
from gap_imputation_benchmark.benchmark.records import FEATURES_BASIC_NORMALIZED
from gap_imputation_benchmark.domains.base import DomainSpec
from gap_imputation_benchmark.imputers.registry import EYE_TRACKING_IMPUTERS


EYE_TRACKING_METHOD_LABELS = {
    "forward_fill": "Forward fill",
    "nearest_boundary": "Nearest boundary",
    "linear": "Linear interpolation",
    "pchip": "PCHIP",
    "local_natural_cubic_spline": "Local natural cubic spline",
    "polyfit_bic": "Polyfit (BIC)",
    "template": "Template reconstruction",
}


@dataclass(frozen=True)
class EyeTrackingBenchmarkConfig:
    """Frozen design parameters for the four-dataset Eye-Tracking benchmark."""

    min_gap_duration_ms: float
    max_gap_duration_ms: float
    n_gaps_per_recording: int
    n_strata: int
    min_context_valid_fraction: float
    random_state: int
    participants_per_dataset: int
    recordings_per_participant: int
    pedrotti_recordings_per_participant: int
    pedrotti_gaps_per_participant: int
    max_gap_sampling_attempts: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_gap_duration_ms) or self.min_gap_duration_ms < 0:
            raise ValueError("min_gap_duration_ms must be finite and non-negative.")
        if not math.isfinite(self.max_gap_duration_ms) or self.max_gap_duration_ms <= self.min_gap_duration_ms:
            raise ValueError("max_gap_duration_ms must exceed min_gap_duration_ms.")
        if self.n_gaps_per_recording < 1 or self.n_strata < 1:
            raise ValueError("n_gaps_per_recording and n_strata must be positive.")
        if not 0 < self.min_context_valid_fraction <= 1:
            raise ValueError("min_context_valid_fraction must be in (0, 1].")
        counts = (
            self.participants_per_dataset,
            self.recordings_per_participant,
            self.pedrotti_recordings_per_participant,
            self.pedrotti_gaps_per_participant,
        )
        if any(value < 1 for value in counts):
            raise ValueError("All recording and participant counts must be positive.")
        if self.pedrotti_recordings_per_participant < self.pedrotti_gaps_per_participant:
            raise ValueError("pedrotti_recordings_per_participant must cover all requested Pedrotti gaps.")
        if self.max_gap_sampling_attempts < 1:
            raise ValueError("max_gap_sampling_attempts must be positive.")


_CONFIG_TABLE = "benchmark"
_EXPECTED_CONFIG_KEYS = frozenset(EyeTrackingBenchmarkConfig.__dataclass_fields__)


def load_eye_tracking_benchmark_config(path: str | Path) -> EyeTrackingBenchmarkConfig:
    """Load a strict, reviewable TOML configuration for this domain."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility path.
        import tomli as tomllib

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Eye-Tracking configuration does not exist: {path}")
    try:
        with path.open("rb") as handle:
            document: dict[str, Any] = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid TOML Eye-Tracking configuration: {path}") from error
    values = document.get(_CONFIG_TABLE)
    if not isinstance(values, dict):
        raise ValueError(f"Configuration must define a [{_CONFIG_TABLE}] table.")
    supplied_keys = frozenset(values)
    missing, unknown = sorted(_EXPECTED_CONFIG_KEYS - supplied_keys), sorted(supplied_keys - _EXPECTED_CONFIG_KEYS)
    if missing or unknown:
        problems = ([f"missing keys: {missing}"] if missing else []) + ([f"unknown keys: {unknown}"] if unknown else [])
        raise ValueError("Invalid Eye-Tracking configuration (" + "; ".join(problems) + ").")
    return EyeTrackingBenchmarkConfig(**values)


def portable_source_reference(file_path: str | Path, raw_data_dir: str | Path) -> str:
    """Return a raw-data-root-relative reference with no machine-specific prefix."""
    try:
        return Path(file_path).resolve().relative_to(Path(raw_data_dir).resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Source file must be below the configured raw-data directory: {file_path}") from error


def replace_source_reference(recording: pd.DataFrame, source_reference: str) -> pd.DataFrame:
    """Copy a recording and replace its local source path with a portable reference."""
    if "source_file" not in recording.columns:
        raise ValueError("Recording is missing source_file.")
    result = recording.copy()
    result["source_file"] = source_reference
    return result


def select_balanced_recordings(recordings: Sequence[pd.DataFrame], config: EyeTrackingBenchmarkConfig) -> list[pd.DataFrame]:
    """Select the deterministic, reference-protocol input subset for this domain."""
    selected: list[pd.DataFrame] = []
    by_dataset: dict[str, list[pd.DataFrame]] = {}
    for recording in recordings:
        if recording.empty or "dataset_id" not in recording.columns:
            raise ValueError("Each recording must have a non-empty dataset_id column.")
        values = recording["dataset_id"].dropna().unique()
        if len(values) != 1:
            raise ValueError("Each recording must have exactly one dataset_id.")
        by_dataset.setdefault(str(values[0]), []).append(recording)

    required_datasets = {"GazeBase", "GazeBaseVR", "ZuCo", "Pedrotti"}
    missing_datasets = sorted(required_datasets - set(by_dataset))
    if missing_datasets:
        raise ValueError(f"Missing required Eye-Tracking datasets: {missing_datasets}")
    for dataset_id in sorted(required_datasets):
        by_participant: dict[str, list[pd.DataFrame]] = {}
        for recording in by_dataset[dataset_id]:
            values = recording.get("participant_id", pd.Series(dtype=object)).dropna().unique()
            if len(values) != 1:
                raise ValueError("Each recording must have exactly one participant_id.")
            by_participant.setdefault(str(values[0]), []).append(recording)
        required = config.pedrotti_recordings_per_participant if dataset_id == "Pedrotti" else config.recordings_per_participant
        eligible = sorted(participant_id for participant_id, frames in by_participant.items() if len(frames) >= required)
        if len(eligible) < config.participants_per_dataset:
            raise ValueError(f"{dataset_id} has {len(eligible)} eligible participants; {config.participants_per_dataset} are required.")
        participant_rng = random.Random(f"{config.random_state}|participants|{dataset_id}")
        for participant_id in sorted(participant_rng.sample(eligible, config.participants_per_dataset)):
            frames = sorted(by_participant[participant_id], key=lambda frame: (str(frame["recording_id"].iat[0]), str(frame["source_file"].iat[0])))
            recording_rng = random.Random(f"{config.random_state}|recordings|{dataset_id}|{participant_id}")
            selected.extend(recording_rng.sample(frames, required))
    return selected


EYE_TRACKING_DOMAIN = DomainSpec(
    name="eye_tracking",
    feature_columns=FEATURES_BASIC_NORMALIZED,
    methods=EYE_TRACKING_IMPUTERS,
    scale_dependent_feature_numerators=RAW_FEATURE_NUMERATOR_COLUMNS,
    method_labels=EYE_TRACKING_METHOD_LABELS,
)
