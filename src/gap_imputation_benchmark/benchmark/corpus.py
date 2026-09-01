"""Generate deterministic corpus-level benchmark records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from gap_imputation_benchmark.benchmark.config import CorpusBenchmarkConfig
from gap_imputation_benchmark.benchmark.gaps import (
    ArtificialGap,
    GapSamplingError,
    create_artificial_gap,
    sample_stratified_random_gaps,
)
from gap_imputation_benchmark.benchmark.records import (
    FEATURES_BASIC_NORMALIZED,
    build_gap_benchmark_record,
    validate_features_basic_normalized,
)
from gap_imputation_benchmark.benchmark.provenance import (
    recording_provenance,
    recording_seed,
)
from gap_imputation_benchmark.benchmark.splits import (
    ParticipantSplits,
    assign_recordings_to_splits,
    split_participants,
)


@dataclass(frozen=True)
class CorpusBenchmarkResult:
    """In-memory corpus benchmark records and the split state that produced them."""

    train_records: tuple[dict[str, object], ...]
    validation_records: tuple[dict[str, object], ...]
    test_records: tuple[dict[str, object], ...]
    participant_splits: ParticipantSplits
    scale_floor: float
    dataset_ids: tuple[str, ...]
    config: CorpusBenchmarkConfig


_SPLIT_NAMES = ("train", "validation", "test")
def _one_non_null_identifier(recording: pd.DataFrame, column_name: str) -> str:
    if column_name not in recording.columns:
        raise ValueError(f"Missing required identifier column: {column_name}")
    values = recording[column_name].dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"Each recording must contain exactly one non-null {column_name}."
        )
    return str(values[0])


def _validate_recordings(recordings: Sequence[pd.DataFrame]) -> tuple[str, ...]:
    if len(recordings) == 0:
        raise ValueError("recordings must not be empty.")
    dataset_ids: set[str] = set()
    for recording in recordings:
        if not isinstance(recording, pd.DataFrame):
            raise ValueError("Each recording must be a pandas DataFrame.")
        dataset_ids.add(_one_non_null_identifier(recording, "dataset_id"))
        _one_non_null_identifier(recording, "participant_id")
        _one_non_null_identifier(recording, "recording_id")
    return tuple(sorted(dataset_ids))


def _sample_recording_gaps(
    recording: pd.DataFrame,
    config: CorpusBenchmarkConfig,
) -> list[ArtificialGap]:
    provenance = recording_provenance(recording)
    try:
        candidates = sample_stratified_random_gaps(
            recording=recording,
            n_gaps=config.n_gaps_per_recording,
            min_gap_duration_ms=config.min_gap_duration_ms,
            max_gap_duration_ms=config.max_gap_duration_ms,
            n_strata=config.n_strata,
            context_multiplier=config.context_multiplier,
            min_context_valid_fraction=config.min_context_valid_fraction,
            random_state=recording_seed(config.random_state, provenance),
        )
    except GapSamplingError as error:
        sampling_config = (
            f"n_gaps={config.n_gaps_per_recording}, "
            f"min_gap_duration_ms={config.min_gap_duration_ms}, "
            f"max_gap_duration_ms={config.max_gap_duration_ms}, "
            f"n_strata={config.n_strata}, "
            f"context_multiplier={config.context_multiplier}, "
            f"min_context_valid_fraction={config.min_context_valid_fraction}"
        )
        raise GapSamplingError(
            config.n_gaps_per_recording,
            0,
            f"Recording provenance: {', '.join(provenance)}. "
            f"Sampling configuration: {sampling_config}. {error}",
        ) from error
    if len(candidates) != config.n_gaps_per_recording:
        raise ValueError(
            "Sampler returned an unexpected number of candidates for recording "
            f"provenance: {', '.join(provenance)}"
        )
    return [create_artificial_gap(recording, candidate) for candidate in candidates]


def _sorted_recordings(recordings: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
    return sorted(recordings, key=recording_provenance)


def _records_for_split(
    artificial_gaps: Sequence[ArtificialGap],
    scale_floor: float,
    split_name: str,
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for artificial_gap in sorted(
        artificial_gaps,
        key=lambda gap: (
            recording_provenance(gap.masked_recording),
            gap.candidate.gap_start_idx,
            gap.candidate.gap_end_idx,
        ),
    ):
        record = build_gap_benchmark_record(artificial_gap, scale_floor=scale_floor)
        record["split"] = split_name
        records.append(record)
    return tuple(records)


def generate_corpus_benchmark(
    recordings: Sequence[pd.DataFrame],
    config: CorpusBenchmarkConfig,
) -> CorpusBenchmarkResult:
    """Generate deterministic, participant-split benchmark records in memory."""
    dataset_ids = _validate_recordings(recordings)
    participant_splits = split_participants(
        recordings,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        random_state=config.random_state,
    )
    assigned_recordings = assign_recordings_to_splits(recordings, participant_splits)
    split_gaps = {
        split_name: [
            artificial_gap
            for recording in _sorted_recordings(assigned_recordings[split_name])
            for artificial_gap in _sample_recording_gaps(recording, config)
        ]
        for split_name in _SPLIT_NAMES
    }

    # Local import avoids a configuration/scaling import cycle while retaining
    # the standalone estimator as the single definition of its numerical rule.
    from gap_imputation_benchmark.benchmark.scaling import estimate_scale_floor

    scale_floor = estimate_scale_floor(
        split_gaps["train"],
        quantile=config.scale_floor_quantile,
    )
    records_by_split = {
        split_name: _records_for_split(
            split_gaps[split_name],
            scale_floor=scale_floor,
            split_name=split_name,
        )
        for split_name in _SPLIT_NAMES
    }
    for split_name in _SPLIT_NAMES:
        expected_count = (
            len(assigned_recordings[split_name]) * config.n_gaps_per_recording
        )
        if len(records_by_split[split_name]) != expected_count:
            raise ValueError(f"Unexpected {split_name} benchmark record count.")
        if any(record.get("split") != split_name for record in records_by_split[split_name]):
            raise ValueError(f"Unexpected split label in {split_name} benchmark records.")
        if any(key.endswith("_predictions") for record in records_by_split[split_name] for key in record):
            raise ValueError("Benchmark records must not retain prediction arrays.")
    validate_features_basic_normalized()

    return CorpusBenchmarkResult(
        train_records=records_by_split["train"],
        validation_records=records_by_split["validation"],
        test_records=records_by_split["test"],
        participant_splits=participant_splits,
        scale_floor=scale_floor,
        dataset_ids=dataset_ids,
        config=config,
    )
