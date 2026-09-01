from __future__ import annotations

import random

import numpy as np
import pandas as pd

from gap_imputation_benchmark.benchmark.eye_tracking import (
    EyeTrackingBenchmarkConfig,
    select_balanced_recordings,
)
from scripts.build_eye_tracking_benchmark import _sample_reference_recording_gaps


def config(**overrides: object) -> EyeTrackingBenchmarkConfig:
    values: dict[str, object] = {
        "min_gap_duration_ms": 0.0,
        "max_gap_duration_ms": 50.0,
        "n_gaps_per_recording": 2,
        "n_strata": 2,
        "min_context_valid_fraction": 0.8,
        "random_state": 42,
        "participants_per_dataset": 1,
        "recordings_per_participant": 2,
        "pedrotti_recordings_per_participant": 2,
        "pedrotti_gaps_per_participant": 2,
        "max_gap_sampling_attempts": 100,
    }
    values.update(overrides)
    return EyeTrackingBenchmarkConfig(**values)


def recording(dataset_id: str, participant_id: str, recording_id: str, length: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": [dataset_id] * length,
            "participant_id": [participant_id] * length,
            "recording_id": [recording_id] * length,
            "session_id": ["session_1"] * length,
            "source_file": [f"{dataset_id}/{recording_id}.csv"] * length,
            "gaze_x": np.arange(length, dtype=float),
            "is_valid": [True] * length,
            "sampling_rate_hz": [1_000.0] * length,
        }
    )


def test_reference_protocol_records_a_shortfall_instead_of_raising() -> None:
    accepted, skipped = _sample_reference_recording_gaps(
        recording("Pedrotti", "participant", "short", length=5),
        config(min_gap_duration_ms=50.0, max_gap_duration_ms=51.0, n_strata=1),
        target_strata=[1],
    )

    assert accepted == []
    assert len(skipped) == 1
    assert skipped[0]["requested_gap_number"] == 1
    assert skipped[0]["duration_stratum"] == 1
    assert skipped[0]["attempts_used"] == 100
    assert skipped[0]["reason"] == "no_eligible_nonoverlapping_position_in_duration_stratum"


def test_reference_selection_retains_legacy_random_sample_order() -> None:
    source = []
    for dataset_id in ("GazeBase", "GazeBaseVR", "ZuCo"):
        source.extend(recording(dataset_id, "participant", f"recording_{index}") for index in range(4))
    source.extend(recording("Pedrotti", "participant", f"trial_{index}") for index in range(4))

    selected = select_balanced_recordings(source, config())
    observed = [frame["recording_id"].iat[0] for frame in selected]
    expected = []
    for dataset_id, prefix in (
        ("GazeBase", "recording"),
        ("GazeBaseVR", "recording"),
        ("Pedrotti", "trial"),
        ("ZuCo", "recording"),
    ):
        available = [f"{prefix}_{index}" for index in range(4)]
        expected.extend(
            random.Random(f"42|recordings|{dataset_id}|participant").sample(available, 2)
        )

    assert observed == expected
