from __future__ import annotations

import pandas as pd

from gap_imputation_benchmark.benchmark.provenance import (
    recording_provenance,
    recording_seed,
)


def test_recording_provenance_is_stable_for_one_recording() -> None:
    recording = pd.DataFrame(
        {
            "dataset_id": ["dataset_a", "dataset_a"],
            "participant_id": ["participant_1", "participant_1"],
            "recording_id": ["recording_1", "recording_1"],
            "session_id": ["session_1", "session_1"],
            "source_file": ["dataset_a/recording_1.csv"] * 2,
        }
    )

    assert recording_provenance(recording) == (
        "dataset_id=dataset_a",
        "participant_id=participant_1",
        "recording_id=recording_1",
        "session_id=session_1",
        "source_file=dataset_a/recording_1.csv",
    )


def test_recording_seed_is_repeatable_and_recording_specific() -> None:
    first = recording_seed(42, ("dataset_id=a", "recording_id=1"))
    second = recording_seed(42, ("dataset_id=a", "recording_id=1"))
    other = recording_seed(42, ("dataset_id=a", "recording_id=2"))

    assert first == second
    assert first != other
