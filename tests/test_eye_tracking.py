from __future__ import annotations

import pandas as pd
import pytest

from gap_imputation_benchmark.benchmark.eye_tracking import (
    EyeTrackingBenchmarkConfig,
    load_eye_tracking_benchmark_config,
    portable_source_reference,
    replace_source_reference,
    select_balanced_recordings,
)


def config(**overrides: object) -> EyeTrackingBenchmarkConfig:
    values: dict[str, object] = {
        "min_gap_duration_ms": 1.0,
        "max_gap_duration_ms": 250.0,
        "n_gaps_per_recording": 2,
        "n_strata": 2,
        "min_context_valid_fraction": 0.8,
        "random_state": 42,
        "participants_per_dataset": 1,
        "recordings_per_participant": 2,
        "pedrotti_recordings_per_participant": 3,
        "pedrotti_gaps_per_participant": 3,
        "max_gap_sampling_attempts": 100,
    }
    values.update(overrides)
    return EyeTrackingBenchmarkConfig(**values)


def recording(dataset_id: str, participant_id: str, recording_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": [dataset_id] * 4,
            "participant_id": [participant_id] * 4,
            "recording_id": [recording_id] * 4,
            "session_id": ["session_1"] * 4,
            "source_file": [f"{dataset_id}/{recording_id}.csv"] * 4,
        }
    )


def test_portable_source_reference_removes_machine_specific_prefix(tmp_path):
    raw_dir = tmp_path / "raw"
    source = raw_dir / "GazeBase_v2_0" / "S_001_S1_TEX.csv"
    source.parent.mkdir(parents=True)
    source.touch()

    assert portable_source_reference(source, raw_dir) == "GazeBase_v2_0/S_001_S1_TEX.csv"
    with pytest.raises(ValueError, match="configured raw-data directory"):
        portable_source_reference(tmp_path / "outside.csv", raw_dir)


def test_configuration_loader_rejects_unknown_keys(tmp_path):
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(
        "[benchmark]\n"
        "min_gap_duration_ms = 1.0\n"
        "max_gap_duration_ms = 2.0\n"
        "n_gaps_per_recording = 1\n"
        "n_strata = 1\n"
        "min_context_valid_fraction = 0.8\n"
        "random_state = 42\n"
        "participants_per_dataset = 1\n"
        "recordings_per_participant = 1\n"
        "pedrotti_recordings_per_participant = 1\n"
        "pedrotti_gaps_per_participant = 1\n"
        "max_gap_sampling_attempts = 100\n"
        "unexpected = true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown keys"):
        load_eye_tracking_benchmark_config(config_path)


def test_zero_duration_lower_bound_represents_the_smallest_sampled_gap():
    assert config(min_gap_duration_ms=0.0).min_gap_duration_ms == 0.0


def test_replace_source_reference_does_not_mutate_loader_output():
    original = recording("GazeBase", "01", "source")
    updated = replace_source_reference(original, "GazeBase_v2_0/source.csv")

    assert original["source_file"].iat[0] == "GazeBase/source.csv"
    assert updated["source_file"].iat[0] == "GazeBase_v2_0/source.csv"


def test_balanced_selection_is_deterministic_and_preserves_all_domains():
    source = []
    for dataset_id in ("GazeBase", "GazeBaseVR", "ZuCo"):
        source.extend(recording(dataset_id, "participant", f"recording_{index}") for index in range(2))
    source.extend(recording("Pedrotti", "participant", f"trial_{index}") for index in range(3))

    first = select_balanced_recordings(source, config())
    second = select_balanced_recordings(list(reversed(source)), config())

    assert [frame["recording_id"].iat[0] for frame in first] == [
        frame["recording_id"].iat[0] for frame in second
    ]
    assert {frame["dataset_id"].iat[0] for frame in first} == {
        "GazeBase", "GazeBaseVR", "ZuCo", "Pedrotti"
    }
