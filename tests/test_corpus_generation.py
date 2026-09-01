from __future__ import annotations

import pandas as pd
import pytest

from gap_imputation_benchmark.benchmark.config import CorpusBenchmarkConfig
from gap_imputation_benchmark.benchmark.corpus import generate_corpus_benchmark
from gap_imputation_benchmark.benchmark.gaps import GapSamplingError
from gap_imputation_benchmark.benchmark.records import FEATURES_BASIC_NORMALIZED


def recording(
    dataset_id: str,
    participant_id: str,
    recording_id: str,
    session_id: str = "S1",
) -> pd.DataFrame:
    values = list(range(60))
    return pd.DataFrame(
        {
            "dataset_id": [dataset_id] * len(values),
            "participant_id": [participant_id] * len(values),
            "recording_id": [recording_id] * len(values),
            "session_id": [session_id] * len(values),
            "source_file": [f"{recording_id}.csv"] * len(values),
            "timestamp_ms": [float(index * 10) for index in values],
            "sampling_rate_hz": [100.0] * len(values),
            "gaze_x": [float(value) for value in values],
            "is_valid": [True] * len(values),
        }
    )


def source_recordings() -> list[pd.DataFrame]:
    return [
        recording("GazeBase", "participant_1", "recording_1", "S1"),
        recording("GazeBase", "participant_1", "recording_2", "S2"),
        recording("GazeBase", "participant_2", "recording_3"),
        recording("GazeBase", "participant_3", "recording_4"),
        recording("GazeBase", "participant_4", "recording_5"),
        recording("GazeBase", "participant_5", "recording_6"),
    ]


def config(**overrides: object) -> CorpusBenchmarkConfig:
    values: dict[str, object] = {
        "min_gap_duration_ms": 20.0,
        "max_gap_duration_ms": 30.0,
        "n_strata": 1,
        "n_gaps_per_recording": 2,
        "random_state": 31,
    }
    values.update(overrides)
    return CorpusBenchmarkConfig(**values)


def records(result: object) -> tuple[dict[str, object], ...]:
    return result.train_records + result.validation_records + result.test_records


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="recordings"):
        generate_corpus_benchmark([], config())


def test_generation_is_deterministic_and_independent_of_input_order():
    source = source_recordings()

    first = generate_corpus_benchmark(source, config())
    second = generate_corpus_benchmark(source, config())
    reversed_order = generate_corpus_benchmark(list(reversed(source)), config())

    assert first == second == reversed_order


def test_generation_produces_expected_count_split_labels_and_provenance():
    source = source_recordings()
    result = generate_corpus_benchmark(source, config())

    assigned_recording_counts = {
        "train": sum(
            (frame["dataset_id"].iloc[0], frame["participant_id"].iloc[0])
            in result.participant_splits.train
            for frame in source
        ),
        "validation": sum(
            (frame["dataset_id"].iloc[0], frame["participant_id"].iloc[0])
            in result.participant_splits.validation
            for frame in source
        ),
        "test": sum(
            (frame["dataset_id"].iloc[0], frame["participant_id"].iloc[0])
            in result.participant_splits.test
            for frame in source
        ),
    }
    for split_name, split_records in (
        ("train", result.train_records),
        ("validation", result.validation_records),
        ("test", result.test_records),
    ):
        assert len(split_records) == assigned_recording_counts[split_name] * 2
        assert all(record["split"] == split_name for record in split_records)
        assert all(record["recording_id"].startswith("recording_") for record in split_records)
        assert all(not key.endswith("_predictions") for record in split_records for key in record)


def test_sampling_uses_config_values_and_stable_distinct_recording_seeds(monkeypatch):
    import gap_imputation_benchmark.benchmark.corpus as corpus_module

    original_sampler = corpus_module.sample_stratified_random_gaps
    calls: list[dict[str, object]] = []

    def sample(**kwargs: object):
        calls.append(kwargs)
        return original_sampler(**kwargs)

    monkeypatch.setattr(corpus_module, "sample_stratified_random_gaps", sample)
    benchmark_config = config(min_gap_duration_ms=20.0, max_gap_duration_ms=27.0, n_strata=2)
    generate_corpus_benchmark(source_recordings(), benchmark_config)
    first_seeds = [call["random_state"] for call in calls]

    assert all(call["min_gap_duration_ms"] == 20.0 for call in calls)
    assert all(call["max_gap_duration_ms"] == 27.0 for call in calls)
    assert all(call["n_strata"] == 2 for call in calls)
    assert len(set(first_seeds)) == len(first_seeds)

    calls.clear()
    generate_corpus_benchmark(list(reversed(source_recordings())), benchmark_config)
    assert first_seeds == [call["random_state"] for call in calls]


def test_scale_floor_uses_only_training_gaps_and_is_shared_by_all_records(monkeypatch):
    import gap_imputation_benchmark.benchmark.scaling as scaling_module

    captured_training_gaps: list[object] = []

    def estimate(training_gaps: object, quantile: float) -> float:
        captured_training_gaps.extend(training_gaps)
        assert quantile == config().scale_floor_quantile
        return 2.5

    monkeypatch.setattr(scaling_module, "estimate_scale_floor", estimate)
    result = generate_corpus_benchmark(source_recordings(), config())
    training_participants = set(result.participant_splits.train)

    assert captured_training_gaps
    assert {
        (
            gap.masked_recording["dataset_id"].iloc[0],
            gap.masked_recording["participant_id"].iloc[0],
        )
        for gap in captured_training_gaps
    } <= training_participants
    assert result.scale_floor == 2.5
    assert all(record["split"] in {"train", "validation", "test"} for record in records(result))


def test_participant_key_uses_dataset_and_participant_and_feature_allowlist_stays_clean():
    source = [
        recording("GazeBase", "same", "gb_same"),
        recording("GazeBaseVR", "same", "vr_same"),
        recording("GazeBase", "one", "one"),
        recording("GazeBaseVR", "two", "two"),
    ]
    result = generate_corpus_benchmark(source, config(n_gaps_per_recording=1))
    split_sets = [
        set(result.participant_splits.train),
        set(result.participant_splits.validation),
        set(result.participant_splits.test),
    ]

    assert result.dataset_ids == ("GazeBase", "GazeBaseVR")
    assert ("GazeBase", "same") in set().union(*split_sets)
    assert ("GazeBaseVR", "same") in set().union(*split_sets)
    assert set(FEATURES_BASIC_NORMALIZED).isdisjoint(
        {"split", "rmse", "mae", "best_method_by_rmse", "participant_id"}
    )


def test_infeasible_recording_reports_provenance_and_configuration(monkeypatch):
    import gap_imputation_benchmark.benchmark.corpus as corpus_module

    def fail(**_: object):
        raise GapSamplingError(2, 0, "infeasible")

    monkeypatch.setattr(corpus_module, "sample_stratified_random_gaps", fail)
    source = source_recordings()

    with pytest.raises(GapSamplingError, match="Recording provenance.*min_gap_duration_ms=20.0"):
        generate_corpus_benchmark(source, config())
