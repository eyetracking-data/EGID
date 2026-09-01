from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gap_imputation_benchmark.benchmark.config import CorpusBenchmarkConfig
from gap_imputation_benchmark.benchmark.gaps import (
    GapSamplingError,
    create_artificial_gap,
    find_gap_candidates,
)
from gap_imputation_benchmark.benchmark.scaling import (
    estimate_scale_floor,
    estimate_scale_floor_from_split,
)


def recording(
    values: list[float],
    participant_id: str = "participant_1",
    recording_id: str = "recording_1",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": ["GazeBase"] * len(values),
            "participant_id": [participant_id] * len(values),
            "recording_id": [recording_id] * len(values),
            "session_id": ["S1"] * len(values),
            "source_file": [f"{recording_id}.csv"] * len(values),
            "gaze_x": values,
            "is_valid": [True] * len(values),
            "sampling_rate_hz": [100.0] * len(values),
        }
    )


def artificial_gap(values: list[float]) -> object:
    source = recording(values)
    candidate = find_gap_candidates(source, gap_duration_ms=20, context_multiplier=1.0)[0]
    return create_artificial_gap(source, candidate)


def test_default_estimator_returns_exact_first_percentile():
    gaps = [
        artificial_gap([0, 0, 1, 2, 2, 2, 3]),
        artificial_gap([0, 0, 1, 2, 4, 4, 4]),
    ]

    expected = float(np.quantile([2.0, 4.0], 0.01))

    assert estimate_scale_floor(gaps) == pytest.approx(expected)


def test_custom_quantile_is_used():
    gaps = [
        artificial_gap([0, 0, 1, 2, 2, 2, 3]),
        artificial_gap([0, 0, 1, 2, 4, 4, 4]),
    ]

    assert estimate_scale_floor(gaps, quantile=0.5) == pytest.approx(3.0)


def test_only_positive_finite_context_iqrs_are_used_and_zero_iqr_is_ignored():
    positive = artificial_gap([0, 0, 1, 2, 4, 4, 4])
    zero = artificial_gap([5, 5, 1, 2, 5, 5, 5])
    invalid = artificial_gap([0, 0, 1, 2, 4, 4, 4])
    invalid.masked_recording.loc[[0, 1, 4, 5], "is_valid"] = False

    assert estimate_scale_floor([zero, invalid, positive]) == pytest.approx(4.0)


def test_empty_input_and_absence_of_positive_iqr_are_rejected():
    with pytest.raises(ValueError, match="training_gaps"):
        estimate_scale_floor([])
    with pytest.raises(ValueError, match="No positive"):
        estimate_scale_floor([artificial_gap([5, 5, 1, 2, 5, 5, 5])])


@pytest.mark.parametrize("quantile", [0, 1, -0.01, 1.01])
def test_invalid_quantile_is_rejected(quantile: float):
    with pytest.raises(ValueError, match="quantile"):
        estimate_scale_floor([artificial_gap([0, 0, 1, 2, 4, 4, 4])], quantile)


def test_estimation_never_accesses_ground_truth_and_leaves_gap_unchanged():
    gap = artificial_gap([0, 0, 1, 2, 4, 4, 4])
    original_recording = gap.masked_recording.copy(deep=True)
    original_candidate = gap.candidate

    class GroundTruthForbiddenGap:
        candidate = gap.candidate
        masked_recording = gap.masked_recording

        @property
        def ground_truth(self) -> np.ndarray:
            raise AssertionError("ground_truth must not be read")

    assert estimate_scale_floor([GroundTruthForbiddenGap()]) == pytest.approx(4.0)
    assert gap.candidate == original_candidate
    assert gap.masked_recording.equals(original_recording)


def benchmark_config(**overrides: object) -> CorpusBenchmarkConfig:
    values: dict[str, object] = {
        "min_gap_duration_ms": 12.5,
        "max_gap_duration_ms": 37.5,
        "n_strata": 2,
        "n_gaps_per_recording": 2,
        "random_state": 77,
    }
    values.update(overrides)
    return CorpusBenchmarkConfig(**values)


def test_split_helper_uses_stable_per_recording_seeds_and_forwards_config(monkeypatch):
    source = [
        recording(list(range(20)), participant_id="one", recording_id="one"),
        recording(list(range(20)), participant_id="two", recording_id="two"),
    ]
    candidate = find_gap_candidates(source[0], gap_duration_ms=20)[0]
    calls: list[dict[str, object]] = []

    def sample(**kwargs: object) -> list[object]:
        calls.append(kwargs)
        return [candidate, candidate]

    monkeypatch.setattr(
        "gap_imputation_benchmark.benchmark.scaling.sample_stratified_random_gaps",
        sample,
    )
    result = estimate_scale_floor_from_split(source, benchmark_config())
    first_run_seeds = [call["random_state"] for call in calls]

    calls.clear()
    repeat = estimate_scale_floor_from_split(source, benchmark_config())

    assert result == repeat
    assert first_run_seeds == [call["random_state"] for call in calls]
    assert first_run_seeds[0] != first_run_seeds[1]
    assert all(call["min_gap_duration_ms"] == 12.5 for call in calls)
    assert all(call["max_gap_duration_ms"] == 37.5 for call in calls)
    assert all(call["n_strata"] == 2 for call in calls)
    assert all(call["n_gaps"] == 2 for call in calls)


def test_split_helper_reports_provenance_for_sampling_failure(monkeypatch):
    source = recording(list(range(20)), participant_id="participant_x", recording_id="recording_x")

    def fail(**_: object) -> list[object]:
        raise GapSamplingError(2, 0, "infeasible")

    monkeypatch.setattr(
        "gap_imputation_benchmark.benchmark.scaling.sample_stratified_random_gaps",
        fail,
    )

    with pytest.raises(GapSamplingError, match="participant_x.*recording_x"):
        estimate_scale_floor_from_split([source], benchmark_config())
