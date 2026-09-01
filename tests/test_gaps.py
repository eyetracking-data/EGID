from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gap_imputation_benchmark.benchmark.gaps import (
    GapCandidate,
    GapSamplingError,
    create_artificial_gap,
    find_gap_candidates,
    sample_non_overlapping_candidates,
    sample_random_gaps,
    sample_stratified_random_gaps,
)


def recording(length: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gaze_x": np.arange(length, dtype=float),
            "is_valid": [True] * length,
            "sampling_rate_hz": [100.0] * length,
        }
    )


def test_context_threshold_is_enforced_independently_per_side():
    data = recording()
    data.loc[0, "is_valid"] = False  # Left context: 75% valid.

    candidates = find_gap_candidates(
        data,
        gap_duration_ms=10,
        context_multiplier=4,
        min_context_valid_fraction=0.80,
    )

    assert not any(candidate.gap_start_idx == 4 for candidate in candidates)

    data = recording()
    data.loc[6, "is_valid"] = False  # Right context: 75% valid.
    candidates = find_gap_candidates(
        data,
        gap_duration_ms=10,
        context_multiplier=4,
        min_context_valid_fraction=0.80,
    )

    assert not any(candidate.gap_start_idx == 2 for candidate in candidates)


def test_only_true_is_valid_and_finite_gaze_values_are_valid():
    data = recording(5)
    data.loc[2, "gaze_x"] = np.inf

    candidates = find_gap_candidates(data, gap_duration_ms=10)

    assert candidates == []

    data = recording(5)
    data.loc[2, "is_valid"] = False
    assert find_gap_candidates(data, gap_duration_ms=10) == []


def test_candidate_records_requested_and_realized_duration():
    candidates = find_gap_candidates(recording(), gap_duration_ms=15)

    candidate = candidates[0]
    assert candidate.requested_gap_duration_ms == 15
    assert candidate.sampling_rate_hz == 100
    assert candidate.gap_length_samples == 2
    assert candidate.realized_gap_duration_ms == 20


def test_one_sample_gap_has_at_least_two_context_samples_per_side():
    candidate = find_gap_candidates(recording(), gap_duration_ms=10)[0]

    assert candidate.gap_length_samples == 1
    assert candidate.gap_start_idx - candidate.left_context_start_idx == 2
    assert candidate.right_context_end_idx - candidate.gap_end_idx == 2


def test_create_artificial_gap_masks_values_and_preserves_ground_truth():
    data = recording()
    candidate = find_gap_candidates(data, gap_duration_ms=20)[0]

    artificial_gap = create_artificial_gap(data, candidate)
    start, end = candidate.gap_start_idx, candidate.gap_end_idx

    np.testing.assert_array_equal(
        artificial_gap.ground_truth,
        data.iloc[start:end]["gaze_x"].to_numpy(),
    )
    assert artificial_gap.masked_recording.loc[start:end - 1, "gaze_x"].isna().all()
    assert artificial_gap.masked_recording.loc[start:end - 1, "original_is_valid"].eq(True).all()
    assert artificial_gap.masked_recording.loc[start:end - 1, "is_valid"].eq(False).all()
    assert artificial_gap.masked_recording.loc[start:end - 1, "is_artificial_gap"].eq(True).all()
    assert data.loc[start:end - 1, "gaze_x"].notna().all()


@pytest.mark.parametrize(
    ("candidate", "invalid_row", "invalid_column"),
    [
        (
            GapCandidate(0, 2, 20, 20, 100, 2, 0, 4, 1, 1, 1),
            None,
            None,
        ),
        (
            GapCandidate(1, 3, 20, 20, 100, 3, 0, 4, 1, 1, 1),
            None,
            None,
        ),
        (
            GapCandidate(1, 3, 20, 20, 100, 2, 0, 4, 1, 1, 1),
            1,
            "gaze_x",
        ),
        (
            GapCandidate(1, 3, 20, 20, 100, 2, 0, 4, 1, 1, 1),
            0,
            "is_valid",
        ),
    ],
)
def test_create_artificial_gap_rejects_invalid_manual_candidates(
    candidate: GapCandidate,
    invalid_row: int | None,
    invalid_column: str | None,
):
    data = recording()
    if invalid_row is not None:
        data.loc[invalid_row, invalid_column] = (
            np.inf if invalid_column == "gaze_x" else False
        )

    with pytest.raises(ValueError):
        create_artificial_gap(data, candidate)


def test_random_gap_sampling_is_reproducible():
    data = recording(100)

    first = sample_random_gaps(data, 4, 10, 30, random_state=123)
    second = sample_random_gaps(data, 4, 10, 30, random_state=123)

    assert first == second


def test_sampled_full_windows_do_not_overlap():
    selected = sample_random_gaps(recording(100), 6, 10, 30, random_state=4)

    for index, candidate in enumerate(selected):
        for other in selected[index + 1 :]:
            assert (
                candidate.right_context_end_idx <= other.left_context_start_idx
                or other.right_context_end_idx <= candidate.left_context_start_idx
            )


@pytest.mark.parametrize(
    "sampler",
    [
        lambda data: sample_non_overlapping_candidates(
            find_gap_candidates(data, 10), 2, random_state=1
        ),
        lambda data: sample_random_gaps(data, 2, 10, 10, random_state=1),
        lambda data: sample_stratified_random_gaps(
            data, 2, 10, 20, random_state=1
        ),
    ],
)
def test_samplers_raise_an_explicit_error_for_shortfalls(sampler):
    with pytest.raises(GapSamplingError, match="requested 2, selected"):
        sampler(recording(5))


def test_stratified_sampling_handles_non_divisible_gap_counts():
    selected = sample_stratified_random_gaps(
        recording(1_000),
        n_gaps=6,
        min_gap_duration_ms=10,
        max_gap_duration_ms=50,
        random_state=9,
    )

    assert len(selected) == 6
    assert sum(candidate.gap_length_samples for candidate in selected) > 0
    assert {candidate.duration_stratum for candidate in selected} <= {1, 2, 3, 4, 5}


def test_stratified_sampling_allocates_gaps_approximately_equally():
    selected = sample_stratified_random_gaps(
        recording(1_000),
        n_gaps=9,
        min_gap_duration_ms=10,
        max_gap_duration_ms=50,
        n_strata=4,
        random_state=9,
    )
    bin_edges = np.linspace(10, 50, 5)
    counts = [0, 0, 0, 0]

    for candidate in selected:
        stratum = min(
            np.searchsorted(
                bin_edges,
                candidate.requested_gap_duration_ms,
                side="right",
            )
            - 1,
            3,
        )
        counts[stratum] += 1

    assert sum(counts) == 9
    assert max(counts) - min(counts) <= 1


def test_stratified_sampler_reuses_candidates_for_identical_sample_lengths(monkeypatch):
    import gap_imputation_benchmark.benchmark.gaps as gaps_module

    original = gaps_module.find_gap_candidates
    calls = 0

    def count_calls(*args: object, **kwargs: object) -> list[GapCandidate]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gaps_module, "find_gap_candidates", count_calls)
    sample_stratified_random_gaps(
        recording(1_000),
        n_gaps=6,
        min_gap_duration_ms=10,
        max_gap_duration_ms=19,
        n_strata=1,
        random_state=13,
    )

    # At 100 Hz this entire range maps to one or two discrete sample lengths.
    assert calls <= 2
