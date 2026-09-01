from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gap_imputation_benchmark.benchmark.gaps import GapCandidate, create_artificial_gap, find_gap_candidates
from gap_imputation_benchmark.benchmark.imputers import (
    SeasonalPeriodConfig,
    impute_forward_fill,
    impute_linear,
    impute_local_natural_cubic_spline,
    impute_nearest_boundary,
    impute_pchip,
    impute_polyfit_bic,
    impute_seasonal_periodic,
    impute_template,
)


def artificial_gap(gap_length: int, timestamps: list[float] | None = None):
    length = gap_length * 3 + 1
    recording = pd.DataFrame(
        {
            "gaze_x": np.arange(length, dtype=float) * 10.0,
            "is_valid": [True] * length,
            "sampling_rate_hz": [100.0] * length,
        }
    )
    if timestamps is not None:
        recording["timestamp_ms"] = timestamps
    candidate = find_gap_candidates(
        recording,
        gap_duration_ms=gap_length * 10,
    )[0]
    return create_artificial_gap(recording, candidate)


def calendar_artificial_gap():
    """Create a two-sample 2023 gap with two past and two future years."""
    timestamps = []
    values = []
    annual_values = {
        2021: [0.0, 10.0, 20.0, 30.0],
        2022: [0.0, 20.0, 30.0, 40.0],
        2023: [0.0, 100.0, 110.0, 0.0],
        2024: [0.0, 40.0, 50.0, 60.0],
        2025: [0.0, 50.0, 60.0, 70.0],
    }
    for year, annual_series in annual_values.items():
        for day, value in enumerate(annual_series, start=1):
            timestamps.append(pd.Timestamp(year, 1, day, 12))
            values.append(value)
    recording = pd.DataFrame(
        {
            "gaze_x": values,
            "is_valid": True,
            "timestamp": timestamps,
            "sampling_rate_hz": 1.0,
        }
    )
    start = 2 * 4 + 1
    candidate = GapCandidate(
        gap_start_idx=start,
        gap_end_idx=start + 2,
        requested_gap_duration_ms=2_000.0,
        realized_gap_duration_ms=2_000.0,
        sampling_rate_hz=1.0,
        gap_length_samples=2,
        left_context_start_idx=start - 1,
        right_context_end_idx=start + 3,
        left_context_valid_fraction=1.0,
        right_context_valid_fraction=1.0,
        context_valid_fraction=1.0,
    )
    return create_artificial_gap(recording, candidate)


def test_linear_imputation_interpolates_between_immediate_boundaries():
    result = impute_linear(artificial_gap(2))

    assert result.method_name == "linear"
    assert result.method_applicable
    np.testing.assert_allclose(result.predictions, [20.0, 30.0])


def test_forward_fill_has_known_output_for_an_even_gap():
    result = impute_forward_fill(artificial_gap(2))

    assert result.method_applicable
    np.testing.assert_allclose(result.predictions, [10.0, 10.0])


def test_nearest_boundary_prefers_left_at_an_odd_gap_midpoint():
    result = impute_nearest_boundary(artificial_gap(3))

    assert result.method_applicable
    np.testing.assert_allclose(result.predictions, [20.0, 20.0, 60.0])


def test_nearest_boundary_uses_valid_timestamps_over_sample_positions():
    result = impute_nearest_boundary(
        artificial_gap(2, [0.0, 10.0, 90.0, 95.0, 100.0, 110.0, 120.0])
    )

    assert result.metadata["distance_basis"] == "timestamp_ms"
    np.testing.assert_allclose(result.predictions, [40.0, 40.0])


@pytest.mark.parametrize(
    "imputer",
    [impute_linear, impute_forward_fill, impute_nearest_boundary],
)
@pytest.mark.parametrize(
    ("column", "value"),
    [("gaze_x", np.inf), ("is_valid", False)],
)
def test_imputers_report_invalid_boundaries_without_fallback(
    imputer,
    column,
    value,
):
    gap = artificial_gap(2)
    gap.masked_recording.loc[1, column] = value

    result = imputer(gap)

    assert not result.method_applicable
    assert result.failure_reason is not None
    assert result.predictions.size == 0


@pytest.mark.parametrize(
    "imputer",
    [impute_linear, impute_forward_fill, impute_nearest_boundary],
)
@pytest.mark.parametrize("gap_length", [2, 3])
def test_imputers_return_finite_prediction_per_gap_sample(imputer, gap_length):
    gap = artificial_gap(gap_length)

    result = imputer(gap)

    assert result.method_applicable
    assert len(result.predictions) == gap_length
    assert np.isfinite(result.predictions).all()


@pytest.mark.parametrize("imputer", [impute_pchip, impute_local_natural_cubic_spline])
def test_local_context_interpolators_are_applicable_with_full_context(imputer):
    result = imputer(artificial_gap(2))

    assert result.method_applicable
    assert result.metadata["left_support_point_count"] == 2
    assert result.metadata["right_support_point_count"] == 2


def test_only_cubic_spline_records_a_boundary_condition():
    assert "spline_boundary_condition" not in impute_pchip(artificial_gap(2)).metadata
    assert impute_local_natural_cubic_spline(artificial_gap(2)).metadata[
        "spline_boundary_condition"
    ] == "natural"


@pytest.mark.parametrize(
    "imputer",
    [impute_pchip, impute_local_natural_cubic_spline],
)
def test_local_context_interpolators_accept_incomplete_sufficient_context(imputer):
    gap = artificial_gap(2)
    gap.masked_recording.loc[0, "gaze_x"] = np.nan
    gap.masked_recording.loc[5, "is_valid"] = False

    result = imputer(gap)

    assert result.method_applicable
    assert result.metadata["left_support_point_count"] == 1
    assert result.metadata["right_support_point_count"] == 1


@pytest.mark.parametrize(
    "imputer",
    [impute_pchip, impute_local_natural_cubic_spline],
)
def test_local_context_interpolators_reject_duplicate_timestamps(imputer):
    gap = artificial_gap(2, [0.0, 1.0, 1.0, 3.0, 4.0, 5.0, 6.0])

    result = imputer(gap)

    assert not result.method_applicable
    assert "strictly increasing" in result.failure_reason


@pytest.mark.parametrize(
    "imputer",
    [impute_pchip, impute_local_natural_cubic_spline],
)
def test_local_context_interpolators_have_finite_gap_length_output(imputer):
    result = imputer(artificial_gap(3))

    assert result.method_applicable
    assert len(result.predictions) == 3
    assert np.isfinite(result.predictions).all()


@pytest.mark.parametrize(
    "imputer",
    [impute_pchip, impute_local_natural_cubic_spline],
)
def test_local_context_interpolators_do_not_use_points_outside_context(imputer):
    baseline = imputer(artificial_gap(2))
    gap = artificial_gap(2)
    gap.masked_recording.loc[6, "gaze_x"] = -1_000_000.0

    result = imputer(gap)

    assert result.method_applicable
    np.testing.assert_allclose(result.predictions, baseline.predictions)


def set_context_values(gap, value_function):
    candidate = gap.candidate
    context_indices = np.concatenate(
        [
            np.arange(candidate.left_context_start_idx, candidate.gap_start_idx),
            np.arange(candidate.gap_end_idx, candidate.right_context_end_idx),
        ]
    )
    gap.masked_recording.loc[context_indices, "gaze_x"] = value_function(
        context_indices
    )


def test_polyfit_bic_selects_linear_synthetic_context():
    gap = artificial_gap(2)
    set_context_values(gap, lambda time: 2.0 * time + 1.0)

    result = impute_polyfit_bic(gap)

    assert result.method_applicable
    assert result.metadata["selected_degree"] == 1
    assert set(result.metadata["bic_by_degree"]) == {1, 2, 3}
    np.testing.assert_allclose(result.predictions, [5.0, 7.0])


def test_polyfit_bic_selects_quadratic_synthetic_context():
    gap = artificial_gap(2)
    set_context_values(gap, lambda time: time.astype(float) ** 2)

    result = impute_polyfit_bic(gap)

    assert result.method_applicable
    assert result.metadata["selected_degree"] == 2
    np.testing.assert_allclose(result.predictions, [4.0, 9.0])


def test_polyfit_bic_rejects_insufficient_context_support():
    gap = artificial_gap(2)
    gap.masked_recording.loc[[0, 1], "is_valid"] = False

    result = impute_polyfit_bic(gap)

    assert not result.method_applicable
    assert "both sides" in result.failure_reason


def test_polyfit_bic_scales_large_timestamps_and_returns_finite_gap_output():
    timestamps = 1_000_000_000_000.0 + np.arange(7) * 10.0
    gap = artificial_gap(2, timestamps.tolist())
    set_context_values(
        gap,
        lambda time: 3.0 * (time.astype(float) - 2.0) + 5.0,
    )

    result = impute_polyfit_bic(gap)

    assert result.method_applicable
    assert result.metadata["time_center"] > 1_000_000_000_000.0
    assert result.metadata["time_scale"] > 0
    assert len(result.predictions) == 2
    assert np.isfinite(result.predictions).all()


def set_template_context_values(gap, left_values, right_values):
    candidate = gap.candidate
    gap.masked_recording.loc[
        np.arange(candidate.left_context_start_idx, candidate.gap_start_idx), "gaze_x"
    ] = left_values
    gap.masked_recording.loc[
        np.arange(candidate.gap_end_idx, candidate.right_context_end_idx), "gaze_x"
    ] = right_values


def test_template_reconstructs_a_repeated_synthetic_pattern():
    gap = artificial_gap(3)
    set_template_context_values(gap, [0.0, 2.0, 0.0], [0.0, 2.0, 0.0])

    result = impute_template(gap)

    assert result.method_applicable
    assert result.metadata["template_candidate_sides"] == ["left", "right"]
    np.testing.assert_allclose(result.predictions, [0.0, 2.0, 0.0])


def test_template_accepts_only_a_left_candidate():
    gap = artificial_gap(3)
    gap.masked_recording.loc[[7, 8], "is_valid"] = False

    result = impute_template(gap)

    assert result.method_applicable
    assert result.metadata["number_of_template_candidates"] == 1
    assert result.metadata["template_candidate_sides"] == ["left"]


def test_template_accepts_only_a_right_candidate():
    gap = artificial_gap(3)
    gap.masked_recording.loc[[0, 1], "is_valid"] = False

    result = impute_template(gap)

    assert result.method_applicable
    assert result.metadata["number_of_template_candidates"] == 1
    assert result.metadata["template_candidate_sides"] == ["right"]


def test_template_records_candidates_from_both_sides():
    result = impute_template(artificial_gap(3))

    assert result.method_applicable
    assert result.metadata["number_of_template_candidates"] == 2
    assert result.metadata["template_candidate_sides"] == ["left", "right"]


def test_template_reports_no_valid_candidate_without_linear_fallback():
    gap = artificial_gap(3)
    gap.masked_recording.loc[[0, 1, 7, 8], "is_valid"] = False

    result = impute_template(gap)

    assert not result.method_applicable
    assert "No fully observed finite template candidate" in result.failure_reason
    assert result.metadata["fallback_used"] is False


def test_template_rejects_invalid_or_incomplete_candidate_sequences():
    gap = artificial_gap(3)
    gap.masked_recording.loc[0, "gaze_x"] = np.nan
    gap.masked_recording.loc[7, "is_valid"] = False

    result = impute_template(gap)

    assert not result.method_applicable


def test_template_has_finite_gap_length_output_and_tapers_at_both_ends():
    gap = artificial_gap(3)
    set_template_context_values(gap, [0.0, 2.0, 0.0], [0.0, 2.0, 0.0])

    result = impute_template(gap)
    baseline = np.linspace(0.0, 0.0, 5)[1:-1]

    assert result.method_applicable
    assert len(result.predictions) == 3
    assert np.isfinite(result.predictions).all()
    assert result.predictions[0] == baseline[0]
    assert result.predictions[-1] == baseline[-1]


def test_template_never_uses_artificial_gap_ground_truth():
    gap = artificial_gap(3)
    expected = impute_template(gap).predictions
    gap.ground_truth[:] = 1_000_000.0

    result = impute_template(gap)

    assert result.method_applicable
    np.testing.assert_allclose(result.predictions, expected)


def test_template_never_searches_outside_predefined_context():
    baseline = impute_template(artificial_gap(3)).predictions
    gap = artificial_gap(3)
    gap.masked_recording.loc[9, "gaze_x"] = -1_000_000.0

    result = impute_template(gap)

    assert result.method_applicable
    np.testing.assert_allclose(result.predictions, baseline)


def test_seasonal_periodic_averages_calendar_matched_past_and_future_segments():
    result = impute_seasonal_periodic(
        calendar_artificial_gap(),
        SeasonalPeriodConfig(
            period_strategy="calendar_year",
            period_value=1,
            candidates_per_direction=2,
            min_candidates=3,
            max_offsets=2,
        ),
        timestamp_column="timestamp",
    )

    assert result.method_applicable
    np.testing.assert_allclose(result.predictions, [30.0, 40.0])
    assert result.metadata["candidate_count"] == 4
    assert result.metadata["past_candidate_count"] == 2
    assert result.metadata["future_candidate_count"] == 2
    assert result.metadata["candidate_directions"] == ["past", "future", "past", "future"]


def test_seasonal_periodic_can_be_restricted_to_past_references():
    result = impute_seasonal_periodic(
        calendar_artificial_gap(),
        SeasonalPeriodConfig(
            period_strategy="calendar_year",
            period_value=1,
            candidates_per_direction=2,
            min_candidates=2,
            max_offsets=2,
            mode="past_only",
        ),
        timestamp_column="timestamp",
    )

    assert result.method_applicable
    np.testing.assert_allclose(result.predictions, [15.0, 25.0])
    assert result.metadata["candidate_directions"] == ["past", "past"]


def test_seasonal_periodic_reports_not_applicable_without_enough_references():
    gap = calendar_artificial_gap()
    gap.masked_recording.loc[[5, 6, 13, 14], "is_valid"] = False

    result = impute_seasonal_periodic(
        gap,
        SeasonalPeriodConfig(
            period_strategy="calendar_year",
            period_value=1,
            candidates_per_direction=1,
            min_candidates=2,
            max_offsets=1,
        ),
        timestamp_column="timestamp",
    )

    assert not result.method_applicable
    assert result.metadata["candidate_count"] == 0
    assert "Fewer than min_candidates" in result.failure_reason
