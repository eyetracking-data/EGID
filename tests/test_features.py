import numpy as np
import pandas as pd
import pytest

from gap_imputation_benchmark.benchmark.features import extract_basic_gap_features
from gap_imputation_benchmark.benchmark.gaps import create_artificial_gap, find_gap_candidates


def artificial_gap(gap_length: int = 2) -> object:
    length = gap_length * 3 + 1
    recording = pd.DataFrame(
        {
            "gaze_x": np.arange(length, dtype=float) * 2.0,
            "is_valid": [True] * length,
            "sampling_rate_hz": [100.0] * length,
            "timestamp_ms": np.arange(length, dtype=float) * 1_000.0,
        }
    )
    candidate = find_gap_candidates(recording, gap_duration_ms=gap_length * 10)[0]
    return create_artificial_gap(recording, candidate)


def test_basic_gap_features_have_known_synthetic_values():
    features = extract_basic_gap_features(artificial_gap(), scale_floor=1.0)

    assert set(features) == {
        "realized_gap_duration_ms",
        "left_context_valid_fraction",
        "right_context_valid_fraction",
        "normalized_boundary_jump",
        "normalized_mean_difference_right_minus_left",
        "local_std_over_scale",
        "local_range_over_scale",
        "normalized_trend_before",
        "normalized_trend_after",
        "normalized_trend_difference",
        "trend_before_r2",
        "trend_after_r2",
        "normalized_median_absolute_velocity_left",
        "normalized_median_absolute_velocity_right",
        "normalized_p90_absolute_velocity_left",
        "normalized_p90_absolute_velocity_right",
    }
    assert features["realized_gap_duration_ms"] == 20.0
    assert features["normalized_boundary_jump"] == pytest.approx(6.0 / 7.0)
    assert features["normalized_mean_difference_right_minus_left"] == pytest.approx(
        8.0 / 7.0
    )
    assert features["local_std_over_scale"] == pytest.approx(np.std([0, 2, 8, 10]) / 7)
    assert features["local_range_over_scale"] == pytest.approx(10.0 / 7.0)
    assert features.diagnostics["n_valid_left"] == 2.0
    assert features.diagnostics["n_valid_right"] == 2.0


def test_basic_features_never_use_ground_truth():
    gap = artificial_gap()
    expected = extract_basic_gap_features(gap, scale_floor=1.0)
    gap.ground_truth[:] = -1_000_000.0

    assert extract_basic_gap_features(gap, scale_floor=1.0) == expected


def test_basic_features_allow_incomplete_valid_context():
    gap = artificial_gap(3)
    gap.masked_recording.loc[0, "is_valid"] = False
    gap.masked_recording.loc[8, "gaze_x"] = np.nan

    features = extract_basic_gap_features(gap, scale_floor=1.0)

    assert features.diagnostics["n_valid_left"] == 2.0
    assert features.diagnostics["n_valid_right"] == 2.0
    assert np.isfinite(list(features.values())).all()


def test_basic_features_use_scale_floor_for_zero_iqr():
    gap = artificial_gap()
    gap.masked_recording.loc[[0, 1, 4, 5], "gaze_x"] = 5.0

    features = extract_basic_gap_features(gap, scale_floor=2.0)

    assert features.diagnostics["local_iqr"] == 0.0
    assert features["normalized_boundary_jump"] == 0.0
    assert features["local_std_over_scale"] == 0.0
    assert features["local_range_over_scale"] == 0.0


def test_basic_features_ignore_values_outside_predefined_context():
    baseline = extract_basic_gap_features(artificial_gap(), scale_floor=1.0)
    gap = artificial_gap()
    gap.masked_recording.loc[6, "gaze_x"] = 1_000_000.0

    assert extract_basic_gap_features(gap, scale_floor=1.0) == baseline


def test_basic_features_reject_non_positive_scale_floor():
    with pytest.raises(ValueError, match="scale_floor"):
        extract_basic_gap_features(artificial_gap(), scale_floor=0.0)


def test_temporal_features_recover_exact_linear_trends():
    features = extract_basic_gap_features(artificial_gap(), scale_floor=1.0)

    assert features["normalized_trend_before"] == pytest.approx(2.0 / 7.0)
    assert features["normalized_trend_after"] == pytest.approx(2.0 / 7.0)
    assert features["normalized_trend_difference"] == pytest.approx(0.0)
    assert features["trend_before_r2"] == pytest.approx(1.0)
    assert features["trend_after_r2"] == pytest.approx(1.0)
    assert features.diagnostics["trend_before"] == pytest.approx(2.0)


def test_temporal_features_capture_zero_trend():
    gap = artificial_gap()
    gap.masked_recording.loc[[0, 1, 4, 5], "gaze_x"] = 5.0

    features = extract_basic_gap_features(gap, scale_floor=2.0)

    assert features["normalized_trend_before"] == pytest.approx(0.0)
    assert features["normalized_trend_after"] == pytest.approx(0.0)
    assert features["trend_before_r2"] == pytest.approx(1.0)


def test_temporal_features_compute_known_r_squared():
    recording = pd.DataFrame(
        {
            "gaze_x": [0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 8.0, 10.0, 12.0, 14.0],
            "is_valid": [True] * 10,
            "sampling_rate_hz": [100.0] * 10,
            "timestamp_ms": np.arange(10, dtype=float) * 1_000.0,
        }
    )
    candidate = find_gap_candidates(recording, gap_duration_ms=30)[0]
    gap = create_artificial_gap(recording, candidate)

    features = extract_basic_gap_features(gap, scale_floor=1.0)

    assert features["trend_before_r2"] == pytest.approx(0.75)


def test_temporal_features_sort_irregular_timestamps_before_fitting():
    gap = artificial_gap()
    gap.masked_recording.loc[:, "timestamp_ms"] = [2_000, 0, 3_000, 4_000, 5_000, 4_000, 6_000]

    features = extract_basic_gap_features(gap, scale_floor=1.0)

    assert features.diagnostics["trend_before"] == pytest.approx(-1.0)
    assert features.diagnostics["trend_after"] == pytest.approx(-2.0)


def test_temporal_features_reject_duplicate_timestamps():
    gap = artificial_gap()
    gap.masked_recording.loc[[0, 1], "timestamp_ms"] = [0.0, 0.0]

    with pytest.raises(ValueError, match="strictly increasing"):
        extract_basic_gap_features(gap, scale_floor=1.0)


def test_temporal_features_compute_known_velocity_quantiles():
    recording = pd.DataFrame(
        {
            "gaze_x": [0.0, 2.0, 6.0, 7.0, 8.0, 9.0, 10.0, 14.0, 15.0, 16.0],
            "is_valid": [True] * 10,
            "sampling_rate_hz": [100.0] * 10,
            "timestamp_ms": [0, 1_000, 3_000, 4_000, 5_000, 5_500, 6_000, 7_000, 9_000, 10_000],
        }
    )
    candidate = find_gap_candidates(recording, gap_duration_ms=30)[0]
    gap = create_artificial_gap(recording, candidate)

    features = extract_basic_gap_features(gap, scale_floor=1.0)

    assert features.diagnostics["median_absolute_velocity_left"] == pytest.approx(2.0)
    assert features.diagnostics["p90_absolute_velocity_right"] == pytest.approx(3.65)
    assert features["normalized_median_absolute_velocity_left"] == pytest.approx(0.2)


def test_temporal_features_reject_insufficient_side_support():
    gap = artificial_gap()
    gap.masked_recording.loc[0, "is_valid"] = False

    with pytest.raises(ValueError, match="At least two"):
        extract_basic_gap_features(gap, scale_floor=1.0)
