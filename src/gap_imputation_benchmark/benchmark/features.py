"""Leakage-safe observable features for predicting reconstruction quality.

Feature extraction uses only a masked gap's geometry, valid boundaries, and
predefined local contexts.  Raw numerators are retained so later folds can
apply their own feature-scale floor without consulting held-out targets.
"""

from __future__ import annotations

import numpy as np

from gap_imputation_benchmark.benchmark.gaps import ArtificialGap


# A versioned benchmark stores these raw quantities beside its normalised
# features.  This lets each evaluation fold apply its own feature-scale floor
# without reconstructing a numerator from a previous normalisation choice.
RAW_FEATURE_NUMERATOR_COLUMNS = {
    "normalized_boundary_jump": "raw_absolute_boundary_difference",
    "normalized_mean_difference_right_minus_left": "raw_mean_difference_right_minus_left",
    "local_std_over_scale": "raw_local_std",
    "local_range_over_scale": "raw_local_range",
    "normalized_trend_before": "raw_trend_before",
    "normalized_trend_after": "raw_trend_after",
    "normalized_trend_difference": "raw_trend_difference",
    "normalized_median_absolute_velocity_left": "raw_median_absolute_velocity_left",
    "normalized_median_absolute_velocity_right": "raw_median_absolute_velocity_right",
    "normalized_p90_absolute_velocity_left": "raw_p90_absolute_velocity_left",
    "normalized_p90_absolute_velocity_right": "raw_p90_absolute_velocity_right",
}


class BasicGapFeatures(dict[str, float]):
    """Primary gap features with separately retained diagnostic values."""

    def __init__(
        self,
        primary_features: dict[str, float],
        diagnostics: dict[str, float],
    ) -> None:
        super().__init__(primary_features)
        self.diagnostics = diagnostics


def _valid_finite_values(
    recording: object,
    indices: np.ndarray,
) -> np.ndarray:
    """Return values that are both marked valid and finite at given indices."""
    valid_indices, gaze_x = _valid_finite_indices(recording, indices)
    return gaze_x[valid_indices]


def _valid_finite_indices(
    recording: object,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid finite indices and the numeric gaze series."""
    is_valid = recording["is_valid"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    ).to_numpy(dtype=bool)
    try:
        gaze_x = recording["gaze_x"].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("gaze_x must contain numeric values.") from error

    valid_indices = indices[is_valid[indices] & np.isfinite(gaze_x[indices])]
    return valid_indices, gaze_x


def _finite_valid_boundary(
    recording: object,
    index: int,
    side: str,
) -> float:
    """Return one immediate valid finite boundary value."""
    is_valid = recording.iloc[index]["is_valid"]
    if not isinstance(is_valid, (bool, np.bool_)) or not is_valid:
        raise ValueError(f"{side} boundary must be valid and finite.")
    try:
        value = float(recording.iloc[index]["gaze_x"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{side} boundary must be valid and finite.") from error
    if not np.isfinite(value):
        raise ValueError(f"{side} boundary must be valid and finite.")
    return value


def _temporal_side_diagnostics(
    timestamps_ms: np.ndarray,
    gaze_x: np.ndarray,
    indices: np.ndarray,
    side: str,
) -> dict[str, float]:
    """Compute intercept OLS slope, R², and absolute velocity statistics."""
    if len(indices) < 2:
        raise ValueError(f"At least two valid finite {side} context points are required.")

    times = timestamps_ms[indices]
    values = gaze_x[indices]
    if not np.isfinite(times).all():
        raise ValueError(f"{side} context timestamps must be finite.")

    order = np.argsort(times, kind="stable")
    times_seconds = times[order] / 1000.0
    values = values[order]
    time_differences = np.diff(times_seconds)
    if not np.all(time_differences > 0):
        raise ValueError(
            f"{side} context timestamps must be unique and strictly increasing."
        )

    centered_times = times_seconds - np.mean(times_seconds)
    design = np.column_stack([np.ones(len(centered_times)), centered_times])
    intercept, slope = np.linalg.lstsq(design, values, rcond=None)[0]
    fitted = intercept + slope * centered_times
    residual_sum_squares = float(np.sum((values - fitted) ** 2))
    total_sum_squares = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = (
        1.0
        if total_sum_squares == 0 and np.isclose(residual_sum_squares, 0.0)
        else 0.0
        if total_sum_squares == 0
        else 1.0 - residual_sum_squares / total_sum_squares
    )
    absolute_velocities = np.abs(np.diff(values) / time_differences)
    return {
        "slope": float(slope),
        "r_squared": float(r_squared),
        "median_absolute_velocity": float(np.median(absolute_velocities)),
        "p90_absolute_velocity": float(np.percentile(absolute_velocities, 90)),
    }


def extract_basic_gap_features(
    artificial_gap: ArtificialGap,
    scale_floor: float,
) -> BasicGapFeatures:
    """Extract scale-normalized features from valid finite predefined context."""
    if scale_floor <= 0:
        raise ValueError("scale_floor must be positive.")

    recording = artificial_gap.masked_recording
    candidate = artificial_gap.candidate
    start = candidate.gap_start_idx
    end = candidate.gap_end_idx
    left_start = candidate.left_context_start_idx
    right_end = candidate.right_context_end_idx

    if (
        left_start < 0
        or left_start >= start
        or start < 1
        or start >= end
        or end >= right_end
        or right_end > len(recording)
    ):
        raise ValueError("Predefined context or boundary indices are invalid.")

    required_columns = {"gaze_x", "is_valid"}
    missing_columns = required_columns - set(recording.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if "timestamp_ms" not in recording.columns:
        raise ValueError("timestamp_ms is required for temporal gap features.")
    try:
        timestamps_ms = recording["timestamp_ms"].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("timestamp_ms must contain numeric values.") from error

    left_indices, gaze_x = _valid_finite_indices(
        recording,
        np.arange(left_start, start),
    )
    right_indices, _ = _valid_finite_indices(
        recording,
        np.arange(end, right_end),
    )
    left_values = gaze_x[left_indices]
    right_values = gaze_x[right_indices]
    if len(left_values) == 0 or len(right_values) == 0:
        raise ValueError("Valid finite context samples are required on both sides.")

    left_boundary = _finite_valid_boundary(recording, start - 1, "Left")
    right_boundary = _finite_valid_boundary(recording, end, "Right")
    local_values = np.concatenate([left_values, right_values])
    local_iqr = float(np.percentile(local_values, 75) - np.percentile(local_values, 25))
    local_scale = max(local_iqr, float(scale_floor))
    left_mean = float(np.mean(left_values))
    right_mean = float(np.mean(right_values))
    local_std = float(np.std(local_values))
    local_range = float(np.max(local_values) - np.min(local_values))
    boundary_difference = right_boundary - left_boundary
    left_temporal = _temporal_side_diagnostics(
        timestamps_ms,
        gaze_x,
        left_indices,
        "left",
    )
    right_temporal = _temporal_side_diagnostics(
        timestamps_ms,
        gaze_x,
        right_indices,
        "right",
    )

    primary_features = {
        "realized_gap_duration_ms": candidate.realized_gap_duration_ms,
        "left_context_valid_fraction": candidate.left_context_valid_fraction,
        "right_context_valid_fraction": candidate.right_context_valid_fraction,
        "normalized_boundary_jump": abs(boundary_difference) / local_scale,
        "normalized_mean_difference_right_minus_left": (
            right_mean - left_mean
        )
        / local_scale,
        "local_std_over_scale": local_std / local_scale,
        "local_range_over_scale": local_range / local_scale,
        "normalized_trend_before": left_temporal["slope"] / local_scale,
        "normalized_trend_after": right_temporal["slope"] / local_scale,
        "normalized_trend_difference": (
            right_temporal["slope"] - left_temporal["slope"]
        )
        / local_scale,
        "trend_before_r2": left_temporal["r_squared"],
        "trend_after_r2": right_temporal["r_squared"],
        "normalized_median_absolute_velocity_left": (
            left_temporal["median_absolute_velocity"] / local_scale
        ),
        "normalized_median_absolute_velocity_right": (
            right_temporal["median_absolute_velocity"] / local_scale
        ),
        "normalized_p90_absolute_velocity_left": (
            left_temporal["p90_absolute_velocity"] / local_scale
        ),
        "normalized_p90_absolute_velocity_right": (
            right_temporal["p90_absolute_velocity"] / local_scale
        ),
    }
    diagnostics = {
        "local_iqr": local_iqr,
        "local_std": local_std,
        "local_range": local_range,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "boundary_difference": boundary_difference,
        "absolute_boundary_difference": abs(boundary_difference),
        "mean_difference_right_minus_left": right_mean - left_mean,
        "n_valid_left": float(len(left_values)),
        "n_valid_right": float(len(right_values)),
        "gap_length_samples": float(candidate.gap_length_samples),
        "sampling_rate_hz": candidate.sampling_rate_hz,
        "trend_before": left_temporal["slope"],
        "trend_after": right_temporal["slope"],
        "median_absolute_velocity_left": left_temporal[
            "median_absolute_velocity"
        ],
        "median_absolute_velocity_right": right_temporal[
            "median_absolute_velocity"
        ],
        "p90_absolute_velocity_left": left_temporal["p90_absolute_velocity"],
        "p90_absolute_velocity_right": right_temporal["p90_absolute_velocity"],
    }
    return BasicGapFeatures(primary_features, diagnostics)


def selector_feature_values(
    artificial_gap: ArtificialGap,
    scale_floor: float,
) -> BasicGapFeatures:
    """Return model features plus explicit raw numerators for one observable gap.

    The input needs only the masked recording and gap geometry.  Ground-truth
    samples and imputation errors are deliberately never accessed.
    """
    features = extract_basic_gap_features(artificial_gap, scale_floor)
    diagnostics = features.diagnostics
    raw_values = {
        "raw_absolute_boundary_difference": diagnostics["absolute_boundary_difference"],
        "raw_mean_difference_right_minus_left": diagnostics["mean_difference_right_minus_left"],
        "raw_local_std": diagnostics["local_std"],
        "raw_local_range": diagnostics["local_range"],
        "raw_trend_before": diagnostics["trend_before"],
        "raw_trend_after": diagnostics["trend_after"],
        "raw_trend_difference": diagnostics["trend_after"] - diagnostics["trend_before"],
        "raw_median_absolute_velocity_left": diagnostics["median_absolute_velocity_left"],
        "raw_median_absolute_velocity_right": diagnostics["median_absolute_velocity_right"],
        "raw_p90_absolute_velocity_left": diagnostics["p90_absolute_velocity_left"],
        "raw_p90_absolute_velocity_right": diagnostics["p90_absolute_velocity_right"],
    }
    return BasicGapFeatures({**features, **raw_values}, dict(diagnostics))
