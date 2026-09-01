"""Candidate gap-reconstruction methods with explicit applicability outcomes.

Each method operates only on the supplied masked recording and predefined
context windows.  It returns a structured result instead of silently falling
back when the local evidence is insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

from gap_imputation_benchmark.benchmark.gaps import ArtificialGap


@dataclass(frozen=True)
class ImputationResult:
    """Outcome of applying one imputation method to an artificial gap."""

    method_name: str
    predictions: np.ndarray
    method_applicable: bool
    failure_reason: str | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class SeasonalPeriodConfig:
    """Configuration for the reusable periodic-reference imputer.

    ``calendar_year`` shifts actual calendar dates and is appropriate for
    annual weather cycles. ``fixed_timedelta`` shifts timestamps by a fixed
    duration (for example, seven days for traffic). ``sample_offset`` shifts
    regular sample positions and does not require timestamps.
    """

    period_strategy: Literal[
        "calendar_year", "fixed_timedelta", "sample_offset"
    ]
    period_value: int | str | pd.Timedelta
    candidates_per_direction: int = 3
    min_candidates: int = 3
    max_offsets: int = 8
    aggregation: Literal["mean", "median"] = "mean"
    leap_day_policy: Literal["skip", "feb28", "mar1"] = "skip"
    mode: Literal["bidirectional", "past_only"] = "bidirectional"


def _not_applicable(
    method_name: str,
    failure_reason: str,
    metadata: dict[str, object] | None = None,
) -> ImputationResult:
    return ImputationResult(
        method_name=method_name,
        predictions=np.array([], dtype=float),
        method_applicable=False,
        failure_reason=failure_reason,
        metadata=metadata or {},
    )


def _validated_boundaries(
    artificial_gap: ArtificialGap,
    method_name: str,
) -> tuple[int, int, float, float] | ImputationResult:
    """Validate gap indices and immediate observed boundaries only."""
    recording = artificial_gap.masked_recording
    candidate = artificial_gap.candidate
    start = candidate.gap_start_idx
    end = candidate.gap_end_idx

    if start < 1 or end >= len(recording) or start >= end:
        return _not_applicable(method_name, "Gap boundaries are unavailable.")

    required_columns = {"gaze_x", "is_valid"}
    missing_columns = required_columns - set(recording.columns)
    if missing_columns:
        return _not_applicable(
            method_name,
            f"Missing required columns: {sorted(missing_columns)}",
        )

    boundary_values: list[float] = []
    for label, index in (("Left", start - 1), ("Right", end)):
        is_valid = recording.iloc[index]["is_valid"]
        if not isinstance(is_valid, (bool, np.bool_)) or not is_valid:
            return _not_applicable(
                method_name,
                f"{label} boundary is not valid.",
            )

        try:
            value = float(recording.iloc[index]["gaze_x"])
        except (TypeError, ValueError):
            return _not_applicable(
                method_name,
                f"{label} boundary is not finite.",
            )

        if not np.isfinite(value):
            return _not_applicable(
                method_name,
                f"{label} boundary is not finite.",
            )
        boundary_values.append(value)

    return start, end, boundary_values[0], boundary_values[1]


def _successful_result(
    method_name: str,
    predictions: np.ndarray,
    metadata: dict[str, object] | None = None,
) -> ImputationResult:
    return ImputationResult(
        method_name=method_name,
        predictions=predictions,
        method_applicable=True,
        failure_reason=None,
        metadata=metadata or {},
    )


def impute_linear(
    artificial_gap: ArtificialGap,
) -> ImputationResult:
    """
    Linearly interpolate the artificial gap using the immediate
    observed values before and after the gap.

    Returns an explicit imputation outcome with one prediction per missing
    sample when both immediate boundaries are valid and finite.
    """
    method_name = "linear"
    boundaries = _validated_boundaries(artificial_gap, method_name)
    if isinstance(boundaries, ImputationResult):
        return boundaries

    start, end, left_value, right_value = boundaries

    gap_length = end - start

    # Include both boundary points, then remove them.
    predictions = np.linspace(
        left_value,
        right_value,
        gap_length + 2,
    )[1:-1]

    return _successful_result(
        method_name,
        predictions,
        {"left_boundary_value": left_value, "right_boundary_value": right_value},
    )


def impute_forward_fill(
    artificial_gap: ArtificialGap,
) -> ImputationResult:
    """Fill every gap sample with the immediate left boundary value."""
    method_name = "forward_fill"
    boundaries = _validated_boundaries(artificial_gap, method_name)
    if isinstance(boundaries, ImputationResult):
        return boundaries

    start, end, left_value, right_value = boundaries
    predictions = np.full(end - start, left_value, dtype=float)
    return _successful_result(
        method_name,
        predictions,
        {"left_boundary_value": left_value, "right_boundary_value": right_value},
    )


def impute_nearest_boundary(
    artificial_gap: ArtificialGap,
) -> ImputationResult:
    """Fill from the nearest immediate boundary, preferring left on ties."""
    method_name = "nearest_boundary"
    boundaries = _validated_boundaries(artificial_gap, method_name)
    if isinstance(boundaries, ImputationResult):
        return boundaries

    start, end, left_value, right_value = boundaries
    recording = artificial_gap.masked_recording
    positions = np.arange(start, end)
    left_position = start - 1
    right_position = end
    distance_basis = "sample_position"

    if "timestamp_ms" in recording.columns:
        try:
            timestamps = recording["timestamp_ms"].to_numpy(dtype=float)
        except (TypeError, ValueError):
            timestamps = np.array([], dtype=float)

        timestamp_window = timestamps[left_position : right_position + 1]
        if (
            len(timestamp_window) == end - start + 2
            and np.isfinite(timestamp_window).all()
            and np.all(np.diff(timestamp_window) > 0)
        ):
            positions = timestamps[start:end]
            left_position = timestamps[start - 1]
            right_position = timestamps[end]
            distance_basis = "timestamp_ms"

    left_distances = np.abs(positions - left_position)
    right_distances = np.abs(right_position - positions)
    predictions = np.where(
        left_distances <= right_distances,
        left_value,
        right_value,
    ).astype(float)

    return _successful_result(
        method_name,
        predictions,
        {
            "left_boundary_value": left_value,
            "right_boundary_value": right_value,
            "distance_basis": distance_basis,
        },
    )


def _context_support_points(
    artificial_gap: ArtificialGap,
    method_name: str,
    spline_boundary_condition: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]] | ImputationResult:
    """Return valid finite support points from the predefined contexts only."""
    recording = artificial_gap.masked_recording
    candidate = artificial_gap.candidate
    start = candidate.gap_start_idx
    end = candidate.gap_end_idx
    left_start = candidate.left_context_start_idx
    right_end = candidate.right_context_end_idx

    if (
        left_start < 0
        or left_start >= start
        or start >= end
        or end >= right_end
        or right_end > len(recording)
    ):
        return _not_applicable(
            method_name,
            "Predefined context indices are unavailable.",
        )

    required_columns = {"gaze_x", "is_valid"}
    missing_columns = required_columns - set(recording.columns)
    if missing_columns:
        return _not_applicable(
            method_name,
            f"Missing required columns: {sorted(missing_columns)}",
        )

    try:
        gaze_x = recording["gaze_x"].to_numpy(dtype=float)
    except (TypeError, ValueError):
        return _not_applicable(method_name, "Context gaze values are not numeric.")

    is_valid = recording["is_valid"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    ).to_numpy(dtype=bool)

    coordinates = np.arange(len(recording), dtype=float)
    coordinate_basis = "sample_position"
    if "timestamp_ms" in recording.columns:
        try:
            coordinates = recording["timestamp_ms"].to_numpy(dtype=float)
        except (TypeError, ValueError):
            return _not_applicable(method_name, "Timestamps are not numeric.")

        local_coordinates = coordinates[left_start:right_end]
        if (
            len(local_coordinates) != right_end - left_start
            or not np.isfinite(local_coordinates).all()
            or not np.all(np.diff(local_coordinates) > 0)
        ):
            return _not_applicable(
                method_name,
                "Timestamps must be finite and strictly increasing.",
            )
        coordinate_basis = "timestamp_ms"

    left_indices = np.arange(left_start, start)
    right_indices = np.arange(end, right_end)
    left_mask = is_valid[left_indices] & np.isfinite(gaze_x[left_indices])
    right_mask = is_valid[right_indices] & np.isfinite(gaze_x[right_indices])
    left_support_indices = left_indices[left_mask]
    right_support_indices = right_indices[right_mask]
    metadata = {
        "left_support_point_count": len(left_support_indices),
        "right_support_point_count": len(right_support_indices),
        "support_point_count": len(left_support_indices) + len(right_support_indices),
        "coordinate_basis": coordinate_basis,
    }
    if spline_boundary_condition is not None:
        metadata["spline_boundary_condition"] = spline_boundary_condition

    if len(left_support_indices) == 0 or len(right_support_indices) == 0:
        return _not_applicable(
            method_name,
            "Interpolation requires valid finite support on both sides of the gap.",
            metadata,
        )

    support_indices = np.concatenate([left_support_indices, right_support_indices])
    return (
        coordinates[support_indices],
        gaze_x[support_indices],
        coordinates[start:end],
        metadata,
    )


def _interpolation_result(
    method_name: str,
    artificial_gap: ArtificialGap,
    interpolator_factory: object,
    spline_boundary_condition: str | None,
) -> ImputationResult:
    """Interpolate only from valid finite points in the predefined contexts."""
    support = _context_support_points(
        artificial_gap,
        method_name,
        spline_boundary_condition,
    )
    if isinstance(support, ImputationResult):
        return support

    support_x, support_y, gap_x, metadata = support
    try:
        predictions = np.asarray(
            interpolator_factory(support_x, support_y)(gap_x),
            dtype=float,
        )
    except ValueError as error:
        return _not_applicable(method_name, str(error), metadata)

    if len(predictions) != len(gap_x) or not np.isfinite(predictions).all():
        return _not_applicable(
            method_name,
            "Interpolation did not produce finite predictions for every gap sample.",
            metadata,
        )

    return _successful_result(method_name, predictions, metadata)


def impute_pchip(
    artificial_gap: ArtificialGap,
) -> ImputationResult:
    """Interpolate with PCHIP using only valid finite local-context support."""
    return _interpolation_result(
        "pchip",
        artificial_gap,
        PchipInterpolator,
        None,
    )


def impute_local_natural_cubic_spline(
    artificial_gap: ArtificialGap,
) -> ImputationResult:
    """Interpolate with a natural cubic spline using local-context support."""
    return _interpolation_result(
        "local_natural_cubic_spline",
        artificial_gap,
        lambda x, y: CubicSpline(x, y, bc_type="natural", extrapolate=False),
        "natural",
    )


def impute_polyfit_bic(
    artificial_gap: ArtificialGap,
) -> ImputationResult:
    """Select a degree-1, 2, or 3 local polynomial by the smallest BIC."""
    method_name = "polyfit_bic"
    support = _context_support_points(
        artificial_gap,
        method_name,
        None,
    )
    if isinstance(support, ImputationResult):
        return support

    support_x, support_y, gap_x, context_metadata = support
    time_center = float(np.mean(support_x))
    time_scale = float(np.max(np.abs(support_x - time_center)))
    metadata = {
        **context_metadata,
        "time_center": time_center,
        "time_scale": time_scale,
        "bic_by_degree": {},
        "rss_by_degree": {},
    }

    if not np.isfinite(time_scale) or time_scale <= 0:
        return _not_applicable(
            method_name,
            "Support timestamps cannot be centered and scaled.",
            metadata,
        )

    scaled_support_x = (support_x - time_center) / time_scale
    scaled_gap_x = (gap_x - time_center) / time_scale
    epsilon = np.finfo(float).eps
    candidates: list[tuple[float, int, np.ndarray]] = []

    for degree in (1, 2, 3):
        if len(support_x) < degree + 1:
            continue

        try:
            coefficients = np.polyfit(scaled_support_x, support_y, degree)
        except (np.linalg.LinAlgError, ValueError) as error:
            return _not_applicable(method_name, str(error), metadata)

        fitted_support = np.polyval(coefficients, scaled_support_x)
        rss = float(np.sum((support_y - fitted_support) ** 2))
        bic = float(
            len(support_x) * np.log(max(rss / len(support_x), epsilon))
            + (degree + 1) * np.log(len(support_x))
        )
        metadata["rss_by_degree"][degree] = rss
        metadata["bic_by_degree"][degree] = bic
        candidates.append((bic, degree, coefficients))

    if not candidates:
        return _not_applicable(
            method_name,
            "At least two valid finite support points are required.",
            metadata,
        )

    selected_bic, selected_degree, selected_coefficients = min(
        candidates,
        key=lambda candidate: candidate[0],
    )
    predictions = np.asarray(
        np.polyval(selected_coefficients, scaled_gap_x),
        dtype=float,
    )

    if len(predictions) != len(gap_x) or not np.isfinite(predictions).all():
        return _not_applicable(
            method_name,
            "Polynomial fitting did not produce finite predictions for every gap sample.",
            metadata,
        )

    metadata["selected_degree"] = selected_degree
    metadata["selected_bic"] = selected_bic
    return _successful_result(method_name, predictions, metadata)


def _template_candidate(
    gaze_x: np.ndarray,
    valid: np.ndarray,
    candidate_starts: range,
    gap_length: int,
) -> tuple[int, np.ndarray] | None:
    """Return the first fully observed finite candidate in notebook scan order."""
    for candidate_start in candidate_starts:
        candidate_end = candidate_start + gap_length
        values = gaze_x[candidate_start:candidate_end]
        if len(values) == gap_length and valid[candidate_start:candidate_end].all():
            return candidate_start, values.copy()
    return None


def impute_template(
    artificial_gap: ArtificialGap,
    template_noise_strength: float = 1.0,
) -> ImputationResult:
    """Reconstruct a gap with tapered local template movement."""
    method_name = "template"
    if template_noise_strength < 0:
        return _not_applicable(
            method_name,
            "template_noise_strength must be non-negative.",
        )
    boundaries = _validated_boundaries(artificial_gap, method_name)
    if isinstance(boundaries, ImputationResult):
        return boundaries

    start, end, left_value, right_value = boundaries
    candidate = artificial_gap.candidate
    left_start = candidate.left_context_start_idx
    right_end = candidate.right_context_end_idx
    recording = artificial_gap.masked_recording
    gap_length = end - start
    base_metadata = {
        "number_of_template_candidates": 0,
        "template_candidate_positions": [],
        "template_candidate_sides": [],
        "template_aggregation_method": "positionwise_mean",
        "envelope_type": "sin_squared",
        "template_noise_strength": template_noise_strength,
        "fallback_used": False,
    }

    if (
        left_start < 0
        or left_start >= start
        or end >= right_end
        or right_end > len(recording)
    ):
        return _not_applicable(
            method_name,
            "Predefined context indices are unavailable.",
            base_metadata,
        )

    try:
        gaze_x = recording["gaze_x"].to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError):
        return _not_applicable(method_name, "Context gaze values are not numeric.", base_metadata)

    current_valid = recording["is_valid"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    ).to_numpy(dtype=bool)
    original_valid = current_valid
    if "original_is_valid" in recording.columns:
        original_valid = recording["original_is_valid"].map(
            lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
        ).to_numpy(dtype=bool)
    valid = current_valid & original_valid & np.isfinite(gaze_x)

    left_candidate = _template_candidate(
        gaze_x,
        valid,
        range(start - gap_length, left_start - 1, -1),
        gap_length,
    )
    right_candidate = _template_candidate(
        gaze_x,
        valid,
        range(end, right_end - gap_length + 1),
        gap_length,
    )

    candidates: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []
    sides: list[str] = []
    for side, selected_candidate in (
        ("left", left_candidate),
        ("right", right_candidate),
    ):
        if selected_candidate is None:
            continue
        candidate_start, values = selected_candidate
        candidates.append(values)
        positions.append((candidate_start, candidate_start + gap_length))
        sides.append(side)

    base_metadata.update(
        {
            "number_of_template_candidates": len(candidates),
            "template_candidate_positions": positions,
            "template_candidate_sides": sides,
        }
    )
    if not candidates:
        return _not_applicable(
            method_name,
            "No fully observed finite template candidate exists in the predefined contexts.",
            base_metadata,
        )

    baseline = np.linspace(left_value, right_value, gap_length + 2)[1:-1]
    template = np.mean(candidates, axis=0)
    template_trend = np.linspace(template[0], template[-1], gap_length)
    movement = template_noise_strength * (template - template_trend)
    envelope = np.sin(np.pi * np.linspace(0.0, 1.0, gap_length)) ** 2
    predictions = baseline + envelope * movement

    if len(predictions) != gap_length or not np.isfinite(predictions).all():
        return _not_applicable(
            method_name,
            "Template reconstruction did not produce finite predictions for every gap sample.",
            base_metadata,
        )

    return _successful_result(method_name, predictions, base_metadata)


def _validate_seasonal_config(config: SeasonalPeriodConfig) -> str | None:
    """Return a validation message for an invalid seasonal configuration."""
    if config.candidates_per_direction <= 0:
        return "candidates_per_direction must be positive."
    if config.min_candidates <= 0:
        return "min_candidates must be positive."
    if config.min_candidates > (
        config.candidates_per_direction
        * (2 if config.mode == "bidirectional" else 1)
    ):
        return "min_candidates exceeds the maximum number of allowed candidates."
    if config.max_offsets <= 0:
        return "max_offsets must be positive."
    if config.period_strategy == "calendar_year":
        if not isinstance(config.period_value, int) or config.period_value <= 0:
            return "calendar_year period_value must be a positive integer number of years."
    elif config.period_strategy == "fixed_timedelta":
        try:
            if pd.Timedelta(config.period_value) <= pd.Timedelta(0):
                return "fixed_timedelta period_value must be positive."
        except (TypeError, ValueError):
            return "fixed_timedelta period_value must be convertible to pandas Timedelta."
    elif config.period_strategy == "sample_offset":
        if not isinstance(config.period_value, int) or config.period_value <= 0:
            return "sample_offset period_value must be a positive integer."
    else:
        return f"Unknown period strategy: {config.period_strategy}."
    return None


def _periodic_candidate_timestamps(
    target_timestamps: pd.DatetimeIndex,
    direction: int,
    offset: int,
    config: SeasonalPeriodConfig,
) -> pd.DatetimeIndex | None:
    """Shift target timestamps under a configured calendar or fixed period."""
    if config.period_strategy == "fixed_timedelta":
        return target_timestamps + direction * offset * pd.Timedelta(config.period_value)

    years = direction * offset * int(config.period_value)
    shifted = target_timestamps + pd.DateOffset(years=years)
    if config.leap_day_policy == "skip":
        changed_calendar_day = (
            (target_timestamps.month == 2)
            & (target_timestamps.day == 29)
            & ((shifted.month != 2) | (shifted.day != 29))
        )
        if changed_calendar_day.any():
            return None
    elif config.leap_day_policy == "mar1":
        shifted = pd.DatetimeIndex(
            [
                pd.Timestamp(timestamp + pd.DateOffset(years=years)).replace(
                    month=3, day=1
                )
                if timestamp.month == 2
                and timestamp.day == 29
                and not pd.Timestamp(timestamp + pd.DateOffset(years=years)).is_leap_year
                else timestamp + pd.DateOffset(years=years)
                for timestamp in target_timestamps
            ]
        )
    return pd.DatetimeIndex(shifted)


def impute_seasonal_periodic(
    artificial_gap: ArtificialGap,
    config: SeasonalPeriodConfig,
    *,
    value_column: str = "gaze_x",
    validity_column: str = "is_valid",
    timestamp_column: str | None = None,
) -> ImputationResult:
    """Fill a gap from complete, seasonally corresponding reference segments.

    Candidates are collected in chronological offsets around the target gap. The
    method is domain agnostic: it supports calendar-aware annual periods,
    fixed-duration cycles, and regular sample offsets. It never uses target-gap
    samples and returns ``not applicable`` rather than hiding failure behind a
    fallback to another imputer.
    """
    method_name = "seasonal_periodic"
    configuration_error = _validate_seasonal_config(config)
    if configuration_error:
        return _not_applicable(method_name, configuration_error)

    recording = artificial_gap.masked_recording
    candidate = artificial_gap.candidate
    start, end = candidate.gap_start_idx, candidate.gap_end_idx
    required_columns = {value_column, validity_column}
    if config.period_strategy != "sample_offset":
        if timestamp_column is None:
            return _not_applicable(
                method_name,
                "A timestamp_column is required for calendar and fixed-duration periods.",
            )
        required_columns.add(timestamp_column)
    missing_columns = required_columns - set(recording.columns)
    if missing_columns:
        return _not_applicable(
            method_name,
            f"Missing required columns: {sorted(missing_columns)}.",
        )
    if start < 0 or end > len(recording) or start >= end:
        return _not_applicable(method_name, "Gap indices are unavailable.")

    values = pd.to_numeric(recording[value_column], errors="coerce").to_numpy(dtype=float)
    # Pandas may expose a read-only NumPy view here.  The periodic method
    # intersects it with original validity below, so it requires ownership.
    valid = recording[validity_column].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    ).to_numpy(dtype=bool).copy()
    if "original_is_valid" in recording.columns:
        valid &= recording["original_is_valid"].map(
            lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
        ).to_numpy(dtype=bool)
    valid &= np.isfinite(values)

    metadata: dict[str, object] = {
        "period_strategy": config.period_strategy,
        "period_value": str(config.period_value),
        "mode": config.mode,
        "aggregation": config.aggregation,
        "leap_day_policy": config.leap_day_policy,
        "candidates_per_direction": config.candidates_per_direction,
        "min_candidates": config.min_candidates,
        "max_offsets": config.max_offsets,
        "candidate_offsets": [],
        "candidate_directions": [],
        "candidate_positions": [],
    }

    timestamp_index: pd.DatetimeIndex | None = None
    timestamp_positions: pd.Series | None = None
    if config.period_strategy != "sample_offset":
        timestamps = pd.to_datetime(recording[timestamp_column], errors="coerce")
        timestamp_index = pd.DatetimeIndex(timestamps)
        if timestamp_index.hasnans or not timestamp_index.is_monotonic_increasing:
            return _not_applicable(
                method_name,
                "Timestamps must be finite and monotonically increasing.",
                metadata,
            )
        if not timestamp_index.is_unique:
            return _not_applicable(
                method_name,
                "Timestamps must be unique for periodic candidate matching.",
                metadata,
            )
        timestamp_positions = pd.Series(np.arange(len(recording)), index=timestamp_index)

    directions = (-1, 1) if config.mode == "bidirectional" else (-1,)
    direction_counts = {direction: 0 for direction in directions}
    candidates: list[np.ndarray] = []
    gap_positions = set(range(start, end))
    gap_length = end - start

    for offset in range(1, config.max_offsets + 1):
        for direction in directions:
            if direction_counts[direction] >= config.candidates_per_direction:
                continue

            if config.period_strategy == "sample_offset":
                shifted_start = start + direction * offset * int(config.period_value)
                positions = np.arange(shifted_start, shifted_start + gap_length)
                if positions[0] < 0 or positions[-1] >= len(recording):
                    continue
            else:
                assert timestamp_index is not None and timestamp_positions is not None
                shifted_times = _periodic_candidate_timestamps(
                    timestamp_index[start:end], direction, offset, config
                )
                if shifted_times is None:
                    continue
                positions = timestamp_positions.reindex(shifted_times).to_numpy()
                if pd.isna(positions).any():
                    continue
                positions = positions.astype(int)

            if len(positions) != gap_length or gap_positions.intersection(positions):
                continue
            if not valid[positions].all():
                continue

            candidates.append(values[positions].copy())
            direction_counts[direction] += 1
            metadata["candidate_offsets"].append(offset)
            metadata["candidate_directions"].append("past" if direction < 0 else "future")
            metadata["candidate_positions"].append((int(positions[0]), int(positions[-1]) + 1))

        if all(count >= config.candidates_per_direction for count in direction_counts.values()):
            break

    metadata["candidate_count"] = len(candidates)
    metadata["past_candidate_count"] = direction_counts.get(-1, 0)
    metadata["future_candidate_count"] = direction_counts.get(1, 0)
    if len(candidates) < config.min_candidates:
        return _not_applicable(
            method_name,
            "Fewer than min_candidates complete periodic reference segments are available.",
            metadata,
        )

    stacked_candidates = np.vstack(candidates)
    predictions = (
        np.mean(stacked_candidates, axis=0)
        if config.aggregation == "mean"
        else np.median(stacked_candidates, axis=0)
    )
    if len(predictions) != gap_length or not np.isfinite(predictions).all():
        return _not_applicable(
            method_name,
            "Periodic aggregation did not produce finite predictions for every gap sample.",
            metadata,
        )
    return _successful_result(method_name, predictions.astype(float), metadata)
