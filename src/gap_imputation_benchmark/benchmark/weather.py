"""Weather-specific artificial-gap helpers using hourly sample counts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gap_imputation_benchmark.benchmark.gaps import GapCandidate


HOURLY_SAMPLING_RATE_HZ = 1 / 3600


def find_hourly_temperature_gap_candidates(
    recording: pd.DataFrame,
    gap_length_samples: int,
    min_context_samples: int = 2,
    min_context_valid_fraction: float = 0.80,
) -> list[GapCandidate]:
    """Find valid DWD-temperature gaps specified directly in hourly samples.

    One weather sample is one hour.  The returned ``GapCandidate`` is kept
    compatible with the shared evaluator, but callers of this weather function
    never supply or need to handle a duration in milliseconds.
    """
    if isinstance(gap_length_samples, bool) or gap_length_samples < 1:
        raise ValueError("gap_length_samples must be a positive integer.")
    if int(gap_length_samples) != gap_length_samples:
        raise ValueError("gap_length_samples must be an integer.")
    if isinstance(min_context_samples, bool) or min_context_samples < 1:
        raise ValueError("min_context_samples must be a positive integer.")
    if not 0 <= min_context_valid_fraction <= 1:
        raise ValueError("min_context_valid_fraction must be between 0 and 1.")

    required_columns = {"gaze_x", "is_valid", "sampling_rate_hz"}
    missing_columns = required_columns - set(recording.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if recording.empty:
        return []

    sampling_rates = recording["sampling_rate_hz"].dropna().unique()
    if len(sampling_rates) != 1 or not np.isclose(
        float(sampling_rates[0]), HOURLY_SAMPLING_RATE_HZ
    ):
        raise ValueError("Weather recordings must have sampling_rate_hz = 1 / 3600.")

    gap_length_samples = int(gap_length_samples)
    context_length_samples = max(gap_length_samples, int(min_context_samples))
    values = pd.to_numeric(recording["gaze_x"], errors="coerce").to_numpy(dtype=float)
    is_valid = recording["is_valid"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    ).to_numpy(dtype=bool)
    valid = is_valid & np.isfinite(values)

    candidates: list[GapCandidate] = []
    first_start = context_length_samples
    last_start = len(recording) - gap_length_samples - context_length_samples
    for start in range(first_start, last_start + 1):
        end = start + gap_length_samples
        left_start = start - context_length_samples
        right_end = end + context_length_samples
        if not valid[start:end].all() or not (valid[start - 1] and valid[end]):
            continue

        left_fraction = float(valid[left_start:start].mean())
        right_fraction = float(valid[end:right_end].mean())
        if (
            left_fraction < min_context_valid_fraction
            or right_fraction < min_context_valid_fraction
        ):
            continue

        # These compatibility fields are consumed only by generic downstream
        # utilities. Weather manifests expose gap lengths in samples and hours.
        duration_ms = gap_length_samples * 3600 * 1000.0
        candidates.append(
            GapCandidate(
                gap_start_idx=start,
                gap_end_idx=end,
                requested_gap_duration_ms=duration_ms,
                realized_gap_duration_ms=duration_ms,
                sampling_rate_hz=HOURLY_SAMPLING_RATE_HZ,
                gap_length_samples=gap_length_samples,
                left_context_start_idx=left_start,
                right_context_end_idx=right_end,
                left_context_valid_fraction=left_fraction,
                right_context_valid_fraction=right_fraction,
                context_valid_fraction=float(
                    np.concatenate([valid[left_start:start], valid[end:right_end]]).mean()
                ),
            )
        )
    return candidates
