"""Training-only scale-floor estimation for normalized gap features."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import pandas as pd

from gap_imputation_benchmark.benchmark.config import CorpusBenchmarkConfig
from gap_imputation_benchmark.benchmark.gaps import (
    ArtificialGap,
    GapSamplingError,
    create_artificial_gap,
    sample_stratified_random_gaps,
)
from gap_imputation_benchmark.benchmark.provenance import (
    recording_provenance,
    recording_seed,
)


def _validate_quantile(quantile: float) -> None:
    if not math.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between 0 and 1.")


def _local_context_iqr(artificial_gap: ArtificialGap) -> float | None:
    """Return a positive finite IQR from an artificial gap's observed context."""
    recording = artificial_gap.masked_recording
    candidate = artificial_gap.candidate
    required_columns = {"gaze_x", "is_valid"}
    missing_columns = required_columns - set(recording.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    left_indices = np.arange(
        candidate.left_context_start_idx,
        candidate.gap_start_idx,
    )
    right_indices = np.arange(
        candidate.gap_end_idx,
        candidate.right_context_end_idx,
    )
    context_indices = np.concatenate([left_indices, right_indices])
    if len(context_indices) == 0:
        return None

    is_valid = recording["is_valid"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    ).to_numpy(dtype=bool)
    gaze_x = pd.to_numeric(recording["gaze_x"], errors="coerce").to_numpy(dtype=float)
    observed_context = gaze_x[
        context_indices[
            is_valid[context_indices] & np.isfinite(gaze_x[context_indices])
        ]
    ]
    if len(observed_context) == 0:
        return None

    local_iqr = float(
        np.percentile(observed_context, 75) - np.percentile(observed_context, 25)
    )
    return local_iqr if math.isfinite(local_iqr) and local_iqr > 0 else None


def estimate_scale_floor(
    training_gaps: Sequence[ArtificialGap],
    quantile: float = 0.01,
) -> float:
    """Estimate one positive scale floor solely from training-gap context."""
    _validate_quantile(quantile)
    if not training_gaps:
        raise ValueError("training_gaps must not be empty.")

    local_iqrs = [
        local_iqr
        for artificial_gap in training_gaps
        if (local_iqr := _local_context_iqr(artificial_gap)) is not None
    ]
    if not local_iqrs:
        raise ValueError("No positive finite local context IQR values are available.")

    scale_floor = float(np.quantile(local_iqrs, quantile))
    if not math.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("Estimated scale_floor must be finite and positive.")
    return scale_floor


def estimate_scale_floor_from_split(
    split_recordings: Sequence[pd.DataFrame],
    config: CorpusBenchmarkConfig,
) -> float:
    """Sample training recordings and estimate their shared scale floor.

    Call this helper only with the training split. It intentionally receives no
    validation or test recordings and performs no split assignment itself.
    """
    training_gaps: list[ArtificialGap] = []
    for recording in split_recordings:
        provenance = recording_provenance(recording)
        try:
            candidates = sample_stratified_random_gaps(
                recording=recording,
                n_gaps=config.n_gaps_per_recording,
                min_gap_duration_ms=config.min_gap_duration_ms,
                max_gap_duration_ms=config.max_gap_duration_ms,
                n_strata=config.n_strata,
                context_multiplier=config.context_multiplier,
                min_context_valid_fraction=config.min_context_valid_fraction,
                random_state=recording_seed(config.random_state, provenance),
            )
        except GapSamplingError as error:
            raise GapSamplingError(
                config.n_gaps_per_recording,
                0,
                f"Recording provenance: {', '.join(provenance)}. {error}",
            ) from error
        training_gaps.extend(create_artificial_gap(recording, candidate) for candidate in candidates)

    return estimate_scale_floor(training_gaps, quantile=config.scale_floor_quantile)
