"""Leakage-safe, fold-local feature scaling for gap selectors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gap_imputation_benchmark.benchmark.features import RAW_FEATURE_NUMERATOR_COLUMNS
from gap_imputation_benchmark.benchmark.records import FEATURES_BASIC_NORMALIZED


SCALE_DEPENDENT_FEATURES = (
    "normalized_boundary_jump",
    "normalized_mean_difference_right_minus_left",
    "local_std_over_scale",
    "local_range_over_scale",
    "normalized_trend_before",
    "normalized_trend_after",
    "normalized_trend_difference",
    "normalized_median_absolute_velocity_left",
    "normalized_median_absolute_velocity_right",
    "normalized_p90_absolute_velocity_left",
    "normalized_p90_absolute_velocity_right",
)


def learn_feature_scale_floor(training_gaps: pd.DataFrame) -> float:
    """Learn a floor from positive local-context IQRs in training data only."""
    local_iqrs = pd.to_numeric(training_gaps["local_iqr"], errors="coerce").to_numpy(float)
    positive_local_iqrs = local_iqrs[np.isfinite(local_iqrs) & (local_iqrs > 0)]
    if not len(positive_local_iqrs):
        raise ValueError("Training fold contains no positive local-context IQR values.")
    floor = float(np.percentile(positive_local_iqrs, 1))
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError(f"Invalid learned feature scale floor: {floor}")
    return floor


def fold_local_feature_frame(
    gaps: pd.DataFrame,
    feature_scale_floor: float,
    *,
    feature_columns: tuple[str, ...] = FEATURES_BASIC_NORMALIZED,
    scale_dependent_feature_numerators: dict[str, str] = RAW_FEATURE_NUMERATOR_COLUMNS,
) -> pd.DataFrame:
    """Reconstruct and rescale selector features using a fixed training-fold floor.

    New benchmark tables retain explicit raw numerators; this function divides
    them by ``max(local_iqr, feature_scale_floor)``.  Older tables are supported
    by reconstructing the numerator from their original scale-normalised value
    and ``scale_i``. Error targets are never changed.
    """
    if not np.isfinite(feature_scale_floor) or feature_scale_floor <= 0:
        raise ValueError("feature_scale_floor must be finite and positive.")
    required = {"local_iqr", "scale_i", *feature_columns}
    missing = required - set(gaps.columns)
    if missing:
        raise ValueError(f"Missing columns for fold-local feature scaling: {sorted(missing)}")

    local_iqr = pd.to_numeric(gaps["local_iqr"], errors="coerce").to_numpy(float)
    local_scale = np.maximum(local_iqr, feature_scale_floor)
    if not np.isfinite(local_scale).all() or (local_scale <= 0).any():
        raise ValueError("Invalid local feature scale.")

    result = gaps.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").copy()
    raw_columns = set(scale_dependent_feature_numerators.values())
    if raw_columns.issubset(gaps.columns):
        for column, raw_column in scale_dependent_feature_numerators.items():
            result[column] = pd.to_numeric(gaps[raw_column], errors="coerce").to_numpy(float) / local_scale
    else:
        original_scale = pd.to_numeric(gaps["scale_i"], errors="coerce").to_numpy(float)
        for column in scale_dependent_feature_numerators:
            result[column] = result[column].to_numpy(float) * original_scale / local_scale
    if not np.isfinite(result.to_numpy(float)).all():
        raise ValueError("Non-finite features after fold-local scaling.")
    return result
