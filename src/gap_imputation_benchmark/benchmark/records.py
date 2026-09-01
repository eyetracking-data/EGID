"""Convert one artificial gap into a portable, leakage-controlled benchmark row.

Rows combine provenance, observable selector features, method outcomes, and
normalised targets while retaining raw feature numerators for fold-local
rescaling in later evaluation and training workflows.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from gap_imputation_benchmark.benchmark.evaluation import (
    Imputer,
    evaluate_imputers_on_gap,
)
from gap_imputation_benchmark.benchmark.features import selector_feature_values
from gap_imputation_benchmark.benchmark.gaps import ArtificialGap
from gap_imputation_benchmark.benchmark.unit_robust import (
    iqr,
    local_scale,
    normalized_rmse,
    recording_iqr_leave_gap_out,
)


FEATURES_BASIC_NORMALIZED: Final[tuple[str, ...]] = (
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
)


def validate_features_basic_normalized() -> None:
    """Ensure the model feature allowlist excludes provenance and outcomes."""
    forbidden = {
        "dataset_id",
        "participant_id",
        "recording_id",
        "session_id",
        "source_file",
        "gap_start_idx",
        "gap_end_idx",
        "gap_length_samples",
        "sampling_rate_hz",
        "ground_truth",
        "rmse",
        "mae",
        "method_applicable",
        "failure_reason",
        "best_method_by_rmse",
        "best_method_by_mae",
        "best_rmse",
        "best_mae",
        "number_of_applicable_methods",
        "polyfit_selected_degree",
        "number_of_template_candidates",
    }
    overlap = set(FEATURES_BASIC_NORMALIZED) & forbidden
    if overlap:
        raise ValueError(f"Model feature list contains forbidden names: {sorted(overlap)}")


def build_gap_benchmark_record(
    artificial_gap: ArtificialGap,
    scale_floor: float,
    imputer_methods: dict[str, Imputer] | None = None,
) -> dict[str, object]:
    """Build one leakage-safe benchmark record for a single artificial gap.

    ``scale_floor`` is intentionally explicit. In later corpus work it must be
    estimated from the training split only, never from this gap or test data.
    Ties in best-method summaries are resolved by the supplied registry's
    insertion order. ``None`` retains the fixed Eye-Tracking registry.
    """
    if scale_floor <= 0:
        raise ValueError("scale_floor must be positive.")
    validate_features_basic_normalized()

    recording = artificial_gap.masked_recording
    required_columns = {
        "dataset_id",
        "participant_id",
        "recording_id",
        "session_id",
        "source_file",
    }
    missing_columns = required_columns - set(recording.columns)
    if missing_columns:
        raise ValueError(f"Missing provenance columns: {sorted(missing_columns)}")

    candidate = artificial_gap.candidate
    gaze_x = recording["gaze_x"].to_numpy(dtype=float, na_value=float("nan"))
    is_valid = recording["is_valid"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    ).to_numpy(dtype=bool)
    context_indices = np.r_[
        candidate.left_context_start_idx:candidate.gap_start_idx,
        candidate.gap_end_idx:candidate.right_context_end_idx,
    ]
    context_values = gaze_x[
        context_indices[
            is_valid[context_indices] & np.isfinite(gaze_x[context_indices])
        ]
    ]
    local_context_iqr = iqr(context_values)
    recording_iqr = recording_iqr_leave_gap_out(
        gaze_x,
        is_valid,
        candidate.gap_start_idx,
        candidate.gap_end_idx,
    )
    normalization_scale = local_scale(local_context_iqr, recording_iqr)
    extracted_features = selector_feature_values(artificial_gap, scale_floor)
    evaluation = evaluate_imputers_on_gap(
        artificial_gap,
        imputer_methods=imputer_methods,
    )
    diagnostics = dict(extracted_features.diagnostics)
    diagnostics["trend_difference"] = diagnostics["trend_after"] - diagnostics["trend_before"]
    record: dict[str, object] = {
        "dataset_id": recording["dataset_id"].iloc[0],
        "participant_id": recording["participant_id"].iloc[0],
        "recording_id": recording["recording_id"].iloc[0],
        "session_id": recording["session_id"].iloc[0],
        "source_file": recording["source_file"].iloc[0],
        "gap_start_idx": candidate.gap_start_idx,
        "gap_end_idx": candidate.gap_end_idx,
        "gap_length_samples": candidate.gap_length_samples,
        "requested_gap_duration_ms": candidate.requested_gap_duration_ms,
        "realized_gap_duration_ms": candidate.realized_gap_duration_ms,
        "sampling_rate_hz": candidate.sampling_rate_hz,
        "left_context_start_idx": candidate.left_context_start_idx,
        "right_context_end_idx": candidate.right_context_end_idx,
        "duration_stratum": candidate.duration_stratum,
        "local_context_iqr": local_context_iqr,
        "recording_iqr_leave_gap_out": recording_iqr,
        "normalization_scale": normalization_scale,
        # These stable aliases are consumed by the final selector workflow.
        # They retain the precise local/recording-aware nRMSE definition above.
        "local_iqr": local_context_iqr,
        "scale_i": normalization_scale,
        "diagnostics": diagnostics,
        "best_method_by_rmse": evaluation.best_method_by_rmse,
        "best_rmse": evaluation.best_rmse,
        "best_method_by_mae": evaluation.best_method_by_mae,
        "best_mae": evaluation.best_mae,
        "number_of_applicable_methods": evaluation.number_of_applicable_methods,
    }
    record.update({name: extracted_features[name] for name in FEATURES_BASIC_NORMALIZED})
    record.update(
        {
            name: value
            for name, value in extracted_features.items()
            if name.startswith("raw_")
        }
    )

    for method_result in evaluation.method_results:
        method_name = method_result.method_name
        record[f"{method_name}_applicable"] = method_result.method_applicable
        record[f"{method_name}_failure_reason"] = method_result.failure_reason
        record[f"{method_name}_rmse"] = method_result.rmse
        record[f"{method_name}_mae"] = method_result.mae
        record[f"{method_name}_nrmse"] = (
            normalized_rmse(method_result.rmse, normalization_scale)
            if method_result.rmse is not None
            else None
        )

    scored_nrmse = [
        (method_result.method_name, record[f"{method_result.method_name}_nrmse"])
        for method_result in evaluation.method_results
        if record[f"{method_result.method_name}_nrmse"] is not None
        and np.isfinite(record[f"{method_result.method_name}_nrmse"])
    ]
    best_nrmse = min(scored_nrmse, key=lambda item: item[1], default=None)
    record["best_method_by_nrmse"] = best_nrmse[0] if best_nrmse else None
    record["best_nrmse"] = best_nrmse[1] if best_nrmse else None

    return record
