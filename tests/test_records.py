import numpy as np
import pandas as pd
import pytest

from gap_imputation_benchmark.benchmark.evaluation import GapEvaluation, ImputerEvaluation
from gap_imputation_benchmark.benchmark.features import RAW_FEATURE_NUMERATOR_COLUMNS
from gap_imputation_benchmark.benchmark.gaps import create_artificial_gap, find_gap_candidates
from gap_imputation_benchmark.benchmark.records import (
    FEATURES_BASIC_NORMALIZED,
    build_gap_benchmark_record,
    validate_features_basic_normalized,
)


def artificial_gap() -> object:
    recording = pd.DataFrame(
        {
            "dataset_id": ["test_dataset"] * 7,
            "participant_id": ["participant_1"] * 7,
            "recording_id": ["recording_1"] * 7,
            "session_id": ["session_1"] * 7,
            "source_file": ["test.csv"] * 7,
            "gaze_x": np.arange(7, dtype=float) * 2.0,
            "is_valid": [True] * 7,
            "sampling_rate_hz": [100.0] * 7,
            "timestamp_ms": np.arange(7, dtype=float) * 1_000.0,
        }
    )
    candidate = find_gap_candidates(recording, gap_duration_ms=20)[0]
    return create_artificial_gap(recording, candidate)


def test_builds_exactly_one_record_with_preserved_provenance():
    gap = artificial_gap()
    record = build_gap_benchmark_record(gap, scale_floor=1.0)

    assert isinstance(record, dict)
    assert record["dataset_id"] == "test_dataset"
    assert record["participant_id"] == "participant_1"
    assert record["recording_id"] == "recording_1"
    assert record["session_id"] == "session_1"
    assert record["source_file"] == "test.csv"
    assert record["gap_start_idx"] == gap.candidate.gap_start_idx
    assert record["gap_end_idx"] == gap.candidate.gap_end_idx
    assert record["local_iqr"] == record["local_context_iqr"]
    assert record["scale_i"] == record["normalization_scale"]


def test_feature_list_has_exact_stable_contents_and_no_leakage():
    assert FEATURES_BASIC_NORMALIZED == (
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
    validate_features_basic_normalized()
    forbidden = {
        "dataset_id",
        "participant_id",
        "recording_id",
        "source_file",
        "gap_length_samples",
        "sampling_rate_hz",
        "ground_truth",
        "rmse",
        "mae",
        "best_method_by_rmse",
        "number_of_template_candidates",
    }
    assert set(FEATURES_BASIC_NORMALIZED).isdisjoint(forbidden)


def test_record_flattens_all_current_method_results():
    record = build_gap_benchmark_record(artificial_gap(), scale_floor=1.0)
    method_names = (
        "forward_fill",
        "nearest_boundary",
        "linear",
        "pchip",
        "local_natural_cubic_spline",
        "polyfit_bic",
        "template",
    )

    for method_name in method_names:
        assert f"{method_name}_applicable" in record
        assert f"{method_name}_failure_reason" in record
        assert f"{method_name}_rmse" in record
        assert f"{method_name}_mae" in record
        assert f"{method_name}_nrmse" in record
        assert f"{method_name}_predictions" not in record


def test_record_normalizes_rmse_by_local_iqr_with_recording_floor():
    record = build_gap_benchmark_record(artificial_gap(), scale_floor=1.0)

    assert record["local_context_iqr"] == pytest.approx(7.0)
    assert record["recording_iqr_leave_gap_out"] == pytest.approx(8.0)
    assert record["normalization_scale"] == pytest.approx(7.0)
    assert record["forward_fill_nrmse"] == pytest.approx(
        record["forward_fill_rmse"] / record["normalization_scale"]
    )
    assert record["best_method_by_nrmse"] == record["best_method_by_rmse"]
    assert record["best_nrmse"] == pytest.approx(
        record[f"{record['best_method_by_nrmse']}_nrmse"]
    )


def test_record_keeps_diagnostics_outside_the_model_feature_list():
    record = build_gap_benchmark_record(artificial_gap(), scale_floor=1.0)

    assert "diagnostics" in record
    assert {
        "local_iqr",
        "local_std",
        "local_range",
        "left_mean",
        "right_mean",
        "boundary_difference",
        "absolute_boundary_difference",
        "n_valid_left",
        "n_valid_right",
        "trend_before",
        "trend_after",
        "trend_difference",
        "median_absolute_velocity_left",
        "median_absolute_velocity_right",
        "p90_absolute_velocity_left",
        "p90_absolute_velocity_right",
    }.issubset(record["diagnostics"])
    assert set(record["diagnostics"]).isdisjoint(FEATURES_BASIC_NORMALIZED)
    assert set(RAW_FEATURE_NUMERATOR_COLUMNS.values()).issubset(record)


def test_record_uses_evaluation_best_method_and_excludes_non_applicable(monkeypatch):
    results = [
        ImputerEvaluation("forward_fill", True, None, np.array([1.0, 2.0]), 1.0, 1.0, {}),
        ImputerEvaluation("linear", True, None, np.array([1.0, 2.0]), 1.0, 1.0, {}),
        ImputerEvaluation("template", False, "No candidate", np.array([]), None, None, {}),
    ]
    evaluation = GapEvaluation(
        method_results=results,
        best_method_by_rmse="forward_fill",
        best_rmse=1.0,
        best_method_by_mae="forward_fill",
        best_mae=1.0,
        number_of_applicable_methods=2,
    )
    monkeypatch.setattr(
        "gap_imputation_benchmark.benchmark.records.evaluate_imputers_on_gap",
        lambda *_args, **_kwargs: evaluation,
    )

    record = build_gap_benchmark_record(artificial_gap(), scale_floor=1.0)

    assert record["best_method_by_rmse"] == "forward_fill"
    assert record["best_method_by_mae"] == "forward_fill"
    assert record["number_of_applicable_methods"] == 2
    assert record["template_applicable"] is False
    assert record["template_rmse"] is None


def test_record_rejects_invalid_scale_floor():
    with pytest.raises(ValueError, match="scale_floor"):
        build_gap_benchmark_record(artificial_gap(), scale_floor=0.0)
