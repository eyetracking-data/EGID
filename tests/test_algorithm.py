from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gap_imputation_benchmark.algorithm import (
    DomainImputationConfig,
    RunMetadata,
    impute_with_rf_selector,
)
from gap_imputation_benchmark.benchmark.imputers import ImputationResult
import gap_imputation_benchmark.algorithm.missing_values as missing_values


def test_algorithm_records_portable_provenance_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise the pipeline without depending on a downloaded release model."""
    def load_test_selector(domain: str) -> tuple[dict[str, object], dict[str, object]]:
        assert domain == "eye_tracking"
        return (
            {"feature_columns": [], "feature_scale_floor": {"value": 1.0}},
            {"artifact": {"path": "artifacts/eyetracking/selector.joblib"}},
        )

    monkeypatch.setattr(missing_values, "_load_selector", load_test_selector)
    monkeypatch.setattr(
        missing_values,
        "_predict_method",
        lambda artifact, feature_values: ("linear", {"linear": 0.0}),
    )

    frame = pd.DataFrame({"x": np.sin(np.linspace(0, 10, 300))})
    frame.loc[150:151, "x"] = np.nan

    result, provenance = impute_with_rf_selector(
        frame,
        "x",
        config=DomainImputationConfig(sampling_rate_hz=1_000.0),
        run_metadata=RunMetadata(executor_name="Oliver", executor_responsible_person="Jenny"),
        output_path=tmp_path / "imputed.csv",
        provenance_path=tmp_path / "provenance.json",
    )

    assert result["x"].notna().all()
    assert provenance["activities"]["missing_value_imputation"]["details"]["imputation_model"]["path"] == (
        "artifacts/eyetracking/selector.joblib"
    )
    assert provenance["run"]["pipeline_implementation"]["path"] == (
        "src/gap_imputation_benchmark/algorithm/rf_domain_imputation.py"
    )
    gap = provenance["activities"]["missing_value_imputation"]["details"]["gaps"][0]
    assert gap["status"] == "filled"
    assert gap["attempted_methods"] == [{
        "method": "linear", "applicable": True, "outcome": "applied", "reason": None,
    }]
    assert provenance["summary"]["filled_gap_count"] == 1
    assert provenance["activities"]["standardization"]["details"]["file_after_standardization"]["path"] == "external"


def test_algorithm_marks_successful_lower_ranked_method_as_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def load_test_selector(domain: str) -> tuple[dict[str, object], dict[str, object]]:
        assert domain == "eye_tracking"
        return (
            {"feature_columns": [], "feature_scale_floor": {"value": 1.0}},
            {"artifact": {"path": "artifacts/eyetracking/selector.joblib"}},
        )

    def apply_test_method(method: str, runtime: object, domain_policy: object) -> ImputationResult:
        if method == "linear":
            return ImputationResult("linear", np.array([]), False, "test inapplicable", {})
        return ImputationResult("forward_fill", np.array([0.1, 0.2]), True, None, {})

    monkeypatch.setattr(missing_values, "_load_selector", load_test_selector)
    monkeypatch.setattr(
        missing_values,
        "_predict_method",
        lambda artifact, feature_values: ("linear", {"linear": 0.0, "forward_fill": 1.0}),
    )
    monkeypatch.setattr(missing_values, "_apply_method", apply_test_method)
    frame = pd.DataFrame({"x": np.sin(np.linspace(0, 10, 300))})
    frame.loc[150:151, "x"] = np.nan

    _, provenance = impute_with_rf_selector(
        frame,
        "x",
        config=DomainImputationConfig(sampling_rate_hz=1_000.0),
        output_path=tmp_path / "imputed.csv",
        provenance_path=tmp_path / "provenance.json",
    )

    gap = provenance["activities"]["missing_value_imputation"]["details"]["gaps"][0]
    assert gap["status"] == "fallback"
    assert gap["used_method"] == "forward_fill"
    assert gap["fallback_reason"] == "higher-ranked RF method did not yield an applicable finite reconstruction"
    assert gap["attempted_methods"] == [
        {"method": "linear", "applicable": False, "outcome": "not_applicable", "reason": "test inapplicable"},
        {"method": "forward_fill", "applicable": True, "outcome": "applied", "reason": None},
    ]
    assert provenance["summary"]["filled_gap_count"] == 1


def test_algorithm_records_invalid_predictions_and_exceptions_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def load_test_selector(domain: str) -> tuple[dict[str, object], dict[str, object]]:
        return (
            {"feature_columns": [], "feature_scale_floor": {"value": 1.0}},
            {"artifact": {"path": "artifacts/eyetracking/selector.joblib"}},
        )

    def apply_test_method(method: str, runtime: object, domain_policy: object) -> ImputationResult:
        if method == "linear":
            return ImputationResult("linear", np.array([np.nan, 0.2]), True, None, {})
        if method == "forward_fill":
            raise RuntimeError("test failure")
        return ImputationResult("nearest_boundary", np.array([0.1, 0.2]), True, None, {})

    monkeypatch.setattr(missing_values, "_load_selector", load_test_selector)
    monkeypatch.setattr(
        missing_values,
        "_predict_method",
        lambda artifact, feature_values: (
            "linear", {"linear": 0.0, "forward_fill": 1.0, "nearest_boundary": 2.0},
        ),
    )
    monkeypatch.setattr(missing_values, "_apply_method", apply_test_method)
    frame = pd.DataFrame({"x": np.sin(np.linspace(0, 10, 300))})
    frame.loc[150:151, "x"] = np.nan

    _, provenance = impute_with_rf_selector(
        frame,
        "x",
        config=DomainImputationConfig(sampling_rate_hz=1_000.0),
        output_path=tmp_path / "imputed.csv",
        provenance_path=tmp_path / "provenance.json",
    )

    gap = provenance["activities"]["missing_value_imputation"]["details"]["gaps"][0]
    assert gap["status"] == "fallback"
    assert gap["attempted_methods"] == [
        {"method": "linear", "applicable": True, "outcome": "invalid_prediction", "reason": "non_finite_predictions"},
        {"method": "forward_fill", "applicable": None, "outcome": "exception", "reason": "RuntimeError: test failure"},
        {"method": "nearest_boundary", "applicable": True, "outcome": "applied", "reason": None},
    ]
