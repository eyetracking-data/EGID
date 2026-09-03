from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gap_imputation_benchmark.algorithm import (
    DomainImputationConfig,
    RunMetadata,
    impute_with_rf_selector,
)
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
    assert provenance["activities"]["standardization"]["details"]["file_after_standardization"]["path"] == "external"
