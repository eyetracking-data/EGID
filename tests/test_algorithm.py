from pathlib import Path

import numpy as np
import pandas as pd

from gap_imputation_benchmark.algorithm import (
    DomainImputationConfig,
    RunMetadata,
    impute_with_rf_selector,
)


def test_algorithm_uses_portable_artifact_and_provenance_paths(tmp_path: Path) -> None:
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
