"""Train the final 3.0 native seven-method selector after global macro LODO.

The artifact is a deployment model, not an additional unbiased evaluation.
It exactly matches the successful 3.0 native seven-method LODO formulation:
one shared multi-output RandomForestRegressor with absolute method nRMSE
targets and selection by the smallest predicted nRMSE.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from gap_imputation_benchmark.benchmark.fold_local_scaling import fold_local_feature_frame, learn_feature_scale_floor
from gap_imputation_benchmark.benchmark.output_contract import (
    SELECTOR_ARTIFACT_OUTPUTS,
    validate_output_contract,
)
from gap_imputation_benchmark.domains.eyetracking import EYE_TRACKING_DOMAIN
from gap_imputation_benchmark.paths import project_relative_path_or_label
from gap_imputation_benchmark.selection.native import (
    HYPERPARAMETER_CANDIDATES,
    RANDOM_STATE,
    fit_predict_method_errors,
    global_lodo_score as shared_global_lodo_score,
    load_learnable_gaps,
    target_columns as selector_target_columns,
)


DOMAIN = "eye_tracking"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def target_columns() -> list[str]:
    return selector_target_columns(EYE_TRACKING_DOMAIN)


def load_gaps(benchmark_dir: Path) -> pd.DataFrame:
    return load_learnable_gaps(benchmark_dir, EYE_TRACKING_DOMAIN)


def predict_errors(train: pd.DataFrame, test: pd.DataFrame, parameters: dict[str, object]) -> tuple[np.ndarray, float]:
    return fit_predict_method_errors(train, test, EYE_TRACKING_DOMAIN, parameters)


def macro_lodo_score(gaps: pd.DataFrame, parameters: dict[str, object]) -> tuple[float, dict[str, float]]:
    return shared_global_lodo_score(gaps, EYE_TRACKING_DOMAIN, parameters)


def main() -> None:
    args = parse_args()
    benchmark_dir, artifact_dir = args.benchmark_dir.resolve(), args.artifact_dir.resolve()
    if artifact_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Artifact directory already exists: {artifact_dir}")
    gaps = load_gaps(benchmark_dir)
    if gaps["dataset_id"].nunique() < 2:
        raise ValueError("Global LODO requires at least two datasets.")

    rows = []
    for index, parameters in enumerate(HYPERPARAMETER_CANDIDATES, start=1):
        score, per_dataset = macro_lodo_score(gaps, parameters)
        rows.append({"candidate_index": index, **parameters, "macro_lodo_selected_nrmse": score,
                     **{f"held_out_{dataset}_selected_nrmse": value for dataset, value in per_dataset.items()}})
    tuning = pd.DataFrame(rows).sort_values("macro_lodo_selected_nrmse").reset_index(drop=True)
    parameters = HYPERPARAMETER_CANDIDATES[int(tuning.loc[0, "candidate_index"]) - 1]

    feature_scale_floor = learn_feature_scale_floor(gaps)
    imputer = SimpleImputer(strategy="median")
    x_all = imputer.fit_transform(fold_local_feature_frame(
        gaps,
        feature_scale_floor,
        feature_columns=EYE_TRACKING_DOMAIN.feature_columns,
        scale_dependent_feature_numerators=dict(EYE_TRACKING_DOMAIN.scale_dependent_feature_numerators),
    ))
    model = RandomForestRegressor(**parameters, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(x_all, gaps.loc[:, target_columns()])

    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    artifact = {
        # Keep this metadata beside the model so loading code can reject a
        # feature, method, or domain mismatch before any inference happens.
        "artifact_type": "selector_artifact",
        "domain": DOMAIN,
        "model": model,
        "feature_imputer": imputer,
        "feature_columns": list(EYE_TRACKING_DOMAIN.feature_columns),
        "method_names": list(EYE_TRACKING_DOMAIN.methods),
        "hyperparameters": parameters, "random_state": RANDOM_STATE,
        "training_rows": len(gaps), "training_datasets": sorted(gaps["dataset_id"].unique()),
        "target_definition": "absolute seven-method per-gap locally normalised nRMSE",
        "selection_rule": "method with smallest predicted absolute nRMSE",
        "model_type": "native shared multi-output RandomForestRegressor",
        "feature_scale_floor": {
            "value": feature_scale_floor,
            "source": "first percentile of positive local-context IQRs across final training gaps",
            "test_data_contribute_to_floor": False,
        },
    }
    joblib.dump(artifact, artifact_dir / "selector.joblib")
    metadata = {key: value for key, value in artifact.items() if key not in {"model", "feature_imputer"}}
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    validate_output_contract(artifact_dir, SELECTOR_ARTIFACT_OUTPUTS)
    print(tuning.to_string(index=False))
    output_location = project_relative_path_or_label(
        artifact_dir,
        fallback_label="configured artifact directory",
    )
    print(f"\nSaved final native seven-method selector to: {output_location}")


if __name__ == "__main__":
    main()
