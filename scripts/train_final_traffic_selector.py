"""Train and save the final LargeST Traffic multi-output selector.

This is a deployment fit on every eligible benchmark gap after the completed
nested LODO evaluation.  It is not an additional performance estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from gap_imputation_benchmark.domains.traffic import (
    TRAFFIC_FEATURE_COLUMNS,
    load_traffic_benchmark_config,
)
from gap_imputation_benchmark.domains.traffic.workflow import (
    TrafficPanel,
    learn_feature_scale_floor,
    model_parameters_from_row,
    resolve_traffic_data_dir,
    traffic_feature_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--lodo-results-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def select_final_parameters(
    tuning_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Select one candidate by mean inner-LODO score across outer folds."""
    tuning = pd.read_csv(tuning_path)
    required = {
        "candidate_index",
        "inner_lodo_mean_selected_nrmse",
        "n_estimators",
        "min_samples_leaf",
        "max_depth",
        "max_features",
    }
    if missing := required - set(tuning):
        raise ValueError(f"Invalid LODO tuning table; missing columns: {sorted(missing)}")
    candidate_means = tuning.groupby("candidate_index", as_index=False).agg(
        inner_lodo_mean_selected_nrmse=("inner_lodo_mean_selected_nrmse", "mean"),
        n_estimators=("n_estimators", "first"),
        min_samples_leaf=("min_samples_leaf", "first"),
        max_depth=("max_depth", "first"),
        max_features=("max_features", "first"),
    )
    selected = candidate_means.sort_values(
        "inner_lodo_mean_selected_nrmse", kind="stable"
    ).iloc[0]
    provenance = {
        "selection_source": "mean inner-LODO nRMSE across completed outer LODO folds",
        "selected_candidate_index": int(selected.candidate_index),
        "mean_inner_lodo_selected_nrmse": float(
            selected.inner_lodo_mean_selected_nrmse
        ),
    }
    return model_parameters_from_row(selected), provenance


def load_final_training_rows(
    benchmark_dir: Path,
) -> tuple[pd.DataFrame, tuple[str, ...], list[str]]:
    """Load every gap with a finite target for each frozen candidate method."""
    benchmark_metadata = json.loads(
        (benchmark_dir / "metadata.json").read_text(encoding="utf-8")
    )
    methods = tuple(benchmark_metadata["method_names"])
    target_columns = [f"{method}_nrmse" for method in methods]
    gaps = pd.read_csv(benchmark_dir / "learnable_gap_table.csv", low_memory=False)
    required = {
        "gap_id", "district", "year", "panel_column", "gap_start_in_year",
        "gap_end_in_year", "left_context_start_in_year",
        "right_context_end_in_year", "gap_length_steps", "duration_stratum",
        "local_context_iqr", "left_context_valid_fraction",
        "right_context_valid_fraction", *target_columns,
    }
    if missing := required - set(gaps):
        raise ValueError(f"Invalid evaluated Traffic table; missing columns: {sorted(missing)}")
    targets = gaps.loc[:, target_columns].apply(pd.to_numeric, errors="coerce")
    eligible = np.isfinite(targets.to_numpy(float)).all(axis=1)
    return gaps.loc[eligible].copy().reset_index(drop=True), methods, target_columns


def fit_final_selector(
    gaps: pd.DataFrame,
    target_columns: list[str],
    panel_dir: Path,
    quantile: float,
    parameters: dict[str, object],
) -> tuple[RandomForestRegressor, SimpleImputer, float]:
    """Build features from the full eligible data and fit one shared multi-output forest."""
    scale_floor = learn_feature_scale_floor(gaps, quantile)
    panel = TrafficPanel(panel_dir)
    try:
        features = traffic_feature_frame(gaps, panel, scale_floor)
    finally:
        panel.close()
    feature_imputer = SimpleImputer(strategy="median")
    model = RandomForestRegressor(**parameters, random_state=42, n_jobs=-1)
    model.fit(feature_imputer.fit_transform(features), gaps.loc[:, target_columns])
    return model, feature_imputer, scale_floor


def write_artifact(
    artifact_dir: Path,
    *,
    model: RandomForestRegressor,
    feature_imputer: SimpleImputer,
    methods: tuple[str, ...],
    parameters: dict[str, object],
    selection: dict[str, object],
    gaps: pd.DataFrame,
    scale_floor: float,
    quantile: float,
) -> None:
    """Persist the model and a readable, path-free description of its contract."""
    if artifact_dir.exists():
        raise FileExistsError(f"Artifact directory already exists: {artifact_dir}")
    artifact_dir.mkdir(parents=True)
    artifact = {
        "artifact_type": "selector_artifact",
        "domain": "traffic",
        "model": model,
        "feature_imputer": feature_imputer,
        "feature_columns": list(TRAFFIC_FEATURE_COLUMNS),
        "method_names": list(methods),
        "hyperparameters": parameters,
        "hyperparameter_selection": selection,
        "random_state": 42,
        "training_rows": len(gaps),
        "training_districts": sorted(int(value) for value in gaps["district"].unique()),
        "target_definition": "absolute eight-method per-gap locally normalised nRMSE",
        "selection_rule": "method with smallest predicted absolute nRMSE",
        "model_type": "native shared multi-output RandomForestRegressor",
        "feature_scale_floor": {
            "value": scale_floor,
            "quantile": quantile,
            "source": "all final-training district gaps",
        },
    }
    joblib.dump(artifact, artifact_dir / "selector.joblib")
    manifest = {
        key: value
        for key, value in artifact.items()
        if key not in {"model", "feature_imputer"}
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_traffic_benchmark_config(args.config)
    gaps, methods, target_columns = load_final_training_rows(args.benchmark_dir.resolve())
    parameters, selection = select_final_parameters(
        args.lodo_results_dir.resolve() / "lodo_hyperparameter_tuning.csv"
    )
    panel_dir = (
        resolve_traffic_data_dir(args.data_dir)
        / "interim"
        / "largest_flow_panel_selected"
    )
    model, feature_imputer, scale_floor = fit_final_selector(
        gaps, target_columns, panel_dir, config.feature_scale_floor_quantile, parameters
    )
    write_artifact(
        args.artifact_dir.resolve(),
        model=model,
        feature_imputer=feature_imputer,
        methods=methods,
        parameters=parameters,
        selection=selection,
        gaps=gaps,
        scale_floor=scale_floor,
        quantile=config.feature_scale_floor_quantile,
    )
    print(f"Final Traffic selector saved: {args.artifact_dir.name}")


if __name__ == "__main__":
    main()
