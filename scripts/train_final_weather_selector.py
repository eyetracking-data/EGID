"""Train the final eight-method Weather selector after reviewed LOSO results.

This deployment script intentionally separates input loading, final parameter
selection, feature construction, model fitting, and artifact writing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from gap_imputation_benchmark.domains.weather import (
    WEATHER_FEATURE_COLUMNS,
    load_weather_benchmark_config,
)
from gap_imputation_benchmark.domains.weather.workflow import (
    learn_feature_scale_floor,
    load_benchmark_gaps,
    load_weather_stations,
    model_parameters_from_row,
    resolve_weather_data_dir,
    weather_feature_frame,
)
from gap_imputation_benchmark.paths import project_relative_path_or_label


def parse_args() -> argparse.Namespace:
    """Parse the reviewed benchmark, LOSO output, and local artifact paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--lodo-results-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_final_parameters(tuning_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Select the best candidate by mean inner-LOSO selection error.

    Outer-test scores are never used here; this is a deployment choice based on
    the inner tuning records produced during the reviewed LOSO run.
    """
    tuning = pd.read_csv(tuning_path)
    required = {
        "candidate_index",
        "inner_loso_mean_selected_nrmse",
        "n_estimators",
        "min_samples_leaf",
        "max_depth",
        "max_features",
    }
    missing = sorted(required - set(tuning.columns))
    if missing:
        raise ValueError(f"LOSO tuning table is missing required columns: {missing}")
    aggregated = tuning.groupby("candidate_index", as_index=False).agg(
        inner_loso_mean_selected_nrmse=("inner_loso_mean_selected_nrmse", "mean"),
        n_estimators=("n_estimators", "first"),
        min_samples_leaf=("min_samples_leaf", "first"),
        max_depth=("max_depth", "first"),
        max_features=("max_features", "first"),
    )
    best = aggregated.sort_values("inner_loso_mean_selected_nrmse", kind="stable").iloc[0]
    selection = {
        "selected_candidate_index": int(best["candidate_index"]),
        "mean_inner_loso_selected_nrmse": float(best["inner_loso_mean_selected_nrmse"]),
        "selection_source": "mean inner-LOSO nRMSE across completed outer LOSO folds",
    }
    return model_parameters_from_row(best), selection


def fit_final_selector(
    gaps: pd.DataFrame,
    *,
    stations: Mapping[str, pd.DataFrame],
    target_columns: Sequence[str],
    parameters: Mapping[str, object],
    feature_scale_floor: float,
    random_state: int,
) -> tuple[RandomForestRegressor, SimpleImputer]:
    """Fit the one deployment model on all eligible Weather gaps."""
    features = weather_feature_frame(gaps, stations, feature_scale_floor)
    imputer = SimpleImputer(strategy="median")
    model = RandomForestRegressor(**parameters, random_state=random_state, n_jobs=-1)
    model.fit(imputer.fit_transform(features), gaps.loc[:, target_columns])
    return model, imputer


def build_artifact_payload(
    *,
    model: RandomForestRegressor,
    feature_imputer: SimpleImputer,
    methods: Sequence[str],
    parameters: Mapping[str, object],
    parameter_selection: Mapping[str, object],
    training_rows: int,
    training_stations: Sequence[str],
    random_state: int,
    feature_scale_floor: float,
    feature_scale_floor_quantile: float,
) -> dict[str, object]:
    """Create the serialised selector and its human-readable metadata contract."""
    return {
        "artifact_type": "selector_artifact",
        "domain": "weather",
        "model": model,
        "feature_imputer": feature_imputer,
        "feature_columns": list(WEATHER_FEATURE_COLUMNS),
        "method_names": list(methods),
        "hyperparameters": dict(parameters),
        "hyperparameter_selection": dict(parameter_selection),
        "random_state": random_state,
        "training_rows": training_rows,
        "training_stations": list(training_stations),
        "target_definition": "absolute eight-method per-gap locally normalised nRMSE",
        "selection_rule": "method with smallest predicted absolute nRMSE",
        "model_type": "native shared multi-output RandomForestRegressor",
        "feature_scale_floor": {
            "value": feature_scale_floor,
            "quantile": feature_scale_floor_quantile,
            "source": "all final-training station gaps",
        },
    }


def write_artifact(artifact_dir: Path, artifact: Mapping[str, object]) -> None:
    """Write the binary model and inspectable metadata without model objects."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(artifact), artifact_dir / "selector.joblib")
    metadata = {
        key: value
        for key, value in artifact.items()
        if key not in {"model", "feature_imputer"}
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_weather_benchmark_config(args.config)
    benchmark_dir = args.benchmark_dir.resolve()
    results_dir = args.lodo_results_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    if artifact_dir.exists() and any(artifact_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Artifact directory already exists: {artifact_dir}")

    gaps, methods, target_columns = load_benchmark_gaps(benchmark_dir)
    stations = load_weather_stations(
        resolve_weather_data_dir(args.data_dir),
        gaps["station_id"].unique(),
        config,
    )
    feature_scale_floor = learn_feature_scale_floor(
        gaps,
        config.feature_scale_floor_quantile,
    )
    parameters, parameter_selection = select_final_parameters(
        results_dir / "lodo_hyperparameter_tuning.csv"
    )
    model, feature_imputer = fit_final_selector(
        gaps,
        stations=stations,
        target_columns=target_columns,
        parameters=parameters,
        feature_scale_floor=feature_scale_floor,
        random_state=config.random_state,
    )
    artifact = build_artifact_payload(
        model=model,
        feature_imputer=feature_imputer,
        methods=methods,
        parameters=parameters,
        parameter_selection=parameter_selection,
        training_rows=len(gaps),
        training_stations=sorted(stations),
        random_state=config.random_state,
        feature_scale_floor=feature_scale_floor,
        feature_scale_floor_quantile=config.feature_scale_floor_quantile,
    )
    write_artifact(artifact_dir, artifact)
    location = project_relative_path_or_label(
        artifact_dir,
        fallback_label="configured artifact directory",
    )
    print(f"Weather selector written: {location}")


if __name__ == "__main__":
    main()
