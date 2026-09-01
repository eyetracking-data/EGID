"""Reproduce the frozen DWD Weather leave-one-station-out selector evaluation.

The functions below correspond to the review protocol: loading, inner tuning,
outer-station prediction, summary calculation, and output writing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

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


HYPERPARAMETER_CANDIDATES = (
    {"n_estimators": 300, "min_samples_leaf": 2, "max_depth": None, "max_features": 1.0},
    {"n_estimators": 400, "min_samples_leaf": 5, "max_depth": None, "max_features": "sqrt"},
    {"n_estimators": 300, "min_samples_leaf": 10, "max_depth": None, "max_features": 1.0},
)


def parse_args() -> argparse.Namespace:
    """Parse explicit, portable workflow inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fit_and_predict_method_errors(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    stations: Mapping[str, pd.DataFrame],
    target_columns: Sequence[str],
    parameters: Mapping[str, object],
    feature_scale_floor: float,
    random_state: int,
) -> np.ndarray:
    """Fit only on a fold's training stations and predict all method errors."""
    imputer = SimpleImputer(strategy="median")
    train_features = weather_feature_frame(train, stations, feature_scale_floor)
    test_features = weather_feature_frame(test, stations, feature_scale_floor)
    x_train = imputer.fit_transform(train_features)
    x_test = imputer.transform(test_features)
    model = RandomForestRegressor(**parameters, random_state=random_state, n_jobs=-1)
    model.fit(x_train, train.loc[:, target_columns])
    return np.asarray(model.predict(x_test), dtype=float)


def tune_outer_fold(
    outer_train: pd.DataFrame,
    *,
    target_station: str,
    stations: Mapping[str, pd.DataFrame],
    target_columns: Sequence[str],
    feature_scale_floor_quantile: float,
    random_state: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Select one candidate with inner LOSO inside the outer training data."""
    rows: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(HYPERPARAMETER_CANDIDATES, start=1):
        scores = []
        for validation_station in sorted(outer_train["station_id"].unique()):
            inner_train = outer_train.loc[outer_train["station_id"].ne(validation_station)]
            validation = outer_train.loc[outer_train["station_id"].eq(validation_station)]
            floor = learn_feature_scale_floor(inner_train, feature_scale_floor_quantile)
            predicted = fit_and_predict_method_errors(
                inner_train,
                validation,
                stations=stations,
                target_columns=target_columns,
                parameters=candidate,
                feature_scale_floor=floor,
                random_state=random_state,
            )
            actual = validation.loc[:, target_columns].to_numpy(float)
            selected = np.argmin(predicted, axis=1)
            scores.append(float(actual[np.arange(len(actual)), selected].mean()))
        rows.append({
            "outer_target_station": target_station,
            "candidate_index": candidate_index,
            **candidate,
            "inner_loso_mean_selected_nrmse": float(np.mean(scores)),
        })
    tuning = pd.DataFrame(rows).sort_values("inner_loso_mean_selected_nrmse").reset_index(drop=True)
    return model_parameters_from_row(tuning.iloc[0]), tuning


def evaluate_outer_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    methods: Sequence[str],
    target_columns: Sequence[str],
    stations: Mapping[str, pd.DataFrame],
    parameters: Mapping[str, object],
    feature_scale_floor: float,
    random_state: int,
    general_oracle_method: str,
) -> pd.DataFrame:
    """Predict a held-out station and expose all pre-registered comparisons."""
    predicted = fit_and_predict_method_errors(
        train,
        test,
        stations=stations,
        target_columns=target_columns,
        parameters=parameters,
        feature_scale_floor=feature_scale_floor,
        random_state=random_state,
    )
    actual = test.loc[:, target_columns].to_numpy(float)
    selected_indices = np.argmin(predicted, axis=1)
    oracle_indices = np.argmin(actual, axis=1)
    result = test.loc[:, [
        "gap_id", "station_id", "year", "duration_stratum",
        "gap_length_samples", "gap_length_hours",
    ]].copy()
    for index, method in enumerate(methods):
        result[f"actual_{method}_nrmse"] = actual[:, index]
        result[f"predicted_{method}_nrmse"] = predicted[:, index]
    result["selected_method"] = np.asarray(methods)[selected_indices]
    result["oracle_method"] = np.asarray(methods)[oracle_indices]
    result["selected_nrmse"] = actual[np.arange(len(test)), selected_indices]
    result["pchip_nrmse"] = actual[:, methods.index("pchip")]
    result["general_oracle_method"] = general_oracle_method
    result["general_oracle_nrmse"] = actual[:, methods.index(general_oracle_method)]
    result["oracle_nrmse"] = actual[np.arange(len(test)), oracle_indices]
    result["improvement_vs_pchip_nrmse"] = result["pchip_nrmse"] - result["selected_nrmse"]
    result["improvement_vs_general_oracle_nrmse"] = (
        result["general_oracle_nrmse"] - result["selected_nrmse"]
    )
    result["selected_regret_vs_oracle_nrmse"] = result["selected_nrmse"] - result["oracle_nrmse"]
    result["matches_oracle"] = result["selected_method"].eq(result["oracle_method"])
    result["comparison_to_pchip"] = np.select(
        [
            result["improvement_vs_pchip_nrmse"] > 1e-12,
            result["improvement_vs_pchip_nrmse"] < -1e-12,
        ],
        ["better", "worse"],
        default="equal",
    )
    return result


def summarize_predictions(
    predictions: pd.DataFrame,
    *,
    methods: Sequence[str],
    station_order: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Produce station, macro, and method-selection summaries."""
    actual_columns = [f"actual_{method}_nrmse" for method in methods]
    general_method = (
        str(predictions.loc[:, actual_columns].mean().idxmin())
        .removeprefix("actual_")
        .removesuffix("_nrmse")
    )
    predictions["general_oracle_method"] = general_method
    predictions["general_oracle_nrmse"] = predictions[f"actual_{general_method}_nrmse"]
    predictions["improvement_vs_general_oracle_nrmse"] = (
        predictions["general_oracle_nrmse"] - predictions["selected_nrmse"]
    )
    summary = predictions.groupby("station_id").agg(
        n_gaps=("gap_id", "size"),
        mean_selected_nrmse=("selected_nrmse", "mean"),
        mean_pchip_nrmse=("pchip_nrmse", "mean"),
        mean_general_oracle_nrmse=("general_oracle_nrmse", "mean"),
        mean_oracle_nrmse=("oracle_nrmse", "mean"),
        mean_regret_vs_oracle_nrmse=("selected_regret_vs_oracle_nrmse", "mean"),
        oracle_match_rate=("matches_oracle", "mean"),
    ).reset_index()
    summary["absolute_improvement_vs_pchip_nrmse"] = summary["mean_pchip_nrmse"] - summary["mean_selected_nrmse"]
    summary["relative_improvement_vs_pchip_percent"] = (
        100 * summary["absolute_improvement_vs_pchip_nrmse"]
        / summary["mean_pchip_nrmse"]
    )
    summary["absolute_improvement_vs_general_oracle_nrmse"] = (
        summary["mean_general_oracle_nrmse"] - summary["mean_selected_nrmse"]
    )
    summary["relative_improvement_vs_general_oracle_percent"] = (
        100 * summary["absolute_improvement_vs_general_oracle_nrmse"]
        / summary["mean_general_oracle_nrmse"]
    )
    summary["general_oracle_method"] = general_method
    macro = summary.drop(columns=["station_id", "n_gaps"]).mean(numeric_only=True).to_dict()
    macro.update({"station_id": "macro_mean_across_stations", "n_gaps": int(summary["n_gaps"].sum())})
    summary = pd.concat([summary, pd.DataFrame([macro])], ignore_index=True)

    index = pd.MultiIndex.from_product([station_order, methods], names=["station_id", "selected_method"])
    selected_counts = (
        predictions.groupby(["station_id", "selected_method"])
        .size()
        .rename("count")
        .reindex(index, fill_value=0)
        .reset_index()
    )
    selected_counts["share"] = (
        selected_counts["count"]
        / selected_counts.groupby("station_id")["count"].transform("sum")
    )
    overall_counts = (
        predictions["selected_method"]
        .value_counts()
        .reindex(methods, fill_value=0)
        .rename_axis("selected_method")
        .reset_index(name="count")
    )
    overall_counts["share"] = overall_counts["count"] / len(predictions)
    return summary, selected_counts, overall_counts, general_method


def write_results(
    results_dir: Path,
    *,
    predictions: pd.DataFrame,
    tuning: pd.DataFrame,
    feature_scale_floors: pd.DataFrame,
    summary: pd.DataFrame,
    selected_counts: pd.DataFrame,
    overall_counts: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    """Write all review artifacts with stable file names."""
    predictions.to_csv(results_dir / "lodo_gap_predictions.csv", index=False)
    tuning.to_csv(results_dir / "lodo_hyperparameter_tuning.csv", index=False)
    feature_scale_floors.to_csv(results_dir / "lodo_feature_scale_floors.csv", index=False)
    summary.to_csv(results_dir / "lodo_summary.csv", index=False)
    selected_counts.to_csv(results_dir / "lodo_selected_method_counts.csv", index=False)
    overall_counts.to_csv(results_dir / "lodo_selected_method_counts_overall.csv", index=False)
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_weather_benchmark_config(args.config)
    benchmark_dir, results_dir = args.benchmark_dir.resolve(), args.results_dir.resolve()
    if results_dir.exists() and any(results_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Results directory already exists: {results_dir}")

    gaps, methods, target_columns = load_benchmark_gaps(benchmark_dir)
    stations = load_weather_stations(resolve_weather_data_dir(args.data_dir), gaps["station_id"].unique(), config)
    station_order = sorted(gaps["station_id"].unique())
    general_method = str(gaps.loc[:, target_columns].mean().idxmin()).removesuffix("_nrmse")
    predictions_by_station, tuning_by_station, floor_rows = [], [], []
    selected_parameters: dict[str, dict[str, object]] = {}

    for held_out_station in station_order:
        outer_train = gaps.loc[gaps["station_id"].ne(held_out_station)].copy()
        outer_test = gaps.loc[gaps["station_id"].eq(held_out_station)].copy()
        floor = learn_feature_scale_floor(outer_train, config.feature_scale_floor_quantile)
        parameters, tuning = tune_outer_fold(
            outer_train,
            target_station=held_out_station,
            stations=stations,
            target_columns=target_columns,
            feature_scale_floor_quantile=config.feature_scale_floor_quantile,
            random_state=config.random_state,
        )
        predictions_by_station.append(evaluate_outer_fold(
            outer_train,
            outer_test,
            methods=methods,
            target_columns=target_columns,
            stations=stations,
            parameters=parameters,
            feature_scale_floor=floor,
            random_state=config.random_state,
            general_oracle_method=general_method,
        ))
        tuning_by_station.append(tuning)
        floor_rows.append({"held_out_station": held_out_station, "feature_scale_floor": floor})
        selected_parameters[held_out_station] = parameters

    predictions = pd.concat(predictions_by_station, ignore_index=True)
    summary, selected_counts, overall_counts, general_method = summarize_predictions(
        predictions,
        methods=methods,
        station_order=station_order,
    )
    metadata = {
        "artifact_type": "nested_evaluation",
        "domain": "weather",
        "workflow": "leave_one_station_out",
        "benchmark_dir": project_relative_path_or_label(benchmark_dir, fallback_label="configured benchmark directory"),
        "method_names": list(methods),
        "feature_columns": list(WEATHER_FEATURE_COLUMNS),
        "feature_scale_floor": {
            "quantile": config.feature_scale_floor_quantile,
            "source": "outer training stations only",
        },
        "outer_evaluation": {
            "unit": "station", "groups": station_order,
            "macro_row": "macro_mean_across_stations",
        },
        "hyperparameter_candidates": list(HYPERPARAMETER_CANDIDATES),
        "selected_parameters_by_outer_group": selected_parameters,
        "selection_rule": "method with the lowest predicted absolute nRMSE",
        "comparison_baselines": {"general_oracle_method": general_method},
        "random_state": config.random_state,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    write_results(
        results_dir,
        predictions=predictions,
        tuning=pd.concat(tuning_by_station, ignore_index=True),
        feature_scale_floors=pd.DataFrame(floor_rows),
        summary=summary,
        selected_counts=selected_counts,
        overall_counts=overall_counts,
        metadata=metadata,
    )
    location = project_relative_path_or_label(results_dir, fallback_label="configured results directory")
    print(f"Weather LOSO results written: {location}")


if __name__ == "__main__":
    main()
