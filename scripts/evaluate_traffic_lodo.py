"""Run the frozen nested leave-one-district-out Traffic selector evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from gap_imputation_benchmark.domains.traffic import TRAFFIC_DOMAIN, TRAFFIC_FEATURE_COLUMNS, load_traffic_benchmark_config
from gap_imputation_benchmark.paths import project_relative_path_or_label
from gap_imputation_benchmark.domains.traffic.workflow import (
    TrafficPanel,
    learn_feature_scale_floor,
    model_parameters_from_row,
    resolve_traffic_data_dir,
    traffic_feature_frame,
)


# Every serialized target matrix follows the canonical domain-method contract.
METHODS = tuple(TRAFFIC_DOMAIN.methods)
HYPERPARAMETER_CANDIDATES = (
    {"n_estimators": 300, "min_samples_leaf": 2, "max_depth": None, "max_features": 1.0},
    {"n_estimators": 400, "min_samples_leaf": 5, "max_depth": None, "max_features": "sqrt"},
    {"n_estimators": 300, "min_samples_leaf": 10, "max_depth": None, "max_features": 1.0},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def load_evaluated_gaps(benchmark_dir: Path) -> tuple[pd.DataFrame, list[str], Path]:
    """Load and validate outcomes plus context fractions needed for feature extraction."""
    target_columns = [f"{method}_nrmse" for method in METHODS]
    gaps = pd.read_csv(benchmark_dir / "learnable_gap_table.csv", low_memory=False)
    required = {"gap_id", "district", "year", "sensor_id2", "panel_column", "gap_start_in_year", "gap_end_in_year", "left_context_start_in_year", "right_context_end_in_year", "gap_length_steps", "gap_duration_minutes", "duration_stratum", "local_context_iqr", "left_context_valid_fraction", "right_context_valid_fraction", *target_columns}
    if missing := required - set(gaps):
        raise ValueError(f"Invalid evaluated Traffic table; missing columns: {sorted(missing)}")
    targets = gaps.loc[:, target_columns].apply(pd.to_numeric, errors="coerce")
    eligible = np.isfinite(targets.to_numpy(float)).all(axis=1)
    return gaps.loc[eligible].copy().reset_index(drop=True), target_columns


def fit_and_predict_method_errors(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    panel: TrafficPanel,
    target_columns: Sequence[str],
    parameters: Mapping[str, object],
    feature_scale_floor: float,
    random_state: int,
) -> np.ndarray:
    """Fit on outer/inner training districts and predict all method errors."""
    feature_imputer = SimpleImputer(strategy="median")
    x_train = feature_imputer.fit_transform(traffic_feature_frame(train, panel, feature_scale_floor))
    x_test = feature_imputer.transform(traffic_feature_frame(test, panel, feature_scale_floor))
    model = RandomForestRegressor(**parameters, random_state=random_state, n_jobs=-1)
    model.fit(x_train, train.loc[:, target_columns])
    return np.asarray(model.predict(x_test), dtype=float)


def tune_outer_fold(
    outer_train: pd.DataFrame,
    *,
    target_district: int,
    panel: TrafficPanel,
    target_columns: Sequence[str],
    feature_scale_floor_quantile: float,
    random_state: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Choose a model candidate using inner leave-one-district-out folds only."""
    rows: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(HYPERPARAMETER_CANDIDATES, start=1):
        scores = []
        for validation_district in sorted(outer_train["district"].unique()):
            inner_train = outer_train.loc[outer_train["district"].ne(validation_district)]
            validation = outer_train.loc[outer_train["district"].eq(validation_district)]
            floor = learn_feature_scale_floor(inner_train, feature_scale_floor_quantile)
            predicted = fit_and_predict_method_errors(inner_train, validation, panel=panel, target_columns=target_columns, parameters=candidate, feature_scale_floor=floor, random_state=random_state)
            actual = validation.loc[:, target_columns].to_numpy(float)
            scores.append(float(actual[np.arange(len(actual)), np.argmin(predicted, axis=1)].mean()))
        rows.append({"outer_target_district": int(target_district), "candidate_index": candidate_index, **candidate, "inner_lodo_mean_selected_nrmse": float(np.mean(scores))})
    tuning = pd.DataFrame(rows).sort_values("inner_lodo_mean_selected_nrmse").reset_index(drop=True)
    return model_parameters_from_row(tuning.iloc[0]), tuning


def evaluate_outer_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    panel: TrafficPanel,
    target_columns: Sequence[str],
    parameters: Mapping[str, object],
    feature_scale_floor: float,
    random_state: int,
    general_oracle_method: str,
) -> pd.DataFrame:
    """Predict one held-out district and calculate pre-specified comparisons."""
    predicted = fit_and_predict_method_errors(train, test, panel=panel, target_columns=target_columns, parameters=parameters, feature_scale_floor=feature_scale_floor, random_state=random_state)
    actual = test.loc[:, target_columns].to_numpy(float)
    selected, oracle = np.argmin(predicted, axis=1), np.argmin(actual, axis=1)
    result = test.loc[:, ["gap_id", "district", "year", "sensor_id2", "duration_stratum", "gap_length_steps", "gap_duration_minutes"]].copy()
    for index, method in enumerate(METHODS):
        result[f"actual_{method}_nrmse"] = actual[:, index]
        result[f"predicted_{method}_nrmse"] = predicted[:, index]
    result["selected_method"] = np.asarray(METHODS)[selected]
    result["oracle_method"] = np.asarray(METHODS)[oracle]
    result["selected_nrmse"] = actual[np.arange(len(test)), selected]
    result["pchip_nrmse"] = actual[:, METHODS.index("pchip")]
    result["general_oracle_method"] = general_oracle_method
    result["general_oracle_nrmse"] = actual[:, METHODS.index(general_oracle_method)]
    result["oracle_nrmse"] = actual[np.arange(len(test)), oracle]
    result["improvement_vs_pchip_nrmse"] = result["pchip_nrmse"] - result["selected_nrmse"]
    result["improvement_vs_general_oracle_nrmse"] = result["general_oracle_nrmse"] - result["selected_nrmse"]
    result["selected_regret_vs_oracle_nrmse"] = result["selected_nrmse"] - result["oracle_nrmse"]
    result["matches_oracle"] = result["selected_method"].eq(result["oracle_method"])
    return result


def summarize_predictions(predictions: pd.DataFrame, *, district_order: Sequence[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create district/macro performance and selected-method frequency tables."""
    summary = predictions.groupby("district").agg(n_gaps=("gap_id", "size"), mean_selected_nrmse=("selected_nrmse", "mean"), mean_pchip_nrmse=("pchip_nrmse", "mean"), mean_general_oracle_nrmse=("general_oracle_nrmse", "mean"), mean_oracle_nrmse=("oracle_nrmse", "mean"), mean_regret_vs_oracle_nrmse=("selected_regret_vs_oracle_nrmse", "mean"), oracle_match_rate=("matches_oracle", "mean")).reset_index()
    summary["absolute_improvement_vs_pchip_nrmse"] = summary["mean_pchip_nrmse"] - summary["mean_selected_nrmse"]
    summary["relative_improvement_vs_pchip_percent"] = 100 * summary["absolute_improvement_vs_pchip_nrmse"] / summary["mean_pchip_nrmse"]
    macro = summary.drop(columns=["district", "n_gaps"]).mean(numeric_only=True).to_dict()
    macro.update({"district": "macro_mean_across_districts", "n_gaps": int(summary["n_gaps"].sum())})
    summary = pd.concat([summary, pd.DataFrame([macro])], ignore_index=True)
    index = pd.MultiIndex.from_product([district_order, METHODS], names=["district", "selected_method"])
    selected_counts = predictions.groupby(["district", "selected_method"]).size().rename("count").reindex(index, fill_value=0).reset_index()
    selected_counts["share"] = selected_counts["count"] / selected_counts.groupby("district")["count"].transform("sum")
    overall_counts = predictions["selected_method"].value_counts().reindex(METHODS, fill_value=0).rename_axis("selected_method").reset_index(name="count")
    overall_counts["share"] = overall_counts["count"] / len(predictions)
    return summary, selected_counts, overall_counts


def global_fixed_method_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare each outer district with the fixed method best across LODO outputs."""
    actual_columns = [f"actual_{method}_nrmse" for method in METHODS]
    best_method = predictions.loc[:, actual_columns].mean().idxmin().removeprefix("actual_").removesuffix("_nrmse")
    baseline = f"actual_{best_method}_nrmse"
    summary = predictions.groupby("district").agg(n_gaps=("gap_id", "size"), mean_selected_nrmse=("selected_nrmse", "mean"), mean_global_best_method_nrmse=(baseline, "mean"), mean_oracle_nrmse=("oracle_nrmse", "mean")).reset_index()
    summary.insert(1, "global_best_method", best_method)
    summary["absolute_improvement_vs_global_best_nrmse"] = summary["mean_global_best_method_nrmse"] - summary["mean_selected_nrmse"]
    summary["relative_improvement_vs_global_best_percent"] = 100 * summary["absolute_improvement_vs_global_best_nrmse"] / summary["mean_global_best_method_nrmse"]
    macro = summary.drop(columns=["district", "global_best_method", "n_gaps"]).mean(numeric_only=True).to_dict()
    macro.update({"district": "macro_mean_across_districts", "global_best_method": best_method, "n_gaps": int(summary["n_gaps"].sum())})
    return pd.concat([summary, pd.DataFrame([macro])], ignore_index=True)


def main() -> None:
    args = parse_args()
    config = load_traffic_benchmark_config(args.config)
    benchmark_dir, results_dir = args.benchmark_dir.resolve(), args.results_dir.resolve()
    if results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(f"Results directory already exists: {results_dir}")
    gaps, target_columns = load_evaluated_gaps(benchmark_dir)
    district_order = sorted(int(value) for value in gaps["district"].unique())
    if len(district_order) < 3:
        raise ValueError("Nested LODO requires at least three districts.")
    general_oracle_method = str(gaps.loc[:, target_columns].mean().idxmin()).removesuffix("_nrmse")
    panel_dir = resolve_traffic_data_dir(args.data_dir) / "interim" / "largest_flow_panel_selected"
    panel = TrafficPanel(panel_dir)
    predictions_by_district: list[pd.DataFrame] = []
    tuning_by_district: list[pd.DataFrame] = []
    floor_rows: list[dict[str, object]] = []
    selected_parameters: dict[str, dict[str, object]] = {}
    try:
        for district in district_order:
            train, test = gaps.loc[gaps["district"].ne(district)].copy(), gaps.loc[gaps["district"].eq(district)].copy()
            floor = learn_feature_scale_floor(train, config.feature_scale_floor_quantile)
            parameters, tuning = tune_outer_fold(train, target_district=district, panel=panel, target_columns=target_columns, feature_scale_floor_quantile=config.feature_scale_floor_quantile, random_state=42)
            predictions_by_district.append(evaluate_outer_fold(train, test, panel=panel, target_columns=target_columns, parameters=parameters, feature_scale_floor=floor, random_state=42, general_oracle_method=general_oracle_method))
            tuning_by_district.append(tuning)
            floor_rows.append({"outer_target_district": district, "feature_scale_floor": floor, "quantile": config.feature_scale_floor_quantile, "source": "outer training districts only"})
            selected_parameters[str(district)] = parameters
    finally:
        panel.close()
    results_dir.mkdir(parents=True)
    predictions, tuning = pd.concat(predictions_by_district, ignore_index=True), pd.concat(tuning_by_district, ignore_index=True)
    summary, selected_counts, overall_counts = summarize_predictions(predictions, district_order=district_order)
    predictions.to_csv(results_dir / "lodo_gap_predictions.csv", index=False)
    tuning.to_csv(results_dir / "lodo_hyperparameter_tuning.csv", index=False)
    pd.DataFrame(floor_rows).to_csv(results_dir / "lodo_feature_scale_floors.csv", index=False)
    summary.to_csv(results_dir / "lodo_summary.csv", index=False)
    selected_counts.to_csv(results_dir / "lodo_selected_method_counts.csv", index=False)
    overall_counts.to_csv(results_dir / "lodo_selected_method_counts_overall.csv", index=False)
    metadata = {
        "artifact_type": "nested_evaluation",
        "domain": "traffic",
        "workflow": "leave_one_district_out",
        "benchmark_dir": project_relative_path_or_label(
            benchmark_dir,
            fallback_label="configured traffic benchmark directory",
        ),
        "method_names": list(METHODS),
        "feature_columns": list(TRAFFIC_FEATURE_COLUMNS),
        "feature_scale_floor": {
            "quantile": config.feature_scale_floor_quantile,
            "source": "outer training districts only",
        },
        "outer_evaluation": {
            "unit": "district", "groups": district_order,
            "macro_row": "macro_mean_across_districts",
        },
        "hyperparameter_candidates": list(HYPERPARAMETER_CANDIDATES),
        "selected_parameters_by_outer_group": selected_parameters,
        "selection_rule": "method with the lowest predicted absolute nRMSE",
        "comparison_baselines": {
            "general_oracle_method": general_oracle_method,
        },
        "random_state": 42,
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Traffic LODO results written: {results_dir.name} ({len(predictions):,} predictions)")


if __name__ == "__main__":
    main()
