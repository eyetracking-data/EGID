"""Domain-independent training and evaluation for native method selectors."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from gap_imputation_benchmark.benchmark.fold_local_scaling import (
    fold_local_feature_frame,
    learn_feature_scale_floor,
)
from gap_imputation_benchmark.domains.base import DomainSpec


RANDOM_STATE = 42
HYPERPARAMETER_CANDIDATES: tuple[dict[str, object], ...] = (
    {"n_estimators": 400, "min_samples_leaf": 5, "max_depth": None, "max_features": "sqrt"},
    {"n_estimators": 300, "min_samples_leaf": 2, "max_depth": None, "max_features": 1.0},
    {"n_estimators": 300, "min_samples_leaf": 5, "max_depth": None, "max_features": 1.0},
    {"n_estimators": 300, "min_samples_leaf": 10, "max_depth": None, "max_features": 1.0},
    {"n_estimators": 300, "min_samples_leaf": 5, "max_depth": 12, "max_features": 0.7},
    {"n_estimators": 500, "min_samples_leaf": 2, "max_depth": None, "max_features": 0.7},
    {"n_estimators": 500, "min_samples_leaf": 10, "max_depth": 12, "max_features": 0.7},
)


def target_columns(domain: DomainSpec) -> list[str]:
    """Return the ordered, method-specific nRMSE target columns for a domain."""
    return [f"{method}_nrmse" for method in domain.methods]


def load_learnable_gaps(benchmark_dir: str | Path, domain: DomainSpec) -> pd.DataFrame:
    """Load valid selector rows using the shared benchmark-table contract."""
    gaps = pd.read_csv(Path(benchmark_dir) / "learnable_gap_table.csv")
    targets = target_columns(domain)
    legacy_targets = [column.removesuffix("_nrmse") + "_normalized_rmse" for column in targets]
    if set(legacy_targets).issubset(gaps.columns) and not set(targets).issubset(gaps.columns):
        gaps = gaps.rename(columns=dict(zip(legacy_targets, targets, strict=True)))
    required = {"gap_id", "dataset_id", "participant_id", "local_iqr", "scale_i", *domain.feature_columns, *targets}
    missing = required - set(gaps.columns)
    if missing:
        raise ValueError(f"Benchmark is missing required columns: {sorted(missing)}")
    if gaps["gap_id"].duplicated().any():
        raise ValueError("gap_id must be unique.")
    numeric = gaps.loc[:, [*domain.feature_columns, "local_iqr", "scale_i", *targets]].apply(pd.to_numeric, errors="coerce")
    return gaps.loc[np.isfinite(numeric.to_numpy(float)).all(axis=1)].reset_index(drop=True)


def fit_predict_method_errors(
    train: pd.DataFrame,
    test: pd.DataFrame,
    domain: DomainSpec,
    parameters: dict[str, object],
) -> tuple[np.ndarray, float]:
    """Fit all preprocessing on ``train`` and predict each candidate's error."""
    feature_scale_floor = learn_feature_scale_floor(train)
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(fold_local_feature_frame(
        train, feature_scale_floor,
        feature_columns=domain.feature_columns,
        scale_dependent_feature_numerators=dict(domain.scale_dependent_feature_numerators),
    ))
    x_test = imputer.transform(fold_local_feature_frame(
        test, feature_scale_floor,
        feature_columns=domain.feature_columns,
        scale_dependent_feature_numerators=dict(domain.scale_dependent_feature_numerators),
    ))
    model = RandomForestRegressor(**parameters, random_state=RANDOM_STATE, n_jobs=-1)
    prediction = np.asarray(model.fit(x_train, train.loc[:, target_columns(domain)]).predict(x_test), dtype=float)
    expected_shape = (len(test), len(domain.methods))
    if prediction.shape != expected_shape:
        raise ValueError(f"Unexpected multi-output prediction shape: {prediction.shape}; expected {expected_shape}")
    return prediction, feature_scale_floor


def tune_nested_lodo_parameters(
    outer_train: pd.DataFrame,
    outer_target_dataset: str,
    domain: DomainSpec,
    candidates: Sequence[dict[str, object]] = HYPERPARAMETER_CANDIDATES,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Choose hyperparameters using inner LODO across outer-training datasets."""
    source_datasets = sorted(outer_train["dataset_id"].unique())
    if len(source_datasets) < 2:
        raise ValueError("Nested LODO requires at least two source datasets in each outer fold.")
    rows: list[dict[str, object]] = []
    for candidate_index, parameters in enumerate(candidates, start=1):
        scores: dict[str, float] = {}
        for validation_dataset in source_datasets:
            train = outer_train.loc[outer_train["dataset_id"].ne(validation_dataset)]
            validation = outer_train.loc[outer_train["dataset_id"].eq(validation_dataset)]
            prediction, _ = fit_predict_method_errors(train, validation, domain, parameters)
            actual = validation.loc[:, target_columns(domain)].to_numpy(float)
            selected = np.argmin(prediction, axis=1)
            scores[str(validation_dataset)] = float(actual[np.arange(len(actual)), selected].mean())
        rows.append({
            "outer_target_dataset": outer_target_dataset,
            "candidate_index": candidate_index,
            **parameters,
            "inner_lodo_macro_mean_selected_nrmse": float(np.mean(list(scores.values()))),
            **{f"inner_validation_{dataset}_mean_selected_nrmse": score for dataset, score in scores.items()},
        })
    tuning = pd.DataFrame(rows).sort_values("inner_lodo_macro_mean_selected_nrmse").reset_index(drop=True)
    return dict(candidates[int(tuning.loc[0, "candidate_index"]) - 1]), tuning


def evaluate_lodo_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    domain: DomainSpec,
    parameters: dict[str, object],
) -> tuple[pd.DataFrame, float]:
    """Evaluate a selected-method policy on one held-out domain."""
    prediction, feature_scale_floor = fit_predict_method_errors(train, test, domain, parameters)
    actual = test.loc[:, target_columns(domain)].to_numpy(float)
    methods = np.asarray(tuple(domain.methods))
    selected_index, oracle_index = np.argmin(prediction, axis=1), np.argmin(actual, axis=1)
    pchip_index = tuple(domain.methods).index("pchip")
    columns = [column for column in ("gap_id", "dataset_id", "participant_id", "recording_id", "duration_stratum", "realized_gap_duration_ms") if column in test]
    result = test.loc[:, columns].copy()
    result["selected_method"], result["oracle_method"] = methods[selected_index], methods[oracle_index]
    result["selected_nrmse"] = actual[np.arange(len(test)), selected_index]
    result["pchip_nrmse"], result["oracle_nrmse"] = actual[:, pchip_index], actual[np.arange(len(test)), oracle_index]
    result["improvement_vs_pchip_nrmse"] = result["pchip_nrmse"] - result["selected_nrmse"]
    result["selected_regret_vs_oracle_nrmse"] = result["selected_nrmse"] - result["oracle_nrmse"]
    result["matches_oracle"] = result["selected_method"].eq(result["oracle_method"])
    for index, method in enumerate(methods):
        result[f"predicted_{method}_nrmse"] = prediction[:, index]
        result[f"actual_{method}_nrmse"] = actual[:, index]
    return result, feature_scale_floor


def summarize_lodo_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build per-domain and macro LODO summaries from gap-level predictions."""
    rows: list[dict[str, object]] = []
    for dataset_id, frame in predictions.groupby("dataset_id", sort=True):
        selected_mean, pchip_mean = float(frame["selected_nrmse"].mean()), float(frame["pchip_nrmse"].mean())
        rows.append({
            "evaluation_set": dataset_id, "n_gaps": len(frame),
            "mean_selected_nrmse": selected_mean, "mean_pchip_nrmse": pchip_mean,
            "mean_oracle_nrmse": float(frame["oracle_nrmse"].mean()),
            "absolute_improvement_vs_pchip_nrmse": pchip_mean - selected_mean,
            "mean_regret_vs_oracle_nrmse": float(frame["selected_regret_vs_oracle_nrmse"].mean()),
            "oracle_match_rate": float(frame["matches_oracle"].mean()),
        })
    summary = pd.DataFrame(rows)
    macro = summary.drop(columns=["evaluation_set", "n_gaps"]).mean(numeric_only=True).to_dict()
    macro["relative_improvement_vs_pchip_percent"] = 100 * macro["absolute_improvement_vs_pchip_nrmse"] / macro["mean_pchip_nrmse"]
    macro.update({"evaluation_set": "macro_mean_across_datasets", "n_gaps": int(summary["n_gaps"].sum())})
    summary["relative_improvement_vs_pchip_percent"] = 100 * summary["absolute_improvement_vs_pchip_nrmse"] / summary["mean_pchip_nrmse"]
    return pd.concat([summary, pd.DataFrame([macro])], ignore_index=True)


def global_lodo_score(
    gaps: pd.DataFrame,
    domain: DomainSpec,
    parameters: dict[str, object],
) -> tuple[float, dict[str, float]]:
    """Score one parameter set by macro mean selected error across all domains."""
    scores: dict[str, float] = {}
    for held_out in sorted(gaps["dataset_id"].unique()):
        train, test = gaps.loc[gaps["dataset_id"].ne(held_out)], gaps.loc[gaps["dataset_id"].eq(held_out)]
        predicted, _ = fit_predict_method_errors(train, test, domain, parameters)
        actual = test.loc[:, target_columns(domain)].to_numpy(float)
        scores[str(held_out)] = float(actual[np.arange(len(test)), np.argmin(predicted, axis=1)].mean())
    return float(np.mean(list(scores.values()))), scores
