#!/usr/bin/env python3
"""Evaluate the Eye-Tracking selector with nested leave-one-dataset-out CV.

This is the confirmatory evaluation step. It writes a separate result
directory and never overwrites it unless --overwrite is passed explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from gap_imputation_benchmark.domains.eyetracking import EYE_TRACKING_DOMAIN
from gap_imputation_benchmark.paths import project_relative_path_or_label
from gap_imputation_benchmark.benchmark.output_contract import (
    EVALUATION_OUTPUTS,
    validate_output_contract,
)
from gap_imputation_benchmark.selection.native import (
    HYPERPARAMETER_CANDIDATES,
    RANDOM_STATE,
    evaluate_lodo_fold,
    fit_predict_method_errors,
    load_learnable_gaps,
    summarize_lodo_predictions,
    target_columns as selector_target_columns,
    tune_nested_lodo_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def target_columns() -> list[str]:
    return selector_target_columns(EYE_TRACKING_DOMAIN)


def load_gaps(benchmark_dir: Path) -> pd.DataFrame:
    return load_learnable_gaps(benchmark_dir, EYE_TRACKING_DOMAIN)


def predict_method_errors(
    train: pd.DataFrame,
    test: pd.DataFrame,
    parameters: dict[str, object],
) -> tuple[np.ndarray, float]:
    return fit_predict_method_errors(train, test, EYE_TRACKING_DOMAIN, parameters)


def tune_parameters(
    outer_train: pd.DataFrame,
    outer_target_dataset: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    return tune_nested_lodo_parameters(outer_train, outer_target_dataset, EYE_TRACKING_DOMAIN)


def evaluate_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    parameters: dict[str, object],
) -> tuple[pd.DataFrame, float]:
    return evaluate_lodo_fold(train, test, EYE_TRACKING_DOMAIN, parameters)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    return summarize_lodo_predictions(predictions)


def main() -> None:
    args = parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    results_dir = args.results_dir.resolve()
    if results_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Results directory already exists: {results_dir}")
    gaps = load_gaps(benchmark_dir)
    datasets = sorted(gaps["dataset_id"].unique())
    if len(datasets) < 3:
        raise ValueError("Nested LODO requires at least three datasets.")

    predictions_by_dataset: list[pd.DataFrame] = []
    tuning_by_dataset: list[pd.DataFrame] = []
    floor_rows: list[dict[str, object]] = []
    selected_parameters: dict[str, dict[str, object]] = {}
    for target_dataset in datasets:
        train = gaps.loc[gaps["dataset_id"].ne(target_dataset)].copy()
        test = gaps.loc[gaps["dataset_id"].eq(target_dataset)].copy()
        parameters, tuning = tune_parameters(train, str(target_dataset))
        predictions, floor = evaluate_fold(train, test, parameters)
        selected_parameters[str(target_dataset)] = parameters
        predictions_by_dataset.append(predictions)
        tuning_by_dataset.append(tuning)
        floor_rows.append(
            {
                "held_out_dataset": target_dataset,
                "training_gaps": len(train),
                "evaluation_gaps": len(test),
                "feature_scale_floor": floor,
            }
        )

    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    predictions = pd.concat(predictions_by_dataset, ignore_index=True)
    tuning = pd.concat(tuning_by_dataset, ignore_index=True)
    summary = summarize(predictions)
    choice_counts = predictions.groupby(["dataset_id", "selected_method"]).size().rename("count").reset_index()
    choice_counts["share"] = choice_counts["count"] / choice_counts.groupby("dataset_id")["count"].transform("sum")
    overall_choice_counts = (
        predictions["selected_method"]
        .value_counts()
        .reindex(EYE_TRACKING_DOMAIN.methods, fill_value=0)
        .rename_axis("selected_method")
        .reset_index(name="count")
    )
    overall_choice_counts["share"] = overall_choice_counts["count"] / len(predictions)
    predictions.to_csv(results_dir / "lodo_gap_predictions.csv", index=False)
    tuning.to_csv(results_dir / "lodo_hyperparameter_tuning.csv", index=False)
    pd.DataFrame(floor_rows).to_csv(results_dir / "lodo_feature_scale_floors.csv", index=False)
    summary.to_csv(results_dir / "lodo_summary.csv", index=False)
    choice_counts.to_csv(results_dir / "lodo_selected_method_counts.csv", index=False)
    overall_choice_counts.to_csv(
        results_dir / "lodo_selected_method_counts_overall.csv", index=False
    )
    metadata = {
        "artifact_type": "nested_evaluation",
        "domain": "eye_tracking",
        "workflow": "nested_leave_one_dataset_out",
        "random_state": RANDOM_STATE,
        "benchmark_dir": project_relative_path_or_label(
            benchmark_dir, fallback_label="configured eye-tracking benchmark directory"
        ),
        "feature_columns": list(EYE_TRACKING_DOMAIN.feature_columns),
        "method_names": list(EYE_TRACKING_DOMAIN.methods),
        "target_definition": "per-gap normalized RMSE for each candidate method",
        "selection_rule": "method with smallest predicted nRMSE",
        "outer_evaluation": {
            "unit": "dataset", "groups": datasets,
            "macro_row": "macro_mean_across_datasets",
        },
        "feature_scale_floor": {"source": "learned from each training fold only"},
        "hyperparameter_candidates": list(HYPERPARAMETER_CANDIDATES),
        "selected_parameters_by_outer_group": selected_parameters,
        "comparison_baselines": {"pchip": "fixed baseline"},
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    validate_output_contract(results_dir, EVALUATION_OUTPUTS)
    print(summary.to_string(index=False))
    output_location = project_relative_path_or_label(
        results_dir,
        fallback_label="configured evaluation directory",
    )
    print(f"\nWrote nested LODO results to: {output_location}")


if __name__ == "__main__":
    main()
