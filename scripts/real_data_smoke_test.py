#!/usr/bin/env python3
"""Run the current per-gap imputation evaluation on one real file per dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gap_imputation_benchmark.benchmark.configuration import load_corpus_benchmark_config
from gap_imputation_benchmark.benchmark.config import CorpusBenchmarkConfig
from gap_imputation_benchmark.benchmark.evaluation import evaluate_all_imputers_on_gap
from gap_imputation_benchmark.benchmark.gaps import create_artificial_gap, sample_stratified_random_gaps
from gap_imputation_benchmark.domains.eyetracking.loaders import (
    load_gazebase_reading,
    load_gazebase_vr_reading,
)
from gap_imputation_benchmark.paths import PROJECT_ROOT, require_raw_data_dir


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "reference_benchmark.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a complete TOML benchmark configuration.",
    )
    return parser.parse_args()


def _first_recording(data_dir: Path, pattern: str, dataset_name: str) -> Path:
    # Data locations are configured per machine instead of being embedded in code.
    recordings = sorted(data_dir.glob(pattern))
    if not recordings:
        raise FileNotFoundError(
            f"No {dataset_name} Reading/Text recordings found under "
            f"{data_dir} (pattern: {pattern})."
        )
    return recordings[0]


def evaluate_recording(
    recording: pd.DataFrame,
    config: CorpusBenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all current imputers on reproducibly sampled artificial gaps."""
    candidates = sample_stratified_random_gaps(
        recording=recording,
        n_gaps=config.n_gaps_per_recording,
        min_gap_duration_ms=config.min_gap_duration_ms,
        max_gap_duration_ms=config.max_gap_duration_ms,
        n_strata=config.n_strata,
        context_multiplier=config.context_multiplier,
        min_context_valid_fraction=config.min_context_valid_fraction,
        random_state=config.random_state,
    )
    rows: list[dict[str, object]] = []
    gap_summaries: list[dict[str, object]] = []

    for candidate in candidates:
        artificial_gap = create_artificial_gap(recording, candidate)
        evaluation = evaluate_all_imputers_on_gap(artificial_gap)
        gap_fields = {
            "dataset_id": recording["dataset_id"].iloc[0],
            "participant_id": recording["participant_id"].iloc[0],
            "recording_id": recording["recording_id"].iloc[0],
            "gap_start_idx": candidate.gap_start_idx,
            "gap_length_samples": candidate.gap_length_samples,
            "requested_gap_duration_ms": candidate.requested_gap_duration_ms,
            "realized_gap_duration_ms": candidate.realized_gap_duration_ms,
            "left_context_valid_fraction": candidate.left_context_valid_fraction,
            "right_context_valid_fraction": candidate.right_context_valid_fraction,
        }
        gap_summaries.append(
            {
                **gap_fields,
                "best_method_by_rmse": evaluation.best_method_by_rmse,
                "best_rmse": evaluation.best_rmse,
                "best_method_by_mae": evaluation.best_method_by_mae,
                "best_mae": evaluation.best_mae,
                "number_of_applicable_methods": evaluation.number_of_applicable_methods,
            }
        )
        for method_result in evaluation.method_results:
            rows.append(
                {
                    **gap_fields,
                    "method_name": method_result.method_name,
                    "method_applicable": method_result.method_applicable,
                    "failure_reason": method_result.failure_reason,
                    "rmse": method_result.rmse,
                    "mae": method_result.mae,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(gap_summaries)


def _print_dataset_summary(
    evaluations: pd.DataFrame,
    gap_summaries: pd.DataFrame,
) -> None:
    dataset_id = evaluations["dataset_id"].iloc[0]
    print(f"\n{dataset_id}: {len(gap_summaries)} evaluated gaps")
    print("Applicability counts per method:")
    print(evaluations.groupby("method_name")["method_applicable"].sum().to_string())
    print("Mean RMSE and MAE per method:")
    print(
        evaluations.groupby("method_name")[["rmse", "mae"]]
        .mean()
        .to_string()
    )
    print("Best method per gap:")
    print(
        gap_summaries[
            ["gap_start_idx", "best_method_by_rmse", "best_method_by_mae"]
        ].to_string(index=False)
    )


def main() -> None:
    args = parse_args()
    config = load_corpus_benchmark_config(args.config)
    data_dir = require_raw_data_dir()
    gazebase_path = _first_recording(
        data_dir,
        "raw/GazeBase_v2_0/**/S_*_TEX.csv",
        "GazeBase",
    )
    gazebase_vr_path = _first_recording(
        data_dir,
        "raw/gazebasevr/data/S_*_TEX.csv",
        "GazeBaseVR",
    )

    datasets = [
        (load_gazebase_reading(gazebase_path), gazebase_path),
        (load_gazebase_vr_reading(gazebase_vr_path), gazebase_vr_path),
    ]
    all_evaluations: list[pd.DataFrame] = []

    for recording, source_path in datasets:
        print(f"Loading {recording['dataset_id'].iloc[0]}: {source_path}")
        evaluations, gap_summaries = evaluate_recording(recording, config)
        _print_dataset_summary(evaluations, gap_summaries)
        all_evaluations.append(evaluations)

    print(f"\nTotal evaluated gaps: {sum(len(frame) // 7 for frame in all_evaluations)}")


if __name__ == "__main__":
    main()
