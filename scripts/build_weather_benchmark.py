"""Build the frozen DWD hourly-temperature artificial-gap benchmark.

This is a direct, portable implementation of the reviewed Weather reference
procedure.  It contains no workstation path: ``--data-dir`` points to the
external ``weather`` data directory containing ``raw/`` and/or ``processed/``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from gap_imputation_benchmark.benchmark.gaps import create_artificial_gap
from gap_imputation_benchmark.benchmark.records import FEATURES_BASIC_NORMALIZED, build_gap_benchmark_record
from gap_imputation_benchmark.benchmark.unit_robust import iqr
from gap_imputation_benchmark.benchmark.weather import find_hourly_temperature_gap_candidates
from gap_imputation_benchmark.domains.weather import (
    WEATHER_DOMAIN,
    WEATHER_FEATURE_COLUMNS,
    load_dwd_station,
    load_weather_benchmark_config,
    prepare_dwd_temperature_data,
)
from gap_imputation_benchmark.imputers.registry import WEATHER_SEASONAL_CONFIG
from gap_imputation_benchmark.paths import load_local_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None, help="External Weather data directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-data", action="store_true", help="Recreate processed/ from raw DWD ZIP files first.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_seed(random_state: int, *parts: object) -> int:
    text = "|".join(map(str, (random_state, *parts)))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def translate_candidate(candidate, offset: int, stratum: int):
    return replace(
        candidate,
        gap_start_idx=candidate.gap_start_idx + offset,
        gap_end_idx=candidate.gap_end_idx + offset,
        left_context_start_idx=candidate.left_context_start_idx + offset,
        right_context_end_idx=candidate.right_context_end_idx + offset,
        duration_stratum=stratum,
    )


def overlaps(candidate, selected: list) -> bool:
    return any(
        candidate.left_context_start_idx < other.right_context_end_idx
        and candidate.right_context_end_idx > other.left_context_start_idx
        for other in selected
    )


def sample_station_year_gaps(frame: pd.DataFrame, station_id: str, year: int, config) -> tuple[list, list[dict[str, object]]]:
    year_positions = np.flatnonzero(frame["timestamp"].dt.year.eq(year).to_numpy())
    year_offset = int(year_positions[0])
    year_frame = frame.iloc[year_positions].reset_index(drop=True)
    rng = random.Random(stable_seed(config.random_state, station_id, year))
    selected, skipped = [], []
    tasks = [stratum for stratum in range(1, len(config.gap_strata_hours) + 1) for _ in range(config.gaps_per_stratum)]
    rng.shuffle(tasks)
    candidate_cache: dict[int, list] = {}
    for gap_number, stratum in enumerate(tasks, start=1):
        lower, upper = config.gap_strata_hours[stratum - 1]
        accepted = None
        requested_hours: int | None = None
        for attempts_used in range(1, config.max_gap_attempts + 1):
            requested_hours = rng.randint(lower, upper)
            if requested_hours not in candidate_cache:
                candidate_cache[requested_hours] = find_hourly_temperature_gap_candidates(
                    year_frame,
                    requested_hours,
                    min_context_samples=config.min_context_samples,
                    min_context_valid_fraction=config.min_context_valid_fraction,
                )
            candidates = [
                translate_candidate(candidate, year_offset, stratum)
                for candidate in candidate_cache[requested_hours]
            ]
            candidates = [candidate for candidate in candidates if not overlaps(candidate, selected)]
            if candidates:
                accepted = rng.choice(candidates)
                selected.append(accepted)
                break
        if accepted is None:
            skipped.append({
                "station_id": station_id, "year": year, "gap_number": gap_number,
                "duration_stratum": stratum, "duration_stratum_lower_hours": lower,
                "duration_stratum_upper_hours": upper, "requested_gap_hours": requested_hours,
                "attempts_used": attempts_used,
                "reason": "no_eligible_nonoverlapping_position_in_duration_stratum",
            })
    return sorted(selected, key=lambda candidate: candidate.gap_start_idx), skipped


def compact_gap_for_evaluation(station: pd.DataFrame, candidate):
    """Retain exactly the local and annual-reference observations legacy code used."""
    timestamps = pd.DatetimeIndex(station["timestamp"])
    target_times = timestamps[candidate.gap_start_idx:candidate.gap_end_idx]
    reference_positions: list[int] = []
    for offset in range(1, WEATHER_SEASONAL_CONFIG.max_offsets + 1):
        for direction in (-1, 1):
            shifted = target_times + pd.DateOffset(years=direction * offset)
            if (WEATHER_SEASONAL_CONFIG.leap_day_policy == "skip"
                    and any(time.month == 2 and time.day == 29 for time in target_times)
                    and any(time.month != 2 or time.day != 29 for time in shifted)):
                continue
            positions = timestamps.get_indexer(shifted)
            reference_positions.extend(positions[positions >= 0].tolist())
    local_positions = np.arange(candidate.left_context_start_idx, candidate.right_context_end_idx)
    positions = np.unique(np.concatenate([local_positions, np.asarray(reference_positions, dtype=int)]))
    compact = station.iloc[positions].reset_index(drop=True).copy()
    mapping = {int(original): index for index, original in enumerate(positions)}
    compact_candidate = replace(
        candidate,
        gap_start_idx=mapping[candidate.gap_start_idx],
        gap_end_idx=mapping[candidate.gap_end_idx - 1] + 1,
        left_context_start_idx=mapping[candidate.left_context_start_idx],
        right_context_end_idx=mapping[candidate.right_context_end_idx - 1] + 1,
    )
    return create_artificial_gap(compact, compact_candidate)


def full_station_iqr_excluding_gap(station: pd.DataFrame, candidate) -> float:
    values = station["gaze_x"].to_numpy(float).copy()
    values[candidate.gap_start_idx:candidate.gap_end_idx] = np.nan
    return iqr(values[np.isfinite(values)])


def _write_standard_tables(
    output_dir: Path,
    *,
    input_manifest: pd.DataFrame,
    skipped: pd.DataFrame,
) -> None:
    """Write the shared benchmark-table names with their common semantics."""
    input_manifest.to_csv(output_dir / "dataset_summary.csv", index=False)
    input_manifest.to_csv(output_dir / "selected_recordings.csv", index=False)
    skipped.to_csv(output_dir / "excluded_gaps.csv", index=False)


def main() -> None:
    args = parse_args()
    config = load_weather_benchmark_config(args.config)
    load_local_environment()
    data_dir = args.data_dir or (Path(__import__("os").environ["WEATHER_DATA_DIR"]) if __import__("os").environ.get("WEATHER_DATA_DIR") else None)
    if data_dir is None:
        raise RuntimeError("Set WEATHER_DATA_DIR or supply --data-dir.")
    data_dir, output_dir = Path(data_dir).resolve(), args.output_dir.resolve()
    processed_dir = data_dir / "processed" / "temperature_2000_2025"
    if args.prepare_data:
        processed_dir = prepare_dwd_temperature_data(data_dir, start=config.start, end=config.end)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    station_paths = sorted(processed_dir.glob("produkt_tu_stunde_*.txt"))
    if not station_paths:
        raise FileNotFoundError(f"No processed DWD files found in {processed_dir}.")
    stations = {
        path.stem.split("_")[-1]: load_dwd_station(
            path, start=config.start, end=config.end,
            source_file=(Path("data/weather/processed/temperature_2000_2025") / path.name).as_posix(),
        )
        for path in station_paths
    }
    input_manifest = pd.DataFrame([{
        "station_id": station_id, "source_file": frame["source_file"].iat[0],
        "first_timestamp": frame["timestamp"].iat[0], "last_timestamp": frame["timestamp"].iat[-1],
        "n_hourly_rows": len(frame), "natural_missing_hours": int((~frame["is_valid"]).sum()),
    } for station_id, frame in stations.items()]).sort_values("station_id").reset_index(drop=True)

    gap_rows: list[dict[str, object]] = []
    learning_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    ground_truth_values: list[np.ndarray] = []
    ground_truth_next_offset = 0
    years = range(pd.Timestamp(config.start).year, pd.Timestamp(config.end).year + 1)
    for station_id, station in sorted(stations.items()):
        for year in years:
            candidates, skipped = sample_station_year_gaps(station, station_id, year, config)
            skipped_rows.extend(skipped)
            for gap_number, candidate in enumerate(candidates, start=1):
                artificial_gap = compact_gap_for_evaluation(station, candidate)
                lower, upper = config.gap_strata_hours[candidate.duration_stratum - 1]
                gap_row = {
                    "gap_id": f"{station_id}_{year}_gap_{gap_number:03d}", "station_id": station_id, "year": year,
                    "gap_start_timestamp": station["timestamp"].iat[candidate.gap_start_idx],
                    "gap_end_timestamp": station["timestamp"].iat[candidate.gap_end_idx - 1],
                    "duration_stratum": candidate.duration_stratum,
                    "duration_stratum_lower_hours": lower, "duration_stratum_upper_hours": upper,
                    "gap_length_samples": candidate.gap_length_samples, "gap_length_hours": candidate.gap_length_samples,
                    "gap_start_idx": candidate.gap_start_idx, "gap_end_idx": candidate.gap_end_idx,
                    "left_context_start_idx": candidate.left_context_start_idx,
                    "right_context_end_idx": candidate.right_context_end_idx,
                    "left_context_valid_fraction": candidate.left_context_valid_fraction,
                    "right_context_valid_fraction": candidate.right_context_valid_fraction,
                    "ground_truth_offset": ground_truth_next_offset,
                    "ground_truth_value_count": len(artificial_gap.ground_truth),
                }
                ground_truth_values.append(artificial_gap.ground_truth)
                ground_truth_next_offset += len(artificial_gap.ground_truth)
                gap_rows.append(gap_row)
                record = build_gap_benchmark_record(artificial_gap, scale_floor=np.finfo(float).tiny, imputer_methods=WEATHER_DOMAIN.methods)
                station_iqr = full_station_iqr_excluding_gap(station, candidate)
                scale = max(record["local_context_iqr"], 0.05 * station_iqr)
                record["recording_iqr_leave_gap_out"] = station_iqr
                record["normalization_scale"] = scale
                for method in WEATHER_DOMAIN.methods:
                    rmse = record[f"{method}_rmse"]
                    record[f"{method}_nrmse"] = rmse / scale if pd.notna(rmse) else np.nan
                scores = {method: record[f"{method}_nrmse"] for method in WEATHER_DOMAIN.methods if np.isfinite(record[f"{method}_nrmse"])}
                record["best_method_by_nrmse"] = min(scores, key=scores.get) if scores else None
                record["best_nrmse"] = scores[record["best_method_by_nrmse"]] if scores else np.nan
                record.update(gap_row)
                record["feature_scale_floor_status"] = "deferred_to_loso_training_stations"
                learning_rows.append(record)

    gap_manifest, learnable = pd.DataFrame(gap_rows), pd.DataFrame(learning_rows)
    skipped_manifest = pd.DataFrame(skipped_rows)
    feature_columns_to_remove = [
        column
        for column in FEATURES_BASIC_NORMALIZED
        if column not in {"left_context_valid_fraction", "right_context_valid_fraction"}
    ]
    learnable = learnable.drop(
        columns=[
            *feature_columns_to_remove,
            "requested_gap_duration_ms",
            "realized_gap_duration_ms",
        ],
        errors="ignore",
    )
    requested = len(stations) * len(years) * len(config.gap_strata_hours) * config.gaps_per_stratum
    expected = pd.MultiIndex.from_product([sorted(stations), years], names=["station_id", "year"]).to_frame(index=False)
    counts = gap_manifest.groupby(["station_id", "year"]).size().rename("successful_gaps").reset_index()
    coverage = expected.merge(counts, on=["station_id", "year"], how="left").fillna({"successful_gaps": 0})
    coverage["successful_gaps"] = coverage["successful_gaps"].astype(int)
    coverage["requested_gaps"] = len(config.gap_strata_hours) * config.gaps_per_stratum
    coverage["shortfall"] = coverage["requested_gaps"] - coverage["successful_gaps"]
    metadata = {
        "artifact_type": "benchmark", "domain": "weather",
        "workflow": "weather_benchmark", "dataset_id": "DWD_hourly_temperature", "variable": "TT_TU",
        "period": {"start": config.start, "end": config.end}, "sampling_unit": "station_year",
        "stations": len(stations), "years_per_station": len(years), "requested_gaps": requested,
        "generated_gaps": len(gap_manifest), "excluded_gaps": len(skipped_manifest),
        "gap_strata_hours": config.gap_strata_hours, "gaps_per_stratum": config.gaps_per_stratum,
        "method_names": list(WEATHER_DOMAIN.methods), "feature_columns": list(WEATHER_FEATURE_COLUMNS),
        "counts": {"requested_gaps": requested, "benchmark_rows": len(learnable), "sampling_exclusions": len(skipped_manifest)},
        "method_configuration": {"seasonal_periodic": {"period_strategy": "calendar_year", "period_value": 1, "candidates_per_direction": 3, "min_candidates": 3, "max_offsets": 8, "aggregation": "mean", "leap_day_policy": "skip", "mode": "bidirectional"}},
        "random_state": config.random_state,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    input_manifest.to_csv(output_dir / "input_manifest.csv", index=False)
    learnable.to_csv(output_dir / "learnable_gap_table.csv", index=False)
    coverage.to_csv(output_dir / "coverage_table.csv", index=False)
    _write_standard_tables(
        output_dir,
        input_manifest=input_manifest,
        skipped=skipped_manifest,
    )
    print(f"Weather benchmark written: {output_dir.name} ({len(gap_manifest):,}/{requested:,} gaps)")


if __name__ == "__main__":
    main()
