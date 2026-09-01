"""Shared, reviewable building blocks for the frozen Weather workflows."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from gap_imputation_benchmark.benchmark.features import extract_basic_gap_features
from gap_imputation_benchmark.benchmark.gaps import GapCandidate, create_artificial_gap
from gap_imputation_benchmark.benchmark.records import FEATURES_BASIC_NORMALIZED
from gap_imputation_benchmark.domains.weather.adapter import WEATHER_FEATURE_COLUMNS, WeatherBenchmarkConfig
from gap_imputation_benchmark.domains.weather.loaders import load_dwd_station
from gap_imputation_benchmark.paths import load_local_environment


HOUR_MS = 3_600_000.0


def resolve_weather_data_dir(value: Path | None) -> Path:
    """Resolve the external Weather data root without storing it in source code."""
    load_local_environment()
    if value is not None:
        return Path(value).resolve()
    configured = os.getenv("WEATHER_DATA_DIR")
    if not configured:
        raise RuntimeError("Set WEATHER_DATA_DIR or supply --data-dir.")
    return Path(configured).resolve()


def load_benchmark_gaps(benchmark_dir: Path) -> tuple[pd.DataFrame, tuple[str, ...], list[str]]:
    """Load the canonical table and retain only gaps evaluable by every method."""
    benchmark_dir = Path(benchmark_dir)
    metadata = json.loads((benchmark_dir / "metadata.json").read_text(encoding="utf-8"))
    methods = tuple(metadata["method_names"])
    target_columns = [f"{method}_nrmse" for method in methods]
    gaps = pd.read_csv(benchmark_dir / "learnable_gap_table.csv", low_memory=False)
    gaps["station_id"] = (
        pd.to_numeric(gaps["station_id"], errors="raise")
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )
    targets = gaps.loc[:, target_columns].apply(pd.to_numeric, errors="coerce")
    eligible = np.isfinite(targets.to_numpy(float)).all(axis=1)
    return gaps.loc[eligible].copy().reset_index(drop=True), methods, target_columns


def load_weather_stations(
    data_dir: Path,
    station_ids: Sequence[str],
    config: WeatherBenchmarkConfig,
) -> dict[str, pd.DataFrame]:
    """Load the exact DWD stations referenced by the benchmark table."""
    processed_dir = Path(data_dir) / "processed" / "temperature_2000_2025"
    paths = {
        path.stem.split("_")[-1]: path
        for path in processed_dir.glob("produkt_tu_stunde_*.txt")
    }
    requested = set(station_ids)
    missing = sorted(requested - set(paths))
    if missing:
        raise FileNotFoundError(f"Processed Weather files are missing stations: {missing}")
    return {
        station_id: load_dwd_station(
            paths[station_id],
            start=config.start,
            end=config.end,
            source_file="",
        )
        for station_id in sorted(requested)
    }


def candidate_from_gap_row(row: pd.Series) -> GapCandidate:
    """Reconstruct one artificial-gap geometry from its portable manifest row."""
    left_fraction = float(row["left_context_valid_fraction"])
    right_fraction = float(row["right_context_valid_fraction"])
    length = int(row["gap_length_samples"])
    return GapCandidate(
        gap_start_idx=int(row["gap_start_idx"]),
        gap_end_idx=int(row["gap_end_idx"]),
        requested_gap_duration_ms=length * HOUR_MS,
        realized_gap_duration_ms=length * HOUR_MS,
        sampling_rate_hz=1 / 3600,
        gap_length_samples=length,
        left_context_start_idx=int(row["left_context_start_idx"]),
        right_context_end_idx=int(row["right_context_end_idx"]),
        left_context_valid_fraction=left_fraction,
        right_context_valid_fraction=right_fraction,
        context_valid_fraction=(left_fraction + right_fraction) / 2,
        duration_stratum=int(row["duration_stratum"]),
    )


def weather_feature_row(
    row: pd.Series,
    stations: Mapping[str, pd.DataFrame],
    feature_scale_floor: float,
) -> dict[str, float]:
    """Extract observable selector features from only the predefined local window."""
    candidate = candidate_from_gap_row(row)
    station = stations[str(row["station_id"])]
    window = station.iloc[
        candidate.left_context_start_idx:candidate.right_context_end_idx
    ].reset_index(drop=True).copy()
    local_candidate = replace(
        candidate,
        gap_start_idx=candidate.gap_start_idx - candidate.left_context_start_idx,
        gap_end_idx=candidate.gap_end_idx - candidate.left_context_start_idx,
        left_context_start_idx=0,
        right_context_end_idx=len(window),
    )
    features = extract_basic_gap_features(
        create_artificial_gap(window, local_candidate),
        feature_scale_floor,
    )
    result = {name: features[name] for name in FEATURES_BASIC_NORMALIZED}
    result["realized_gap_duration_hours"] = (
        result.pop("realized_gap_duration_ms") / HOUR_MS
    )
    return result


def weather_feature_frame(
    gaps: pd.DataFrame,
    stations: Mapping[str, pd.DataFrame],
    feature_scale_floor: float,
) -> pd.DataFrame:
    """Return model features in the frozen Weather feature order."""
    rows = [
        weather_feature_row(row, stations, feature_scale_floor)
        for _, row in gaps.iterrows()
    ]
    return pd.DataFrame(rows, index=gaps.index).loc[:, WEATHER_FEATURE_COLUMNS]


def learn_feature_scale_floor(gaps: pd.DataFrame, quantile: float) -> float:
    """Learn the positive local-context-IQR floor from the supplied training rows."""
    values = pd.to_numeric(gaps["local_context_iqr"], errors="coerce").to_numpy(float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        raise ValueError("No positive training local-context IQR is available.")
    return float(np.quantile(values, quantile))


def model_parameters_from_row(row: pd.Series) -> dict[str, object]:
    """Restore typed scikit-learn parameters after CSV round-tripping."""
    max_features = row["max_features"]
    if isinstance(max_features, str):
        max_features = max_features.strip()
        if max_features not in {"sqrt", "log2"}:
            max_features = float(max_features)
    else:
        max_features = float(max_features)
    return {
        "n_estimators": int(row["n_estimators"]),
        "min_samples_leaf": int(row["min_samples_leaf"]),
        "max_depth": None if pd.isna(row["max_depth"]) else int(row["max_depth"]),
        "max_features": max_features,
    }
