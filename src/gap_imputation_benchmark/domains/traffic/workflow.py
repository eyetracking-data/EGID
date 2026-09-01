"""Named building blocks shared by Traffic benchmark, LODO, and final training."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from gap_imputation_benchmark.benchmark.features import extract_basic_gap_features
from gap_imputation_benchmark.benchmark.gaps import GapCandidate, create_artificial_gap
from gap_imputation_benchmark.benchmark.records import FEATURES_BASIC_NORMALIZED
from gap_imputation_benchmark.domains.traffic.adapter import TRAFFIC_FEATURE_COLUMNS, TrafficBenchmarkConfig
from gap_imputation_benchmark.domains.traffic.loaders import require_tables
from gap_imputation_benchmark.paths import load_local_environment


SAMPLE_MS = 300_000.0
WEEK_STEPS = 2_016


def resolve_traffic_data_dir(value: Path | None) -> Path:
    """Resolve the ignored local Traffic data root without recording its path."""
    load_local_environment()
    if value is not None:
        return Path(value).resolve()
    configured = os.getenv("TRAFFIC_DATA_DIR")
    if not configured:
        raise RuntimeError("Set TRAFFIC_DATA_DIR or supply --data-dir; see data/traffic.md.")
    return Path(configured).resolve()


def stable_rng(random_state: int, *parts: object) -> np.random.Generator:
    """Match the historical content-addressed random-number streams exactly."""
    text = "|".join(map(str, (random_state, *parts)))
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def read_panel_columns(flow, row_start: int, row_stop: int, panel_columns: Sequence[int]) -> np.ndarray:
    """Read selected panel columns in weekly chunks, matching the reference I/O order."""
    columns = np.asarray(panel_columns, dtype=np.int64)
    values = np.empty((row_stop - row_start, len(columns)), dtype=np.float32)
    destination = 0
    for source_start in range(row_start, row_stop, WEEK_STEPS):
        source_stop = min(source_start + WEEK_STEPS, row_stop)
        block = flow[source_start:source_stop, :][:, columns]
        values[destination:destination + len(block)] = block
        destination += len(block)
    return values


def sample_sensor_year_gaps(
    values: np.ndarray,
    *,
    district: int,
    year: int,
    sensor_id2: int,
    panel_column: int,
    config: TrafficBenchmarkConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Sample one non-overlapping fully observed gap per Traffic duration stratum."""
    valid = np.isfinite(values)
    prefix = np.r_[0, np.cumsum(valid, dtype=np.int64)]
    rng = stable_rng(config.random_state, "gap-positions", district, year, sensor_id2)
    selected_windows: list[tuple[int, int]] = []
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for stratum, (lower, upper) in enumerate(config.gap_strata_steps, start=1):
        gap_steps = int(rng.integers(lower, upper + 1))
        context_steps = max(gap_steps, config.min_context_steps)
        last_start = len(values) - gap_steps - context_steps
        if last_start < context_steps:
            skipped.append({"district": district, "year": year, "sensor_id2": sensor_id2, "duration_stratum": stratum, "reason": "Series too short for gap and context."})
            continue
        for attempt in range(config.max_gap_attempts):
            start = int(rng.integers(context_steps, last_start + 1))
            end, left_start, right_end = start + gap_steps, start - context_steps, start + gap_steps + context_steps
            count_valid = lambda lower_index, upper_index: int(prefix[upper_index] - prefix[lower_index])
            if count_valid(start, end) != gap_steps or not (valid[start - 1] and valid[end]):
                continue
            left_fraction = count_valid(left_start, start) / context_steps
            right_fraction = count_valid(end, right_end) / context_steps
            if left_fraction < config.min_context_valid_fraction or right_fraction < config.min_context_valid_fraction:
                continue
            if any(left_start < old_right and right_end > old_left for old_left, old_right in selected_windows):
                continue
            selected_windows.append((left_start, right_end))
            rows.append({"district": district, "year": year, "sensor_id2": sensor_id2, "panel_column": panel_column, "duration_stratum": stratum, "gap_start_in_year": start, "gap_end_in_year": end, "gap_length_steps": gap_steps, "gap_duration_minutes": gap_steps * 5, "left_context_start_in_year": left_start, "right_context_end_in_year": right_end, "left_context_valid_fraction": left_fraction, "right_context_valid_fraction": right_fraction})
            break
        else:
            skipped.append({"district": district, "year": year, "sensor_id2": sensor_id2, "duration_stratum": stratum, "reason": f"No eligible non-overlapping position in {config.max_gap_attempts} attempts."})
    return rows, skipped


def create_gap_manifest(panel_dir: Path, config: TrafficBenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the reference sampling manifest from an already selected HDF5 panel."""
    tables = require_tables()
    panel_dir = Path(panel_dir)
    time_manifest = pd.read_csv(panel_dir / "time_manifest.csv")
    all_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    unit_rows: list[dict[str, object]] = []
    for district in config.districts:
        metadata = pd.read_csv(panel_dir / f"district_{district:02d}_selected_sensor_metadata.csv").sort_values("ID2").reset_index(drop=True)
        with tables.open_file(panel_dir / f"district_{district:02d}_flow_panel.h5", mode="r") as handle:
            flow = handle.root.flow
            for year in config.years:
                year_row = time_manifest.loc[(time_manifest["district"] == district) & (time_manifest["year"] == year)].iloc[0]
                columns = np.sort(stable_rng(config.random_state, "sensor-units", district, year).choice(len(metadata), size=config.sensors_per_district_year, replace=False))
                values = read_panel_columns(flow, int(year_row.row_start), int(year_row.row_stop_exclusive), columns)
                for local_column, column in enumerate(columns):
                    sensor_id2 = int(metadata.iloc[int(column)]["ID2"])
                    gaps, skipped = sample_sensor_year_gaps(values[:, local_column], district=district, year=year, sensor_id2=sensor_id2, panel_column=int(column), config=config)
                    all_rows.extend(gaps)
                    skipped_rows.extend(skipped)
                    unit_rows.append({"district": district, "year": year, "sensor_id2": sensor_id2, "panel_column": int(column), "n_gaps_written": len(gaps)})
    gaps = pd.DataFrame(all_rows).sort_values(["district", "year", "sensor_id2", "duration_stratum"]).reset_index(drop=True)
    gaps.insert(0, "gap_id", [f"traffic_{index:06d}" for index in range(1, len(gaps) + 1)])
    expected = pd.MultiIndex.from_product([config.districts, config.years, range(1, len(config.gap_strata_steps) + 1)], names=["district", "year", "duration_stratum"]).to_frame(index=False)
    observed = gaps.groupby(["district", "year", "duration_stratum"]).size().rename("n_gaps_written").reset_index()
    coverage = expected.merge(observed, how="left").fillna({"n_gaps_written": 0})
    coverage["n_gaps_written"] = coverage["n_gaps_written"].astype(int)
    coverage["n_gaps_requested"] = config.sensors_per_district_year
    coverage["shortfall"] = coverage["n_gaps_requested"] - coverage["n_gaps_written"]
    return gaps, pd.DataFrame(skipped_rows), pd.DataFrame(unit_rows), coverage


def candidate_from_gap_row(row: pd.Series, *, year_start: int, local: bool = False) -> GapCandidate:
    """Reconstruct the exact local artificial-gap geometry from a manifest row."""
    start = int(row["gap_start_in_year"])
    end = int(row["gap_end_in_year"])
    left = int(row["left_context_start_in_year"])
    right = int(row["right_context_end_in_year"])
    if not local:
        start, end, left, right = (year_start + value for value in (start, end, left, right))
    length = int(row["gap_length_steps"])
    left_fraction, right_fraction = float(row["left_context_valid_fraction"]), float(row["right_context_valid_fraction"])
    return GapCandidate(start, end, length * SAMPLE_MS, length * SAMPLE_MS, 1 / 300, length, left, right, left_fraction, right_fraction, (left_fraction + right_fraction) / 2, int(row["duration_stratum"]))


def compact_gap_for_evaluation(frame: pd.DataFrame, candidate: GapCandidate):
    """Keep local observations and every weekly reference needed by the seasonal method."""
    references: list[int] = []
    for offset in range(1, 9):
        for direction in (-1, 1):
            start = candidate.gap_start_idx + direction * offset * WEEK_STEPS
            end = start + candidate.gap_length_samples
            if start >= 0 and end <= len(frame):
                references.extend(range(start, end))
    positions = np.unique(np.r_[np.arange(candidate.left_context_start_idx, candidate.right_context_end_idx), np.asarray(references, dtype=int)])
    mapping = {int(original): index for index, original in enumerate(positions)}
    compact = frame.iloc[positions].reset_index(drop=True).copy()
    local_candidate = replace(candidate, gap_start_idx=mapping[candidate.gap_start_idx], gap_end_idx=mapping[candidate.gap_end_idx - 1] + 1, left_context_start_idx=mapping[candidate.left_context_start_idx], right_context_end_idx=mapping[candidate.right_context_end_idx - 1] + 1)
    return create_artificial_gap(compact, local_candidate)


def compact_traffic_series_for_evaluation(
    series: np.ndarray,
    candidate: GapCandidate,
    *,
    district: int,
    sensor_id2: int,
    panel_column: int,
):
    """Build only the rows needed for one Traffic-gap evaluation.

    The original implementation first materialised an entire 525,888-row
    sensor DataFrame and then retained this same compact subset.  Constructing
    the subset directly preserves the protocol while substantially reducing
    per-sensor memory allocation and pandas indexing work.
    """
    references: list[int] = []
    for offset in range(1, 9):
        for direction in (-1, 1):
            start = candidate.gap_start_idx + direction * offset * WEEK_STEPS
            end = start + candidate.gap_length_samples
            if start >= 0 and end <= len(series):
                references.extend(range(start, end))
    positions = np.unique(
        np.r_[
            np.arange(candidate.left_context_start_idx, candidate.right_context_end_idx),
            np.asarray(references, dtype=int),
        ]
    )
    mapping = {int(original): index for index, original in enumerate(positions)}
    values = np.asarray(series[positions], dtype=float)
    compact = pd.DataFrame(
        {
            "gaze_x": values,
            "is_valid": np.isfinite(values),
            "timestamp": pd.Timestamp("2017-01-01") + pd.to_timedelta(positions * 5, unit="min"),
            "timestamp_ms": positions.astype(np.int64) * int(SAMPLE_MS),
            "dataset_id": f"PeMS_D{district}",
            "participant_id": str(sensor_id2),
            "recording_id": f"D{district}_{panel_column}",
            "session_id": "2017_2021",
            "source_file": f"interim/largest_flow_panel_selected/district_{district:02d}_flow_panel.h5",
        }
    )
    local_candidate = replace(
        candidate,
        gap_start_idx=mapping[candidate.gap_start_idx],
        gap_end_idx=mapping[candidate.gap_end_idx - 1] + 1,
        left_context_start_idx=mapping[candidate.left_context_start_idx],
        right_context_end_idx=mapping[candidate.right_context_end_idx - 1] + 1,
    )
    return create_artificial_gap(compact, local_candidate)


def learn_feature_scale_floor(gaps: pd.DataFrame, quantile: float) -> float:
    """Learn a fold-local positive IQR floor using training gaps only."""
    values = pd.to_numeric(gaps["local_context_iqr"], errors="coerce").to_numpy(float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        raise ValueError("No positive training local-context IQR is available.")
    return float(np.quantile(values, quantile))


class TrafficPanel:
    """Lazily opened selected panels with explicit closure after each workflow."""

    def __init__(self, panel_dir: Path):
        self.panel_dir = Path(panel_dir)
        self.year_starts = {(int(row.district), int(row.year)): int(row.row_start) for _, row in pd.read_csv(self.panel_dir / "time_manifest.csv").iterrows()}
        self._handles: dict[int, object] = {}

    def values(self, district: int, start: int, end: int, column: int) -> np.ndarray:
        tables = require_tables()
        if district not in self._handles:
            self._handles[district] = tables.open_file(self.panel_dir / f"district_{district:02d}_flow_panel.h5", mode="r")
        return np.asarray(self._handles[district].root.flow[start:end, column], dtype=float)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def traffic_feature_row(row: pd.Series, panel: TrafficPanel, feature_scale_floor: float) -> dict[str, float]:
    """Build observable local features for one gap from the external panel."""
    district, year = int(row.district), int(row.year)
    year_start = panel.year_starts[(district, year)]
    start, end = year_start + int(row.left_context_start_in_year), year_start + int(row.right_context_end_in_year)
    values = panel.values(district, start, end, int(row.panel_column))
    local_row = row.copy()
    local_candidate = candidate_from_gap_row(local_row, year_start=0, local=True)
    local_candidate = replace(local_candidate, gap_start_idx=local_candidate.gap_start_idx - int(row.left_context_start_in_year), gap_end_idx=local_candidate.gap_end_idx - int(row.left_context_start_in_year), left_context_start_idx=0, right_context_end_idx=len(values))
    frame = pd.DataFrame({"gaze_x": values, "is_valid": np.isfinite(values), "timestamp_ms": np.arange(start, end, dtype=float) * SAMPLE_MS})
    features = extract_basic_gap_features(create_artificial_gap(frame, local_candidate), feature_scale_floor)
    result = {name: features[name] for name in FEATURES_BASIC_NORMALIZED}
    result["realized_gap_duration_minutes"] = result.pop("realized_gap_duration_ms") / 60_000
    return result


def traffic_feature_frame(gaps: pd.DataFrame, panel: TrafficPanel, feature_scale_floor: float) -> pd.DataFrame:
    """Return Traffic selector inputs in the frozen column order."""
    return pd.DataFrame([traffic_feature_row(row, panel, feature_scale_floor) for _, row in gaps.iterrows()], index=gaps.index).loc[:, TRAFFIC_FEATURE_COLUMNS]


def model_parameters_from_row(row: pd.Series) -> dict[str, object]:
    """Restore typed Random-Forest parameters after CSV serialization."""
    max_features = row["max_features"]
    if isinstance(max_features, str):
        max_features = max_features.strip()
        if max_features not in {"sqrt", "log2"}:
            max_features = float(max_features)
    else:
        max_features = float(max_features)
    return {"n_estimators": int(row["n_estimators"]), "min_samples_leaf": int(row["min_samples_leaf"]), "max_depth": None if pd.isna(row["max_depth"]) else int(row["max_depth"]), "max_features": max_features}
