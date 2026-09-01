"""External LargeST archive access and portable selected-panel preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


INTERVAL_MINUTES = 5
ROW_CHUNK_SIZE = 288


def require_tables():
    """Import PyTables only when Traffic data are actually accessed."""
    try:
        import tables
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError("Traffic workflows require PyTables. Install with: pip install -e '.[traffic]'") from error
    return tables


def raw_files(raw_dir: Path, years: tuple[int, ...]) -> list[Path]:
    """Return the exact annual files in protocol year order."""
    paths = [Path(raw_dir) / f"ca_his_raw_{year}.h5" for year in years]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing LargeST annual files: {missing}")
    return paths


def flow_matrix_path(path: Path) -> str:
    """Identify the single 2D flow node in a LargeST annual archive."""
    tables = require_tables()
    with tables.open_file(path, mode="r") as handle:
        nodes = [node for node in handle.walk_nodes("/t") if getattr(node, "ndim", None) == 2]
        if len(nodes) != 1:
            labels = [node._v_pathname for node in nodes]
            raise ValueError(f"Expected one 2D flow matrix in {path.name}, found {labels}")
        return str(nodes[0]._v_pathname)


def audit_raw_flow(
    raw_dir: Path,
    *,
    districts: tuple[int, ...],
    years: tuple[int, ...],
) -> pd.DataFrame:
    """Audit coverage and natural missing runs without loading a full matrix."""
    tables = require_tables()
    raw_dir = Path(raw_dir)
    metadata = pd.read_csv(raw_dir / "ca_meta.csv")
    required = {"ID2", "ID", "District", "County", "Fwy"}
    if missing := required - set(metadata):
        raise ValueError(f"Unexpected ca_meta.csv schema; missing {sorted(missing)}")
    if metadata["ID2"].duplicated().any() or set(metadata["ID2"]) != set(range(len(metadata))):
        raise ValueError("ID2 must be a unique zero-based flow-matrix mapping.")
    selected = metadata.loc[metadata["District"].isin(districts)].sort_values("ID2").reset_index(drop=True)
    columns = selected["ID2"].to_numpy(dtype=np.int64)
    n_sensors = len(selected)
    valid_count = np.zeros(n_sensors, dtype=np.int64)
    invalid_count = np.zeros(n_sensors, dtype=np.int64)
    zero_count = np.zeros(n_sensors, dtype=np.int64)
    minimum = np.full(n_sensors, np.inf)
    maximum = np.full(n_sensors, -np.inf)
    active = np.zeros(n_sensors, dtype=np.int64)
    shortest = np.full(n_sensors, np.iinfo(np.int64).max, dtype=np.int64)
    longest = np.zeros(n_sensors, dtype=np.int64)
    n_gaps = np.zeros(n_sensors, dtype=np.int64)
    total_rows = 0
    paths = raw_files(raw_dir, years)
    matrix_path = flow_matrix_path(paths[0])
    for path in paths:
        with tables.open_file(path, mode="r") as handle:
            matrix = handle.get_node(matrix_path)
            if matrix.shape[1] != len(metadata):
                raise ValueError(f"Unexpected matrix width in {path.name}")
            for start in range(0, matrix.shape[0], ROW_CHUNK_SIZE):
                values = matrix[start:min(start + ROW_CHUNK_SIZE, matrix.shape[0]), :][:, columns]
                valid = np.isfinite(values) & (values >= 0)
                invalid = ~valid
                valid_count += valid.sum(axis=0, dtype=np.int64)
                invalid_count += invalid.sum(axis=0, dtype=np.int64)
                zero_count += ((values == 0) & valid).sum(axis=0, dtype=np.int64)
                minimum = np.minimum(minimum, np.where(valid, values, np.inf).min(axis=0))
                maximum = np.maximum(maximum, np.where(valid, values, -np.inf).max(axis=0))
                for row in invalid:
                    active += row
                    longest = np.maximum(longest, active)
                    completed = ~row & (active > 0)
                    shortest[completed] = np.minimum(shortest[completed], active[completed])
                    n_gaps += completed
                    active[completed] = 0
            total_rows += matrix.shape[0]
    unfinished = active > 0
    shortest[unfinished] = np.minimum(shortest[unfinished], active[unfinished])
    n_gaps[unfinished] += 1
    audit = selected.loc[:, ["ID2", "ID", "District", "County", "Fwy"]].copy()
    audit["n_time_steps"] = total_rows
    audit["valid_observations"] = valid_count
    audit["invalid_observations"] = invalid_count
    audit["valid_coverage"] = valid_count / total_rows
    audit["zero_flow_share_of_valid"] = np.divide(zero_count, valid_count, out=np.full(n_sensors, np.nan), where=valid_count > 0)
    audit["minimum_valid_flow"] = np.where(np.isfinite(minimum), minimum, np.nan)
    audit["maximum_valid_flow"] = np.where(np.isfinite(maximum), maximum, np.nan)
    audit["n_natural_gaps"] = n_gaps
    audit["smallest_natural_gap_minutes"] = np.where(shortest == np.iinfo(np.int64).max, np.nan, shortest * INTERVAL_MINUTES)
    audit["largest_natural_gap_minutes"] = np.where(longest == 0, np.nan, longest * INTERVAL_MINUTES)
    return audit


def build_selected_panel(
    raw_dir: Path,
    output_dir: Path,
    audit: pd.DataFrame,
    *,
    districts: tuple[int, ...],
    years: tuple[int, ...],
) -> None:
    """Create selected HDF5 panels, preserving the reference row ordering."""
    tables = require_tables()
    selected = audit.loc[(audit["valid_coverage"] >= 0.95) & (audit["largest_natural_gap_minutes"] <= 24 * 60)].copy()
    expected_counts = {3: 409, 4: 2249, 7: 1787, 11: 632}
    counts = selected.groupby("District").size().to_dict()
    if tuple(districts) == (3, 4, 7, 11) and counts != expected_counts:
        raise ValueError(f"Unexpected selected-sensor counts {counts}; expected {expected_counts}.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = raw_files(raw_dir, years)
    matrix_path = flow_matrix_path(paths[0])
    manifest_rows, summary_rows = [], []
    filters = tables.Filters(complevel=5, complib="blosc:zstd", shuffle=True)
    for district in districts:
        final_path = output_dir / f"district_{district:02d}_flow_panel.h5"
        if final_path.exists():
            raise FileExistsError(f"Refusing to overwrite selected panel: {final_path}")
        sensors = selected.loc[selected["District"].eq(district)].sort_values("ID2").reset_index(drop=True)
        columns = sensors["ID2"].to_numpy(dtype=np.int64)
        expected_valid, expected_invalid = sensors["valid_observations"].to_numpy(), sensors["invalid_observations"].to_numpy()
        partial_path = output_dir / f"district_{district:02d}_flow_panel.partial.h5"
        if partial_path.exists():
            raise FileExistsError(f"Partial panel requires inspection: {partial_path}")
        valid_count, invalid_count = np.zeros(len(columns), dtype=np.int64), np.zeros(len(columns), dtype=np.int64)
        cursor = 0
        with tables.open_file(partial_path, mode="w") as destination:
            flow = destination.create_earray("/", "flow", tables.Float32Atom(), shape=(0, len(columns)), expectedrows=525_888, filters=filters)
            destination.root._v_attrs["district"] = district
            destination.root._v_attrs["interval_minutes"] = INTERVAL_MINUTES
            destination.root._v_attrs["value_semantics"] = "flow; invalid raw values encoded as NaN; zero is valid"
            destination.root._v_attrs["source_years"] = "2017-2021"
            for year, path in zip(years, paths, strict=True):
                with tables.open_file(path, mode="r") as source:
                    matrix = source.get_node(matrix_path)
                    start_cursor = cursor
                    for start in range(0, matrix.shape[0], ROW_CHUNK_SIZE):
                        block = matrix[start:min(start + ROW_CHUNK_SIZE, matrix.shape[0]), :][:, columns]
                        valid = np.isfinite(block) & (block >= 0)
                        valid_count += valid.sum(axis=0, dtype=np.int64)
                        invalid_count += (~valid).sum(axis=0, dtype=np.int64)
                        block = block.astype(np.float32, copy=False)
                        block[~valid] = np.nan
                        flow.append(block)
                    cursor += matrix.shape[0]
                    manifest_rows.append({"district": district, "year": year, "row_start": start_cursor, "row_stop_exclusive": cursor, "start_timestamp": f"{year}-01-01 00:00:00", "end_timestamp": (pd.Timestamp(f"{year}-01-01") + pd.Timedelta(minutes=5 * (matrix.shape[0] - 1))).isoformat()})
        if cursor != 525_888 or not np.array_equal(valid_count, expected_valid) or not np.array_equal(invalid_count, expected_invalid):
            raise RuntimeError(f"Audit mismatch during panel build for District {district}; retained {partial_path}.")
        partial_path.replace(final_path)
        sensors.to_csv(output_dir / f"district_{district:02d}_selected_sensor_metadata.csv", index=False)
        summary_rows.append({"district": district, "n_sensors": len(sensors), "n_time_steps": cursor, "valid_coverage": valid_count.sum() / (cursor * len(sensors)), "panel_path": f"interim/largest_flow_panel_selected/{final_path.name}"})
    pd.DataFrame(manifest_rows).to_csv(output_dir / "time_manifest.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "build_summary.csv", index=False)
