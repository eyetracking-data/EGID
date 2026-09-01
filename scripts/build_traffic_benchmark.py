"""Build the frozen LargeST Traffic benchmark and eight-method outcome table.

The raw archive remains external.  By default this script reuses the prepared
selected panel under ``TRAFFIC_DATA_DIR/interim``.  ``--prepare-panel`` runs
the slower raw audit and panel build first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from gap_imputation_benchmark.benchmark.records import FEATURES_BASIC_NORMALIZED, build_gap_benchmark_record
from gap_imputation_benchmark.benchmark.unit_robust import iqr
from gap_imputation_benchmark.domains.traffic import (
    TRAFFIC_DOMAIN,
    TRAFFIC_FEATURE_COLUMNS,
    load_traffic_benchmark_config,
)
from gap_imputation_benchmark.domains.traffic.loaders import audit_raw_flow, build_selected_panel, require_tables
from gap_imputation_benchmark.domains.traffic.workflow import (
    TrafficPanel,
    candidate_from_gap_row,
    compact_traffic_series_for_evaluation,
    create_gap_manifest,
    resolve_traffic_data_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None, help="External directory with raw/ and interim/.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-panel", action="store_true", help="Audit raw archive and rebuild selected panel before sampling.")
    return parser.parse_args()


def prepare_panel_if_requested(data_dir: Path, config, prepare_panel: bool) -> Path:
    """Return an existing panel or deliberately recreate it from the raw archive."""
    panel_dir = data_dir / "interim" / "largest_flow_panel_selected"
    if not prepare_panel:
        required = [panel_dir / "time_manifest.csv", *(panel_dir / f"district_{district:02d}_flow_panel.h5" for district in config.districts)]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Selected panel is missing {missing}. Re-run with --prepare-panel.")
        return panel_dir
    audit = audit_raw_flow(data_dir / "raw", districts=config.districts, years=config.years)
    audit_dir = data_dir / "interim"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_dir / "largest_raw_flow_gap_audit_by_sensor.csv", index=False)
    build_selected_panel(data_dir / "raw", panel_dir, audit, districts=config.districts, years=config.years)
    return panel_dir


def _report_progress(completed: int, total: int, started_at: float) -> None:
    """Print periodic evaluation status that also works in notebook output."""
    interval = max(1, total // 100)
    if completed != total and completed % interval:
        return
    elapsed = monotonic() - started_at
    rate = completed / elapsed if elapsed else 0.0
    remaining = (total - completed) / rate if rate else float("inf")
    eta = "--" if not np.isfinite(remaining) else f"{remaining / 60:.1f} min"
    bar_width = 24
    filled = round(bar_width * completed / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(
        f"[evaluate] [{bar}] {completed:,}/{total:,} ({completed / total:.0%}) "
        f"| {rate:.2f} gaps/s | ETA {eta}",
        flush=True,
    )


def evaluate_methods(gaps: pd.DataFrame, panel_dir: Path) -> pd.DataFrame:
    """Evaluate every registered imputer using the historical compact windows."""
    panel = TrafficPanel(panel_dir)
    time_manifest = pd.read_csv(panel_dir / "time_manifest.csv")
    rows: list[dict[str, object]] = []
    total_gaps = len(gaps)
    completed = 0
    started_at = monotonic()
    print(f"[evaluate] [{'-' * 24}] 0/{total_gaps:,} (0%)", flush=True)
    try:
        for district in sorted(gaps["district"].unique()):
            metadata = pd.read_csv(panel_dir / f"district_{district:02d}_selected_sensor_metadata.csv").sort_values("ID2").reset_index(drop=True)
            district_gaps = gaps.loc[gaps["district"].eq(district)]
            for column, column_gaps in district_gaps.groupby("panel_column", sort=True):
                series = panel.values(int(district), 0, 525_888, int(column))
                for _, gap in column_gaps.iterrows():
                    year_row = time_manifest.loc[(time_manifest["district"] == district) & (time_manifest["year"] == gap.year)].iloc[0]
                    candidate = candidate_from_gap_row(gap, year_start=int(year_row.row_start))
                    artificial_gap = compact_traffic_series_for_evaluation(
                        series,
                        candidate,
                        district=int(district),
                        sensor_id2=int(metadata.iloc[int(column)].ID2),
                        panel_column=int(column),
                    )
                    record = build_gap_benchmark_record(artificial_gap, np.finfo(float).tiny, imputer_methods=TRAFFIC_DOMAIN.methods)
                    observed = np.concatenate((series[:candidate.gap_start_idx], series[candidate.gap_end_idx:]))
                    recording_iqr = iqr(observed[np.isfinite(observed)])
                    scale = max(record["local_context_iqr"], 0.05 * recording_iqr)
                    record["recording_iqr_leave_gap_out"] = recording_iqr
                    record["normalization_scale"] = scale
                    for method in TRAFFIC_DOMAIN.methods:
                        rmse = record[f"{method}_rmse"]
                        record[f"{method}_nrmse"] = rmse / scale if pd.notna(rmse) else np.nan
                    scores = {method: record[f"{method}_nrmse"] for method in TRAFFIC_DOMAIN.methods if np.isfinite(record[f"{method}_nrmse"])}
                    record["best_method_by_nrmse"] = min(scores, key=scores.get) if scores else None
                    record["best_nrmse"] = scores.get(record["best_method_by_nrmse"], np.nan)
                    record.update(gap.to_dict())
                    rows.append(record)
                    completed += 1
                    _report_progress(completed, total_gaps, started_at)
    finally:
        panel.close()
    return pd.DataFrame(rows).drop(columns=list(FEATURES_BASIC_NORMALIZED), errors="ignore")


def write_benchmark(output_dir: Path, *, skipped: pd.DataFrame, units: pd.DataFrame, coverage: pd.DataFrame, evaluated: pd.DataFrame, config) -> None:
    """Write portable tables and protocol metadata without local paths."""
    output_dir.mkdir(parents=True)
    units.to_csv(output_dir / "selected_recordings.csv", index=False)
    skipped.to_csv(output_dir / "excluded_gaps.csv", index=False)
    coverage.to_csv(output_dir / "coverage_table.csv", index=False)
    evaluated.to_csv(output_dir / "learnable_gap_table.csv", index=False)
    dataset_summary = coverage.groupby("district", as_index=False).agg(
        requested_gaps=("n_gaps_requested", "sum"),
        generated_gaps=("n_gaps_written", "sum"),
        shortfall=("shortfall", "sum"),
    )
    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    input_manifest = dataset_summary.assign(
        source_file=lambda frame: frame["district"].map(
            lambda district: f"interim/largest_flow_panel_selected/district_{district:02d}_flow_panel.h5"
        )
    )
    input_manifest.to_csv(output_dir / "input_manifest.csv", index=False)
    metadata = {
        "artifact_type": "benchmark",
        "domain": "traffic",
        "workflow": "traffic_benchmark",
        "sampling_unit": "sensor_x_calendar_year",
        "districts": list(config.districts),
        "years": list(config.years),
        "sensors_per_district_year": config.sensors_per_district_year,
        "gap_strata_steps_inclusive": [list(row) for row in config.gap_strata_steps],
        "interval_minutes": 5,
        "min_context_valid_fraction": config.min_context_valid_fraction,
        "context_policy": "max(gap_length_steps, 2) steps per side",
        "max_gap_attempts": config.max_gap_attempts,
        "random_state": config.random_state,
        "method_names": list(TRAFFIC_DOMAIN.methods),
        "feature_columns": list(TRAFFIC_FEATURE_COLUMNS),
        "counts": {
            "requested_gaps": int(coverage["n_gaps_requested"].sum()),
            "benchmark_rows": len(evaluated),
            "sampling_exclusions": int(coverage["shortfall"].sum()),
        },
        "method_configuration": {"seasonal_periodic": {"period_strategy": "weekly", "candidates_per_direction": 3, "max_offsets": 8, "min_candidates": 3, "aggregation": "median", "mode": "bidirectional"}},
        "nrmse_scale": "max(IQR(local_context), 0.05 * IQR(full_sensor_series_excluding_artificial_gap))",
        "generated_gaps": len(evaluated),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_traffic_benchmark_config(args.config)
    data_dir = resolve_traffic_data_dir(args.data_dir)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    require_tables()
    panel_dir = prepare_panel_if_requested(data_dir, config, args.prepare_panel)
    gaps, skipped, units, coverage = create_gap_manifest(panel_dir, config)
    evaluated = evaluate_methods(gaps, panel_dir)
    context_fractions = gaps.loc[:, [
        "gap_id",
        "left_context_valid_fraction",
        "right_context_valid_fraction",
    ]]
    evaluated = evaluated.merge(
        context_fractions,
        on="gap_id",
        how="left",
        validate="one_to_one",
    )
    write_benchmark(
        output_dir,
        skipped=skipped,
        units=units,
        coverage=coverage,
        evaluated=evaluated,
        config=config,
    )
    print(f"Traffic benchmark written: {output_dir.name} ({len(evaluated):,} evaluated gaps)")


if __name__ == "__main__":
    main()
