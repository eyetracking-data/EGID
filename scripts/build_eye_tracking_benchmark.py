#!/usr/bin/env python3
"""Build the reproducible four-dataset Eye-Tracking benchmark table.

The script implements notebook 01 without notebook-only state. It never
modifies raw data and writes only to an explicitly supplied output directory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Callable

import numpy as np
import pandas as pd

from gap_imputation_benchmark.domains.eyetracking import (
    EyeTrackingBenchmarkConfig,
    load_eye_tracking_benchmark_config,
    portable_source_reference,
    replace_source_reference,
    select_balanced_recordings,
)
from gap_imputation_benchmark.benchmark.gaps import (
    GapCandidate,
    create_artificial_gap,
    duration_ms_to_samples,
    find_gap_candidates,
)
from gap_imputation_benchmark.benchmark.provenance import recording_provenance
from gap_imputation_benchmark.benchmark.output_contract import (
    BENCHMARK_OUTPUTS,
    validate_output_contract,
)
from gap_imputation_benchmark.benchmark.records import (
    FEATURES_BASIC_NORMALIZED,
    build_gap_benchmark_record,
)
from gap_imputation_benchmark.benchmark.unit_robust import iqr, local_scale, recording_iqr_leave_gap_out
from gap_imputation_benchmark.domains.eyetracking.loaders import (
    load_gazebase_reading,
    load_gazebase_vr_reading,
    load_pedrotti_reading,
    load_zuco_normal_reading,
)
from gap_imputation_benchmark.imputers.registry import EYE_TRACKING_METHODS
from gap_imputation_benchmark.paths import PROJECT_ROOT, require_raw_data_dir


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "eye_tracking_final.toml"
METHODS = EYE_TRACKING_METHODS


@dataclass(frozen=True)
class SampledGap:
    """One accepted gap with its immutable reference-protocol numbering."""

    recording: pd.DataFrame
    candidate: GapCandidate
    requested_gap_number: int
    duration_stratum_lower_ms: float
    duration_stratum_upper_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source loading and balanced selection without sampling or writing files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _report_progress(stage: str, index: int, total: int, dataset_id: str) -> None:
    """Print compact progress updates that remain useful in terminals and notebooks."""
    interval = max(1, total // 20)
    if index == 1 or index == total or index % interval == 0:
        print(f"[{stage}] {dataset_id}: {index}/{total}", flush=True)


def _load_file(
    file_path: Path,
    raw_data_dir: Path,
    loader: Callable[[Path], pd.DataFrame],
) -> pd.DataFrame:
    recording = loader(file_path)
    return replace_source_reference(
        recording,
        portable_source_reference(file_path, raw_data_dir),
    )


def load_eye_tracking_sources(raw_data_dir: Path) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Load the four source datasets and retain a portable input manifest."""
    recordings: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    datasets: tuple[tuple[str, str, Callable[[Path], pd.DataFrame]], ...] = (
        ("GazeBase", "GazeBase_v2_0/**/S_*_TEX.csv", load_gazebase_reading),
        ("GazeBaseVR", "gazebasevr/data/S_*_TEX.csv", load_gazebase_vr_reading),
        ("ZuCo", "ZUCO/*/*_NR*_ET.mat", load_zuco_normal_reading),
    )
    for dataset_id, pattern, loader in datasets:
        paths = sorted(raw_data_dir.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No {dataset_id} files found with pattern: {pattern}")
        print(f"[load] {dataset_id}: {len(paths)} source files", flush=True)
        for index, file_path in enumerate(paths, start=1):
            _report_progress("load", index, len(paths), dataset_id)
            source_file = portable_source_reference(file_path, raw_data_dir)
            try:
                recording = _load_file(file_path, raw_data_dir, loader)
            except Exception as error:
                manifest_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "source_file": source_file,
                        "status": "excluded_loader_error",
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            recordings.append(recording)
            manifest_rows.append(
                {
                    "dataset_id": dataset_id,
                    "source_file": source_file,
                    "status": "loaded",
                    "reason": None,
                }
            )

    pedrotti_paths = sorted(raw_data_dir.glob("Pedrotti/[0-9][0-9].txt"))
    if not pedrotti_paths:
        raise FileNotFoundError("No Pedrotti files found below Pedrotti/[0-9][0-9].txt")
    print(f"[load] Pedrotti: {len(pedrotti_paths)} source files", flush=True)
    for index, file_path in enumerate(pedrotti_paths, start=1):
        _report_progress("load", index, len(pedrotti_paths), "Pedrotti")
        source_file = portable_source_reference(file_path, raw_data_dir)
        try:
            trials = load_pedrotti_reading(file_path)
        except Exception as error:
            manifest_rows.append(
                {
                    "dataset_id": "Pedrotti",
                    "source_file": source_file,
                    "status": "excluded_loader_error",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        for trial in trials:
            trial = replace_source_reference(trial, source_file)
            condition = str(trial["stimulus_condition"].iat[0])
            status = "excluded_number_stimulus" if condition == "number" else "loaded"
            manifest_rows.append(
                {
                    "dataset_id": "Pedrotti",
                    "source_file": source_file,
                    "recording_id": trial["recording_id"].iat[0],
                    "status": status,
                    "reason": None,
                }
            )
            if status == "loaded":
                recordings.append(trial)

    return recordings, pd.DataFrame(manifest_rows)


def selection_summary(recordings: list[pd.DataFrame]) -> pd.DataFrame:
    """Summarize the deterministic selected input before any gap is sampled."""
    rows = [
        {
            "dataset_id": recording["dataset_id"].iat[0],
            "participant_id": recording["participant_id"].iat[0],
            "recording_id": recording["recording_id"].iat[0],
        }
        for recording in recordings
    ]
    return pd.DataFrame(rows).groupby("dataset_id").agg(
        participants=("participant_id", "nunique"),
        recordings=("recording_id", "nunique"),
    ).reset_index()


def planned_gap_summary(
    recordings: list[pd.DataFrame],
    config: EyeTrackingBenchmarkConfig,
) -> pd.DataFrame:
    """Report requested gaps before expensive sampling begins."""
    rows: list[dict[str, object]] = []
    for dataset_id, frame in selection_summary(recordings).set_index("dataset_id").iterrows():
        requested_gaps = (
            int(frame["participants"]) * config.pedrotti_gaps_per_participant
            if dataset_id == "Pedrotti"
            else int(frame["recordings"]) * config.n_gaps_per_recording
        )
        rows.append({"dataset_id": dataset_id, "requested_gaps": requested_gaps})
    return pd.DataFrame(rows)


def _pedrotti_strata(config: EyeTrackingBenchmarkConfig, participant_id: str) -> list[int]:
    """Reproduce the original balanced Pedrotti stratum assignment exactly."""
    base, remainder = divmod(config.pedrotti_gaps_per_participant, config.n_strata)
    strata = [index for index in range(1, config.n_strata + 1) for _ in range(base)]
    stratum_rng = random.Random(f"{config.random_state}|Pedrotti-strata|{participant_id}")
    strata.extend(index + 1 for index in stratum_rng.sample(range(config.n_strata), remainder))
    random.Random(f"{config.random_state}|Pedrotti-task-order|{participant_id}").shuffle(strata)
    return strata


def _reference_recording_seed(recording: pd.DataFrame, random_state: int) -> int:
    """Return the exact recording seed used by the original benchmark notebook."""
    text = "|".join(
        (
            str(random_state),
            str(recording["dataset_id"].iat[0]),
            str(recording["recording_id"].iat[0]),
        )
    )
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _sampling_exclusion(
    recording: pd.DataFrame,
    requested_gap_number: int,
    stratum: int,
    lower: float,
    upper: float,
    requested_duration_ms: float,
    attempts_used: int,
) -> dict[str, object]:
    """Describe a reference-protocol shortfall without changing the sample."""
    return {
        "dataset_id": recording["dataset_id"].iat[0],
        "participant_id": recording["participant_id"].iat[0],
        "recording_id": recording["recording_id"].iat[0],
        "source_file": recording["source_file"].iat[0],
        "requested_gap_number": requested_gap_number,
        "duration_stratum": stratum,
        "duration_stratum_lower_ms": lower,
        "duration_stratum_upper_ms": upper,
        "requested_gap_duration_ms": requested_duration_ms,
        "attempts_used": attempts_used,
        "reason": "no_eligible_nonoverlapping_position_in_duration_stratum",
    }


def _sample_reference_recording_gaps(
    recording: pd.DataFrame,
    config: EyeTrackingBenchmarkConfig,
    target_strata: list[int] | None = None,
) -> tuple[list[SampledGap], list[dict[str, object]]]:
    """Use the original notebook's sampling protocol, including shortfalls.

    A sampling shortfall is an observed property of the reference benchmark,
    not a reason to replace a recording or alter a duration draw. It is written
    to the exclusion table and subsequent requested gaps retain their original
    numbering.
    """
    rng = random.Random(_reference_recording_seed(recording, config.random_state))
    selected: list[SampledGap] = []
    skipped: list[dict[str, object]] = []
    selected_windows: list[GapCandidate] = []
    boundaries = np.linspace(
        config.min_gap_duration_ms,
        config.max_gap_duration_ms,
        config.n_strata + 1,
    )

    if target_strata is None:
        target_counts = [config.n_gaps_per_recording // config.n_strata] * config.n_strata
        for index in rng.sample(range(config.n_strata), config.n_gaps_per_recording % config.n_strata):
            target_counts[index] += 1
    else:
        if any(index not in range(1, config.n_strata + 1) for index in target_strata):
            raise ValueError("target_strata contains an invalid stratum index.")
        target_counts = [target_strata.count(index) for index in range(1, config.n_strata + 1)]

    requested_gap_number = 0
    sampling_rate_hz = float(recording["sampling_rate_hz"].iat[0])
    for stratum, target_count in enumerate(target_counts, start=1):
        lower, upper = float(boundaries[stratum - 1]), float(boundaries[stratum])
        for _ in range(target_count):
            requested_gap_number += 1
            accepted: GapCandidate | None = None
            requested_duration_ms = float("nan")
            attempts_used = 0
            for attempts_used in range(1, config.max_gap_sampling_attempts + 1):
                upper_draw = np.nextafter(upper, lower) if stratum < config.n_strata else upper
                requested_duration_ms = max(float(np.finfo(float).eps), rng.uniform(lower, upper_draw))
                requested_gap_samples = duration_ms_to_samples(requested_duration_ms, sampling_rate_hz)
                context_multiplier = 2.0 if requested_gap_samples == 1 else 1.0
                candidates = find_gap_candidates(
                    recording,
                    requested_duration_ms,
                    context_multiplier=context_multiplier,
                    min_context_valid_fraction=config.min_context_valid_fraction,
                )
                candidates = [
                    candidate
                    for candidate in candidates
                    if all(
                        candidate.right_context_end_idx <= chosen.left_context_start_idx
                        or chosen.right_context_end_idx <= candidate.left_context_start_idx
                        for chosen in selected_windows
                    )
                ]
                if candidates:
                    accepted = rng.choice(candidates)
                    break

            if accepted is None:
                skipped.append(
                    _sampling_exclusion(
                        recording,
                        requested_gap_number,
                        stratum,
                        lower,
                        upper,
                        requested_duration_ms,
                        attempts_used,
                    )
                )
                continue

            selected_windows.append(accepted)
            selected.append(
                SampledGap(
                    recording=recording,
                    candidate=replace(accepted, duration_stratum=stratum),
                    requested_gap_number=requested_gap_number,
                    duration_stratum_lower_ms=lower,
                    duration_stratum_upper_ms=upper,
                )
            )
    return selected, skipped


def sample_benchmark_candidates(
    recordings: list[pd.DataFrame],
    config: EyeTrackingBenchmarkConfig,
) -> tuple[list[SampledGap], list[dict[str, object]]]:
    """Sample gaps in the exact task order of the reference notebook."""
    tasks: list[tuple[pd.DataFrame, list[int] | None]] = []
    for dataset_id in sorted({str(recording["dataset_id"].iat[0]) for recording in recordings}):
        dataset_recordings = [
            recording for recording in recordings if recording["dataset_id"].iat[0] == dataset_id
        ]
        if dataset_id != "Pedrotti":
            tasks.extend((recording, None) for recording in dataset_recordings)
            continue
        for participant_id in sorted(
            {str(recording["participant_id"].iat[0]) for recording in dataset_recordings}
        ):
            participant_recordings = [
                recording
                for recording in dataset_recordings
                if str(recording["participant_id"].iat[0]) == participant_id
            ]
            strata = _pedrotti_strata(config, participant_id)
            if len(participant_recordings) != len(strata):
                raise ValueError("Selected Pedrotti trials and requested gaps must have identical counts.")
            tasks.extend(zip(participant_recordings, ([stratum] for stratum in strata), strict=True))

    sampled: list[SampledGap] = []
    skipped: list[dict[str, object]] = []
    for index, (recording, target_strata) in enumerate(tasks, start=1):
        _report_progress("sample", index, len(tasks), str(recording["dataset_id"].iat[0]))
        accepted, shortfalls = _sample_reference_recording_gaps(recording, config, target_strata)
        sampled.extend(accepted)
        skipped.extend(shortfalls)
    return sampled, skipped


def _feature_scale(artificial_gap: object) -> float:
    recording = artificial_gap.masked_recording
    candidate = artificial_gap.candidate
    context_indices = np.r_[
        candidate.left_context_start_idx:candidate.gap_start_idx,
        candidate.gap_end_idx:candidate.right_context_end_idx,
    ]
    gaze_x = recording["gaze_x"].to_numpy(dtype=float, na_value=float("nan"))
    is_valid = recording["is_valid"].to_numpy(dtype=bool)
    context_values = gaze_x[context_indices]
    context_valid = is_valid[context_indices] & np.isfinite(context_values)
    local_context_iqr = iqr(context_values[context_valid])
    recording_iqr = recording_iqr_leave_gap_out(
        gaze_x,
        is_valid,
        candidate.gap_start_idx,
        candidate.gap_end_idx,
    )
    scale = local_scale(local_context_iqr, recording_iqr)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The gap has no positive finite local normalization scale.")
    return scale


def _record_id(sample: SampledGap) -> str:
    """Keep gap identifiers compatible with the reference benchmark table."""
    return f"{sample.recording['recording_id'].iat[0]}_gap_{sample.requested_gap_number:03d}"


def build_records(
    sampled: list[SampledGap],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate each sampled gap and separate valid learning rows from exclusions."""
    records: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    targets = [f"{method}_nrmse" for method in METHODS]
    for index, sample in enumerate(sampled, start=1):
        recording = sample.recording
        candidate = sample.candidate
        _report_progress("evaluate", index, len(sampled), str(recording["dataset_id"].iat[0]))
        try:
            artificial_gap = create_artificial_gap(recording, candidate)
            record = build_gap_benchmark_record(artificial_gap, scale_floor=_feature_scale(artificial_gap))
            record["gap_id"] = _record_id(sample)
            record["duration_stratum_lower_ms"] = sample.duration_stratum_lower_ms
            record["duration_stratum_upper_ms"] = sample.duration_stratum_upper_ms
            records.append(record)
        except Exception as error:
            exclusions.append(
                {
                    "dataset_id": recording["dataset_id"].iat[0],
                    "participant_id": recording["participant_id"].iat[0],
                    "recording_id": recording["recording_id"].iat[0],
                    "source_file": recording["source_file"].iat[0],
                    "gap_start_idx": candidate.gap_start_idx,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    all_records = pd.DataFrame(records)
    if all_records.empty:
        return all_records, pd.DataFrame(exclusions)
    required = ["gap_id", "dataset_id", "participant_id", "local_iqr", "scale_i", *FEATURES_BASIC_NORMALIZED, *targets]
    valid = np.isfinite(all_records.loc[:, required[3:]].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1)
    for _, row in all_records.loc[~valid].iterrows():
        exclusions.append(
            {
                "dataset_id": row["dataset_id"],
                "participant_id": row["participant_id"],
                "recording_id": row["recording_id"],
                "source_file": row["source_file"],
                "gap_start_idx": row["gap_start_idx"],
                "reason": "non_finite_feature_or_method_target",
            }
        )
    return all_records.loc[valid].reset_index(drop=True), pd.DataFrame(exclusions)


def main() -> None:
    args = parse_args()
    config = load_eye_tracking_benchmark_config(args.config)
    if not args.dry_run and args.output_dir is None:
        raise ValueError("--output-dir is required unless --dry-run is used.")
    output_dir = args.output_dir.resolve() if args.output_dir is not None else None
    if output_dir is not None and output_dir.exists() and not args.overwrite:
        raise FileExistsError("The configured output directory already exists.")

    raw_data_dir = require_raw_data_dir() / "raw"
    source_recordings, input_manifest = load_eye_tracking_sources(raw_data_dir)
    selected = select_balanced_recordings(source_recordings, config)
    print("[select] balanced input sample", flush=True)
    print(selection_summary(selected).to_string(index=False), flush=True)
    print("[select] requested artificial gaps", flush=True)
    print(planned_gap_summary(selected, config).to_string(index=False), flush=True)
    if args.dry_run:
        print("[dry-run] Source loading and balanced selection completed; no files were written.")
        return

    sampled, sampling_exclusions = sample_benchmark_candidates(selected, config)
    learnable, record_exclusions = build_records(sampled)
    exclusions = pd.concat(
        [pd.DataFrame(sampling_exclusions), record_exclusions],
        ignore_index=True,
        sort=False,
    )
    if learnable.empty:
        raise RuntimeError("No learnable gaps were generated; no output was written.")

    if output_dir is None:  # Defensive narrowing for static type checkers.
        raise AssertionError("output_dir must be set after dry-run handling.")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    selected_rows = [
        {field.split("=", 1)[0]: field.split("=", 1)[1] for field in recording_provenance(recording)}
        for recording in selected
    ]
    summary = learnable.groupby("dataset_id").agg(
        gaps=("gap_id", "size"),
        participants=("participant_id", "nunique"),
        recordings=("recording_id", "nunique"),
    ).reset_index()
    requested = planned_gap_summary(selected, config)
    learnable_counts = learnable.groupby("dataset_id").size().rename("learnable_gaps").reset_index()
    excluded_counts = (
        exclusions.groupby("dataset_id").size().rename("excluded_gaps").reset_index()
        if not exclusions.empty
        else pd.DataFrame(columns=["dataset_id", "excluded_gaps"])
    )
    coverage = requested.merge(learnable_counts, on="dataset_id", how="left").merge(
        excluded_counts,
        on="dataset_id",
        how="left",
    )
    coverage[["learnable_gaps", "excluded_gaps"]] = coverage[
        ["learnable_gaps", "excluded_gaps"]
    ].fillna(0).astype(int)
    learnable.to_csv(output_dir / "learnable_gap_table.csv", index=False)
    input_manifest.to_csv(output_dir / "input_manifest.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output_dir / "selected_recordings.csv", index=False)
    exclusions.to_csv(output_dir / "excluded_gaps.csv", index=False)
    summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    coverage.to_csv(output_dir / "coverage_table.csv", index=False)
    metadata = {
        "artifact_type": "benchmark",
        "domain": "eye_tracking",
        "workflow": "eye_tracking_benchmark",
        "config": asdict(config),
        "random_state": config.random_state,
        "method_names": list(METHODS),
        "feature_columns": list(FEATURES_BASIC_NORMALIZED),
        "counts": {
            "requested_gaps": int(requested["requested_gaps"].sum()),
            "benchmark_rows": len(learnable),
            "sampling_exclusions": len(exclusions),
        },
        "method_configuration": {},
        "sampling_protocol": "reference_notebook_v1",
        "requested_gaps": int(requested["requested_gaps"].sum()),
        "sampled_gaps": len(sampled),
        "learnable_gaps": len(learnable),
        "excluded_gaps": len(exclusions),
        "coverage": coverage.to_dict(orient="records"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    validate_output_contract(output_dir, BENCHMARK_OUTPUTS)
    print(summary.to_string(index=False))
    print("\nWrote reproducible Eye-Tracking benchmark outputs.")


if __name__ == "__main__":
    main()
