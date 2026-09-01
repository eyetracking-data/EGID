"""Public orchestrator for the generic three-stage preprocessing pipeline."""
from __future__ import annotations

import getpass
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
import numpy as np

from .provenance import associated_agents, file_metadata, write_json

Domain = Literal["eye_tracking", "weather", "traffic"]
OutlierMethod = Literal["none", "iqr", "zscore"]
StandardizationMethod = Literal["zscore", "robust", "minmax"] | None
TimestampUnit = Literal["s", "ms", "us"]

ALL_IMPUTATION_METHODS = (
    "forward_fill", "nearest_boundary", "linear", "pchip",
    "local_natural_cubic_spline", "polyfit_bic", "template", "seasonal_periodic",
)
RF_METHODS = ALL_IMPUTATION_METHODS
CONTEXT_MULTIPLIER = 1.0
MIN_CONTEXT_SAMPLES = 2
# The package lives below ``src/``; model artefacts remain in the portable
# repository-level ``artifacts/`` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DomainImputationConfig:
    """Settings that determine the processing of one numeric time series."""

    domain: Domain = "eye_tracking"
    sampling_rate_hz: float | None = None
    timestamp_col: str | None = None
    timestamp_unit: TimestampUnit = "ms"
    validity_col: str | None = None
    outlier_method: OutlierMethod = "none"
    outlier_threshold: float = 2.5
    standardization_method: StandardizationMethod = None
    impute_outside_validated_gap_range: bool = False


@dataclass(frozen=True)
class RunMetadata:
    """Human and execution details recorded without affecting imputation."""

    dataset_creator: str | None = None
    dataset_creator_info_path: str | Path | None = None
    executor_name: str | None = None
    executor_info_path: str | Path | None = None
    executor_responsible_person: str | None = None
    executor_responsible_person_info_path: str | Path | None = None
    execution_notebook_path: str | Path | None = None
    comment: str | None = None


DOMAIN_MODELS = {
    "eye_tracking": {
        "artifact_dir": PROJECT_ROOT / "artifacts/eyetracking",
        "artifact_filename": "selector.joblib",
        "validated_gap_duration_range_ms": (1.0, 250.0),
        "min_context_valid_fraction": 0.80,
        "scale_floor_recording_iqr_fraction": 0.05,
    },
    "weather": {
        "artifact_dir": PROJECT_ROOT / "artifacts/weather",
        "artifact_filename": "selector.joblib",
        "validated_gap_duration_range_ms": (3_600_000.0, 1_584_000_000.0),
        "min_context_valid_fraction": 0.80,
        "scale_floor_recording_iqr_fraction": 0.05,
        "duration_feature_unit": "hours",
        "seasonal_config": {
            "period_strategy": "calendar_year", "period_value": 1,
            "candidates_per_direction": 3, "min_candidates": 3, "max_offsets": 8,
            "aggregation": "mean", "leap_day_policy": "skip", "mode": "bidirectional",
        },
        "requires_calendar_timestamps": True,
    },
    "traffic": {
        "artifact_dir": PROJECT_ROOT / "artifacts/traffic",
        "artifact_filename": "selector.joblib",
        "validated_gap_duration_range_ms": (300_000.0, 86_400_000.0),
        "min_context_valid_fraction": 0.80,
        "scale_floor_recording_iqr_fraction": 0.05,
        "duration_feature_unit": "minutes",
        "seasonal_config": {
            "period_strategy": "fixed_timedelta", "period_value": "7 days",
            "candidates_per_direction": 3, "min_candidates": 3, "max_offsets": 8,
            "aggregation": "median", "leap_day_policy": "skip", "mode": "bidirectional",
        },
        "requires_calendar_timestamps": True,
    },
}


def _write_frame(frame: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _validate(config: DomainImputationConfig, frame: pd.DataFrame, value_col: str) -> None:
    if config.domain not in DOMAIN_MODELS:
        raise ValueError(f"Unsupported domain: {config.domain}")
    if value_col not in frame:
        raise ValueError(f"Column not found: {value_col}")
    if config.domain == "eye_tracking" and config.sampling_rate_hz is not None and config.sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")
    if config.outlier_threshold <= 0:
        raise ValueError("outlier_threshold must be positive.")
    if config.outlier_method not in {"none", "iqr", "zscore"}:
        raise ValueError("outlier_method must be none, iqr, or zscore.")
    if config.standardization_method not in {None, "zscore", "robust", "minmax"}:
        raise ValueError("standardization_method must be None, zscore, robust, or minmax.")
    if config.timestamp_col is not None and config.timestamp_col not in frame:
        raise ValueError(f"Timestamp column not found: {config.timestamp_col}")
    if config.timestamp_unit not in {"s", "ms", "us"}:
        raise ValueError("timestamp_unit must be s, ms, or us.")
    if config.validity_col is not None and config.validity_col not in frame:
        raise ValueError(f"Validity column not found: {config.validity_col}")
    if DOMAIN_MODELS[config.domain].get("requires_calendar_timestamps") and config.timestamp_col is None:
        raise ValueError(f"{config.domain} requires timestamp_col for calendar-aware imputation.")


def _serializable_config(config: DomainImputationConfig) -> dict[str, object]:
    payload = asdict(config)
    if config.domain in {"weather", "traffic"}:
        payload.pop("sampling_rate_hz", None)
    return payload


def _remove_outliers(values: np.ndarray, method: OutlierMethod, threshold: float) -> tuple[np.ndarray, dict[str, object]]:
    values = values.copy()
    log: dict[str, object] = {"method": method, "threshold": threshold, "removed": 0,
                              "outlier_count": 0, "outlier_regions": []}
    if method == "none":
        return values, log
    finite = values[np.isfinite(values)]
    if not len(finite):
        log["reason"] = "no_finite_values"
        return values, log
    if method == "iqr":
        q1, q3 = np.percentile(finite, [25, 75])
        spread = q3 - q1
        log.update({"q1": float(q1), "q3": float(q3), "iqr": float(spread)})
        if not np.isfinite(spread) or spread == 0:
            log["reason"] = "zero_or_nonfinite_iqr"
            return values, log
        mask = (values < q1 - threshold * spread) | (values > q3 + threshold * spread)
    else:
        mean, std = float(np.mean(finite)), float(np.std(finite))
        log.update({"mean": mean, "std": std})
        if not np.isfinite(std) or std == 0:
            log["reason"] = "zero_or_nonfinite_std"
            return values, log
        mask = np.abs((values - mean) / std) > threshold
    values[mask] = np.nan
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True]) + 1
    log["removed"] = int(mask.sum())
    log["outlier_regions"] = [
        {"outlier_number": number, "start_idx_inclusive": int(start),
         "end_idx_exclusive": int(end), "length_samples": int(end - start)}
        for number, (start, end) in enumerate(zip(starts, ends), start=1)
    ]
    log["outlier_count"] = len(log["outlier_regions"])
    return values, log


def _standardize_values(values: np.ndarray, method: StandardizationMethod) -> tuple[np.ndarray, dict[str, object]]:
    standardized = values.copy()
    log: dict[str, object] = {"method": method, "status": "not_applied",
                              "finite_value_count": int(np.isfinite(values).sum())}
    if method is None:
        log["reason"] = "standardization_disabled"
        return standardized, log
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    if not len(finite):
        log.update({"status": "skipped", "reason": "no_finite_values"})
        return standardized, log
    if method == "zscore":
        center, scale = float(np.mean(finite)), float(np.std(finite))
        log["parameters"] = {"mean": center, "std": scale}
    elif method == "robust":
        q1, center, q3 = (float(value) for value in np.percentile(finite, [25, 50, 75]))
        scale = q3 - q1
        log["parameters"] = {"median": center, "q1": q1, "q3": q3, "iqr": scale}
    else:
        minimum, maximum = float(np.min(finite)), float(np.max(finite))
        center, scale = minimum, maximum - minimum
        log["parameters"] = {"min": minimum, "max": maximum, "range": scale}
    if not np.isfinite(scale) or scale == 0:
        log.update({"status": "skipped", "reason": "zero_or_nonfinite_scale"})
        return standardized, log
    standardized[finite_mask] = (finite - center) / scale
    log["status"] = "applied"
    return standardized, log


def _intermediate_path(output_path: str | Path | None, suffix: str) -> Path | None:
    if output_path is None:
        return None
    target = Path(output_path).expanduser()
    return target.with_name(f"{target.stem}_{suffix}.csv")


def _pipeline_configuration(config: DomainImputationConfig) -> dict[str, Any]:
    """Serialize only settings that affect the currently validated pipeline."""
    return _serializable_config(config)


def _gap_count(frame: pd.DataFrame, value_col: str) -> int:
    """Count contiguous non-finite regions in one data state."""
    values = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    return int(np.sum(missing & np.r_[True, ~missing[:-1]]))


def impute_with_rf_selector(
    frame: pd.DataFrame, value_col: str, *, config: DomainImputationConfig | None = None,
    run_metadata: RunMetadata | None = None, input_path: str | Path | None = None,
    output_path: str | Path | None = None, provenance_path: str | Path | None = None,
    preprocessed_outlier_log: dict[str, object] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Run Outlier -> Missing Values -> Standardization with central provenance.

    ``preprocessed_outlier_log`` supports callers that have already completed
    outlier detection; a
    separately preprocessed frame is treated as the data_1 state.
    """
    from . import missing_values, outlier, standardization
    started = datetime.now().astimezone()
    config = config or DomainImputationConfig()
    run_metadata = run_metadata or RunMetadata()
    _validate(config, frame, value_col)
    executor = run_metadata.executor_name or getpass.getuser()
    responsible = run_metadata.executor_responsible_person
    agents = associated_agents(
        executor_name=executor, executor_info_path=run_metadata.executor_info_path,
        responsible_person_name=responsible,
        responsible_person_info_path=run_metadata.executor_responsible_person_info_path,
    )
    output = Path(output_path).expanduser() if output_path else None
    provenance_file = Path(provenance_path).expanduser() if provenance_path else None
    run_id = f"run-{started.strftime('%Y%m%dT%H%M%S%z')}-{uuid4().hex[:8]}"
    log_dir = (provenance_file.with_name(f"{provenance_file.stem}_logs") / run_id) if provenance_file else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    configuration = _pipeline_configuration(config)
    configuration_path = log_dir / "configuration.txt" if log_dir else None
    if configuration_path:
        configuration_path.write_text(
            "# General preprocessing pipeline configuration\n"
            + json.dumps(configuration, indent=2, default=str)
            + "\n",
            encoding="utf-8",
        )
    gaps_before_outlier_detection = _gap_count(frame, value_col)
    data_1_path = _intermediate_path(output, "after_outlier_detection")
    data_2_path = _intermediate_path(output, "after_missing_value_imputation")
    outlier_metadata_path = log_dir / "outlier_metadata.json" if log_dir else None
    missing_metadata_path = log_dir / "missing_value_metadata.json" if log_dir else None
    standardization_metadata_path = log_dir / "standardization_metadata.json" if log_dir else None

    # Explicit component calls define and make the required processing order testable.
    if preprocessed_outlier_log is None:
        data_1, outlier_activity = outlier.run(
            frame, value_col, method=config.outlier_method, threshold=config.outlier_threshold,
            validity_col=config.validity_col, input_path=input_path, output_path=data_1_path,
            metadata_path=outlier_metadata_path,
        )
    else:
        data_1 = frame.copy()
        if data_1_path is not None:
            _write_frame(data_1, data_1_path)
        now = datetime.now().astimezone()
        outlier_activity = outlier.activity_record(
            implementation_path=Path(outlier.__file__), started=now,
            finished=now,
            details=outlier.build_details(preprocessed_outlier_log, input_path=input_path,
                                          output_path=data_1_path),
        )
        if outlier_metadata_path:
            write_json(outlier_metadata_path, outlier_activity)
    data_2, missing_activity = missing_values.run(
        data_1, value_col, config=config, input_path=data_1_path, output_path=data_2_path,
        metadata_path=missing_metadata_path,
    )
    final, standardization_activity = standardization.run(
        data_2, value_col, method=config.standardization_method, input_path=data_2_path,
        output_path=output, metadata_path=standardization_metadata_path,
    )
    finished = datetime.now().astimezone()
    gaps_after_outlier_detection = _gap_count(data_1, value_col)
    implementation_path = Path(__file__).resolve()
    activity_metadata_files = {
        "outlier_detection": outlier_metadata_path,
        "missing_value_imputation": missing_metadata_path,
        "standardization": standardization_metadata_path,
    }
    activity_records = {
        "outlier_detection": outlier_activity,
        "missing_value_imputation": missing_activity,
        "standardization": standardization_activity,
    }
    provenance: dict[str, Any] = {
        "run": {"run_id": run_id, "started_at": started.isoformat(), "finished_at": finished.isoformat(),
                "duration_seconds": (finished - started).total_seconds(),
                "pipeline_implementation": {
                    "path": implementation_path.relative_to(implementation_path.parents[3]).as_posix(),
                    "sha256": file_metadata(implementation_path)["sha256"],
                },
                "execution_notebook": file_metadata(run_metadata.execution_notebook_path),
                "configuration_file": file_metadata(configuration_path),
                "comment": run_metadata.comment},
        "agents": agents,
        "configuration": configuration,
        "activities": {
            name: {"metadata_file": file_metadata(metadata_path), **activity_records[name]}
            for name, metadata_path in activity_metadata_files.items()
        },
        "summary": {"gaps_before_outlier_detection": gaps_before_outlier_detection,
                    "gaps_after_outlier_detection": gaps_after_outlier_detection,
                    "filled_gap_count": missing_activity["details"]["filled_gap_count"],
                    "skipped_gap_count": missing_activity["details"]["skipped_gap_count"]},
    }
    if provenance_file:
        write_json(provenance_file, provenance)
    return (final, provenance) if provenance_file else final


__all__ = ("ALL_IMPUTATION_METHODS", "RF_METHODS", "DomainImputationConfig", "RunMetadata", "impute_with_rf_selector")
