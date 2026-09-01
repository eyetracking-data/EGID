"""Missing-value imputation: data after outlier detection -> imputed data.

This module owns the complete second pipeline activity. It deliberately does
not call the legacy all-in-one deployment function.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .rf_domain_imputation import (
    CONTEXT_MULTIPLIER, DOMAIN_MODELS, MIN_CONTEXT_SAMPLES, _write_frame,
)
from gap_imputation_benchmark.benchmark.features import extract_basic_gap_features
from gap_imputation_benchmark.benchmark.gaps import ArtificialGap, GapCandidate
from gap_imputation_benchmark.benchmark.imputers import (
    SeasonalPeriodConfig, impute_forward_fill, impute_linear,
    impute_local_natural_cubic_spline, impute_nearest_boundary, impute_pchip,
    impute_polyfit_bic, impute_seasonal_periodic, impute_template,
)

from .provenance import activity_record, file_metadata, write_json


_METHODS = {
    "forward_fill": impute_forward_fill,
    "nearest_boundary": impute_nearest_boundary,
    "linear": impute_linear,
    "pchip": impute_pchip,
    "local_natural_cubic_spline": impute_local_natural_cubic_spline,
    "polyfit_bic": impute_polyfit_bic,
    "template": impute_template,
    "seasonal_periodic": impute_seasonal_periodic,
}


def _find_gaps(values: np.ndarray) -> list[tuple[int, int]]:
    missing = ~np.isfinite(values)
    starts = np.flatnonzero(missing & np.r_[True, ~missing[:-1]])
    ends = np.flatnonzero(missing & np.r_[~missing[1:], True]) + 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _runtime_recording(values: np.ndarray, valid: np.ndarray, timestamps_ms: np.ndarray,
                       timestamps: pd.DatetimeIndex | None) -> pd.DataFrame:
    recording = pd.DataFrame({"gaze_x": values, "is_valid": valid,
                              "original_is_valid": valid, "timestamp_ms": timestamps_ms})
    if timestamps is not None:
        recording["timestamp"] = timestamps
    return recording


def _candidate_for_gap(values: np.ndarray, valid: np.ndarray, timestamps_ms: np.ndarray,
                       start: int, end: int, sampling_rate_hz: float,
                       min_context_valid_fraction: float) -> GapCandidate | tuple[None, str]:
    context_length = max(MIN_CONTEXT_SAMPLES, int(np.ceil((end - start) * CONTEXT_MULTIPLIER)))
    left_start, right_end = start - context_length, end + context_length
    if left_start < 0 or right_end > len(values) or start < 1 or end >= len(values):
        return None, "insufficient_edge_context"
    left_fraction = float(valid[left_start:start].mean())
    right_fraction = float(valid[end:right_end].mean())
    if left_fraction < min_context_valid_fraction or right_fraction < min_context_valid_fraction:
        return None, "insufficient_valid_context"
    duration_ms = float(timestamps_ms[end] - timestamps_ms[start])
    return GapCandidate(
        gap_start_idx=start, gap_end_idx=end, requested_gap_duration_ms=duration_ms,
        realized_gap_duration_ms=duration_ms, sampling_rate_hz=sampling_rate_hz,
        gap_length_samples=end - start, left_context_start_idx=left_start,
        right_context_end_idx=right_end, left_context_valid_fraction=left_fraction,
        right_context_valid_fraction=right_fraction,
        context_valid_fraction=(left_fraction + right_fraction) / 2,
    )


def _context_diagnostics(valid: np.ndarray, start: int, end: int,
                         min_context_valid_fraction: float) -> dict[str, Any]:
    """Describe the exact context that made a gap ineligible for imputation."""
    context_length = max(MIN_CONTEXT_SAMPLES, int(np.ceil((end - start) * CONTEXT_MULTIPLIER)))
    left_start, right_end = start - context_length, end + context_length
    left_context = valid[max(0, left_start):start]
    right_context = valid[end:min(len(valid), right_end)]
    return {
        "context_length_samples_per_side": context_length,
        "left_context_valid_sample_count": int(left_context.sum()),
        "right_context_valid_sample_count": int(right_context.sum()),
        "left_context_valid_fraction": float(left_context.mean()) if len(left_context) else None,
        "right_context_valid_fraction": float(right_context.mean()) if len(right_context) else None,
        "required_context_valid_fraction": min_context_valid_fraction,
    }


def _load_selector(domain: str) -> tuple[dict[str, Any], dict[str, Any]]:
    route = DOMAIN_MODELS[domain]
    artifact_dir = Path(route["artifact_dir"])
    artifact_path = artifact_dir / str(route["artifact_filename"])
    manifest_path = artifact_dir / "manifest.json"
    metadata: dict[str, Any] = {
        "domain": domain, "artifact_dir": str(artifact_dir),
        "artifact": file_metadata(artifact_path), "model_manifest": file_metadata(manifest_path),
    }
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Validated selector artifact is missing: {artifact_path}")
    artifact = joblib.load(artifact_path)
    required = {"model", "feature_imputer", "feature_columns", "method_names"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Selector artifact misses keys: {sorted(missing)}")
    metadata["methods"] = artifact["method_names"]
    return artifact, metadata


def _selector_features(extracted: dict[str, float], artifact: dict[str, Any],
                       domain_policy: dict[str, Any]) -> dict[str, float]:
    features = dict(extracted)
    unit = domain_policy.get("duration_feature_unit")
    if unit == "hours":
        features["realized_gap_duration_hours"] = features["realized_gap_duration_ms"] / 3_600_000.0
    elif unit == "minutes":
        features["realized_gap_duration_minutes"] = features["realized_gap_duration_ms"] / 60_000.0
    missing = set(artifact["feature_columns"]) - set(features)
    if missing:
        raise ValueError(f"Cannot construct selector features: {sorted(missing)}")
    return features


def _predict_method(artifact: dict[str, Any], feature_values: dict[str, float]) -> tuple[str, dict[str, float]]:
    columns = list(artifact["feature_columns"])
    row = pd.DataFrame([{column: feature_values[column] for column in columns}], columns=columns)
    transformed = artifact["feature_imputer"].transform(row)
    scores = np.asarray(artifact["model"].predict(transformed), dtype=float)[0]
    methods = list(artifact["method_names"])
    unsupported = set(methods) - set(_METHODS)
    if unsupported:
        raise ValueError(f"Selector artifact contains unsupported methods: {sorted(unsupported)}")
    predicted_scores = {method: float(score) for method, score in zip(methods, scores)}
    return methods[int(np.argmin(scores))], predicted_scores


def _apply_method(method: str, runtime: ArtificialGap, domain_policy: dict[str, Any]):
    if method == "seasonal_periodic":
        seasonal_config = domain_policy.get("seasonal_config")
        if seasonal_config is None:
            raise ValueError("seasonal_periodic was selected without a domain configuration.")
        return impute_seasonal_periodic(
            runtime, config=SeasonalPeriodConfig(**seasonal_config), timestamp_column="timestamp",
        )
    return _METHODS[method](runtime)


def _scale_floor_from_recording(values: np.ndarray, fraction: float) -> float | None:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    recording_iqr = float(np.percentile(finite, 75) - np.percentile(finite, 25))
    scale_floor = fraction * recording_iqr
    return scale_floor if np.isfinite(scale_floor) and scale_floor > 0 else None


def _add_method_metadata(entry: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Expose method diagnostics directly on the gap without static internals."""
    for key, value in metadata.items():
        if key == "coordinate_basis":
            continue
        if key in entry:
            raise ValueError(f"Imputer metadata conflicts with gap field: {key}")
        entry[key] = value


def _timestamps(frame: pd.DataFrame, config: Any) -> tuple[np.ndarray, float, pd.DatetimeIndex | None]:
    if config.timestamp_col is None:
        sampling_rate_hz = config.sampling_rate_hz or 250.0
        return np.arange(len(frame), dtype=float) / sampling_rate_hz * 1000.0, sampling_rate_hz, None
    unit_to_ms = {"s": 1000.0, "ms": 1.0, "us": 0.001}
    raw_timestamps = frame[config.timestamp_col]
    calendar_timestamps: pd.DatetimeIndex | None = None
    if pd.api.types.is_datetime64_any_dtype(raw_timestamps):
        calendar_timestamps = pd.DatetimeIndex(raw_timestamps)
        timestamps_ms = calendar_timestamps.as_unit("ns").asi8.astype(float) / 1_000_000.0
    else:
        numeric = pd.to_numeric(raw_timestamps, errors="coerce")
        if numeric.notna().all():
            timestamps_ms = numeric.to_numpy(dtype=float) * unit_to_ms[config.timestamp_unit]
            calendar_timestamps = pd.to_datetime(timestamps_ms, unit="ms", errors="coerce")
        else:
            calendar_timestamps = pd.DatetimeIndex(pd.to_datetime(raw_timestamps, errors="coerce"))
            timestamps_ms = calendar_timestamps.as_unit("ns").asi8.astype(float) / 1_000_000.0
    if not np.isfinite(timestamps_ms).all() or not np.all(np.diff(timestamps_ms) > 0):
        raise ValueError("timestamp_col must be finite and strictly increasing.")
    sampling_rate_hz = 1000.0 / float(np.median(np.diff(timestamps_ms)))
    return timestamps_ms, sampling_rate_hz, calendar_timestamps


def _impute_values(frame: pd.DataFrame, value_col: str, config: Any) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    domain_policy = DOMAIN_MODELS[config.domain]
    artifact, model_provenance = _load_selector(config.domain)
    values = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype=float).copy()
    original_valid = np.isfinite(values) if config.validity_col is None else (
        frame[config.validity_col].fillna(False).astype(bool).to_numpy() & np.isfinite(values)
    )
    values[~original_valid] = np.nan
    gaps = _find_gaps(values)
    normalization_reference_values = values.copy()
    timestamps_ms, sampling_rate_hz, calendar_timestamps = _timestamps(frame, config)
    logs: list[dict[str, Any]] = []
    for gap_number, (start, end) in enumerate(gaps, start=1):
        valid = np.isfinite(values)
        candidate_or_none = _candidate_for_gap(values, valid, timestamps_ms, start, end, sampling_rate_hz,
                                                float(domain_policy["min_context_valid_fraction"]))
        candidate, reason = candidate_or_none if isinstance(candidate_or_none, tuple) else (candidate_or_none, None)
        entry: dict[str, Any] = {"gap_number": gap_number, "gap_start_idx_inclusive": start,
                                 "gap_end_idx_exclusive": end, "gap_length_samples": end - start}
        if candidate is None:
            entry.update({"status": "skipped", "reason": reason,
                          "model_recommended_method": None, "used_method": None})
            if reason == "insufficient_valid_context":
                entry["realized_gap_duration_ms"] = float(timestamps_ms[end] - timestamps_ms[start])
                entry.update(_context_diagnostics(
                    valid, start, end, float(domain_policy["min_context_valid_fraction"]),
                ))
            logs.append(entry)
            continue
        validated_range = domain_policy["validated_gap_duration_range_ms"]
        in_validated_range = (None if validated_range is None else
                              validated_range[0] <= candidate.realized_gap_duration_ms <= validated_range[1])
        entry.update({"realized_gap_duration_ms": candidate.realized_gap_duration_ms,
                      "within_validated_gap_duration_range": in_validated_range})
        if in_validated_range is False and not config.impute_outside_validated_gap_range:
            entry.update({"status": "skipped", "reason": "outside_validated_gap_duration_range",
                          "model_recommended_method": None, "used_method": None})
            logs.append(entry)
            continue
        trained_scale_floor = artifact.get("feature_scale_floor", {}).get("value") if artifact is not None else None
        scale_floor = (float(trained_scale_floor) if trained_scale_floor is not None else
                       _scale_floor_from_recording(normalization_reference_values,
                                                   float(domain_policy["scale_floor_recording_iqr_fraction"])))
        if scale_floor is None:
            entry.update({"status": "skipped", "reason": "nonpositive_or_nonfinite_recording_scale",
                          "model_recommended_method": None, "used_method": None})
            logs.append(entry)
            continue
        runtime = ArtificialGap(candidate=candidate, ground_truth=np.full(end - start, np.nan),
                               masked_recording=_runtime_recording(values, valid, timestamps_ms, calendar_timestamps))
        try:
            extracted = extract_basic_gap_features(runtime, scale_floor)
            entry["gap_characteristics"] = {key: float(value) for key, value in extracted.items()}
            diagnostics = {key: float(value) for key, value in extracted.diagnostics.items()}
            if config.domain in {"weather", "traffic"}:
                diagnostics.pop("sampling_rate_hz", None)
            entry["feature_diagnostics"] = diagnostics
        except (ValueError, TypeError) as error:
            entry.update({"status": "skipped", "reason": f"feature_extraction_failed: {error}",
                          "model_recommended_method": None, "used_method": None})
            logs.append(entry)
            continue
        selected_method, scores = _predict_method(
            artifact, _selector_features(extracted, artifact, domain_policy),
        )
        methods_to_try = sorted(scores, key=scores.__getitem__)
        entry.update({"selection_mode": "random_forest", "predicted_method_nrmse": scores})
        if in_validated_range is False:
            entry["selection_warning"] = "imputed_outside_validated_gap_duration_range_by_explicit_configuration"
        entry["model_recommended_method"] = selected_method
        entry["attempted_methods"] = []
        for method in methods_to_try:
            result = _apply_method(method, runtime, domain_policy)
            entry["attempted_methods"].append({"method": method, "applicable": result.method_applicable,
                                                "reason": result.failure_reason})
            if result.method_applicable and len(result.predictions) == end - start:
                values[start:end] = result.predictions
                entry.update({"used_method": method, "status": "filled"})
                _add_method_metadata(entry, result.metadata)
                if method != selected_method:
                    entry["fallback_reason"] = "higher-ranked RF method was not applicable"
                break
        else:
            entry.update({"used_method": None, "status": "skipped", "reason": "no_candidate_method_applicable"})
        logs.append(entry)
    return values, logs, model_provenance


def run(frame: pd.DataFrame, value_col: str, *, config: Any, input_path: str | Path | None,
        output_path: str | Path | None, metadata_path: Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Impute data_1 and write its independently useful metadata record."""
    started = datetime.now().astimezone()
    values, gaps, model_provenance = _impute_values(frame, value_col, config)
    result = frame.copy()
    result[value_col] = values
    if output_path is not None:
        _write_frame(result, output_path)
    finished = datetime.now().astimezone()
    details: dict[str, Any] = {
        "imputation_model": model_provenance["artifact"], "gap_count": len(gaps), "gaps": gaps,
        "validated_gap_duration_range_ms": list(DOMAIN_MODELS[config.domain]["validated_gap_duration_range_ms"]),
        "filled_gap_count": sum(gap.get("status") == "filled" for gap in gaps),
        "skipped_gap_count": sum(gap.get("status") == "skipped" for gap in gaps),
        "input_file": file_metadata(input_path),
        "file_after_missing_value_imputation": file_metadata(output_path),
    }
    record = activity_record(implementation_path=Path(__file__), started=started, finished=finished,
                             details=details)
    if metadata_path:
        write_json(metadata_path, record)
    return result, record
