"""Outlier detection: raw input -> data after outlier detection."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .rf_domain_imputation import _remove_outliers, _write_frame
from .provenance import activity_record, file_metadata, write_json


def build_details(outlier_log: dict[str, Any], *, input_path: str | Path | None,
                  output_path: str | Path | None) -> dict[str, Any]:
    """Use unambiguous sample and region counts in the public activity log."""
    details = dict(outlier_log)
    sample_count = details.pop("removed", details.get("outlier_count", 0))
    region_count = details.pop("outlier_region_count", details.pop("outlier_count", None))
    details.pop("outlier_sample_count", None)
    details["outlier_count"] = int(sample_count)
    details["outlier_region_count"] = int(
        len(details.get("outlier_regions", [])) if region_count is None else region_count
    )
    details["input_file"] = file_metadata(input_path)
    details["file_after_outlier_detection"] = file_metadata(output_path)
    return details


def run(frame: pd.DataFrame, value_col: str, *, method: str, threshold: float,
        validity_col: str | None, input_path: str | Path | None,
        output_path: str | Path | None, metadata_path: Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mark invalid values and detected outliers as NaN without imputing them."""
    started = datetime.now().astimezone()
    values = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype=float).copy()
    valid = np.isfinite(values) if validity_col is None else (
        frame[validity_col].fillna(False).astype(bool).to_numpy() & np.isfinite(values)
    )
    values[~valid] = np.nan
    values, details = _remove_outliers(values, method, threshold)
    result = frame.copy()
    result[value_col] = values
    if output_path is not None:
        _write_frame(result, output_path)
    finished = datetime.now().astimezone()
    details = build_details(details, input_path=input_path, output_path=output_path)
    record = activity_record(implementation_path=Path(__file__),
                             started=started, finished=finished,
                             details=details)
    if metadata_path:
        write_json(metadata_path, record)
    return result, record
