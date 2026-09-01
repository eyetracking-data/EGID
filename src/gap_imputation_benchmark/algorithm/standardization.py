"""Optional z-score, robust, and min-max standardization."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .rf_domain_imputation import _standardize_values, _write_frame
from .provenance import activity_record, file_metadata, write_json


def run(frame: pd.DataFrame, value_col: str, *, method: str | None,
        input_path: str | Path | None, output_path: str | Path | None,
        metadata_path: Path | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = datetime.now().astimezone()
    values = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype=float)
    standardized, details = _standardize_values(values, method)
    # Map legacy values to concise English activity statuses.
    if details["status"] == "not_applied":
        details["status"] = "disabled"
    elif details["status"] == "applied":
        details["status"] = "applied"
    elif details["status"] == "skipped":
        details["status"] = "skipped"
    result = frame.copy()
    result[value_col] = standardized
    if output_path is not None:
        _write_frame(result, output_path)
    finished = datetime.now().astimezone()
    details.update({"input_file": file_metadata(input_path),
                    "file_after_standardization": file_metadata(output_path)})
    record = activity_record(implementation_path=Path(__file__),
                             started=started, finished=finished,
                             details=details)
    if metadata_path:
        write_json(metadata_path, record)
    return result, record
