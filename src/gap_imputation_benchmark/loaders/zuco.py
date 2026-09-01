"""Loader for ZuCo 2.0 normal-reading eye-tracking recordings."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat

_FILENAME_PATTERN = re.compile(
    r"(?P<participant>[A-Za-z0-9]+)_NR(?P<session>\d+)_ET\.mat"
)
_REQUIRED_RAW_COLUMNS = {"TIME", "L_GAZE_X", "L_GAZE_Y", "L_AREA"}


def _parse_zuco_path(file_path: Path) -> tuple[str, str]:
    """Validate a supported ZuCo layout and return participant and session IDs."""
    match = _FILENAME_PATTERN.fullmatch(file_path.name)
    if match is None:
        raise ValueError(f"Unexpected ZuCo filename: {file_path.name}")

    participant_id = match.group("participant")
    raw_root = file_path.parent.parent.parent
    is_legacy_raw_root = raw_root.name == "raw" and raw_root.parent.name == "data"
    is_eyetracking_raw_root = (
        raw_root.name == "raw"
        and raw_root.parent.name == "eyetracking"
        and raw_root.parent.parent.name == "data"
    )
    if (
        file_path.parent.name != participant_id
        or file_path.parent.parent.name != "ZUCO"
        or not (is_legacy_raw_root or is_eyetracking_raw_root)
    ):
        raise ValueError(
            "Expected a ZuCo normal-reading file below data/raw/ZUCO or "
            "data/eyetracking/raw/ZUCO."
        )

    return participant_id, f"NR{match.group('session')}"


def load_zuco_normal_reading(file_path: str | Path) -> pd.DataFrame:
    """Load one ZuCo normal-reading recording into the common schema.

    Files matching ``data/raw/ZUCO/*/*_NR*_ET.mat`` or
    ``data/eyetracking/raw/ZUCO/*/*_NR*_ET.mat`` are accepted. Native
    top-left-origin screen-pixel coordinates are retained. A sample is valid
    only when timestamp, horizontal and vertical left-eye gaze coordinates,
    and left pupil area are finite and the pupil area is strictly positive.
    Zero-valued gaze coordinates remain valid screen-edge observations.
    """
    file_path = Path(file_path)
    participant_id, session_id = _parse_zuco_path(file_path)

    mat = loadmat(file_path, simplify_cells=True)
    if "colheader" not in mat or "data" not in mat:
        raise ValueError("ZuCo MAT file must contain 'colheader' and 'data'.")

    column_names = [str(name) for name in np.asarray(mat["colheader"]).ravel()]
    raw_data = np.asarray(mat["data"])
    if raw_data.ndim != 2 or raw_data.shape[1] != len(column_names):
        raise ValueError("ZuCo 'data' must be a 2D array matching 'colheader'.")

    missing_columns = _REQUIRED_RAW_COLUMNS - set(column_names)
    if missing_columns:
        raise ValueError(
            f"Missing required ZuCo columns: {sorted(missing_columns)}"
        )

    column_index = {name: index for index, name in enumerate(column_names)}
    timestamp_ms = raw_data[:, column_index["TIME"]].astype(float)
    gaze_x = raw_data[:, column_index["L_GAZE_X"]].astype(float)
    gaze_y = raw_data[:, column_index["L_GAZE_Y"]].astype(float)
    left_area = raw_data[:, column_index["L_AREA"]].astype(float)

    return pd.DataFrame(
        {
            "dataset_id": "ZuCo",
            "participant_id": participant_id,
            "recording_id": file_path.stem,
            "session_id": session_id,
            "task": "normal_reading",
            "timestamp_ms": timestamp_ms,
            "sampling_rate_hz": 500.0,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "is_valid": (
                np.isfinite(timestamp_ms)
                & np.isfinite(gaze_x)
                & np.isfinite(gaze_y)
                & np.isfinite(left_area)
                & (left_area > 0)
            ),
            "source_file": str(file_path),
            "coordinate_unit": "pixel",
        }
    )
