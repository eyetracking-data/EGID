from pathlib import Path
import re

import pandas as pd


def load_gazebase_vr_reading(file_path: str | Path) -> pd.DataFrame:
    """Load one GazeBaseVR Text recording into the common schema."""
    file_path = Path(file_path)

    match = re.fullmatch(
        r"S_(?P<participant>\d+)_(?P<session>S\d+)_(?P<order>\d+)_TEX\.csv",
        file_path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected GazeBaseVR filename: {file_path.name}")

    raw = pd.read_csv(file_path)

    required_columns = {"n", "x"}
    missing = required_columns - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    participant_id = match.group("participant")
    session_id = match.group("session")

    result = pd.DataFrame(
        {
            "dataset_id": "GazeBaseVR",
            "participant_id": participant_id,
            "recording_id": file_path.stem,
            "session_id": session_id,
            "task": "reading",
            "timestamp_ms": raw["n"].astype(float),
            "sampling_rate_hz": 250.0,
            "gaze_x": raw["x"].astype(float),
            "is_valid": raw["x"].notna(),
            "source_file": str(file_path),
            "coordinate_unit": "degree",
        }
    )

    return result
