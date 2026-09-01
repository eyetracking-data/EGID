from pathlib import Path
from collections.abc import Callable

import pandas as pd


def summarize_recordings(
    files: list[Path],
    loader: Callable[[Path], pd.DataFrame],
) -> pd.DataFrame:
    """Create one summary row per recording."""
    rows = []

    for file_path in files:
        df = loader(file_path)

        n_samples = len(df)
        n_valid = int(df["is_valid"].sum())
        n_invalid = n_samples - n_valid

        duration_ms = (
            float(df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0])
            if n_samples > 1
            else 0.0
        )

        rows.append(
            {
                "dataset_id": df["dataset_id"].iloc[0],
                "participant_id": df["participant_id"].iloc[0],
                "recording_id": df["recording_id"].iloc[0],
                "session_id": df["session_id"].iloc[0],
                "task": df["task"].iloc[0],
                "n_samples": n_samples,
                "duration_ms": duration_ms,
                "n_valid": n_valid,
                "n_invalid": n_invalid,
                "valid_fraction": n_valid / n_samples if n_samples else 0.0,
                "source_file": str(file_path),
            }
        )

    return pd.DataFrame(rows)
