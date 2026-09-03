"""Portable DWD hourly-temperature source preparation and loading."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_dwd_temperature_data(data_dir: Path, *, start: str, end: str) -> Path:
    """Recreate the legacy DWD extraction exactly, outside the repository."""
    data_dir = Path(data_dir)
    raw_dir, extracted_dir = data_dir / "raw", data_dir / "extracted"
    processed_dir = data_dir / "processed" / "temperature_2000_2025"
    zip_paths = sorted(raw_dir.glob("stundenwerte_TU_*_hist.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No DWD temperature ZIP archives found in {raw_dir}.")
    extracted_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in zip_paths:
        match = re.search(r"TU_(\d{5})_", zip_path.name)
        if match is None:
            raise ValueError(f"Cannot determine DWD station ID from {zip_path.name}.")
        destination = extracted_dir / match.group(1)
        destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(destination)
    data_paths = sorted(extracted_dir.glob("*/produkt_tu_stunde_*.txt"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    for path in data_paths:
        raw = pd.read_csv(path, sep=";", na_values=["NaN", -999, "-999"], skipinitialspace=True)
        raw.columns = raw.columns.str.strip()
        # DWD stores timestamps as YYYYMMDDHH integers.  Parsing integers directly
        # makes pandas treat them as nanoseconds since the Unix epoch.
        raw["MESS_DATUM"] = pd.to_datetime(
            raw["MESS_DATUM"].astype(str).str.strip(),
            format="%Y%m%d%H",
            errors="coerce",
        )
        raw.loc[raw["MESS_DATUM"].between(start, end)].to_csv(
            processed_dir / path.name, sep=";", index=False, na_rep="NaN"
        )
    return processed_dir


def load_dwd_station(path: Path, *, start: str, end: str, source_file: str) -> pd.DataFrame:
    """Load one DWD station into the shared observed-recording representation."""
    raw = pd.read_csv(path, sep=";", na_values=["NaN", -999, "-999"], skipinitialspace=True)
    raw.columns = raw.columns.str.strip()
    raw["MESS_DATUM"] = pd.to_datetime(raw["MESS_DATUM"], errors="coerce")
    station_id = path.stem.split("_")[-1]
    raw = raw.loc[raw["MESS_DATUM"].between(start, end), ["MESS_DATUM", "TT_TU"]].copy()
    if raw["MESS_DATUM"].isna().any() or raw["MESS_DATUM"].duplicated().any():
        raise ValueError(f"{station_id}: invalid or duplicate timestamps in {path.name}")
    index = pd.date_range(start, end, freq="h")
    values = pd.to_numeric(raw.set_index("MESS_DATUM")["TT_TU"], errors="coerce").reindex(index)
    frame = pd.DataFrame({"timestamp": index, "gaze_x": values.to_numpy(float)})
    frame["is_valid"] = np.isfinite(frame["gaze_x"])
    frame["timestamp_ms"] = (frame["timestamp"].astype("int64") // 1_000_000).astype(float)
    frame["sampling_rate_hz"] = 1 / 3600
    frame["dataset_id"] = "DWD_hourly_temperature"
    frame["participant_id"] = station_id
    frame["recording_id"] = station_id
    frame["session_id"] = f"{pd.Timestamp(start).year}-{pd.Timestamp(end).year}"
    frame["source_file"] = source_file
    return frame
