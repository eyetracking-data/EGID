"""Loader for the Pedrotti reading-aloud eye-tracking recordings."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

_FILENAME_PATTERN = re.compile(r"(?P<participant>\d{2})\.txt")
_REQUIRED_COLUMNS = {
    "TRIAL_INDEX",
    "LEFT_GAZE_X",
    "LEFT_GAZE_Y",
    "LEFT_PUPIL_SIZE",
    "RIGHT_GAZE_X",
    "RIGHT_GAZE_Y",
    "RIGHT_PUPIL_SIZE",
    "TIMESTAMP",
    "TrialTextShown",
}
_EYE_COLUMNS = {
    "left": ("LEFT_GAZE_X", "LEFT_GAZE_Y", "LEFT_PUPIL_SIZE"),
    "right": ("RIGHT_GAZE_X", "RIGHT_GAZE_Y", "RIGHT_PUPIL_SIZE"),
}

# Transcribed from data/raw/Pedrotti/readme.txt. Numerals are identified from
# the documented numeral notation: every numeral stimulus begins with a digit.
_WORDS = frozenset(
    {
        "amer", "aula", "brut", "déjà", "écho", "être", "fuir", "iris", "saga",
        "très", "unir", "user", "astucieux", "baptiser", "boulangerie",
        "caoutchouc", "carrelage", "dorénavant", "franchir", "fréquemment",
        "impatient", "limonade", "participer", "questionner",
    }
)
_PSEUDOWORDS = frozenset(
    {
        "érul", "fabu", "inor", "iqué", "isan", "muar", "oufé", "spac", "stau",
        "udre", "ujar", "zago", "birtajicer", "carriloge", "daurinfarue",
        "frinchar", "lomanube", "ostéciant", "outigieux", "pannesquier",
        "pondaser", "pouitcheau", "quimmévrant", "tulanévont",
    }
)


def _parse_pedrotti_path(file_path: Path) -> str:
    """Validate a supported local layout and return the participant ID."""
    match = _FILENAME_PATTERN.fullmatch(file_path.name)
    if match is None:
        raise ValueError(f"Unexpected Pedrotti filename: {file_path.name}")
    raw_root = file_path.parent.parent
    is_legacy_raw_root = raw_root.name == "raw" and raw_root.parent.name == "data"
    is_eyetracking_raw_root = (
        raw_root.name == "raw"
        and raw_root.parent.name == "eyetracking"
        and raw_root.parent.parent.name == "data"
    )
    if (
        file_path.parent.name != "Pedrotti"
        or not (is_legacy_raw_root or is_eyetracking_raw_root)
    ):
        raise ValueError(
            "Expected a Pedrotti participant file below data/raw/Pedrotti or "
            "data/eyetracking/raw/Pedrotti."
        )
    return match.group("participant")


def _is_usable_eye(raw: pd.DataFrame, eye: str) -> bool:
    """Return whether every required channel for one eye has observations."""
    return all(raw[column].notna().any() for column in _EYE_COLUMNS[eye])


def _select_tracked_eye(raw: pd.DataFrame) -> str:
    available_eyes = [eye for eye in _EYE_COLUMNS if _is_usable_eye(raw, eye)]
    if len(available_eyes) != 1:
        raise ValueError(
            "Expected exactly one usable Pedrotti eye, found: "
            f"{', '.join(available_eyes) if available_eyes else 'none'}"
        )
    return available_eyes[0]


def _stimulus_condition(stimulus_text: str) -> str:
    if stimulus_text in _WORDS:
        return "word"
    if stimulus_text in _PSEUDOWORDS:
        return "pseudoword"
    if re.fullmatch(r"\d[\d'’]*", stimulus_text):
        return "number"
    raise ValueError(f"Unknown Pedrotti stimulus: {stimulus_text!r}")


def load_pedrotti_reading(file_path: str | Path) -> list[pd.DataFrame]:
    """Load native screen-pixel Pedrotti trials into the common schema."""
    file_path = Path(file_path)
    participant_id = _parse_pedrotti_path(file_path)
    raw = pd.read_csv(file_path, na_values=["."], dtype={"TrialTextShown": "string"})

    missing_columns = _REQUIRED_COLUMNS - set(raw.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required Pedrotti columns: {sorted(missing_columns)}"
        )

    tracked_eye = _select_tracked_eye(raw)
    gaze_x_column, gaze_y_column, pupil_column = _EYE_COLUMNS[tracked_eye]
    timestamp_ms = pd.to_numeric(raw["TIMESTAMP"], errors="coerce")
    gaze_x = pd.to_numeric(raw[gaze_x_column], errors="coerce")
    gaze_y = pd.to_numeric(raw[gaze_y_column], errors="coerce")
    pupil_size = pd.to_numeric(raw[pupil_column], errors="coerce")
    is_valid = (
        np.isfinite(timestamp_ms)
        & np.isfinite(gaze_x)
        & np.isfinite(gaze_y)
        & np.isfinite(pupil_size)
        & (pupil_size > 0)
    )

    recordings: list[pd.DataFrame] = []
    for trial_index, trial_rows in raw.groupby("TRIAL_INDEX", sort=True):
        positions = trial_rows.index
        stimuli = trial_rows["TrialTextShown"].dropna().unique()
        if len(stimuli) != 1:
            raise ValueError(f"Trial {trial_index} must contain exactly one stimulus.")
        stimulus_text = str(stimuli[0])
        trial_id = int(trial_index)
        recordings.append(
            pd.DataFrame(
                {
                    "dataset_id": "Pedrotti",
                    "participant_id": participant_id,
                    "recording_id": f"{participant_id}_trial_{trial_id:03d}",
                    "session_id": "session_1",
                    "task": "reading_aloud",
                    "timestamp_ms": timestamp_ms.loc[positions].to_numpy(),
                    "sampling_rate_hz": 1000.0,
                    "gaze_x": gaze_x.loc[positions].to_numpy(),
                    "gaze_y": gaze_y.loc[positions].to_numpy(),
                    "is_valid": is_valid.loc[positions].to_numpy(),
                    "source_file": str(file_path),
                    "coordinate_unit": "pixel",
                    "tracked_eye": tracked_eye,
                    "stimulus_text": stimulus_text,
                    "stimulus_condition": _stimulus_condition(stimulus_text),
                }
            )
        )
    return recordings
