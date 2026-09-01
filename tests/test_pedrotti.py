from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gap_imputation_benchmark.loaders.pedrotti import load_pedrotti_reading


COLUMNS = [
    "TRIAL_INDEX",
    "LEFT_GAZE_X",
    "LEFT_GAZE_Y",
    "LEFT_PUPIL_SIZE",
    "RIGHT_GAZE_X",
    "RIGHT_GAZE_Y",
    "RIGHT_PUPIL_SIZE",
    "TIMESTAMP",
    "TrialTextShown",
]


def write_pedrotti_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


def left_row(trial: int, timestamp: int, stimulus: str, x, y, pupil):
    return {
        "TRIAL_INDEX": trial,
        "LEFT_GAZE_X": x,
        "LEFT_GAZE_Y": y,
        "LEFT_PUPIL_SIZE": pupil,
        "RIGHT_GAZE_X": ".",
        "RIGHT_GAZE_Y": ".",
        "RIGHT_PUPIL_SIZE": ".",
        "TIMESTAMP": timestamp,
        "TrialTextShown": stimulus,
    }


def right_row(trial: int, timestamp: int, stimulus: str, x, y, pupil):
    return {
        "TRIAL_INDEX": trial,
        "LEFT_GAZE_X": ".",
        "LEFT_GAZE_Y": ".",
        "LEFT_PUPIL_SIZE": ".",
        "RIGHT_GAZE_X": x,
        "RIGHT_GAZE_Y": y,
        "RIGHT_PUPIL_SIZE": pupil,
        "TIMESTAMP": timestamp,
        "TrialTextShown": stimulus,
    }


def test_loader_selects_left_eye_segments_trials_and_preserves_metadata(tmp_path: Path):
    file_path = tmp_path / "data/eyetracking/raw/Pedrotti/01.txt"
    write_pedrotti_csv(
        file_path,
        [
            left_row(1, 100, "amer", -12.0, -5.0, 900.0),
            left_row(1, 101, "amer", ".", 8.0, 900.0),
            left_row(2, 500, "1293", 10.0, 8.0, 0.0),
            left_row(3, 900, "érul", 11.0, 9.0, 910.0),
        ],
    )

    recordings = load_pedrotti_reading(file_path)

    assert len(recordings) == 3
    assert recordings[0].columns.tolist() == [
        "dataset_id",
        "participant_id",
        "recording_id",
        "session_id",
        "task",
        "timestamp_ms",
        "sampling_rate_hz",
        "gaze_x",
        "gaze_y",
        "is_valid",
        "source_file",
        "coordinate_unit",
        "tracked_eye",
        "stimulus_text",
        "stimulus_condition",
    ]
    assert recordings[0]["dataset_id"].eq("Pedrotti").all()
    assert recordings[0]["participant_id"].eq("01").all()
    assert [recording["recording_id"].iat[0] for recording in recordings] == [
        "01_trial_001",
        "01_trial_002",
        "01_trial_003",
    ]
    assert recordings[0]["tracked_eye"].eq("left").all()
    assert recordings[0]["coordinate_unit"].eq("pixel").all()
    assert recordings[0]["gaze_x"].iat[0] == pytest.approx(-12.0)
    assert np.isnan(recordings[0]["gaze_x"].iat[1])
    assert recordings[0]["is_valid"].tolist() == [True, False]
    assert recordings[1]["stimulus_text"].iat[0] == "1293"
    assert recordings[1]["stimulus_condition"].iat[0] == "number"
    assert recordings[1]["is_valid"].tolist() == [False]
    assert recordings[2]["stimulus_condition"].iat[0] == "pseudoword"
    assert recordings[0]["sampling_rate_hz"].eq(1000.0).all()
    assert recordings[0]["session_id"].eq("session_1").all()
    assert recordings[0]["task"].eq("reading_aloud").all()
    assert recordings[0]["timestamp_ms"].tolist() == [100, 101]
    assert recordings[0]["source_file"].eq(str(file_path)).all()


def test_loader_selects_right_eye(tmp_path: Path):
    file_path = tmp_path / "data/eyetracking/raw/Pedrotti/02.txt"
    write_pedrotti_csv(file_path, [right_row(1, 100, "unir", 11.0, 12.0, 700.0)])

    recording = load_pedrotti_reading(file_path)[0]

    assert recording["tracked_eye"].eq("right").all()
    assert recording["gaze_x"].tolist() == pytest.approx([11.0])
    assert recording["stimulus_condition"].iat[0] == "word"


@pytest.mark.parametrize("eye_state", ["both", "neither"])
def test_loader_rejects_ambiguous_or_missing_eye(tmp_path: Path, eye_state: str):
    file_path = tmp_path / "data/eyetracking/raw/Pedrotti/03.txt"
    row = left_row(1, 100, "amer", 1.0, 2.0, 3.0)
    if eye_state == "both":
        row.update(
            {
                "RIGHT_GAZE_X": 4.0,
                "RIGHT_GAZE_Y": 5.0,
                "RIGHT_PUPIL_SIZE": 6.0,
            }
        )
    else:
        row.update(
            {
                "LEFT_GAZE_X": ".",
                "LEFT_GAZE_Y": ".",
                "LEFT_PUPIL_SIZE": ".",
            }
        )
    write_pedrotti_csv(file_path, [row])

    with pytest.raises(ValueError, match="exactly one usable Pedrotti eye"):
        load_pedrotti_reading(file_path)
