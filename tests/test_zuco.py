from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from gap_imputation_benchmark.loaders.zuco import load_zuco_normal_reading


def write_zuco_mat(path: Path) -> None:
    path.parent.mkdir(parents=True)
    savemat(
        path,
        {
            "colheader": np.array(
                ["TIME", "L_GAZE_X", "L_GAZE_Y", "L_AREA"], dtype=object
            ),
            "data": np.array(
                [
                    [100.0, 11.5, 12.0, 700.0],
                    [102.0, 0.0, 12.0, 0.0],
                    [104.0, np.nan, 12.0, 800.0],
                    [106.0, 22.5, 12.0, -1.0],
                    [108.0, 33.0, 0.0, 900.0],
                    [np.nan, 44.0, 12.0, 1_000.0],
                ]
            ),
        },
    )


def test_load_zuco_normal_reading_maps_common_schema(tmp_path: Path):
    file_path = tmp_path / "data/eyetracking/raw/ZUCO/YAC/YAC_NR1_ET.mat"
    write_zuco_mat(file_path)

    result = load_zuco_normal_reading(file_path)

    assert result.columns.tolist() == [
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
    ]
    assert result["dataset_id"].eq("ZuCo").all()
    assert result["participant_id"].eq("YAC").all()
    assert result["recording_id"].eq("YAC_NR1_ET").all()
    assert result["session_id"].eq("NR1").all()
    assert result["task"].eq("normal_reading").all()
    assert result["timestamp_ms"].iloc[:5].tolist() == [100.0, 102.0, 104.0, 106.0, 108.0]
    assert np.isnan(result["timestamp_ms"].iloc[5])
    assert result["sampling_rate_hz"].eq(500.0).all()
    assert result["gaze_x"].iloc[[0, 1, 3, 4]].tolist() == pytest.approx([11.5, 0.0, 22.5, 33.0])
    assert np.isnan(result["gaze_x"].iloc[2])
    assert result["is_valid"].tolist() == [True, False, False, False, True, False]
    assert result["source_file"].eq(str(file_path)).all()
    assert result["coordinate_unit"].eq("pixel").all()


@pytest.mark.parametrize(
    "relative_path",
    [
        "data/raw/ZUCO/YAC/YAC_TSR_ET.mat",
        "data/raw/ZUCO/YAC/YAC_NR1_EEG.mat",
        "data/raw/ZUCO/YAC/OTHER_NR1_ET.mat",
        "data/raw/OTHER/YAC/YAC_NR1_ET.mat",
    ],
)
def test_load_zuco_normal_reading_rejects_non_nr_et_paths(
    tmp_path: Path, relative_path: str
):
    file_path = tmp_path / relative_path

    with pytest.raises(ValueError):
        load_zuco_normal_reading(file_path)
