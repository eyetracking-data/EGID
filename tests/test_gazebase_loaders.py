from pathlib import Path

import pandas as pd

from gap_imputation_benchmark.loaders.gazebase import load_gazebase_reading
from gap_imputation_benchmark.loaders.gazebase_vr import load_gazebase_vr_reading


def test_gazebase_reading_sets_degree_coordinate_unit(tmp_path: Path):
    file_path = tmp_path / "S_1001_S1_TEX.csv"
    pd.DataFrame({"n": [0, 1], "x": [1.5, 2.5], "val": [0, 0]}).to_csv(
        file_path, index=False
    )

    recording = load_gazebase_reading(file_path)

    assert recording["coordinate_unit"].eq("degree").all()


def test_gazebase_vr_reading_sets_degree_coordinate_unit(tmp_path: Path):
    file_path = tmp_path / "S_1001_S1_4_TEX.csv"
    pd.DataFrame({"n": [0, 1], "x": [1.5, 2.5]}).to_csv(file_path, index=False)

    recording = load_gazebase_vr_reading(file_path)

    assert recording["coordinate_unit"].eq("degree").all()
