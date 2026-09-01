"""Small deterministic checks for the portable Traffic protocol code."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gap_imputation_benchmark.domains.traffic import TRAFFIC_FEATURE_COLUMNS
from gap_imputation_benchmark.domains.traffic.workflow import (
    candidate_from_gap_row,
    learn_feature_scale_floor,
    model_parameters_from_row,
    stable_rng,
)


def test_traffic_random_stream_is_stable_for_the_same_sensor_year():
    first = stable_rng(20260824, "gap-positions", 3, 2017, 123).integers(0, 10_000, 4)
    second = stable_rng(20260824, "gap-positions", 3, 2017, 123).integers(0, 10_000, 4)

    assert np.array_equal(first, second)


def test_traffic_gap_candidate_reconstructs_global_panel_geometry():
    row = pd.Series({
        "gap_start_in_year": 10,
        "gap_end_in_year": 13,
        "gap_length_steps": 3,
        "left_context_start_in_year": 7,
        "right_context_end_in_year": 16,
        "left_context_valid_fraction": 0.8,
        "right_context_valid_fraction": 1.0,
        "duration_stratum": 2,
    })

    candidate = candidate_from_gap_row(row, year_start=105_120)

    assert candidate.gap_start_idx == 105_130
    assert candidate.gap_end_idx == 105_133
    assert candidate.requested_gap_duration_ms == 3 * 300_000.0
    assert candidate.context_valid_fraction == 0.9


def test_traffic_scale_floor_and_csv_parameters_follow_the_shared_contract():
    gaps = pd.DataFrame({"local_context_iqr": [np.nan, -1.0, 0.0, 2.0, 4.0]})
    row = pd.Series({
        "n_estimators": 400.0,
        "min_samples_leaf": 5.0,
        "max_depth": np.nan,
        "max_features": "sqrt",
    })

    assert learn_feature_scale_floor(gaps, 0.5) == 3.0
    assert model_parameters_from_row(row) == {
        "n_estimators": 400,
        "min_samples_leaf": 5,
        "max_depth": None,
        "max_features": "sqrt",
    }
    assert TRAFFIC_FEATURE_COLUMNS[0] == "realized_gap_duration_minutes"
