from __future__ import annotations

import numpy as np
import pandas as pd

from gap_imputation_benchmark.domains.weather.workflow import (
    candidate_from_gap_row,
    learn_feature_scale_floor,
    model_parameters_from_row,
)


def test_weather_scale_floor_uses_only_positive_finite_local_iqrs():
    gaps = pd.DataFrame({"local_context_iqr": [np.nan, -1.0, 0.0, 2.0, 4.0]})

    assert learn_feature_scale_floor(gaps, 0.5) == 3.0


def test_weather_gap_candidate_reconstructs_hourly_manifest_geometry():
    row = pd.Series({
        "gap_start_idx": 10,
        "gap_end_idx": 13,
        "gap_length_samples": 3,
        "left_context_start_idx": 7,
        "right_context_end_idx": 16,
        "left_context_valid_fraction": 0.8,
        "right_context_valid_fraction": 1.0,
        "duration_stratum": 2,
    })

    candidate = candidate_from_gap_row(row)

    assert candidate.gap_length_samples == 3
    assert candidate.requested_gap_duration_ms == 3 * 3_600_000.0
    assert candidate.context_valid_fraction == 0.9


def test_weather_model_parameters_restore_csv_types():
    row = pd.Series({
        "n_estimators": 400.0,
        "min_samples_leaf": 5.0,
        "max_depth": np.nan,
        "max_features": "1.0",
    })

    assert model_parameters_from_row(row) == {
        "n_estimators": 400,
        "min_samples_leaf": 5,
        "max_depth": None,
        "max_features": 1.0,
    }
