import numpy as np
import pytest

from gap_imputation_benchmark.benchmark.metrics import calculate_rmse
from gap_imputation_benchmark.benchmark.unit_robust import local_scale, normalized_rmse, recording_iqr_leave_gap_out


def test_positive_affine_scaling_preserves_normalized_rmse_and_oracle():
    gaze = np.array([-3., -2., -1., 0., 3., 5., 8., 13., 21., 34.])
    valid = np.ones(len(gaze), dtype=bool)
    start, end = 3, 6
    local_values = np.r_[gaze[:start], gaze[end:]]
    local_iqr = np.percentile(local_values, 75) - np.percentile(local_values, 25)
    truth = gaze[start:end]
    predictions = {"linear": np.array([0.2, 2.7, 5.4]), "pchip": np.array([-0.4, 3.2, 5.6]), "template": np.array([0.0, 3.8, 5.0])}
    raw = {name: calculate_rmse(truth, pred) for name, pred in predictions.items()}
    recording_iqr = recording_iqr_leave_gap_out(gaze, valid, start, end)
    scale = local_scale(local_iqr, recording_iqr)
    normalized = {name: normalized_rmse(value, scale) for name, value in raw.items()}
    a, b = 7.5, -11.0
    transformed = a * gaze + b
    transformed_truth = a * truth + b
    transformed_local = a * local_values + b
    transformed_iqr = np.percentile(transformed_local, 75) - np.percentile(transformed_local, 25)
    transformed_recording_iqr = recording_iqr_leave_gap_out(transformed, valid, start, end)
    transformed_scale = local_scale(transformed_iqr, transformed_recording_iqr)
    transformed_raw = {name: calculate_rmse(transformed_truth, a * pred + b) for name, pred in predictions.items()}
    assert transformed_iqr == pytest.approx(a * local_iqr)
    assert transformed_recording_iqr == pytest.approx(a * recording_iqr)
    assert transformed_scale == pytest.approx(a * scale)
    for name in predictions:
        assert transformed_raw[name] == pytest.approx(a * raw[name])
        assert normalized_rmse(transformed_raw[name], transformed_scale) == pytest.approx(normalized[name])
    assert min(raw, key=raw.get) == min(transformed_raw, key=transformed_raw.get)


def test_nonpositive_scale_is_excluded_from_normalized_metric():
    assert not np.isnan(local_scale(0.0, 0.0))
    assert local_scale(0.0, 0.0) == 0.0
    assert np.isnan(normalized_rmse(1.0, 0.0))
