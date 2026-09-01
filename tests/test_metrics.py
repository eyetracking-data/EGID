import numpy as np
import pytest

from gap_imputation_benchmark.benchmark.metrics import calculate_mae, calculate_rmse


@pytest.mark.parametrize("metric", [calculate_rmse, calculate_mae])
def test_metrics_calculate_expected_values(metric):
    assert metric([1.0, 3.0], [2.0, 5.0]) == pytest.approx(
        np.sqrt(2.5) if metric is calculate_rmse else 1.5
    )


@pytest.mark.parametrize("metric", [calculate_rmse, calculate_mae])
@pytest.mark.parametrize(
    ("y_true", "y_pred", "match"),
    [
        ([], [], "must not be empty"),
        ([1.0], [1.0, 2.0], "same shape"),
        ([np.nan], [1.0], "finite values"),
        ([1.0], [np.inf], "finite values"),
    ],
)
def test_metrics_reject_invalid_inputs(metric, y_true, y_pred, match):
    with pytest.raises(ValueError, match=match):
        metric(y_true, y_pred)
