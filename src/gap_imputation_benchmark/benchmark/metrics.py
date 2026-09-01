from __future__ import annotations

import numpy as np


def _validated_metric_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert and validate metric inputs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("y_true and y_pred must contain only finite values.")

    return y_true, y_pred


def calculate_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true, y_pred = _validated_metric_inputs(y_true, y_pred)

    return float(
        np.sqrt(np.mean((y_true - y_pred) ** 2))
    )


def calculate_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true, y_pred = _validated_metric_inputs(y_true, y_pred)

    return float(
        np.mean(np.abs(y_true - y_pred))
    )
