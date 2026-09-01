"""Leakage-safe, locally scaled evaluation helpers.

These helpers deliberately do not alter the benchmark's raw-DVA metrics.
"""
from __future__ import annotations

import numpy as np


def finite_valid_values_excluding_gap(
    gaze_x: np.ndarray,
    is_valid: np.ndarray,
    gap_start_idx: int,
    gap_end_idx: int,
) -> np.ndarray:
    """Observed values in one recording, excluding this artificial gap."""
    values = np.asarray(gaze_x, dtype=float)
    valid = np.asarray(is_valid, dtype=bool).copy()
    if not (0 <= gap_start_idx < gap_end_idx <= len(values)):
        raise ValueError("Invalid gap bounds.")
    valid[gap_start_idx:gap_end_idx] = False
    return values[valid & np.isfinite(values)]


def iqr(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def recording_iqr_leave_gap_out(
    gaze_x: np.ndarray,
    is_valid: np.ndarray,
    gap_start_idx: int,
    gap_end_idx: int,
) -> float:
    return iqr(finite_valid_values_excluding_gap(gaze_x, is_valid, gap_start_idx, gap_end_idx))


def local_scale(local_iqr: float, recording_iqr: float, alpha: float = 0.05) -> float:
    if not (np.isfinite(local_iqr) and np.isfinite(recording_iqr) and alpha > 0):
        return float("nan")
    return float(max(local_iqr, alpha * recording_iqr))


def normalized_rmse(rmse: float, scale: float) -> float:
    if not (np.isfinite(rmse) and np.isfinite(scale) and scale > 0):
        return float("nan")
    return float(rmse / scale)
