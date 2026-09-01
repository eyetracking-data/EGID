"""Leakage-free gain regressors for PCHIP challenger experiments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


@dataclass
class GainPrediction:
    """Mean and tree-wise standard deviation of a gain prediction."""

    mean: np.ndarray
    std: np.ndarray


class ExtraTreesGainRegressor:
    """Extra-Trees model whose uncertainty is tree-prediction dispersion.

    This is deliberately a lightweight heuristic, not a calibrated predictive
    interval. The experiment evaluates it out of fold and uses it only to make
    the challenger decision more conservative.
    """

    def __init__(self, random_state: int = 42, n_estimators: int = 300) -> None:
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=3,
            max_features=1.0,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ExtraTreesGainRegressor":
        self.model.fit(x, y)
        return self

    def predict_with_uncertainty(self, x: np.ndarray) -> GainPrediction:
        tree_predictions = np.vstack([tree.predict(x) for tree in self.model.estimators_])
        return GainPrediction(
            mean=tree_predictions.mean(axis=0),
            std=tree_predictions.std(axis=0),
        )
