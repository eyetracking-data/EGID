"""PCHIP-default decision rule for per-method gain challengers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ChallengerDecision:
    selected_method: str
    switch_reason: str
    lower_confidence_bounds: dict[str, float]


def select_challenger(
    applicability: Mapping[str, bool],
    predicted_gain: Mapping[str, float],
    uncertainty: Mapping[str, float],
    *,
    minimum_gain: float,
    z_value: float,
    score: str = "lcb",
) -> ChallengerDecision:
    """Keep PCHIP unless an applicable challenger clears both safeguards."""
    if score not in {"lcb", "mean"}:
        raise ValueError("score must be 'lcb' or 'mean'.")
    lower_bounds: dict[str, float] = {}
    accepted: list[tuple[float, str]] = []
    for method, applicable in applicability.items():
        gain = float(predicted_gain.get(method, np.nan))
        std = float(uncertainty.get(method, np.nan))
        lcb = gain - z_value * std
        lower_bounds[method] = lcb
        if applicable and np.isfinite(gain) and np.isfinite(std) and gain >= minimum_gain and lcb > 0:
            accepted.append(((lcb if score == "lcb" else gain), method))
    if not accepted:
        return ChallengerDecision("pchip", "retain_pchip_no_challenger_passed", lower_bounds)
    _, method = max(accepted, key=lambda item: (item[0], item[1]))
    return ChallengerDecision(method, "switch_highest_" + score, lower_bounds)
