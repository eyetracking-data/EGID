"""Metrics for a PCHIP-default challenger selector."""
from __future__ import annotations

import numpy as np
import pandas as pd


def selection_metrics(table: pd.DataFrame, selected_column: str, pchip_column: str = "pchip_error", oracle_column: str = "oracle_error") -> dict[str, float]:
    """Summarize error, regret, and safety of a selector's decisions."""
    selected = pd.to_numeric(table[selected_column], errors="coerce").to_numpy(float)
    pchip = pd.to_numeric(table[pchip_column], errors="coerce").to_numpy(float)
    oracle = pd.to_numeric(table[oracle_column], errors="coerce").to_numpy(float)
    switched = table["selected_method"].ne("pchip").to_numpy(bool)
    actual_gain = pchip - selected
    meaningful = (table["best_alternative_gain"].to_numpy(float) > 0)
    successful = switched & (actual_gain > 0)
    false_switch = switched & (actual_gain <= 0)
    missed = ~switched & meaningful
    damage = -actual_gain[false_switch]
    benefit = actual_gain[successful]
    return {
        "n_gaps": len(table),
        "mean_nrmse": float(np.mean(selected)),
        "mean_improvement_vs_pchip": float(np.mean(actual_gain)),
        "relative_improvement_vs_pchip": float((np.mean(pchip) - np.mean(selected)) / np.mean(pchip)) if np.mean(pchip) > 0 else np.nan,
        "mean_regret_vs_oracle": float(np.mean(selected - oracle)),
        "switch_rate": float(np.mean(switched)),
        "retain_pchip_rate": float(1 - np.mean(switched)),
        "switch_precision": float(np.mean(successful[switched])) if switched.any() else np.nan,
        "switch_recall": float(np.mean(switched[meaningful])) if meaningful.any() else np.nan,
        "false_switch_rate": float(np.mean(false_switch)),
        "missed_opportunity_rate": float(np.mean(missed)),
        "mean_false_switch_damage": float(np.mean(damage)) if len(damage) else 0.0,
        "median_false_switch_damage": float(np.median(damage)) if len(damage) else 0.0,
        "p90_false_switch_damage": float(np.quantile(damage, .9)) if len(damage) else 0.0,
        "max_false_switch_damage": float(np.max(damage)) if len(damage) else 0.0,
        "mean_successful_switch_benefit": float(np.mean(benefit)) if len(benefit) else 0.0,
        "net_benefit_per_gap": float(np.sum(benefit) - np.sum(damage)) / len(table),
    }
