"""Reusable descriptive plots for any versioned gap-imputation benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


def _plotting() -> Any:
    """Import plotting support only for exploratory notebook use."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Exploratory plots require matplotlib. Install it with: "
            "python -m pip install -e '.[analysis]'"
        ) from error
    return plt


def _labels(methods: Sequence[str], method_labels: Mapping[str, str] | None) -> dict[str, str]:
    return {method: method_labels.get(method, method) if method_labels else method for method in methods}


def plot_coverage(coverage: pd.DataFrame, *, ax: Any | None = None) -> Any:
    """Plot requested, learnable, and excluded gaps for each dataset or domain."""
    required = {"dataset_id", "requested_gaps", "learnable_gaps", "excluded_gaps"}
    missing = required - set(coverage.columns)
    if missing:
        raise ValueError(f"Coverage table is missing columns: {sorted(missing)}")
    plt = _plotting()
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    coverage.set_index("dataset_id")[["requested_gaps", "learnable_gaps", "excluded_gaps"]].plot(kind="bar", ax=ax)
    ax.set(title="Benchmark coverage by dataset", xlabel="Dataset", ylabel="Number of gaps")
    ax.tick_params(axis="x", rotation=0)
    return ax


def plot_oracle_frequency(
    gaps: pd.DataFrame,
    methods: Sequence[str],
    *,
    oracle_column: str = "best_method_by_nrmse",
    method_labels: Mapping[str, str] | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot how often each candidate method is the offline oracle overall."""
    if oracle_column not in gaps:
        raise ValueError(f"Missing oracle column: {oracle_column}")
    labels = _labels(methods, method_labels)
    counts = gaps[oracle_column].value_counts().reindex(methods, fill_value=0).sort_values()
    plt = _plotting()
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    ax.barh([labels[method] for method in counts.index], counts.values, color="#3569a8")
    ax.set(title="Offline oracle frequency", xlabel="Number of artificial gaps", ylabel="Method")
    total = len(gaps)
    for position, count in enumerate(counts.values):
        share = 100 * count / total if total else 0.0
        ax.text(count, position, f"  {count} ({share:.1f}%)", va="center")
    return ax


def plot_oracle_distribution(
    gaps: pd.DataFrame,
    methods: Sequence[str],
    *,
    group_column: str,
    oracle_column: str = "best_method_by_nrmse",
    method_labels: Mapping[str, str] | None = None,
    group_label: str | None = None,
    ax: Any | None = None,
) -> tuple[Any, pd.DataFrame]:
    """Plot offline-oracle shares for any categorical benchmark grouping."""
    required = {group_column, oracle_column}
    missing = required - set(gaps.columns)
    if missing:
        raise ValueError(f"Gap table is missing columns: {sorted(missing)}")
    shares = pd.crosstab(gaps[group_column], gaps[oracle_column], normalize="index").reindex(columns=methods, fill_value=0)
    plt = _plotting()
    ax = ax or plt.subplots(figsize=(10, 5))[1]
    shares.rename(columns=_labels(methods, method_labels)).plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    label = group_label or group_column
    ax.set(title=f"Offline oracle distribution by {label}", xlabel=label, ylabel="Share of gaps")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    return ax, shares


def plot_mean_method_error(
    gaps: pd.DataFrame,
    methods: Sequence[str],
    *,
    group_column: str = "dataset_id",
    error_suffix: str = "_nrmse",
    method_labels: Mapping[str, str] | None = None,
    ax: Any | None = None,
) -> tuple[Any, pd.DataFrame]:
    """Plot mean method error by dataset, duration stratum, or another group."""
    columns = [f"{method}{error_suffix}" for method in methods]
    missing = {group_column, *columns} - set(gaps.columns)
    if missing:
        raise ValueError(f"Gap table is missing columns: {sorted(missing)}")
    means = gaps.groupby(group_column)[columns].mean().T
    means.index = [_labels(methods, method_labels)[method] for method in methods]
    plt = _plotting()
    ax = ax or plt.subplots(figsize=(13, 6))[1]
    means.plot(kind="bar", ax=ax)
    ax.set(title=f"Mean {error_suffix.removeprefix('_')} by method and {group_column}", xlabel="Imputation method", ylabel=f"Mean {error_suffix.removeprefix('_')}")
    ax.legend(title=group_column, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=25)
    return ax, means


def plot_macro_lodo_comparison(
    summary: pd.DataFrame,
    *,
    metric_columns: Mapping[str, str],
    evaluation_set_column: str = "evaluation_set",
    macro_row: str = "macro_mean_across_datasets",
    ax: Any | None = None,
) -> tuple[Any, pd.Series]:
    """Plot macro-average held-out error for named LODO comparison policies.

    ``metric_columns`` maps reader-facing policy labels to columns in the LODO
    summary. The macro row must already be the unweighted mean across held-out
    groups, so this function never pools individual gaps implicitly.
    """
    required = {evaluation_set_column, *metric_columns.values()}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"LODO summary is missing columns: {sorted(missing)}")
    macro = summary.loc[summary[evaluation_set_column].eq(macro_row)]
    if len(macro) != 1:
        raise ValueError(f"Expected exactly one macro LODO row named {macro_row!r}.")
    values = pd.to_numeric(macro.iloc[0].loc[list(metric_columns.values())], errors="coerce")
    if values.isna().any():
        raise ValueError("Macro LODO comparison values must be finite numbers.")
    values.index = list(metric_columns)

    plt = _plotting()
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    bars = ax.bar(values.index, values.values, color=("#6c757d", "#3569a8", "#4c956c"))
    ax.set(title="Macro LODO comparison", xlabel="Policy", ylabel="Mean held-out nRMSE")
    ax.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, values.values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.5f}", ha="center", va="bottom")
    return ax, values


def plot_selected_method_frequency(
    choice_counts: pd.DataFrame,
    methods: Sequence[str],
    *,
    method_column: str = "selected_method",
    count_column: str = "count",
    method_labels: Mapping[str, str] | None = None,
    ax: Any | None = None,
) -> tuple[Any, pd.DataFrame]:
    """Plot methods chosen across all held-out LODO gaps, with their shares."""
    required = {method_column, count_column}
    missing = required - set(choice_counts.columns)
    if missing:
        raise ValueError(f"Selected-method table is missing columns: {sorted(missing)}")
    counts = choice_counts.groupby(method_column)[count_column].sum().reindex(methods, fill_value=0)
    counts = pd.to_numeric(counts, errors="coerce")
    if counts.isna().any() or (counts < 0).any() or not counts.sum():
        raise ValueError("Selected-method counts must be non-negative and sum to more than zero.")
    result = pd.DataFrame({"count": counts, "share": counts / counts.sum()})
    labels = _labels(methods, method_labels)

    plt = _plotting()
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    bars = ax.barh([labels[method] for method in result.index], result["count"], color="#3569a8")
    ax.set(title="Methods selected on held-out gaps", xlabel="Number of held-out gaps", ylabel="Method")
    for bar, (_, row) in zip(bars, result.iterrows(), strict=True):
        ax.text(row["count"], bar.get_y() + bar.get_height() / 2, f"  {int(row['count'])} ({100 * row['share']:.1f}%)", va="center")
    return ax, result
