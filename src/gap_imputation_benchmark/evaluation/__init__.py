"""Evaluation metrics and domain-independent exploratory diagnostics."""

from gap_imputation_benchmark.evaluation.exploration import (
    plot_coverage,
    plot_macro_lodo_comparison,
    plot_mean_method_error,
    plot_oracle_distribution,
    plot_oracle_frequency,
    plot_selected_method_frequency,
)

__all__ = [
    "plot_coverage",
    "plot_macro_lodo_comparison",
    "plot_mean_method_error",
    "plot_oracle_distribution",
    "plot_oracle_frequency",
    "plot_selected_method_frequency",
]
