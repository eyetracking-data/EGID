"""Fairly evaluate registered imputers on identical artificial gaps.

The evaluator isolates method failures, guards against input mutation, and
records comparable RMSE/MAE outcomes without introducing fallback predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from gap_imputation_benchmark.benchmark.gaps import ArtificialGap
from gap_imputation_benchmark.benchmark.imputers import ImputationResult
from gap_imputation_benchmark.imputers.registry import EYE_TRACKING_IMPUTERS
from gap_imputation_benchmark.benchmark.metrics import calculate_mae, calculate_rmse


@dataclass(frozen=True)
class ImputerEvaluation:
    """Evaluation outcome for one imputer on one artificial gap."""

    method_name: str
    method_applicable: bool
    failure_reason: str | None
    predictions: np.ndarray
    rmse: float | None
    mae: float | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class GapEvaluation:
    """All imputer evaluations and their per-gap summary."""

    method_results: list[ImputerEvaluation]
    best_method_by_rmse: str | None
    best_rmse: float | None
    best_method_by_mae: str | None
    best_mae: float | None
    number_of_applicable_methods: int


Imputer = Callable[[ArtificialGap], ImputationResult]

IMPUTER_METHODS: dict[str, Imputer] = EYE_TRACKING_IMPUTERS


def _append_failure_reason(
    existing_reason: str | None,
    new_reason: str,
) -> str:
    return f"{existing_reason}; {new_reason}" if existing_reason else new_reason


def _input_is_unchanged(
    artificial_gap: ArtificialGap,
    original_candidate: object,
    original_recording: object,
) -> bool:
    """Check that methods share, and do not alter, the supplied gap inputs."""
    return (
        artificial_gap.candidate == original_candidate
        and artificial_gap.masked_recording.equals(original_recording)
    )


def evaluate_imputers_on_gap(
    artificial_gap: ArtificialGap,
    imputer_methods: dict[str, Imputer] | None = None,
) -> GapEvaluation:
    """Run a fixed imputer registry on one gap without sharing mutable state.

    ``None`` selects the Eye-Tracking registry. Domain-specific benchmarks can
    pass an explicit registry (for example, the same seven methods plus a
    periodic seasonal method) without changing the Eye-Tracking benchmark.
    """
    methods = IMPUTER_METHODS if imputer_methods is None else imputer_methods
    if not methods:
        raise ValueError("imputer_methods must contain at least one method.")
    original_candidate = artificial_gap.candidate
    original_recording = artificial_gap.masked_recording.copy(deep=True)
    ground_truth = np.asarray(artificial_gap.ground_truth, dtype=float)
    results: list[ImputerEvaluation] = []

    for method_name, imputer in methods.items():
        if not _input_is_unchanged(
            artificial_gap,
            original_candidate,
            original_recording,
        ):
            results.append(
                ImputerEvaluation(
                    method_name=method_name,
                    method_applicable=False,
                    failure_reason="A previous method changed the shared gap input.",
                    predictions=np.array([], dtype=float),
                    rmse=None,
                    mae=None,
                    metadata={},
                )
            )
            continue

        try:
            imputation = imputer(artificial_gap)
        except Exception as error:
            results.append(
                ImputerEvaluation(
                    method_name=method_name,
                    method_applicable=False,
                    failure_reason=f"Method raised {type(error).__name__}: {error}",
                    predictions=np.array([], dtype=float),
                    rmse=None,
                    mae=None,
                    metadata={},
                )
            )
            continue

        predictions = np.asarray(imputation.predictions, dtype=float)
        method_applicable = imputation.method_applicable
        failure_reason = imputation.failure_reason
        rmse: float | None = None
        mae: float | None = None

        if method_applicable and len(predictions) != len(ground_truth):
            method_applicable = False
            failure_reason = _append_failure_reason(
                failure_reason,
                "Prediction length does not match ground-truth length.",
            )
        elif method_applicable and not np.isfinite(predictions).all():
            method_applicable = False
            failure_reason = _append_failure_reason(
                failure_reason,
                "Predictions contain non-finite values.",
            )
        elif method_applicable:
            try:
                rmse = calculate_rmse(ground_truth, predictions)
                mae = calculate_mae(ground_truth, predictions)
            except ValueError as error:
                method_applicable = False
                failure_reason = _append_failure_reason(failure_reason, str(error))

        results.append(
            ImputerEvaluation(
                method_name=method_name,
                method_applicable=method_applicable,
                failure_reason=failure_reason,
                predictions=predictions,
                rmse=rmse,
                mae=mae,
                metadata=imputation.metadata,
            )
        )

    scored_results = [result for result in results if result.rmse is not None]
    best_by_rmse = min(scored_results, key=lambda result: result.rmse, default=None)
    best_by_mae = min(
        (result for result in results if result.mae is not None),
        key=lambda result: result.mae,
        default=None,
    )
    return GapEvaluation(
        method_results=results,
        best_method_by_rmse=best_by_rmse.method_name if best_by_rmse else None,
        best_rmse=best_by_rmse.rmse if best_by_rmse else None,
        best_method_by_mae=best_by_mae.method_name if best_by_mae else None,
        best_mae=best_by_mae.mae if best_by_mae else None,
        number_of_applicable_methods=sum(
            result.method_applicable for result in results
        ),
    )


def evaluate_all_imputers_on_gap(
    artificial_gap: ArtificialGap,
) -> GapEvaluation:
    """Run the unchanged default Eye-Tracking imputer registry on one gap."""
    return evaluate_imputers_on_gap(artificial_gap)
