from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gap_imputation_benchmark.benchmark.evaluation import (
    IMPUTER_METHODS,
    evaluate_all_imputers_on_gap,
)
from gap_imputation_benchmark.benchmark.gaps import create_artificial_gap, find_gap_candidates
from gap_imputation_benchmark.benchmark.imputers import ImputationResult


def artificial_gap() -> object:
    recording = pd.DataFrame(
        {
            "gaze_x": np.arange(10, dtype=float),
            "is_valid": [True] * 10,
            "sampling_rate_hz": [100.0] * 10,
            "timestamp_ms": np.arange(10, dtype=float) * 10.0,
        }
    )
    candidate = find_gap_candidates(recording, gap_duration_ms=20)[0]
    return create_artificial_gap(recording, candidate)


def result(method_name: str, predictions: list[float]) -> ImputationResult:
    return ImputationResult(
        method_name=method_name,
        predictions=np.asarray(predictions, dtype=float),
        method_applicable=True,
        failure_reason=None,
        metadata={"source": method_name},
    )


def test_evaluator_runs_all_methods_on_the_same_gap():
    gap = artificial_gap()
    evaluation = evaluate_all_imputers_on_gap(gap)

    assert [record.method_name for record in evaluation.method_results] == list(
        IMPUTER_METHODS
    )
    assert len(evaluation.method_results) == 7
    assert all(
        len(record.predictions) == len(gap.ground_truth)
        for record in evaluation.method_results
        if record.method_applicable
    )


def test_one_failing_method_does_not_stop_remaining_evaluation(monkeypatch):
    monkeypatch.setitem(IMPUTER_METHODS, "forward_fill", lambda gap: (_ for _ in ()).throw(RuntimeError("boom")))

    evaluation = evaluate_all_imputers_on_gap(artificial_gap())
    records = {record.method_name: record for record in evaluation.method_results}

    assert not records["forward_fill"].method_applicable
    assert "RuntimeError" in records["forward_fill"].failure_reason
    assert any(record.method_applicable for record in evaluation.method_results[1:])


def test_evaluator_selects_best_method_and_calculates_known_metrics(monkeypatch):
    gap = replace(artificial_gap(), ground_truth=np.array([2.0, 4.0]))
    monkeypatch.setattr(
            "gap_imputation_benchmark.benchmark.evaluation.IMPUTER_METHODS",
        {
            "worse": lambda current_gap: result("worse", [3.0, 6.0]),
            "best": lambda current_gap: result("best", [2.0, 4.0]),
        },
    )

    evaluation = evaluate_all_imputers_on_gap(gap)
    worse = evaluation.method_results[0]

    assert worse.rmse == pytest.approx(np.sqrt(2.5))
    assert worse.mae == pytest.approx(1.5)
    assert evaluation.best_method_by_rmse == "best"
    assert evaluation.best_rmse == 0.0
    assert evaluation.best_method_by_mae == "best"
    assert evaluation.best_mae == 0.0


@pytest.mark.parametrize(
    "predictions, reason",
    [([1.0], "Prediction length"), ([1.0, np.inf], "non-finite")],
)
def test_evaluator_rejects_invalid_successful_output(monkeypatch, predictions, reason):
    monkeypatch.setattr(
            "gap_imputation_benchmark.benchmark.evaluation.IMPUTER_METHODS",
        {"invalid": lambda gap: result("invalid", predictions)},
    )

    record = evaluate_all_imputers_on_gap(artificial_gap()).method_results[0]

    assert not record.method_applicable
    assert reason in record.failure_reason
    assert record.rmse is None
    assert record.mae is None


def test_evaluator_preserves_method_metadata_and_no_fallback(monkeypatch):
    metadata = {
        "selected_degree": 2,
        "selected_bic": 1.0,
        "left_support_point_count": 2,
        "template_candidate_sides": ["left", "right"],
        "fallback_used": False,
    }
    monkeypatch.setattr(
            "gap_imputation_benchmark.benchmark.evaluation.IMPUTER_METHODS",
        {
            "polyfit_bic": lambda gap: ImputationResult(
                "polyfit_bic", np.array([2.0, 3.0]), True, None, metadata
            ),
            "template": lambda gap: ImputationResult(
                "template", np.array([], dtype=float), False, "No candidate", metadata
            ),
        },
    )

    evaluation = evaluate_all_imputers_on_gap(artificial_gap())
    records = {record.method_name: record for record in evaluation.method_results}

    assert records["polyfit_bic"].metadata == metadata
    assert records["template"].metadata == metadata
    assert records["template"].rmse is None
    assert records["template"].mae is None
