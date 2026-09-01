from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from gap_imputation_benchmark.artifacts.selector import (
    ArtifactCompatibilityError,
    LoadedSelectorArtifact,
    select_method,
    validate_selector_payload,
)
from gap_imputation_benchmark.benchmark.gaps import create_artificial_gap, find_gap_candidates
from gap_imputation_benchmark.benchmark.imputers import ImputationResult
from gap_imputation_benchmark.benchmark.records import FEATURES_BASIC_NORMALIZED


class FakeModel:
    n_features_in_ = len(FEATURES_BASIC_NORMALIZED)
    n_outputs_ = 7

    def predict(self, values: object) -> object:
        return values


class FakeImputer:
    def transform(self, values: object) -> object:
        return values


def valid_payload() -> dict[str, object]:
    return {
        "domain": "eye_tracking",
        "model": FakeModel(),
        "feature_imputer": FakeImputer(),
        "feature_columns": list(FEATURES_BASIC_NORMALIZED),
        "method_names": [
            "forward_fill",
            "nearest_boundary",
            "linear",
            "pchip",
            "local_natural_cubic_spline",
            "polyfit_bic",
            "template",
        ],
    }


def test_valid_payload_matches_current_eye_tracking_interface():
    artifact = validate_selector_payload(valid_payload(), expected_domain="eye_tracking")

    assert artifact.domain == "eye_tracking"
    assert artifact.feature_columns == FEATURES_BASIC_NORMALIZED
    assert artifact.method_names[-1] == "template"


def test_feature_order_mismatch_is_rejected_before_inference():
    payload = valid_payload()
    payload["feature_columns"] = list(reversed(FEATURES_BASIC_NORMALIZED))

    with pytest.raises(ArtifactCompatibilityError, match="feature_columns"):
        validate_selector_payload(payload)


def test_model_width_mismatch_is_rejected_before_inference():
    payload = valid_payload()
    payload["model"] = type("WrongWidthModel", (FakeModel,), {"n_features_in_": 15})()

    with pytest.raises(ArtifactCompatibilityError, match="input width"):
        validate_selector_payload(payload)


def test_domain_mismatch_is_rejected_before_inference():
    with pytest.raises(ArtifactCompatibilityError, match="not 'weather'"):
        validate_selector_payload(valid_payload(), expected_domain="weather")


def test_real_gap_selection_uses_artifact_floor_and_excludes_inapplicable_methods():
    recording = pd.DataFrame(
        {
            "gaze_x": np.arange(12, dtype=float),
            "timestamp_ms": np.arange(12, dtype=float) * 10,
            "sampling_rate_hz": [100.0] * 12,
            "is_valid": [True] * 12,
        }
    )
    gap = create_artificial_gap(recording, find_gap_candidates(recording, 20.0)[0])

    class DecisionModel:
        n_features_in_ = len(FEATURES_BASIC_NORMALIZED)
        n_outputs_ = 2

        def predict(self, values: object) -> np.ndarray:
            assert len(values) == 1
            return np.array([[0.4, 0.01]])

    artifact = LoadedSelectorArtifact(
        domain="eye_tracking",
        model=DecisionModel(),
        feature_imputer=FakeImputer(),
        feature_columns=FEATURES_BASIC_NORMALIZED,
        method_names=("linear", "template"),
        metadata={"feature_scale_floor": {"value": 1.0}},
    )
    methods = {
        "linear": lambda _: ImputationResult("linear", np.array([1.0, 2.0]), True, None, {}),
        "template": lambda _: ImputationResult("template", np.array([]), False, "No template", {}),
    }

    decision = select_method(gap.masked_recording, gap.candidate, artifact, imputer_methods=methods)

    assert decision.selected_method == "linear"
    assert not decision.applicability["template"]
    assert decision.inapplicability_reasons["template"] == "No template"
    assert set(decision.feature_values) == set(FEATURES_BASIC_NORMALIZED)
