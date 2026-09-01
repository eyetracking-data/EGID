"""Load selector artifacts only when their public contract matches this code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from gap_imputation_benchmark.benchmark.features import selector_feature_values
from gap_imputation_benchmark.benchmark.gaps import ArtificialGap, GapCandidate
from gap_imputation_benchmark.benchmark.imputers import ImputationResult
from gap_imputation_benchmark.benchmark.records import FEATURES_BASIC_NORMALIZED
from gap_imputation_benchmark.domains.base import Imputer
from gap_imputation_benchmark.imputers.registry import (
    EYE_TRACKING_IMPUTERS,
    TRAFFIC_IMPUTERS,
    WEATHER_IMPUTERS,
)


class ArtifactCompatibilityError(ValueError):
    """Raised when an artifact cannot safely be used by the current code."""


class ArtifactUnavailableError(FileNotFoundError):
    """Raised when a requested local artifact does not exist."""


@dataclass(frozen=True)
class LoadedSelectorArtifact:
    """A validated selector model and the metadata needed to use it safely."""

    domain: str
    model: Any
    feature_imputer: Any
    feature_columns: tuple[str, ...]
    method_names: tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SelectorDecision:
    """An observable selector decision for one real missing segment."""

    selected_method: str
    predicted_nrmse: Mapping[str, float]
    applicability: Mapping[str, bool]
    inapplicability_reasons: Mapping[str, str | None]
    feature_values: Mapping[str, float]


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"{label} must be a JSON object.")
    return value


def _require_string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ArtifactCompatibilityError(f"{label} must be a non-empty list of strings.")
    if len(set(value)) != len(value):
        raise ArtifactCompatibilityError(f"{label} must not contain duplicate entries.")
    return tuple(value)


def _expected_features_for_domain(domain: str) -> tuple[str, ...] | None:
    if domain == "eye_tracking":
        return tuple(FEATURES_BASIC_NORMALIZED)
    if domain == "weather":
        return tuple(
            "realized_gap_duration_hours" if feature == "realized_gap_duration_ms" else feature
            for feature in FEATURES_BASIC_NORMALIZED
        )
    if domain == "traffic":
        return tuple(
            "realized_gap_duration_minutes" if feature == "realized_gap_duration_ms" else feature
            for feature in FEATURES_BASIC_NORMALIZED
        )
    return None


def validate_selector_payload(
    payload: Mapping[str, Any],
    *,
    expected_domain: str | None = None,
) -> LoadedSelectorArtifact:
    """Validate metadata, shape, and code-level feature compatibility.

    This does not execute model inference. It establishes that the serialized
    model was trained for the same domain, features, and method ordering that
    the current code exposes.
    """
    payload = _require_mapping(payload, "Selector artifact")
    domain = payload.get("domain")
    if not isinstance(domain, str) or not domain:
        raise ArtifactCompatibilityError("Selector artifact domain must be a non-empty string.")
    if expected_domain is not None and domain != expected_domain:
        raise ArtifactCompatibilityError(
            f"Selector artifact is for domain {domain!r}, not {expected_domain!r}."
        )

    feature_columns = _require_string_list(payload.get("feature_columns"), "feature_columns")
    method_names = _require_string_list(payload.get("method_names"), "method_names")
    expected_features = _expected_features_for_domain(domain)
    if expected_features is not None and feature_columns != expected_features:
        raise ArtifactCompatibilityError(
            "Selector artifact feature_columns do not match the current "
            f"{domain} feature interface."
        )

    model = payload.get("model")
    feature_imputer = payload.get("feature_imputer")
    if not callable(getattr(model, "predict", None)):
        raise ArtifactCompatibilityError("Selector artifact model must provide predict().")
    if not callable(getattr(feature_imputer, "transform", None)):
        raise ArtifactCompatibilityError("Selector artifact feature_imputer must provide transform().")

    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and n_features != len(feature_columns):
        raise ArtifactCompatibilityError(
            "Selector artifact model input width does not match feature_columns."
        )
    n_outputs = getattr(model, "n_outputs_", None)
    if n_outputs is not None and n_outputs != len(method_names):
        raise ArtifactCompatibilityError(
            "Selector artifact model output width does not match method_names."
        )

    return LoadedSelectorArtifact(
        domain=domain,
        model=model,
        feature_imputer=feature_imputer,
        feature_columns=feature_columns,
        method_names=method_names,
        metadata=payload,
    )


def load_selector_artifact(
    path: Path,
    *,
    expected_domain: str | None = None,
) -> LoadedSelectorArtifact:
    """Deserialize and validate one local selector artifact.

    Artifacts are intentionally loaded from an explicit local path. Downloading
    is kept outside the library, so use stays simple and never has hidden
    network access.
    """
    path = Path(path)
    if not path.is_file():
        raise ArtifactUnavailableError(f"Selector artifact does not exist: {path}")

    try:
        import joblib
    except ImportError as error:
        raise ImportError(
            "Loading selector artifacts requires the optional 'rf' dependencies. "
            "Install them with: python -m pip install -e '.[rf]'"
        ) from error
    payload = joblib.load(path)
    return validate_selector_payload(
        payload,
        expected_domain=expected_domain,
    )


def _artifact_feature_scale_floor(artifact: LoadedSelectorArtifact) -> float:
    value = artifact.metadata.get("feature_scale_floor")
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError("Selector artifact is missing feature_scale_floor metadata.")
    floor = value.get("value")
    try:
        floor = float(floor)
    except (TypeError, ValueError) as error:
        raise ArtifactCompatibilityError("Selector artifact feature_scale_floor must be numeric.") from error
    if not np.isfinite(floor) or floor <= 0:
        raise ArtifactCompatibilityError("Selector artifact feature_scale_floor must be finite and positive.")
    return floor


def select_method(
    recording: pd.DataFrame,
    gap: GapCandidate,
    artifact: LoadedSelectorArtifact,
    *,
    imputer_methods: Mapping[str, Imputer] | None = None,
) -> SelectorDecision:
    """Select the lowest-predicted applicable method for a real missing gap.

    ``recording`` must already contain the real missing segment; only its
    observed boundaries and predefined context windows are read.  The stored
    training scale floor is used directly, and methods that cannot reconstruct
    this particular gap are excluded before the predicted errors are compared.
    """
    if artifact.domain not in {"eye_tracking", "weather", "traffic"}:
        raise ArtifactCompatibilityError(
            "select_method currently requires an eye_tracking, weather, or traffic selector artifact."
        )
    default_methods = {
        "eye_tracking": EYE_TRACKING_IMPUTERS,
        "weather": WEATHER_IMPUTERS,
        "traffic": TRAFFIC_IMPUTERS,
    }[artifact.domain]
    methods = default_methods if imputer_methods is None else imputer_methods
    missing_methods = set(artifact.method_names) - set(methods)
    if missing_methods:
        raise ArtifactCompatibilityError(
            f"No implementation is registered for artifact methods: {sorted(missing_methods)}"
        )

    artificial_gap = ArtificialGap(
        candidate=gap,
        ground_truth=np.array([], dtype=float),
        masked_recording=recording,
    )
    values = selector_feature_values(artificial_gap, _artifact_feature_scale_floor(artifact))
    if artifact.domain == "weather":
        values["realized_gap_duration_hours"] = values.pop("realized_gap_duration_ms") / 3_600_000.0
    if artifact.domain == "traffic":
        values["realized_gap_duration_minutes"] = values.pop("realized_gap_duration_ms") / 60_000.0
    missing_features = set(artifact.feature_columns) - set(values)
    if missing_features:
        raise ArtifactCompatibilityError(
            f"Current feature extraction does not provide: {sorted(missing_features)}"
        )
    frame = pd.DataFrame([{column: values[column] for column in artifact.feature_columns}])
    transformed = artifact.feature_imputer.transform(frame)
    prediction = np.asarray(artifact.model.predict(transformed), dtype=float)
    if prediction.shape != (1, len(artifact.method_names)) or not np.isfinite(prediction).all():
        raise ArtifactCompatibilityError("Selector model returned invalid predicted nRMSE values.")

    outcomes: dict[str, ImputationResult] = {}
    for name in artifact.method_names:
        try:
            outcomes[name] = methods[name](artificial_gap)
        except Exception as error:
            outcomes[name] = ImputationResult(name, np.array([], dtype=float), False, f"Method raised {type(error).__name__}: {error}", {})
    applicability = {name: outcome.method_applicable for name, outcome in outcomes.items()}
    reasons = {name: outcome.failure_reason for name, outcome in outcomes.items()}
    predicted_nrmse = dict(zip(artifact.method_names, prediction[0], strict=True))
    applicable_predictions = {
        name: value for name, value in predicted_nrmse.items() if applicability[name]
    }
    if not applicable_predictions:
        raise ArtifactCompatibilityError("No artifact method is applicable to this gap.")
    selected_method = min(applicable_predictions, key=applicable_predictions.__getitem__)
    return SelectorDecision(
        selected_method=selected_method,
        predicted_nrmse=predicted_nrmse,
        applicability=applicability,
        inapplicability_reasons=reasons,
        feature_values={column: float(values[column]) for column in artifact.feature_columns},
    )
