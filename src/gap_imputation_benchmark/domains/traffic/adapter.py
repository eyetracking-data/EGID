"""Frozen configuration and domain contract for 5-minute LargeST flow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gap_imputation_benchmark.benchmark.features import RAW_FEATURE_NUMERATOR_COLUMNS
from gap_imputation_benchmark.benchmark.records import FEATURES_BASIC_NORMALIZED
from gap_imputation_benchmark.domains.base import DomainSpec
from gap_imputation_benchmark.imputers.registry import TRAFFIC_IMPUTERS


TRAFFIC_FEATURE_COLUMNS = tuple(
    "realized_gap_duration_minutes" if name == "realized_gap_duration_ms" else name
    for name in FEATURES_BASIC_NORMALIZED
)
TRAFFIC_METHOD_LABELS = {
    "forward_fill": "Forward fill",
    "nearest_boundary": "Nearest boundary",
    "linear": "Linear interpolation",
    "pchip": "PCHIP",
    "local_natural_cubic_spline": "Local natural cubic spline",
    "polyfit_bic": "Polyfit (BIC)",
    "template": "Template reconstruction",
    "seasonal_periodic": "Weekly seasonal reference",
}


@dataclass(frozen=True)
class TrafficBenchmarkConfig:
    """Exact constants used for the historical Traffic reference benchmark."""

    districts: tuple[int, ...]
    years: tuple[int, ...]
    sensors_per_district_year: int
    gap_strata_steps: tuple[tuple[int, int], ...]
    min_context_valid_fraction: float
    min_context_steps: int
    max_gap_attempts: int
    random_state: int
    feature_scale_floor_quantile: float

    def __post_init__(self) -> None:
        if len(self.districts) < 3 or len(self.years) < 1:
            raise ValueError("Traffic requires at least three districts and one year.")
        if self.sensors_per_district_year < 1 or self.min_context_steps < 1:
            raise ValueError("Traffic sampling counts must be positive.")
        if self.max_gap_attempts < 1 or not 0 < self.min_context_valid_fraction <= 1:
            raise ValueError("Traffic sampling thresholds are invalid.")
        if not 0 < self.feature_scale_floor_quantile < 1:
            raise ValueError("feature_scale_floor_quantile must be in (0, 1).")
        if any(low < 1 or high < low for low, high in self.gap_strata_steps):
            raise ValueError("Traffic gap strata must be positive inclusive step ranges.")


_EXPECTED_KEYS = frozenset(TrafficBenchmarkConfig.__dataclass_fields__)


def load_traffic_benchmark_config(path: str | Path) -> TrafficBenchmarkConfig:
    """Read a strict TOML configuration so protocol changes are explicit."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib
    with Path(path).open("rb") as handle:
        document: dict[str, Any] = tomllib.load(handle)
    values = document.get("benchmark")
    if not isinstance(values, dict):
        raise ValueError("Configuration must define a [benchmark] table.")
    missing, unknown = sorted(_EXPECTED_KEYS - set(values)), sorted(set(values) - _EXPECTED_KEYS)
    if missing or unknown:
        details = ([f"missing keys: {missing}"] if missing else []) + ([f"unknown keys: {unknown}"] if unknown else [])
        raise ValueError("Invalid Traffic configuration (" + "; ".join(details) + ").")
    return TrafficBenchmarkConfig(
        **{
            **values,
            "districts": tuple(int(value) for value in values["districts"]),
            "years": tuple(int(value) for value in values["years"]),
            "gap_strata_steps": tuple(tuple(int(value) for value in pair) for pair in values["gap_strata_steps"]),
        }
    )


TRAFFIC_DOMAIN = DomainSpec(
    name="traffic",
    feature_columns=TRAFFIC_FEATURE_COLUMNS,
    methods=TRAFFIC_IMPUTERS,
    scale_dependent_feature_numerators={
        ("realized_gap_duration_minutes" if feature == "realized_gap_duration_ms" else feature): numerator
        for feature, numerator in RAW_FEATURE_NUMERATOR_COLUMNS.items()
    },
    method_labels=TRAFFIC_METHOD_LABELS,
)
