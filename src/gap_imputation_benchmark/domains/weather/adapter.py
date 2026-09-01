"""Frozen configuration and public contract for the DWD Weather benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gap_imputation_benchmark.benchmark.features import RAW_FEATURE_NUMERATOR_COLUMNS
from gap_imputation_benchmark.benchmark.records import FEATURES_BASIC_NORMALIZED
from gap_imputation_benchmark.domains.base import DomainSpec
from gap_imputation_benchmark.imputers.registry import WEATHER_IMPUTERS


WEATHER_FEATURE_COLUMNS = tuple(
    "realized_gap_duration_hours" if name == "realized_gap_duration_ms" else name
    for name in FEATURES_BASIC_NORMALIZED
)
WEATHER_METHOD_LABELS = {
    "forward_fill": "Forward fill",
    "nearest_boundary": "Nearest boundary",
    "linear": "Linear interpolation",
    "pchip": "PCHIP",
    "local_natural_cubic_spline": "Local natural cubic spline",
    "polyfit_bic": "Polyfit (BIC)",
    "template": "Template reconstruction",
    "seasonal_periodic": "Seasonal periodic reference",
}


@dataclass(frozen=True)
class WeatherBenchmarkConfig:
    """The exact DWD 2000–2025 reference sampling design."""

    start: str
    end: str
    gaps_per_stratum: int
    gap_strata_hours: tuple[tuple[int, int], ...]
    min_context_valid_fraction: float
    min_context_samples: int
    max_gap_attempts: int
    random_state: int
    feature_scale_floor_quantile: float

    def __post_init__(self) -> None:
        if not self.start or not self.end or self.start >= self.end:
            raise ValueError("Weather start must precede end.")
        if self.gaps_per_stratum < 1 or self.min_context_samples < 1 or self.max_gap_attempts < 1:
            raise ValueError("Weather sampling counts must be positive.")
        if not 0 < self.min_context_valid_fraction <= 1:
            raise ValueError("min_context_valid_fraction must be in (0, 1].")
        if not 0 < self.feature_scale_floor_quantile < 1:
            raise ValueError("feature_scale_floor_quantile must be in (0, 1).")
        if not self.gap_strata_hours or any(lower < 1 or upper < lower for lower, upper in self.gap_strata_hours):
            raise ValueError("Weather gap strata must be non-empty positive inclusive ranges.")


_CONFIG_TABLE = "benchmark"
_EXPECTED_CONFIG_KEYS = frozenset(WeatherBenchmarkConfig.__dataclass_fields__)


def load_weather_benchmark_config(path: str | Path) -> WeatherBenchmarkConfig:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib

    path = Path(path)
    with path.open("rb") as handle:
        document: dict[str, Any] = tomllib.load(handle)
    values = document.get(_CONFIG_TABLE)
    if not isinstance(values, dict):
        raise ValueError(f"Configuration must define a [{_CONFIG_TABLE}] table.")
    missing, unknown = sorted(_EXPECTED_CONFIG_KEYS - set(values)), sorted(set(values) - _EXPECTED_CONFIG_KEYS)
    if missing or unknown:
        problems = ([f"missing keys: {missing}"] if missing else []) + ([f"unknown keys: {unknown}"] if unknown else [])
        raise ValueError("Invalid Weather configuration (" + "; ".join(problems) + ").")
    strata = tuple(tuple(int(value) for value in pair) for pair in values["gap_strata_hours"])
    return WeatherBenchmarkConfig(**{**values, "gap_strata_hours": strata})


WEATHER_DOMAIN = DomainSpec(
    name="weather",
    feature_columns=WEATHER_FEATURE_COLUMNS,
    methods=WEATHER_IMPUTERS,
    scale_dependent_feature_numerators={
        ("realized_gap_duration_hours" if feature == "realized_gap_duration_ms" else feature): numerator
        for feature, numerator in RAW_FEATURE_NUMERATOR_COLUMNS.items()
    },
    method_labels=WEATHER_METHOD_LABELS,
)
