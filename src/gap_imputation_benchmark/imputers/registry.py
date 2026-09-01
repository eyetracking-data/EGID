"""Single source of truth for candidate-method names and implementations."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from gap_imputation_benchmark.benchmark.gaps import ArtificialGap
from gap_imputation_benchmark.benchmark.imputers import (
    ImputationResult,
    impute_forward_fill,
    impute_linear,
    impute_local_natural_cubic_spline,
    impute_nearest_boundary,
    impute_pchip,
    impute_polyfit_bic,
    impute_template,
    SeasonalPeriodConfig,
    impute_seasonal_periodic,
)


Imputer = Callable[[ArtificialGap], ImputationResult]

EYE_TRACKING_IMPUTERS: dict[str, Imputer] = {
    "forward_fill": impute_forward_fill,
    "nearest_boundary": impute_nearest_boundary,
    "linear": impute_linear,
    "pchip": impute_pchip,
    "local_natural_cubic_spline": impute_local_natural_cubic_spline,
    "polyfit_bic": impute_polyfit_bic,
    "template": impute_template,
}
EYE_TRACKING_METHODS = tuple(EYE_TRACKING_IMPUTERS)


# Frozen Weather reference protocol.  It deliberately uses the pointwise mean:
# that is the setting used to create the published DWD reference tables.
WEATHER_SEASONAL_CONFIG = SeasonalPeriodConfig(
    period_strategy="calendar_year",
    period_value=1,
    candidates_per_direction=3,
    min_candidates=3,
    max_offsets=8,
    aggregation="mean",
    leap_day_policy="skip",
    mode="bidirectional",
)
WEATHER_IMPUTERS: dict[str, Imputer] = {
    **EYE_TRACKING_IMPUTERS,
    "seasonal_periodic": partial(
        impute_seasonal_periodic,
        config=WEATHER_SEASONAL_CONFIG,
        timestamp_column="timestamp",
    ),
}
WEATHER_METHODS = tuple(WEATHER_IMPUTERS)


# Frozen LargeST reference protocol: 5-minute traffic flow, weekly references,
# up to three complete segments in either direction, and a pointwise median.
TRAFFIC_SEASONAL_CONFIG = SeasonalPeriodConfig(
    period_strategy="fixed_timedelta",
    period_value="7 days",
    candidates_per_direction=3,
    min_candidates=3,
    max_offsets=8,
    aggregation="median",
    mode="bidirectional",
)
TRAFFIC_IMPUTERS: dict[str, Imputer] = {
    **EYE_TRACKING_IMPUTERS,
    "seasonal_periodic": partial(
        impute_seasonal_periodic,
        config=TRAFFIC_SEASONAL_CONFIG,
        timestamp_column="timestamp",
    ),
}
TRAFFIC_METHODS = tuple(TRAFFIC_IMPUTERS)
