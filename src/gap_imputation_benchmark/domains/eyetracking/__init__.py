"""Eye-tracking-specific data adapter and protocol."""

from gap_imputation_benchmark.domains.eyetracking.adapter import (
    EYE_TRACKING_DOMAIN,
    EYE_TRACKING_METHOD_LABELS,
    EyeTrackingBenchmarkConfig,
    load_eye_tracking_benchmark_config,
    portable_source_reference,
    replace_source_reference,
    select_balanced_recordings,
)

__all__ = [
    "EYE_TRACKING_DOMAIN",
    "EYE_TRACKING_METHOD_LABELS",
    "EyeTrackingBenchmarkConfig",
    "load_eye_tracking_benchmark_config",
    "portable_source_reference",
    "replace_source_reference",
    "select_balanced_recordings",
]
