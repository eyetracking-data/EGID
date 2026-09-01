"""Backward-compatible Eye-Tracking imports.

The domain implementation now lives in :mod:`gap_imputation_benchmark.domains`
so that the benchmark core stays domain agnostic.
"""

from gap_imputation_benchmark.domains.eyetracking.adapter import (
    EyeTrackingBenchmarkConfig,
    load_eye_tracking_benchmark_config,
    portable_source_reference,
    replace_source_reference,
    select_balanced_recordings,
)

__all__ = [
    "EyeTrackingBenchmarkConfig",
    "load_eye_tracking_benchmark_config",
    "portable_source_reference",
    "replace_source_reference",
    "select_balanced_recordings",
]
