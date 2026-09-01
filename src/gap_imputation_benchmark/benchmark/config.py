"""Validated parameter objects for corpus-level benchmark generation."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CorpusBenchmarkConfig:
    """User-controlled parameters for corpus-level benchmark generation."""

    min_gap_duration_ms: float
    max_gap_duration_ms: float
    n_gaps_per_recording: int
    n_strata: int = 5
    context_multiplier: float = 1.0
    min_context_valid_fraction: float = 0.80
    random_state: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    scale_floor_quantile: float = 0.01

    def __post_init__(self) -> None:
        """Validate benchmark settings without changing caller-supplied values."""
        if not math.isfinite(self.min_gap_duration_ms) or self.min_gap_duration_ms <= 0:
            raise ValueError("min_gap_duration_ms must be positive.")
        if (
            not math.isfinite(self.max_gap_duration_ms)
            or self.max_gap_duration_ms < self.min_gap_duration_ms
        ):
            raise ValueError(
                "max_gap_duration_ms must be greater than or equal to "
                "min_gap_duration_ms."
            )
        if self.n_strata < 1:
            raise ValueError("n_strata must be at least 1.")
        if self.n_gaps_per_recording < 1:
            raise ValueError("n_gaps_per_recording must be at least 1.")
        if (
            not math.isfinite(self.context_multiplier)
            or self.context_multiplier <= 0
        ):
            raise ValueError("context_multiplier must be positive.")
        if not (
            math.isfinite(self.min_context_valid_fraction)
            and 0 < self.min_context_valid_fraction <= 1
        ):
            raise ValueError(
                "min_context_valid_fraction must be greater than 0 and at most 1."
            )

        split_fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if not all(math.isfinite(fraction) and fraction > 0 for fraction in split_fractions):
            raise ValueError("All split fractions must be positive.")
        if not math.isclose(
            sum(split_fractions),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Split fractions must sum to 1.")
        if not (
            math.isfinite(self.scale_floor_quantile)
            and 0 < self.scale_floor_quantile < 1
        ):
            raise ValueError("scale_floor_quantile must be between 0 and 1.")
