from dataclasses import FrozenInstanceError

import pytest

from gap_imputation_benchmark.benchmark.config import CorpusBenchmarkConfig


def config(**overrides: object) -> CorpusBenchmarkConfig:
    values: dict[str, object] = {
        "min_gap_duration_ms": 12.5,
        "max_gap_duration_ms": 987.5,
        "n_strata": 4,
        "n_gaps_per_recording": 8,
    }
    values.update(overrides)
    return CorpusBenchmarkConfig(**values)


def test_config_keeps_user_controlled_duration_limits_unchanged():
    benchmark_config = config(min_gap_duration_ms=0.125, max_gap_duration_ms=2_500.75)

    assert benchmark_config.min_gap_duration_ms == 0.125
    assert benchmark_config.max_gap_duration_ms == 2_500.75


def test_config_is_frozen_and_accepts_defaults():
    benchmark_config = config()

    assert benchmark_config.context_multiplier == 1.0
    assert benchmark_config.train_fraction == 0.70
    with pytest.raises(FrozenInstanceError):
        benchmark_config.n_strata = 2  # type: ignore[misc]


def test_config_uses_five_duration_strata_by_default():
    benchmark_config = CorpusBenchmarkConfig(
        min_gap_duration_ms=12.5,
        max_gap_duration_ms=987.5,
        n_gaps_per_recording=8,
    )

    assert benchmark_config.n_strata == 5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min_gap_duration_ms": 0}, "min_gap_duration_ms"),
        ({"max_gap_duration_ms": 12.4}, "max_gap_duration_ms"),
        ({"n_strata": 0}, "n_strata"),
        ({"n_gaps_per_recording": 0}, "n_gaps_per_recording"),
        ({"context_multiplier": 0}, "context_multiplier"),
        ({"min_context_valid_fraction": 0}, "min_context_valid_fraction"),
        ({"min_context_valid_fraction": 1.1}, "min_context_valid_fraction"),
        ({"train_fraction": 0}, "split fractions"),
        ({"train_fraction": 0.6}, "sum to 1"),
        ({"scale_floor_quantile": 0}, "scale_floor_quantile"),
        ({"scale_floor_quantile": 1}, "scale_floor_quantile"),
    ],
)
def test_config_rejects_invalid_values(overrides: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        config(**overrides)
