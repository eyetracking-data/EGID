from __future__ import annotations

from pathlib import Path

import pytest

from gap_imputation_benchmark.benchmark.configuration import load_corpus_benchmark_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG = REPOSITORY_ROOT / "configs" / "reference_benchmark.toml"


def test_reference_configuration_loads_expected_values() -> None:
    config = load_corpus_benchmark_config(REFERENCE_CONFIG)

    assert config.random_state == 42
    assert config.n_gaps_per_recording == 8
    assert config.n_strata == 4
    assert config.min_gap_duration_ms == 20.0
    assert config.max_gap_duration_ms == 80.0


def test_configuration_rejects_unknown_keys(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.toml"
    config_file.write_text(
        REFERENCE_CONFIG.read_text(encoding="utf-8") + "unexpected_setting = true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown keys"):
        load_corpus_benchmark_config(config_file)
