"""Load version-controlled benchmark configurations from TOML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility path.
    import tomli as tomllib

from gap_imputation_benchmark.benchmark.config import CorpusBenchmarkConfig


_BENCHMARK_TABLE = "benchmark"
_EXPECTED_KEYS = frozenset(CorpusBenchmarkConfig.__dataclass_fields__)


def load_corpus_benchmark_config(file_path: str | Path) -> CorpusBenchmarkConfig:
    """Load one complete corpus-benchmark configuration from a TOML file.

    The loader rejects missing and unknown keys so experiment settings cannot be
    silently ignored because of a spelling mistake or outdated configuration.
    """
    config_path = Path(file_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Benchmark configuration does not exist: {config_path}")

    try:
        with config_path.open("rb") as config_file:
            document: dict[str, Any] = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid TOML benchmark configuration: {config_path}") from error

    values = document.get(_BENCHMARK_TABLE)
    if not isinstance(values, dict):
        raise ValueError(
            f"Benchmark configuration must define a [{_BENCHMARK_TABLE}] table."
        )

    supplied_keys = frozenset(values)
    missing_keys = sorted(_EXPECTED_KEYS - supplied_keys)
    unknown_keys = sorted(supplied_keys - _EXPECTED_KEYS)
    if missing_keys or unknown_keys:
        problems: list[str] = []
        if missing_keys:
            problems.append(f"missing keys: {missing_keys}")
        if unknown_keys:
            problems.append(f"unknown keys: {unknown_keys}")
        raise ValueError("Invalid benchmark configuration (" + "; ".join(problems) + ").")

    return CorpusBenchmarkConfig(**values)
