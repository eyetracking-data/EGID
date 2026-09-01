from __future__ import annotations

import os

import pytest

import gap_imputation_benchmark.paths as paths
from gap_imputation_benchmark.paths import (
    DATA_DIRECTORY_VARIABLE,
    PROJECT_ROOT,
    get_example_data_dir,
    local_path_or_default,
    load_local_environment,
    project_relative_path_or_label,
    require_raw_data_dir,
)


def test_example_data_dir_is_relative_to_the_repository() -> None:
    # The project must derive its own path without knowing a user's home directory.
    assert get_example_data_dir() == PROJECT_ROOT / "data"


def test_raw_data_dir_requires_explicit_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DATA_DIRECTORY_VARIABLE, raising=False)
    monkeypatch.setattr(paths, "load_local_environment", lambda: None)

    with pytest.raises(RuntimeError, match=DATA_DIRECTORY_VARIABLE):
        require_raw_data_dir()


def test_raw_data_dir_uses_configured_existing_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(DATA_DIRECTORY_VARIABLE, str(tmp_path))

    assert require_raw_data_dir() == tmp_path


def test_local_environment_file_preserves_process_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('BENCHMARK_DATA_DIR="/from-file"\nOTHER_VALUE=loaded\n', encoding="utf-8")
    monkeypatch.setenv(DATA_DIRECTORY_VARIABLE, "/from-process")

    assert load_local_environment(env_file) == env_file
    assert os.environ[DATA_DIRECTORY_VARIABLE] == "/from-process"
    assert os.environ["OTHER_VALUE"] == "loaded"


def test_local_path_or_default_prefers_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    configured = tmp_path / "configured"
    fallback = tmp_path / "fallback"
    monkeypatch.setenv("TEST_WORKFLOW_DIR", str(configured))

    assert local_path_or_default("TEST_WORKFLOW_DIR", fallback) == configured
    monkeypatch.delenv("TEST_WORKFLOW_DIR")
    assert local_path_or_default("TEST_WORKFLOW_DIR", fallback) == fallback


def test_project_relative_path_or_label_hides_external_absolute_paths(tmp_path) -> None:
    assert project_relative_path_or_label(PROJECT_ROOT / "benchmarks", fallback_label="configured output") == "benchmarks"
    assert project_relative_path_or_label(tmp_path, fallback_label="configured output") == "configured output"
