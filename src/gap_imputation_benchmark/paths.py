"""Project-wide path helpers that avoid machine-specific source-code paths."""

from __future__ import annotations

import os
from pathlib import Path

# Resolve the repository root from this file, so source-code paths stay portable.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# This location may hold small public example data in the future.
REPOSITORY_DATA_DIR = PROJECT_ROOT / "data"
DATA_DIRECTORY_VARIABLE = "BENCHMARK_DATA_DIR"


def load_local_environment(
    env_path: Path | None = None,
    *,
    override: bool = False,
) -> Path | None:
    """Load simple ``KEY=VALUE`` entries from the ignored local ``.env`` file.

    Existing process variables take precedence by default, which lets a shell
    or continuous-integration environment override local workstation settings.
    The parser deliberately supports only the small, documented subset needed
    by this project and does not execute content from the file.
    """
    path = PROJECT_ROOT / ".env" if env_path is None else Path(env_path)
    if not path.is_file():
        return None

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()

        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not name.isidentifier():
            raise ValueError(f"Invalid .env entry at {path}:{line_number}")

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or name not in os.environ:
            os.environ[name] = value

    return path


def local_path_or_default(variable_name: str, default_path: Path) -> Path:
    """Resolve an optional local path setting, falling back to a project path.

    Workflow notebooks use this for generated benchmark, evaluation, and
    artifact directories. A local ``.env`` value remains authoritative, while
    a fresh checkout can use a documented, repository-relative default.
    """
    configured_path = os.getenv(variable_name)
    path = Path(configured_path).expanduser() if configured_path else Path(default_path)
    return path.resolve()


def project_relative_path_or_label(path: Path, *, fallback_label: str) -> str:
    """Return a portable project-relative path, or a safe generic label.

    Command-line status messages must not reveal a contributor's absolute
    workstation path when output is intentionally configured outside the
    repository.
    """
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return fallback_label


def get_example_data_dir() -> Path:
    """Return the repository location reserved for small example data."""
    return REPOSITORY_DATA_DIR


def require_raw_data_dir() -> Path:
    """Return the local raw-data directory configured for this machine.

    Raw benchmark data is intentionally kept outside the repository. Failing
    early prevents scripts from silently running against an empty data folder.
    """
    load_local_environment()
    configured_path = os.getenv(DATA_DIRECTORY_VARIABLE)
    if not configured_path:
        raise RuntimeError(
            f"{DATA_DIRECTORY_VARIABLE} is not set. Copy .env.example to .env "
            "or set the variable directly."
        )

    data_dir = Path(configured_path).expanduser()
    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"{DATA_DIRECTORY_VARIABLE} does not point to a directory: {data_dir}"
        )

    return data_dir
