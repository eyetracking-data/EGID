"""Small shared helpers for self-contained provenance records."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def file_metadata(path: str | Path | None) -> dict[str, Any]:
    path = Path(path).expanduser().resolve(strict=False) if path else None
    if path is None:
        portable_path = None
    else:
        try:
            portable_path = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            portable_path = "external"
    result: dict[str, Any] = {
        "path": portable_path,
        "format": path.suffix.lstrip(".").lower() if path else None,
        "size_bytes": path.stat().st_size if path and path.is_file() else None,
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
        if path and path.is_file() else None,
    }
    if path and path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1_048_576), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def associated_agents(*, executor_name: str, executor_info_path: str | Path | None,
                      responsible_person_name: str | None,
                      responsible_person_info_path: str | Path | None) -> dict[str, Any]:
    """Build the document-only agent records shared by every activity."""
    def information_file(path: str | Path | None) -> dict[str, Any]:
        metadata = file_metadata(path)
        return metadata

    return {
        "executor": {"name": executor_name, "information_file": information_file(executor_info_path)},
        "responsible_person": {
            "name": responsible_person_name,
            "information_file": information_file(responsible_person_info_path),
        },
    }


def activity_record(*, implementation_path: Path, started: datetime,
                    finished: datetime,
                    details: dict[str, Any]) -> dict[str, Any]:
    return {
        "implementation": {
            "path": implementation_path.resolve().relative_to(implementation_path.resolve().parents[3]).as_posix(),
            "sha256": file_metadata(implementation_path)["sha256"],
        },
        "timing": {"started_at": started.isoformat(), "finished_at": finished.isoformat(),
                   "duration_seconds": (finished - started).total_seconds()},
        "details": details,
    }
