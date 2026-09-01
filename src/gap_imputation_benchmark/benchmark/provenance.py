"""Stable recording identifiers used for deterministic benchmark sampling."""

from __future__ import annotations

import hashlib
import json

import pandas as pd


# These fields identify a recording across ordering changes in an input collection.
RECORDING_PROVENANCE_FIELDS = (
    "dataset_id",
    "participant_id",
    "recording_id",
    "session_id",
    "source_file",
)


def recording_provenance(recording: pd.DataFrame) -> tuple[str, ...]:
    """Build a stable textual identity for one recording.

    The result is used only for deterministic ordering, reproducible sampling
    seeds, and actionable error messages. It is not a model feature.
    """
    values: list[str] = []
    for field in RECORDING_PROVENANCE_FIELDS:
        if field not in recording.columns:
            values.append(f"{field}=<missing>")
            continue
        unique_values = recording[field].dropna().unique()
        if len(unique_values) == 1:
            values.append(f"{field}={unique_values[0]}")
        elif len(unique_values) == 0:
            values.append(f"{field}=<null>")
        else:
            values.append(f"{field}=<multiple>")
    return tuple(values)


def recording_seed(random_state: int, provenance: tuple[str, ...]) -> int:
    """Derive one stable random seed from a global seed and recording identity."""
    payload = json.dumps([random_state, *provenance], separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
