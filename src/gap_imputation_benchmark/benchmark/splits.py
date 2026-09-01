"""Participant-level split helpers for corpus benchmark inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import random

import pandas as pd


ParticipantKey = tuple[str, str]
_SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class ParticipantSplits:
    """Disjoint participant keys allocated to train, validation, and test."""

    train: tuple[ParticipantKey, ...]
    validation: tuple[ParticipantKey, ...]
    test: tuple[ParticipantKey, ...]

    def __post_init__(self) -> None:
        split_sets = tuple(set(getattr(self, name)) for name in _SPLIT_NAMES)
        if any(len(split) == 0 for split in split_sets):
            raise ValueError("Each participant split must contain at least one key.")
        if sum(len(split) for split in split_sets) != len(set().union(*split_sets)):
            raise ValueError("Participant keys must appear in exactly one split.")


def _validate_split_fractions(
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[float, float, float]:
    fractions = (train_fraction, validation_fraction, test_fraction)
    if not all(math.isfinite(fraction) and fraction > 0 for fraction in fractions):
        raise ValueError("All split fractions must be positive.")
    total = sum(fractions)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to 1.")
    return tuple(fraction / total for fraction in fractions)


def _participant_key(recording: pd.DataFrame) -> ParticipantKey:
    """Return a recording's sole non-null dataset and participant identifiers."""
    required_columns = {"dataset_id", "participant_id"}
    missing_columns = required_columns - set(recording.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required identifier columns: {sorted(missing_columns)}"
        )

    identifiers: list[str] = []
    for column_name in ("dataset_id", "participant_id"):
        values = recording[column_name].dropna().unique()
        if len(values) != 1:
            raise ValueError(
                f"Each recording must contain exactly one non-null {column_name}."
            )
        identifiers.append(str(values[0]))

    return identifiers[0], identifiers[1]


def _largest_remainder_counts(
    n_participants: int,
    fractions: tuple[float, float, float],
) -> tuple[int, int, int]:
    ideal_counts = tuple(n_participants * fraction for fraction in fractions)
    counts = [math.floor(count) for count in ideal_counts]
    remaining = n_participants - sum(counts)
    order = sorted(
        range(len(counts)),
        key=lambda index: (-(ideal_counts[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1

    # Standard largest-remainder allocation can leave a small split empty for
    # small cohorts (for example, 3 participants at 70/15/15). Rebalance only
    # as needed to uphold the public non-empty-split requirement.
    for recipient in (index for index, count in enumerate(counts) if count == 0):
        donor = max(
            (index for index, count in enumerate(counts) if count > 1),
            key=lambda index: (counts[index], -index),
        )
        counts[donor] -= 1
        counts[recipient] += 1
    return tuple(counts)


def split_participants(
    recordings: Sequence[pd.DataFrame],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
) -> ParticipantSplits:
    """Create deterministic, disjoint participant-level train/validation/test splits."""
    fractions = _validate_split_fractions(
        train_fraction,
        validation_fraction,
        test_fraction,
    )
    participant_keys = sorted({_participant_key(recording) for recording in recordings})
    if len(participant_keys) < 3:
        raise ValueError("At least three unique participant keys are required.")

    rng = random.Random(random_state)
    rng.shuffle(participant_keys)
    train_count, validation_count, test_count = _largest_remainder_counts(
        len(participant_keys),
        fractions,
    )
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("Each participant split must contain at least one key.")

    validation_end = train_count + validation_count
    return ParticipantSplits(
        train=tuple(participant_keys[:train_count]),
        validation=tuple(participant_keys[train_count:validation_end]),
        test=tuple(participant_keys[validation_end:]),
    )


def assign_recordings_to_splits(
    recordings: Sequence[pd.DataFrame],
    splits: ParticipantSplits,
) -> dict[str, list[pd.DataFrame]]:
    """Assign complete input recordings to their participant-level split."""
    key_to_split = {
        key: split_name
        for split_name in _SPLIT_NAMES
        for key in getattr(splits, split_name)
    }
    assigned = {split_name: [] for split_name in _SPLIT_NAMES}

    for recording in recordings:
        key = _participant_key(recording)
        try:
            split_name = key_to_split[key]
        except KeyError as error:
            raise ValueError(f"Recording participant key is absent from splits: {key}") from error
        assigned[split_name].append(recording)

    return assigned
