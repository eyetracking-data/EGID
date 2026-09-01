import pandas as pd
import pytest

from gap_imputation_benchmark.benchmark.splits import (
    ParticipantSplits,
    assign_recordings_to_splits,
    split_participants,
)


def recording(dataset_id: str, participant_id: str, session_id: str = "S1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": [dataset_id, dataset_id],
            "participant_id": [participant_id, participant_id],
            "session_id": [session_id, session_id],
            "value": [1.0, 2.0],
        }
    )


def recordings(n_participants: int) -> list[pd.DataFrame]:
    return [recording("GazeBase", f"{index:04d}") for index in range(n_participants)]


def all_keys(splits: ParticipantSplits) -> set[tuple[str, str]]:
    return set(splits.train) | set(splits.validation) | set(splits.test)


def test_seeded_split_is_deterministic_and_independent_of_recording_order():
    source = recordings(20)

    first = split_participants(source, random_state=17)
    second = split_participants(source, random_state=17)
    reversed_order = split_participants(list(reversed(source)), random_state=17)

    assert first == second == reversed_order


def test_default_allocation_is_approximately_70_15_15():
    splits = split_participants(recordings(20), random_state=3)

    assert (len(splits.train), len(splits.validation), len(splits.test)) == (14, 3, 3)


def test_largest_remainder_rounding_assigns_remainders_by_fractional_part():
    splits = split_participants(
        recordings(11),
        train_fraction=0.50,
        validation_fraction=0.30,
        test_fraction=0.20,
        random_state=5,
    )

    assert (len(splits.train), len(splits.validation), len(splits.test)) == (6, 3, 2)


def test_participant_keys_are_exclusive_and_same_identifier_in_datasets_is_distinct():
    source = [
        recording("GazeBase", "same"),
        recording("GazeBaseVR", "same"),
        recording("GazeBase", "other"),
    ]

    splits = split_participants(source, random_state=9)
    split_sets = [set(splits.train), set(splits.validation), set(splits.test)]

    assert all_keys(splits) == {
        ("GazeBase", "same"),
        ("GazeBaseVR", "same"),
        ("GazeBase", "other"),
    }
    assert all(first.isdisjoint(second) for index, first in enumerate(split_sets) for second in split_sets[index + 1 :])


def test_assignment_keeps_all_recordings_for_each_participant_together_and_preserves_objects():
    source = [
        recording("GazeBase", "participant_1", "S1"),
        recording("GazeBase", "participant_1", "S2"),
        recording("GazeBase", "participant_2", "S1"),
        recording("GazeBase", "participant_3", "S1"),
    ]
    splits = split_participants(source, random_state=1)
    assigned = assign_recordings_to_splits(source, splits)

    participant_1_split = next(
        name
        for name, assigned_recordings in assigned.items()
        if any(frame["participant_id"].iloc[0] == "participant_1" for frame in assigned_recordings)
    )
    assert sum(
        frame["participant_id"].iloc[0] == "participant_1"
        for frame in assigned[participant_1_split]
    ) == 2
    assert {
        id(frame) for frames in assigned.values() for frame in frames
    } == {id(frame) for frame in source}
    assert sum(len(frames) for frames in assigned.values()) == len(source)


@pytest.mark.parametrize(
    "malformed",
    [
        pd.DataFrame({"participant_id": ["one"]}),
        pd.DataFrame({"dataset_id": ["GazeBase"]}),
        pd.DataFrame({"dataset_id": [None], "participant_id": ["one"]}),
        pd.DataFrame({"dataset_id": ["GazeBase"], "participant_id": [None]}),
        pd.DataFrame({"dataset_id": ["A", "B"], "participant_id": ["one", "one"]}),
        pd.DataFrame({"dataset_id": ["A", "A"], "participant_id": ["one", "two"]}),
    ],
)
def test_split_rejects_malformed_recording_identifiers(malformed: pd.DataFrame):
    with pytest.raises(ValueError):
        split_participants([malformed, *recordings(3)])


def test_split_rejects_fewer_than_three_unique_participants():
    with pytest.raises(ValueError, match="At least three"):
        split_participants(recordings(2))


def test_split_rejects_fractions_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        split_participants(recordings(5), train_fraction=0.6)


def test_assignment_rejects_recordings_not_present_in_splits():
    splits = split_participants(recordings(3))

    with pytest.raises(ValueError, match="absent from splits"):
        assign_recordings_to_splits([recording("GazeBase", "missing")], splits)
