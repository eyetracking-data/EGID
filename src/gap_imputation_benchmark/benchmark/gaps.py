"""Gap geometry, eligibility checks, deterministic sampling, and masking.

The functions here define the shared artificial-gap protocol used by every
domain: observed samples become held-out ground truth while methods see only
the corresponding masked recording and local context.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import random
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GapCandidate:
    gap_start_idx: int
    gap_end_idx: int
    requested_gap_duration_ms: float
    realized_gap_duration_ms: float
    sampling_rate_hz: float
    gap_length_samples: int

    left_context_start_idx: int
    right_context_end_idx: int

    left_context_valid_fraction: float
    right_context_valid_fraction: float
    context_valid_fraction: float
    duration_stratum: int | None = None


def duration_ms_to_samples(
    duration_ms: float,
    sampling_rate_hz: float,
) -> int:
    """Convert a physical duration in milliseconds to at least one sample."""
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive.")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")

    n_samples = round(duration_ms / 1000.0 * sampling_rate_hz)
    return max(1, int(n_samples))


def find_gap_candidates(
    recording: pd.DataFrame,
    gap_duration_ms: float,
    context_multiplier: float = 1.0,
    min_context_valid_fraction: float = 0.80,
) -> list[GapCandidate]:
    """
    Find all possible artificial-gap positions satisfying:

    1. Every value inside the artificial gap is valid.
    2. The immediate value before and after the gap is valid.
    3. The surrounding context has at least the requested completeness.
    4. Context length on each side is at least two samples and otherwise
       equals context_multiplier * gap length.

    The gap end index is exclusive.
    """
    required_columns = {
        "gaze_x",
        "is_valid",
        "sampling_rate_hz",
    }
    missing_columns = required_columns - set(recording.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if recording.empty:
        return []

    if not 0 <= min_context_valid_fraction <= 1:
        raise ValueError(
            "min_context_valid_fraction must be between 0 and 1."
        )

    if context_multiplier <= 0:
        raise ValueError("context_multiplier must be positive.")

    sampling_rates = recording["sampling_rate_hz"].dropna().unique()

    if len(sampling_rates) != 1:
        raise ValueError(
            "Each recording must contain exactly one sampling rate."
        )

    sampling_rate_hz = float(sampling_rates[0])

    gap_length_samples = duration_ms_to_samples(
        duration_ms=gap_duration_ms,
        sampling_rate_hz=sampling_rate_hz,
    )

    context_length_samples = max(
        2,
        round(context_multiplier * gap_length_samples),
    )

    is_valid = recording["is_valid"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    ).to_numpy(dtype=bool)
    gaze_x = pd.to_numeric(
        recording["gaze_x"],
        errors="coerce",
    ).to_numpy(dtype=float)
    valid = is_valid & np.isfinite(gaze_x)

    first_possible_start = context_length_samples
    last_possible_start = len(recording) - gap_length_samples - context_length_samples
    if last_possible_start < first_possible_start:
        return []

    # Prefix sums preserve the original eligibility rules while replacing the
    # innermost Python loop with vectorized window counts. This matters when a
    # benchmark repeatedly probes long recordings at many gap durations.
    starts = np.arange(first_possible_start, last_possible_start + 1)
    ends = starts + gap_length_samples
    prefix_valid = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    gap_valid_counts = prefix_valid[ends] - prefix_valid[starts]
    left_valid_counts = prefix_valid[starts] - prefix_valid[
        starts - context_length_samples
    ]
    right_valid_counts = prefix_valid[ends + context_length_samples] - prefix_valid[ends]
    eligible = (
        (gap_valid_counts == gap_length_samples)
        & valid[starts - 1]
        & valid[ends]
        & (left_valid_counts / context_length_samples >= min_context_valid_fraction)
        & (right_valid_counts / context_length_samples >= min_context_valid_fraction)
    )

    candidates: list[GapCandidate] = []
    for gap_start_idx, gap_end_idx, left_count, right_count in zip(
        starts[eligible],
        ends[eligible],
        left_valid_counts[eligible],
        right_valid_counts[eligible],
        strict=True,
    ):
        left_fraction = float(left_count / context_length_samples)
        right_fraction = float(right_count / context_length_samples)
        candidates.append(
            GapCandidate(
                gap_start_idx=int(gap_start_idx),
                gap_end_idx=int(gap_end_idx),
                requested_gap_duration_ms=gap_duration_ms,
                realized_gap_duration_ms=(
                    gap_length_samples / sampling_rate_hz * 1000.0
                ),
                sampling_rate_hz=sampling_rate_hz,
                gap_length_samples=gap_length_samples,
                left_context_start_idx=int(gap_start_idx - context_length_samples),
                right_context_end_idx=int(gap_end_idx + context_length_samples),
                left_context_valid_fraction=left_fraction,
                right_context_valid_fraction=right_fraction,
                context_valid_fraction=(left_fraction + right_fraction) / 2.0,
            )
        )
    return candidates



class GapSamplingError(RuntimeError):
    """Raised when the requested number of non-overlapping gaps is infeasible."""

    def __init__(
        self,
        requested_gaps: int,
        selected_gaps: int,
        detail: str = "",
    ) -> None:
        message = (
            "Unable to sample the requested number of non-overlapping gaps: "
            f"requested {requested_gaps}, selected {selected_gaps}."
        )
        if detail:
            message = f"{message} {detail}"
        super().__init__(message)


def _overlaps_selected_window(
    candidate: GapCandidate,
    selected: list[GapCandidate],
) -> bool:
    """Return whether a candidate's full context-plus-gap window overlaps."""
    return any(
        candidate.left_context_start_idx < chosen.right_context_end_idx
        and candidate.right_context_end_idx > chosen.left_context_start_idx
        for chosen in selected
    )


def _sorted_selection_or_error(
    selected: list[GapCandidate],
    n_gaps: int,
    detail: str = "",
) -> list[GapCandidate]:
    """Return a complete selection, never a silent sampling shortfall."""
    if len(selected) != n_gaps:
        raise GapSamplingError(n_gaps, len(selected), detail)
    return sorted(selected, key=lambda candidate: candidate.gap_start_idx)


def sample_non_overlapping_candidates(
    candidates: list[GapCandidate],
    n_gaps: int,
    random_state: int = 42,
) -> list[GapCandidate]:
    """
    Randomly sample gap candidates such that their full windows
    (left context + gap + right context) do not overlap.
    """
    if n_gaps <= 0:
        raise ValueError("n_gaps must be positive.")

    rng = random.Random(random_state)
    shuffled = candidates.copy()
    rng.shuffle(shuffled)

    selected: list[GapCandidate] = []

    for candidate in shuffled:
        if _overlaps_selected_window(candidate, selected):
            continue

        selected.append(candidate)

        if len(selected) == n_gaps:
            break

    return _sorted_selection_or_error(
        selected,
        n_gaps,
        "Provide more feasible candidates or request fewer gaps.",
    )


@dataclass(frozen=True)
class ArtificialGap:
    candidate: GapCandidate
    ground_truth: np.ndarray
    masked_recording: pd.DataFrame


def create_artificial_gap(
    recording: pd.DataFrame,
    candidate: GapCandidate,
) -> ArtificialGap:
    """
    Create an artificial gaze_x gap while preserving the original values
    as ground truth.
    """
    if recording.empty:
        raise ValueError("recording must not be empty.")

    required_columns = {"gaze_x", "is_valid"}
    missing_columns = required_columns - set(recording.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    start = candidate.gap_start_idx
    end = candidate.gap_end_idx

    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, np.integer))
        or not isinstance(end, (int, np.integer))
        or start < 1
        or end >= len(recording)
        or start >= end
        or end - start != candidate.gap_length_samples
        or candidate.left_context_start_idx < 0
        or candidate.left_context_start_idx > start
        or candidate.right_context_end_idx < end
        or candidate.right_context_end_idx > len(recording)
    ):
        raise ValueError("Invalid gap indices.")

    gaze_x = pd.to_numeric(recording["gaze_x"], errors="coerce").to_numpy(
        dtype=float
    )
    is_valid = recording["is_valid"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    ).to_numpy(dtype=bool)
    valid = is_valid & np.isfinite(gaze_x)

    if not valid[start:end].all():
        raise ValueError(
            "Artificial gap must be created only from valid, finite values."
        )

    if not (valid[start - 1] and valid[end]):
        raise ValueError(
            "Artificial gap requires valid, finite boundary values."
        )

    ground_truth = gaze_x[start:end].copy()

    masked_recording = recording.copy()

    masked_recording["original_is_valid"] = masked_recording["is_valid"].copy()
    masked_recording["is_artificial_gap"] = False

    gaze_x_col_idx = masked_recording.columns.get_loc("gaze_x")
    is_valid_col_idx = masked_recording.columns.get_loc("is_valid")
    artificial_gap_col_idx = masked_recording.columns.get_loc(
        "is_artificial_gap"
    )
    masked_recording.iloc[start:end, gaze_x_col_idx] = np.nan
    masked_recording.iloc[start:end, is_valid_col_idx] = False
    masked_recording.iloc[start:end, artificial_gap_col_idx] = True

    return ArtificialGap(
        candidate=candidate,
        ground_truth=ground_truth,
        masked_recording=masked_recording,
    )


def sample_random_gaps(
    recording: pd.DataFrame,
    n_gaps: int,
    min_gap_duration_ms: float,
    max_gap_duration_ms: float,
    context_multiplier: float = 1.0,
    min_context_valid_fraction: float = 0.80,
    random_state: int = 42,
) -> list[GapCandidate]:
    """
    Sample non-overlapping artificial gaps.

    Gap durations are proposed uniformly in physical time between
    min_gap_duration_ms and max_gap_duration_ms. Accepted durations can
    deviate from ideal uniformity when candidate feasibility or non-overlap
    constraints reject proposals.
    """
    if n_gaps <= 0:
        raise ValueError("n_gaps must be positive.")

    if min_gap_duration_ms <= 0:
        raise ValueError("min_gap_duration_ms must be positive.")

    if max_gap_duration_ms < min_gap_duration_ms:
        raise ValueError(
            "max_gap_duration_ms must be >= min_gap_duration_ms."
        )

    rng = random.Random(random_state)
    selected: list[GapCandidate] = []

    # More attempts than requested gaps because sampled durations or
    # positions may be invalid or overlap already selected windows.
    max_attempts = max(100, n_gaps * 100)

    for _ in range(max_attempts):
        if len(selected) >= n_gaps:
            break

        gap_duration_ms = rng.uniform(
            min_gap_duration_ms,
            max_gap_duration_ms,
        )

        candidates = find_gap_candidates(
            recording=recording,
            gap_duration_ms=gap_duration_ms,
            context_multiplier=context_multiplier,
            min_context_valid_fraction=min_context_valid_fraction,
        )

        if not candidates:
            continue

        candidate = rng.choice(candidates)

        if _overlaps_selected_window(candidate, selected):
            continue

        selected.append(candidate)

    return _sorted_selection_or_error(
        selected,
        n_gaps,
        f"Exhausted {max_attempts} sampling attempts.",
    )

def sample_stratified_random_gaps(
    recording: pd.DataFrame,
    n_gaps: int,
    min_gap_duration_ms: float,
    max_gap_duration_ms: float,
    n_strata: int = 5,
    context_multiplier: float = 1.0,
    min_context_valid_fraction: float = 0.80,
    random_state: int = 42,
) -> list[GapCandidate]:
    """
    Sample non-overlapping artificial gaps from equal-probability duration strata.

    The interval [min_gap_duration_ms, max_gap_duration_ms] is divided
    into n_strata equal-width ranges, which are equal-probability strata under
    the required uniform duration distribution. Approximately equal numbers of
    gaps are sampled uniformly from each range. A sampling shortfall is
    reported explicitly rather than silently unbalancing the design.
    """
    if n_gaps <= 0:
        raise ValueError("n_gaps must be positive.")

    if min_gap_duration_ms <= 0:
        raise ValueError("min_gap_duration_ms must be positive.")

    if max_gap_duration_ms <= min_gap_duration_ms:
        raise ValueError(
            "max_gap_duration_ms must be greater than min_gap_duration_ms."
        )

    if n_strata <= 0:
        raise ValueError("n_strata must be positive.")

    rng = random.Random(random_state)
    candidate_cache: dict[int, list[GapCandidate]] = {}

    bin_edges = np.linspace(
        min_gap_duration_ms,
        max_gap_duration_ms,
        n_strata + 1,
    )

    # Example: 20 gaps, 4 strata -> [5, 5, 5, 5].
    # Example: 22 gaps, 4 strata -> [6, 6, 5, 5].
    base_count = n_gaps // n_strata
    remainder = n_gaps % n_strata

    target_counts = [base_count] * n_strata
    for stratum_index in rng.sample(range(n_strata), remainder):
        target_counts[stratum_index] += 1

    selected: list[GapCandidate] = []

    stratum_order = list(range(n_strata))
    rng.shuffle(stratum_order)

    for stratum_index in stratum_order:
        target_count = target_counts[stratum_index]
        lower = float(bin_edges[stratum_index])
        upper = float(bin_edges[stratum_index + 1])

        selected_in_stratum = 0
        max_attempts = max(200, target_count * 200)

        for _ in range(max_attempts):
            if selected_in_stratum >= target_count:
                break

            # Avoid sampling the exact shared upper boundary except
            # in the final stratum.
            if stratum_index < n_strata - 1:
                duration_ms = rng.uniform(
                    lower,
                    np.nextafter(upper, lower),
                )
            else:
                duration_ms = rng.uniform(lower, upper)

            gap_length_samples = duration_ms_to_samples(
                duration_ms,
                float(recording["sampling_rate_hz"].dropna().iloc[0]),
            )
            if gap_length_samples not in candidate_cache:
                candidate_cache[gap_length_samples] = find_gap_candidates(
                    recording=recording,
                    gap_duration_ms=duration_ms,
                    context_multiplier=context_multiplier,
                    min_context_valid_fraction=min_context_valid_fraction,
                )
            candidates = candidate_cache[gap_length_samples]

            if not candidates:
                continue

            candidate = rng.choice(candidates)

            if _overlaps_selected_window(candidate, selected):
                continue

            selected.append(
                replace(
                    candidate,
                    requested_gap_duration_ms=duration_ms,
                    duration_stratum=stratum_index + 1,
                )
            )
            selected_in_stratum += 1

    return _sorted_selection_or_error(
        selected,
        n_gaps,
        "One or more duration strata could not meet their target after "
        "feasibility and non-overlap filtering.",
    )
