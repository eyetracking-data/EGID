# Using the preprocessing algorithm

This guide describes how to apply the repository's provenance-aware preprocessing algorithm to one numeric time series. The pipeline always runs in this order:

1. optional outlier detection;
2. selector-based missing-value imputation;
3. optional standardization.

It is intended for real missing segments. It does not create artificial gaps
and does not retrain the selector. The selector artifact for the chosen domain
must already have been created by the corresponding benchmark workflow.

This is an operational usage guide, not a claim that the benchmark validates
every possible signal or gap length. Read [methods.md](methods.md) for the
evaluation protocol and the corresponding `metadata.json` files for the
configuration and scope of a released artifact.

## Start with the example notebook

Open and run [notebooks/algorithm/synthetic_example/01_example_pipeline_call.ipynb](../notebooks/algorithm/synthetic_example/01_example_pipeline_call.ipynb) from within the repository. It is a self-contained example with a synthetic signal and writes its outputs below `notebooks/algorithm/synthetic_example/outputs/`. The complete output directory is committed as a reference result. Domain-specific examples are available in the `eyetracking/`, `weather/`, and `traffic/` subdirectories.

For a real dataset, replace the synthetic `frame`, `VALUE_COLUMN`, output paths, and `DomainImputationConfig` in that notebook. The notebook is the recommended starting point because it makes the configuration and provenance fields explicit.

## Required setup

Install the project as described in [REPRODUCE.md](../REPRODUCE.md). Before running the algorithm, make sure that the selected domain has a final selector artifact at its repository-relative location:

| Domain | Required artifact |
| --- | --- |
| `eye_tracking` | `artifacts/eyetracking/selector.joblib` |
| `weather` | `artifacts/weather/selector.joblib` |
| `traffic` | `artifacts/traffic/selector.joblib` |

The algorithm reads these paths automatically; no absolute artifact path is
configured. If the required artifact is absent or incompatible with the domain,
the run stops with an error rather than selecting a method silently. Retrieve a
published artifact explicitly with `python scripts/download_artifact.py --all`
from the repository root; the library never downloads a model during import or
pipeline execution.

## Input table

Pass a pandas `DataFrame` and the name of its numeric signal column. Missing values may be represented as `NaN` or any value that pandas cannot convert to a finite number.

Optional columns depend on the configuration:

| Configuration field | Corresponding input requirement |
| --- | --- |
| `timestamp_col` | A finite, strictly increasing timestamp column. It can be datetime-like, parseable dates, or numeric timestamps with `timestamp_unit` set to `s`, `ms`, or `us`. |
| `validity_col` | A Boolean-like column. A value is eligible as observed context only when this column is true and the signal value is finite. |
| `sampling_rate_hz` | Required in practice for Eye Tracking when no timestamp column is supplied. If omitted, the implementation uses 250 Hz. |

Weather and Traffic require `timestamp_col`, because their seasonal candidate method uses calendar-aware timestamps. The time series must be ordered chronologically before it is passed to the algorithm.

All columns other than the signal, timestamp, and optional validity columns are retained unchanged in the returned and written table. The signal column is converted to numeric values; non-numeric values become missing values. The algorithm processes one signal column per call.

## Domain policy and validated scope

The algorithm only fills gaps inside the range that was evaluated for the selected domain. It also requires at least 80% valid finite samples in the predefined context on each side of a gap. The context length is the larger of the gap length and two samples per side.

| Domain | Validated gap duration | Timestamp requirement | Seasonal reference policy |
| --- | --- | --- | --- |
| Eye Tracking | 1–250 ms | Optional; a synthetic timeline is derived from `sampling_rate_hz` when absent. | Not used. |
| Weather | 1 hour–18.33 days | Required. | Calendar-year references; up to three complete segments on either side; pointwise mean; leap days skipped. |
| Traffic | 5 minutes–24 hours | Required. | Weekly references; up to three complete segments on either side; pointwise median. |

By default, a gap outside these duration ranges is retained as missing with the reason `outside_validated_gap_duration_range`. Setting `impute_outside_validated_gap_range=True` overrides this protection, but the provenance report records an explicit warning. It does not make such an extrapolated result validated by the benchmark.

## Minimal Python call

```python
import pandas as pd
from gap_imputation_benchmark.algorithm import (
    DomainImputationConfig,
    RunMetadata,
    impute_with_rf_selector,
)

frame = pd.read_csv("input.csv")

config = DomainImputationConfig(
    domain="eye_tracking",
    sampling_rate_hz=1_000.0,
    outlier_method="zscore",
    outlier_threshold=2.1,
    standardization_method=None,
)

imputed_frame, provenance = impute_with_rf_selector(
    frame,
    "gaze_x",
    config=config,
    run_metadata=RunMetadata(
        executor_name="Your Name",
        comment="Imputation of session 001",
    ),
    input_path="input.csv",
    output_path="outputs/session_001_imputed.csv",
    provenance_path="outputs/session_001_provenance.json",
)
```

For Weather or Traffic, provide the timestamp column and use its respective domain:

```python
config = DomainImputationConfig(
    domain="weather",  # or "traffic"
    timestamp_col="timestamp",
    outlier_method="iqr",
    outlier_threshold=2.5,
    standardization_method="robust",
)
```

## Complete, path-based example

Use this template when running the algorithm on a real CSV file. It exposes every path and all runtime options that affect processing. Change only the values in the **User settings** section; the remaining code can stay unchanged.

```python
from pathlib import Path

import pandas as pd
from gap_imputation_benchmark.algorithm import (
    DomainImputationConfig,
    RunMetadata,
    impute_with_rf_selector,
)

# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------

# Repository root. Change this only when the notebook is started outside
# the repository; selector artifacts are resolved relative to this project.
PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / "pyproject.toml").is_file():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Start Jupyter from inside the cloned repository.")
    PROJECT_ROOT = PROJECT_ROOT.parent

# Source data: keep raw input outside the repository if it is not intended
# for publication. This can be an absolute path.
INPUT_CSV = Path("/absolute/path/to/source-data/recording_001.csv")

# All derived files go to one dedicated, writable run directory.
RUN_DIR = Path("/absolute/path/to/derived-results/recording_001_run")
OUTPUT_CSV = RUN_DIR / "recording_001_imputed.csv"
PROVENANCE_JSON = RUN_DIR / "recording_001_provenance.json"

# Column names in INPUT_CSV.
VALUE_COLUMN = "temperature_c"
TIMESTAMP_COLUMN = "timestamp"
VALIDITY_COLUMN = "is_valid"  # Set to None when no validity column exists.

# Select exactly one domain: "eye_tracking", "weather", or "traffic".
DOMAIN = "weather"

# For Eye Tracking without TIMESTAMP_COLUMN, set the sampling rate, e.g. 1_000.0.
# It is ignored for Weather and Traffic, which require calendar timestamps.
SAMPLING_RATE_HZ = None

# For numeric timestamps only: "s", "ms", or "us". Datetime-like timestamps
# are parsed automatically and do not require conversion.
TIMESTAMP_UNIT = "ms"

# Outlier handling: "none", "iqr", or "zscore". Outliers are set to NaN and
# are then considered by the imputation stage.
OUTLIER_METHOD = "iqr"
OUTLIER_THRESHOLD = 2.5

# Final scaling: None, "zscore", "robust", or "minmax".
STANDARDIZATION_METHOD = "robust"

# Keep False for a benchmark-supported run. True permits an explicit,
# provenance-marked extrapolation beyond the domain's validated gap range.
IMPUTE_OUTSIDE_VALIDATED_GAP_RANGE = False

# Human-readable provenance fields. Information files are optional JSON/text
# files and may be absolute paths. Set fields to None if unavailable.
EXECUTOR_NAME = "Name of person running this notebook"
EXECUTOR_INFO_PATH = Path("/absolute/path/to/executor_info.json")
RESPONSIBLE_PERSON_NAME = "Name of responsible person"
RESPONSIBLE_PERSON_INFO_PATH = Path("/absolute/path/to/responsible_person_info.json")
EXECUTION_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "algorithm" / "your_real_data_run.ipynb"
COMMENT = "Weather-station imputation for station 001; input collected on YYYY-MM-DD."

# ------------------------------------------------------------------
# Load, validate, and run
# ------------------------------------------------------------------

if not INPUT_CSV.is_file():
    raise FileNotFoundError(f"Input CSV does not exist: {INPUT_CSV}")
RUN_DIR.mkdir(parents=True, exist_ok=True)

frame = pd.read_csv(INPUT_CSV)
required_columns = {VALUE_COLUMN}
if TIMESTAMP_COLUMN is not None:
    required_columns.add(TIMESTAMP_COLUMN)
if VALIDITY_COLUMN is not None:
    required_columns.add(VALIDITY_COLUMN)
missing_columns = required_columns - set(frame.columns)
if missing_columns:
    raise ValueError(f"Input is missing columns: {sorted(missing_columns)}")

config = DomainImputationConfig(
    domain=DOMAIN,
    sampling_rate_hz=SAMPLING_RATE_HZ,
    timestamp_col=TIMESTAMP_COLUMN,
    timestamp_unit=TIMESTAMP_UNIT,
    validity_col=VALIDITY_COLUMN,
    outlier_method=OUTLIER_METHOD,
    outlier_threshold=OUTLIER_THRESHOLD,
    standardization_method=STANDARDIZATION_METHOD,
    impute_outside_validated_gap_range=IMPUTE_OUTSIDE_VALIDATED_GAP_RANGE,
)

run_metadata = RunMetadata(
    executor_name=EXECUTOR_NAME,
    executor_info_path=EXECUTOR_INFO_PATH,
    executor_responsible_person=RESPONSIBLE_PERSON_NAME,
    executor_responsible_person_info_path=RESPONSIBLE_PERSON_INFO_PATH,
    execution_notebook_path=EXECUTION_NOTEBOOK_PATH,
    comment=COMMENT,
)

imputed_frame, provenance = impute_with_rf_selector(
    frame,
    VALUE_COLUMN,
    config=config,
    run_metadata=run_metadata,
    input_path=INPUT_CSV,
    output_path=OUTPUT_CSV,
    provenance_path=PROVENANCE_JSON,
)

# ------------------------------------------------------------------
# Inspect the complete run report
# ------------------------------------------------------------------

print("Final output:      ", OUTPUT_CSV)
print("Provenance report: ", PROVENANCE_JSON)
print("Intermediate data: ", OUTPUT_CSV.with_name(f"{OUTPUT_CSV.stem}_after_outlier_detection.csv"))
print("Intermediate data: ", OUTPUT_CSV.with_name(f"{OUTPUT_CSV.stem}_after_missing_value_imputation.csv"))
print("Summary:           ", provenance["summary"])

gap_report = provenance["activities"]["missing_value_imputation"]["details"]["gaps"]
for gap in gap_report:
    print({
        "gap_number": gap["gap_number"],
        "status": gap["status"],
        "used_method": gap.get("used_method"),
        "reason": gap.get("reason"),
        "fallback_reason": gap.get("fallback_reason"),
    })
```

For Eye Tracking with no timestamp column, use `DOMAIN = "eye_tracking"`, set `TIMESTAMP_COLUMN = None`, and give `SAMPLING_RATE_HZ` a positive value. For Traffic, change `DOMAIN` and the signal-column name, retain a strictly increasing timestamp column, and use the Traffic selector artifact created by its completed workflow.

## Configuration reference

`DomainImputationConfig` accepts the following fields:

| Field | Meaning |
| --- | --- |
| `domain` | One of `eye_tracking`, `weather`, or `traffic`. It determines the selector artifact, admissible gap-duration range, and seasonal-method policy. |
| `sampling_rate_hz` | Sampling rate for a series without timestamps; relevant to Eye Tracking. |
| `timestamp_col` / `timestamp_unit` | Timestamp column and unit for numeric timestamps. Datetime columns do not need a unit conversion. |
| `validity_col` | Optional observed-sample indicator. |
| `outlier_method` | `none`, `iqr`, or `zscore`. Detected outliers become missing and are considered by the following imputation stage. |
| `outlier_threshold` | Positive threshold used by the selected outlier method. |
| `standardization_method` | `None`, `zscore`, `robust`, or `minmax`; applied only after imputation. |
| `impute_outside_validated_gap_range` | Defaults to `False`. Set to `True` only to explicitly allow imputation outside the domain's benchmarked duration range; the provenance record marks this as a warning. |

## What happens to each missing segment

For every contiguous missing segment, the algorithm checks that sufficient valid context is available on both sides and that its duration lies inside the domain's validated range. It then extracts the observable gap features, predicts the nRMSE of each candidate method with the domain selector, and attempts the predicted best method first. If that method is inapplicable, it tries the remaining methods in increasing predicted-error order. Segments that cannot safely be handled remain missing and receive a documented reason.

Missing segments are processed in temporal order. After a segment has been filled successfully, its reconstructed values are available as observable context for later segments in the same run. This reflects an operational preprocessing run and means that successive decisions can propagate earlier reconstruction errors. In contrast, the benchmark evaluates artificial gaps independently on their original masked recordings; its reported error estimates do not measure this sequential error propagation.

The common candidate set is forward fill, nearest boundary, linear interpolation, PCHIP, local natural cubic spline, BIC-selected polynomial fitting, and template imputation. Weather and Traffic additionally expose their domain-specific `seasonal_periodic` candidate. The selector predicts a method-specific error for each registered candidate; it does not directly produce the imputed values.

Typical non-filled outcomes are insufficient context at a series edge, less than 80% valid context, a gap outside the validated duration range, failure to extract features, non-positive recording scale, or no applicable candidate method. A fallback from the top-ranked method is not an error: the report records the selected method, the attempted methods, and the method ultimately used.

## Complete run metadata

`RunMetadata` does not alter the signal or the selector decision. It documents the run so that a processing record can identify its data and responsible people.

| Field | Recorded purpose |
| --- | --- |
| `executor_name` / `executor_info_path` | Person or system that executed the run. If no name is supplied, the local user name is recorded. |
| `executor_responsible_person` / `executor_responsible_person_info_path` | Optional responsible person. |
| `execution_notebook_path` | Notebook or script that initiated the run. |
| `comment` | Free-text run note. |

Paths in metadata are made repository-relative where possible; external input locations are represented as external rather than embedding a machine-specific path.

## Outputs and provenance

If `output_path` is supplied, the final table is written there. Two intermediate tables are written next to it:

- `<name>_after_outlier_detection.csv`
- `<name>_after_missing_value_imputation.csv`

If `provenance_path` is supplied, it records the configuration, selected artifact, per-gap method recommendations and fallbacks, skipped-gap reasons, the three processing activities, and the input/output file metadata. Per-stage metadata and the recorded configuration are written in a sibling `<provenance-name>_logs/` directory.

With a provenance path, `impute_with_rf_selector` returns `(imputed_frame, provenance)`; without one, it returns only `imputed_frame`.

Do not overwrite the original input file. Keep the input, final output, and provenance JSON together as one processing record.

## Reading the run report

The top-level provenance JSON has four sections:

| Section | Content |
| --- | --- |
| `run` | Run identifier, start and end time, configuration record, initiating notebook/script, and free-text comment. |
| `agents` | Executor and responsible-person information supplied through `RunMetadata`. |
| `configuration` | The effective `DomainImputationConfig`. |
| `activities` | Separate records for outlier detection, missing-value imputation, and standardization. |
| `summary` | Counts of gaps before outlier processing, after outlier processing, filled gaps, and skipped gaps. |

The `activities.missing_value_imputation.details.gaps` list is the per-gap audit trail. Its `status` is `filled` when the primary recommendation was applied, `fallback` when a lower-ranked candidate was applied, or `skipped` when no imputation was performed. Each `attempted_methods` item records `outcome` as `applied`, `not_applicable`, `invalid_prediction`, or `exception`; `reason` is `null` only for `applied`. For each gap, inspect `reason` when skipped, `fallback_reason` when applicable, `within_validated_gap_duration_range`, `model_recommended_method`, `predicted_method_nrmse`, `attempted_methods`, `used_method`, and any method-specific diagnostics. This list is the authoritative report of what the algorithm did to every gap.

The following notebook cell prints a compact completion report after a run:

```python
summary = provenance["summary"]
gaps = provenance["activities"]["missing_value_imputation"]["details"]["gaps"]

print(summary)
for gap in gaps:
    print(
        gap["gap_number"],
        gap["status"],
        gap.get("used_method"),
        gap.get("reason"),
    )
```

## Operational checks before using a result

1. Confirm that the chosen `domain` matches the data-generating process and that its selector artifact was generated from the final benchmark workflow.
2. Confirm that timestamps are chronological and strictly increasing, and that `validity_col`, if used, has the intended semantics.
3. Inspect the summary and every skipped gap in the provenance report; skipped gaps remain missing by design.
4. Treat outputs obtained with `impute_outside_validated_gap_range=True` as explicitly out-of-scope extrapolations.
5. Retain the final CSV, its two intermediate CSVs, the provenance JSON, and its log directory together.
