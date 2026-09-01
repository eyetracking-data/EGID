# Reproducing the benchmark workflows

This guide reproduces one of the three domain workflows from a fresh clone and
external raw data. Each workflow produces a benchmark table, a nested
group-wise evaluation, and a final selector artifact. The notebook sequence is
the primary interface; equivalent command-line entry points are provided for
automation.

Raw study data is neither included in this repository nor modified by the
workflow. Keep it in an external, user-controlled directory.

## 1. Installation

Use the Python version recorded in [runtime.txt](runtime.txt), then create a
clean environment and verify the installation:

```bash
git clone <repository-url> gap-imputation-benchmark
cd gap-imputation-benchmark

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps
python -m pytest
```

The tests do not require raw data or release artifacts. They should pass before
starting a data-dependent workflow.

Final deployment selectors are GitHub Release assets rather than tracked
binaries. They are not needed to rebuild a benchmark or rerun the nested
evaluation. Download them only when using the released preprocessing algorithm
or when checking the published release:

```bash
python scripts/download_artifact.py --all
```

The command validates each asset's size, checksum, and compatibility against
[artifacts/manifest.json](artifacts/manifest.json).

## 2. Prepare external data and local paths

Read the relevant data guide before running a workflow:

- [Eye Tracking](data/eye_tracking.md)
- [Weather](data/weather.md)
- [Traffic](data/traffic.md)

Create a local `.env` file from `.env.example` and set paths only for the
domain you will run. The real `.env` file is ignored by Git. Output locations
for a full reproduction must be new directories outside the versioned
reference outputs.

```dotenv
# Eye Tracking: a directory containing raw/
BENCHMARK_DATA_DIR="/absolute/path/to/local-data/eyetracking"
EYE_TRACKING_BENCHMARK_DIR="/absolute/path/to/reproduced/benchmarks/eyetracking"
EYE_TRACKING_LODO_RESULTS_DIR="/absolute/path/to/reproduced/results/eyetracking/lodo"
EYE_TRACKING_ARTIFACT_DIR="/absolute/path/to/reproduced/artifacts/eyetracking"

# Weather: a directory containing raw/ and/or processed/
WEATHER_DATA_DIR="/absolute/path/to/local-data/weather"
WEATHER_BENCHMARK_DIR="/absolute/path/to/reproduced/benchmarks/weather"
WEATHER_LODO_RESULTS_DIR="/absolute/path/to/reproduced/results/weather/loso"
WEATHER_ARTIFACT_DIR="/absolute/path/to/reproduced/artifacts/weather"

# Traffic: a directory containing raw/ and/or interim/
TRAFFIC_DATA_DIR="/absolute/path/to/local-data/traffic"
TRAFFIC_BENCHMARK_DIR="/absolute/path/to/reproduced/benchmarks/traffic"
TRAFFIC_LODO_RESULTS_DIR="/absolute/path/to/reproduced/results/traffic/lodo"
TRAFFIC_ARTIFACT_DIR="/absolute/path/to/reproduced/artifacts/traffic"
```

The notebooks load `.env` automatically. To make the same values available to
terminal commands, run:

```bash
set -a
source .env
set +a
```

## 3. Run one domain workflow

Open JupyterLab from the repository root with the environment activated. Run
all cells in each notebook before moving to the next one. The notebooks invoke
the same canonical code as the scripts and contain no machine-specific paths.

| Domain | Build | Explore | Confirmatory evaluation | Final deployment fit |
| --- | --- | --- | --- | --- |
| Eye Tracking | `notebooks/eyetracking/01_build_benchmark.ipynb` | `notebooks/eyetracking/02_exploratory_analysis.ipynb` | `notebooks/eyetracking/03_nested_lodo_evaluation.ipynb` | `notebooks/eyetracking/04_train_final_selector.ipynb` |
| Weather | `notebooks/weather/01_build_benchmark.ipynb` | `notebooks/weather/02_exploratory_analysis.ipynb` | `notebooks/weather/03_nested_loso_evaluation.ipynb` | `notebooks/weather/04_train_final_selector.ipynb` |
| Traffic | `notebooks/traffic/01_build_benchmark.ipynb` | `notebooks/traffic/02_exploratory_analysis.ipynb` | `notebooks/traffic/03_nested_lodo_evaluation.ipynb` | `notebooks/traffic/04_train_final_selector.ipynb` |

The Weather workflow uses leave-one-station-out (LOSO). Some historic output
filenames retain `lodo`; the metadata field `workflow` is the authoritative
description of the evaluation procedure.

For Weather, set `PREPARE_RAW_DATA = True` in notebook 01 only when starting
with the DWD ZIP archives; otherwise use the prepared station files. For
Traffic, set `PREPARE_PANEL = True` only when starting with the raw LargeST
files; otherwise reuse the prepared selected panel. Eye Tracking reads the raw
files directly.

## 4. Equivalent command-line workflows

The following commands use the frozen domain configuration. Substitute fresh
absolute output paths; do not use the versioned reference directories for a
rerun.

### Eye Tracking

```bash
python scripts/build_eye_tracking_benchmark.py \
  --config configs/eye_tracking_final.toml \
  --output-dir /absolute/path/to/reproduced/benchmarks/eyetracking

python scripts/evaluate_eye_tracking_lodo.py \
  --benchmark-dir /absolute/path/to/reproduced/benchmarks/eyetracking \
  --results-dir /absolute/path/to/reproduced/results/eyetracking/lodo

python scripts/train_final_native_7_selector.py \
  --benchmark-dir /absolute/path/to/reproduced/benchmarks/eyetracking \
  --artifact-dir /absolute/path/to/reproduced/artifacts/eyetracking
```

Before a long run, validate source loading and deterministic selection without
sampling gaps or writing files:

```bash
python scripts/build_eye_tracking_benchmark.py \
  --config configs/eye_tracking_final.toml \
  --dry-run
```

### Weather

```bash
python scripts/build_weather_benchmark.py \
  --config configs/weather_final.toml \
  --output-dir /absolute/path/to/reproduced/benchmarks/weather

python scripts/evaluate_weather_loso.py \
  --config configs/weather_final.toml \
  --benchmark-dir /absolute/path/to/reproduced/benchmarks/weather \
  --results-dir /absolute/path/to/reproduced/results/weather/loso

python scripts/train_final_weather_selector.py \
  --config configs/weather_final.toml \
  --benchmark-dir /absolute/path/to/reproduced/benchmarks/weather \
  --lodo-results-dir /absolute/path/to/reproduced/results/weather/loso \
  --artifact-dir /absolute/path/to/reproduced/artifacts/weather
```

Add `--prepare-data` to the build command only to create the processed station
files from the raw DWD archives.

### Traffic

```bash
python scripts/build_traffic_benchmark.py \
  --config configs/traffic_final.toml \
  --output-dir /absolute/path/to/reproduced/benchmarks/traffic

python scripts/evaluate_traffic_lodo.py \
  --config configs/traffic_final.toml \
  --benchmark-dir /absolute/path/to/reproduced/benchmarks/traffic \
  --results-dir /absolute/path/to/reproduced/results/traffic/lodo

python scripts/train_final_traffic_selector.py \
  --config configs/traffic_final.toml \
  --benchmark-dir /absolute/path/to/reproduced/benchmarks/traffic \
  --lodo-results-dir /absolute/path/to/reproduced/results/traffic/lodo \
  --artifact-dir /absolute/path/to/reproduced/artifacts/traffic
```

Add `--prepare-panel` to the build command only to audit the raw LargeST files
and rebuild the selected panel.

## 5. Required outputs and interpretation

The builders validate the complete benchmark-output contract before they
finish. Every benchmark directory must contain:

```text
learnable_gap_table.csv
input_manifest.csv
selected_recordings.csv
excluded_gaps.csv
dataset_summary.csv
coverage_table.csv
metadata.json
```

Each evaluation directory must contain:

```text
lodo_gap_predictions.csv
lodo_hyperparameter_tuning.csv
lodo_feature_scale_floors.csv
lodo_summary.csv
lodo_selected_method_counts.csv
lodo_selected_method_counts_overall.csv
metadata.json
```

The final fitting step writes `selector.joblib` and `metadata.json` to its
artifact directory. Read `metadata.json` after each stage: it records the
domain, method and feature contract, random seed, protocol settings, output
counts, and final training scope.

Before interpreting a rerun, compare its configuration and metadata with the
corresponding committed reference outputs. Inspect exclusions and coverage
before comparing performance. A final selector fitted in stage 4 is a
deployment model, not a new confirmatory estimate.

## 6. What a reproduction can and cannot establish

Reproducing the software workflow verifies that the documented code,
configuration, and external inputs can generate the expected classes of
outputs. Exact numerical equality can still depend on the precise source
archive versions, extraction state, dependency versions, and platform-level
numerical behavior. Preserve the source-version record and all generated
metadata with any rerun.

Follow every upstream data-source licence, attribution requirement, and
redistribution restriction before publishing or sharing derived outputs.
