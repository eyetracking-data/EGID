# Gap Imputation Benchmark

This repository contains the reproducible implementation and published derived
outputs for a benchmark of gap-imputation methods and a gap-specific method
selector. It evaluates three separate domains—Eye Tracking, Weather, and Traffic—using a shared benchmark architecture and one domain-specific selector per domain.


The repository contains no raw study data. Raw inputs stay in user-controlled external directories; only code, configuration, documentation, derived benchmark tables, evaluation outputs, and artifact metadata are included in the repository.


The complete benchmark protocol, including artificial-gap sampling, candidate
methods, normalization, and grouped evaluation, is specified in
[docs/methods.md](docs/methods.md). The 16 observable selector features are
defined in [docs/selector_features.md](docs/selector_features.md), and
[REPRODUCE.md](REPRODUCE.md) describes the domain workflows from external
source data. The provenance-aware preprocessing pipeline for real missing
segments is documented in [docs/algorithm_usage.md](docs/algorithm_usage.md).

## Scope and interpretation

The benchmark compares method-specific normalized reconstruction errors on
artificial gaps drawn only from originally observed samples. A domain-specific
multi-output Random Forest predicts one normalized error per candidate method
from observable context around a gap; the method with the lowest predicted
error is selected. The selector does not reconstruct values itself and does not
use hidden gap values, oracle labels, or group identifiers as input features.

The nested group-wise evaluations provide the confirmatory performance
estimates. Final selector artifacts are fitted after that evaluation on all
eligible gaps in their domain; they are deployment artifacts, not an additional
unbiased performance estimate. See [docs/methods.md](docs/methods.md) for the
complete protocol and limitations.

## Repository layout

```text
src/gap_imputation_benchmark/  Installable package
  benchmark/                   Shared gaps, metrics, features, splits, and output contracts
  domains/                     Eye-Tracking, Weather, and Traffic adapters and workflows
  imputers/                    Candidate-method registries
  selection/                   Selector training, evaluation, and inference
  algorithm/                   Provenance-aware preprocessing pipeline
scripts/                       Terminal entry points for build, evaluation, training, and artifact download
notebooks/                     Four-step workflows per domain and algorithm examples
configs/                       Strict, versioned TOML protocol configurations
tests/                         Automated tests
data/                          External-data documentation only; raw data is ignored
benchmarks/                    Published derived benchmark tables and metadata
results/                       Published nested-evaluation outputs and metadata
artifacts/                     Selector metadata; selector binaries are GitHub Release assets
docs/                          Methods, feature definitions, and algorithm usage
```

Each domain builder produces the same required set of benchmark-output files.
The required files and their purposes are defined in
[docs/methods.md](docs/methods.md#8-exclusions-and-output-artifacts) and
implemented in
[`src/gap_imputation_benchmark/benchmark/output_contract.py`](src/gap_imputation_benchmark/benchmark/output_contract.py).

## Installation and verification

The tested interpreter version is recorded in [runtime.txt](runtime.txt).
Create an isolated environment, install the pinned dependencies, then install
the package itself:

On macOS, install the tested interpreter with Homebrew first if
`python3.12 --version` is unavailable:

```bash
brew install python@3.12
```

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps
python -m pytest
```

The tests cover deterministic sampling, leakage controls, feature extraction,
method applicability, artifact compatibility, and the three domain workflows.
They do not require external raw data or selector binaries and do not download
anything from the network.

## Selector artifacts

Selector binaries are intentionally distributed as GitHub Release assets rather
than committed to Git. Downloading is always explicit; importing the library or
running the algorithm never contacts the network.

```bash
python scripts/download_artifact.py --all
```

To download one selector, use `--domain eye_tracking`, `--domain weather`, or
`--domain traffic`. The command checks the asset size, SHA-256 digest, and
artifact compatibility before placing it at
its configured destination (for example,
`artifacts/eyetracking/selector.joblib` for `--domain eye_tracking`). The
release mapping, expected checksums, and destinations are in
[artifacts/manifest.json](artifacts/manifest.json).
Use `--force` only when intentionally replacing an existing local binary.

## External data and local configuration

Raw inputs must remain outside the clone. Configure only the domain you intend
to run, either as environment variables or in a local `.env` file (ignored by
Git):

```dotenv
BENCHMARK_DATA_DIR="/absolute/path/to/eyetracking-data"
WEATHER_DATA_DIR="/absolute/path/to/weather-data"
TRAFFIC_DATA_DIR="/absolute/path/to/traffic-data"
```

The matching data guide specifies the required layout, source terms, and
citations: [Eye Tracking](data/eye_tracking.md), [Weather](data/weather.md),
and [Traffic](data/traffic.md). The full reproduction guide documents the
optional output-directory variables as well.

Each data-directory variable must name the directory that contains its `raw/`
subdirectory, not the `raw/` directory itself.

For shell commands that should use a local `.env` file, load it explicitly:

```bash
set -a
source .env
set +a
```

Never point a new full reproduction at the versioned `benchmarks/`,
`results/`, or `artifacts/` directories. Use fresh output directories so the
published reference outputs remain unchanged.

## Reproduction workflows

The canonical notebook sequence and command-line instructions are in
[REPRODUCE.md](REPRODUCE.md). The main implementations are also exposed through
the scripts in `scripts/`. The
frozen paper-workflow settings are [Eye Tracking](configs/eye_tracking_final.toml),
[Weather](configs/weather_final.toml), and
[Traffic](configs/traffic_final.toml). Their loaders reject missing and unknown
keys so protocol changes remain explicit.

## Using the preprocessing algorithm

`gap_imputation_benchmark.algorithm` implements a three-stage, provenance-aware
pipeline: optional outlier detection, selector-based missing-value imputation,
and optional standardization. It applies a downloaded selector artifact to real
gaps; it neither creates artificial gaps nor retrains the selector.

Use the self-contained synthetic example at
`notebooks/algorithm/synthetic_example/01_example_pipeline_call.ipynb` or the
complete input, configuration, and provenance guide in
[docs/algorithm_usage.md](docs/algorithm_usage.md). Generated algorithm-example
outputs are deliberately ignored by Git.

## Licence and attribution

The source code is licensed under the [MIT License](LICENSE). Derived tables,
results, and model artifacts remain subject to the licenses and attribution
requirements of their external data sources. Consult the relevant guide under
[data/](data/) before reusing or redistributing a derived output.
