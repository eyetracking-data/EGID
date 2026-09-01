# Data directory

This repository does not contain raw source data. The `data/` directory
contains documentation only; users keep all downloaded data in external local
directories.

## Common setup

Use only the domain that you want to run. Create a local `.env` file (or set
the variables in the current environment) with its external data root:

```bash
BENCHMARK_DATA_DIR="/absolute/path/to/local-data/eyetracking"
WEATHER_DATA_DIR="/absolute/path/to/local-data/weather"
TRAFFIC_DATA_DIR="/absolute/path/to/local-data/traffic"
```

The domain notebooks create only their documented intermediate data below these
external roots. They write versioned benchmark tables, evaluation results, and
final selector artifacts to the repository defaults (`benchmarks/<domain>/`,
`results/<domain>/`, and `artifacts/<domain>/`) unless local output variables
in `.env` redirect them. Use a new output path when rerunning a published
workflow.

## Domain guides

Each guide uses the same structure: local layout, user-provided inputs, source
selection, preparation, license, benchmark outputs, and notebooks.

| Domain | External input supplied by the user | Created automatically |
| --- | --- | --- |
| [Eye Tracking](eye_tracking.md) | Four downloaded source datasets below `raw/` | No preprocessed source directory; notebook 01 reads raw data directly. |
| [Weather](weather.md) | 16 DWD ZIP files, or prepared station files | `extracted/` and `processed/` from the DWD ZIP files. |
| [Traffic](traffic.md) | Five LargeST annual HDF5 files and `ca_meta.csv`, or a prepared panel | The audited selected panel below `interim/`. |

Follow the domain guide before running its first notebook. Raw source data and
local machine paths are never committed to Git.
