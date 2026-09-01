# Traffic data requirements

The Traffic benchmark uses external LargeST five-minute traffic-flow data. Raw
data is not distributed with this repository.

## Local layout

```text
$TRAFFIC_DATA_DIR/
├── raw/                                      # downloaded LargeST files
├── interim/largest_flow_panel_selected/      # generated benchmark input
└── final/                                    # optional local outputs
```

## What a user must provide

1. Use a Python environment with the optional Traffic dependency (PyTables);
   see `README.md`.
2. Set `TRAFFIC_DATA_DIR` to one external data directory in `.env`.
3. Provide either the five annual HDF5 files plus `ca_meta.csv` in `raw/`, or
   a prepared `interim/largest_flow_panel_selected/` directory.
4. Set `TRAFFIC_BENCHMARK_DIR` to an empty output directory when the published
   `benchmarks/traffic/` directory should not be replaced.

Notebook 01 creates `interim/` when `PREPARE_PANEL = True`; `final/` is not an
input to the benchmark workflow.

## Source data and study use

Download LargeST from [Kaggle](https://www.kaggle.com/datasets/liuxu77/largest).
The required raw files are `ca_his_raw_2017.h5` through
`ca_his_raw_2021.h5` and `ca_meta.csv`. The supplied `ca_rn_adj.npy` is not
used by this benchmark.

The reference uses PeMS Districts 3, 4, 7, and 11 for 2017–2021. Districts 5,
6, 8, 10, and 12 are excluded before the sensor audit. Within the four retained
districts, sensors are excluded when valid coverage is below 95% or a natural
gap exceeds 24 hours. Non-finite and negative values are natural gaps; zero is
valid flow.

| District | Audited sensors | Retained | Excluded |
| ---: | ---: | ---: | ---: |
| 3 | 480 | 409 | 71 |
| 4 | 2,352 | 2,249 | 103 |
| 7 | 1,859 | 1,787 | 72 |
| 11 | 716 | 632 | 84 |

The complete sensor-level decision record is written to
`interim/largest_raw_flow_gap_audit_by_sensor.csv`. It is the authoritative
list of excluded sensor locations and their coverage/gap values.

## Download and preparation

Place the six required LargeST files in `raw/`. With `PREPARE_PANEL = True`,
notebook 01 audits the raw data and creates four selected HDF5 panels plus
sensor metadata and `time_manifest.csv` in `interim/`. Later runs reuse only
the prepared panel; raw source files are not modified.

## Version and license record

According to the [official LargeST repository](https://github.com/liuxu77/LargeST),
the dataset is released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); its code is
MIT-licensed. Cite Liu et al., *LargeST: A Benchmark Dataset for Large-Scale
Traffic Forecasting* (NeurIPS 2023), link CC BY-NC 4.0, and state the District,
year, and sensor-filtering steps above when publishing derived data.

The linked LargeST licence information (CC BY-NC 4.0 for data and MIT for code)
was reviewed on **2026-09-01**.

## Release provenance record

The reference workflow used the LargeST files `ca_his_raw_2017.h5` through
`ca_his_raw_2021.h5` plus `ca_meta.csv`, obtained from the Kaggle source linked
above. Their local filesystem modification timestamps range from
**2026-08-24 20:10:33 +02:00** to **2026-08-24 20:11:53 +02:00**
(`Europe/Berlin`). This is the best available local proxy for acquisition date,
not a provider-issued download receipt.

The audited selected panel used by the benchmark was built from these files on
2026-08-24; its `time_manifest.csv` was last written at 21:59:09 +02:00. The
workflow retains Districts 3, 4, 7, and 11 for 2017–2021 and applies the
coverage and natural-gap exclusions documented above. The data licence is
CC BY-NC 4.0; the LargeST code licence is MIT.

## Benchmark design and review outputs

`configs/traffic_final.toml` fixes the protocol: 100 sensor-years per
district-year and one non-overlapping gap in five strata (1–3, 4–12, 13–36,
37–144, 145–288 five-minute steps). This requests 10,000 gaps; the reference
run contains 10,000 generated gaps.

Every benchmark directory contains `learnable_gap_table.csv`,
`input_manifest.csv`, `selected_recordings.csv`, `excluded_gaps.csv`,
`dataset_summary.csv`, `coverage_table.csv`, and `metadata.json`.

## Running the workflow

Run these notebooks in order from the repository root:

1. `notebooks/traffic/01_build_benchmark.ipynb`
2. `notebooks/traffic/02_exploratory_analysis.ipynb`
3. `notebooks/traffic/03_nested_lodo_evaluation.ipynb`
4. `notebooks/traffic/04_train_final_selector.ipynb`
