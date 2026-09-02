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

According to the [official LargeST repository](https://github.com/liuxu77/LargeST), the LargeST benchmark dataset is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), while the accompanying code implementation is released under the MIT License.

Users of LargeST should cite Liu et al., *LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting* (NeurIPS 2023). For reproducibility, any derived dataset or benchmark release based on this workflow should also document the selected PeMS districts, source year, and sensor-filtering steps described above.

The linked LargeST license information was last reviewed on **2026-09-02**.


## Release provenance record

The reference workflow used the LargeST files ca_his_raw_2017.h5 through ca_his_raw_2021.h5 and ca_meta.csv, obtained from the Kaggle source linked above. The recorded access date for these source files is 2026-08-24.

The audited benchmark panel was built from these files using Districts 3, 4, 7, and 11 for 2017–2021 and applying the coverage and natural-gap exclusions documented above.

## Benchmark design and review outputs

`configs/traffic_final.toml` fixes the protocol: for each district-year, 100 sensors are selected. For every selected sensor-year, one fully observed artificial gap is sampled from each of five duration strata (1–3, 4–12, 13–36, 37–144, and 145–288 five-minute steps). The gap and its context window do not overlap with those of another sampled gap for the same sensor-year. This requests 10,000 gaps in total; the reference run generated all 10,000.

The benchmark-output contract is defined in
[docs/methods.md](../docs/methods.md#8-exclusions-and-output-artifacts).

## Running the workflow

Follow the canonical notebook sequence in
[REPRODUCE.md](../REPRODUCE.md#3-run-one-domain-workflow), using the Traffic
notebooks in `notebooks/traffic/`.
