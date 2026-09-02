# Weather data requirements

The Weather benchmark uses external DWD historical hourly air-temperature
data. Raw data is not distributed with this repository.

## Local layout

```text
$WEATHER_DATA_DIR/
├── raw/                                      # downloaded DWD ZIP archives
├── extracted/<station_id>/                   # generated from raw/
└── processed/temperature_2000_2025/          # generated benchmark input
```

## What a user must provide

1. Set `WEATHER_DATA_DIR` to one external data directory in `.env`.
2. Provide either the 16 DWD ZIP archives in `raw/`, or the 16 processed
   station files in `processed/temperature_2000_2025/`.
3. Set `WEATHER_BENCHMARK_DIR` to an empty output directory when the published
   `benchmarks/weather/` directory should not be replaced.

`extracted/` and `processed/` are created automatically from `raw/` by
notebook 01 when `PREPARE_RAW_DATA = True`.

## Source data and study use

Download the archives from the [DWD historical hourly air-temperature archive](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/historical/).
The workflow uses `TT_TU` only and restricts every series to 2000-01-01 00:00
through 2025-12-31 23:00.

| Station ID | Station | Required archive |
| --- | --- | --- |
| 00232 | Augsburg | `stundenwerte_TU_00232_19550101_20251231_hist.zip` |
| 00427 | Berlin Brandenburg | `stundenwerte_TU_00427_19730101_20251231_hist.zip` |
| 01048 | Dresden-Klotzsche | `stundenwerte_TU_01048_19730101_20251231_hist.zip` |
| 01346 | Feldberg/Schwarzwald | `stundenwerte_TU_01346_19520101_20251231_hist.zip` |
| 01420 | Frankfurt/Main | `stundenwerte_TU_01420_19810101_20251231_hist.zip` |
| 01443 | Freiburg | `stundenwerte_TU_01443_19510101_20251231_hist.zip` |
| 01550 | Garmisch-Partenkirchen | `stundenwerte_TU_01550_19480101_20251231_hist.zip` |
| 01691 | Göttingen | `stundenwerte_TU_01691_19480101_20251231_hist.zip` |
| 01975 | Hamburg-Fuhlsbüttel | `stundenwerte_TU_01975_19490101_20251231_hist.zip` |
| 02014 | Hannover | `stundenwerte_TU_02014_19490101_20251231_hist.zip` |
| 02667 | Köln/Bonn | `stundenwerte_TU_02667_19600101_20251231_hist.zip` |
| 03032 | List auf Sylt | `stundenwerte_TU_03032_19490101_20251231_hist.zip` |
| 03668 | Nürnberg | `stundenwerte_TU_03668_19510101_20251231_hist.zip` |
| 04104 | Regensburg | `stundenwerte_TU_04104_19480101_20251231_hist.zip` |
| 04271 | Rostock-Warnemünde | `stundenwerte_TU_04271_19470101_20251231_hist.zip` |
| 05792 | Zugspitze | `stundenwerte_TU_05792_19500101_20251231_hist.zip` |

## Download and preparation

Place the listed ZIP archives in `raw/`. Notebook 01 extracts them to
`extracted/`, maps DWD `-999` values to missing values, and writes the
2000–2025 files to `processed/temperature_2000_2025/`. Later runs reuse only
`processed/`; raw ZIP files are not modified.

## Version and license record

According to the DWD CDC [terms of use](https://opendata.dwd.de/climate_environment/CDC/Terms_of_use.pdf), data provided through the CDC OpenData area are available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Users should attribute DWD/CDC when using these data. For reproducibility, releases based on this workflow should also document the source archives and download date, the restriction to the years 2000–2025, and the recoding of source missing-value indicators described above.

The linked DWD CDC terms and CC BY 4.0 license information were last reviewed on **2026-09-02**.

## Release provenance record

The reference workflow used the 16 DWD ZIP archives listed in the station table above. The recorded access date for these source archives is **2026-08-18**.

The archive filenames indicate historical coverage through `20251231`, and the workflow restricts the analysis period to `2000-01-01 00:00` through `2025-12-31 23:00`. The source data were obtained from the DWD historical hourly air-temperature archive linked above.


## Benchmark design and review outputs

`configs/weather_final.toml` fixes the protocol: four non-overlapping gaps per
station-year in five hourly strata (1–6, 7–24, 25–72, 73–168, 169–440 hours).
This requests 8,320 gaps; the reference run contains 8,283 generated gaps.

The benchmark-output contract is defined in
[docs/methods.md](../docs/methods.md#8-exclusions-and-output-artifacts).

## Running the workflow

Follow the canonical notebook sequence in
[REPRODUCE.md](../REPRODUCE.md#3-run-one-domain-workflow), using the Weather
notebooks in `notebooks/weather/`.
