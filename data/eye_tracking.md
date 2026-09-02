# Eye-Tracking data requirements

The Eye-Tracking benchmark uses four external datasets. Raw data is not
distributed with this repository.

## Local layout

```text
$BENCHMARK_DATA_DIR/
└── raw/
    ├── GazeBase_v2_0/**/S_*_TEX.csv
    ├── gazebasevr/data/S_*_TEX.csv
    ├── ZUCO/*/*_NR*_ET.mat
    └── Pedrotti/[0-9][0-9].txt
```

## What a user must provide

1. Download the four source datasets and arrange them below one external data
   root using the layout above.
2. Set `BENCHMARK_DATA_DIR` to that external root in `.env`.
3. Set `EYE_TRACKING_BENCHMARK_DIR` to a new, non-existent output directory
   when the published `benchmarks/eyetracking/` directory should not be
   replaced.

Notebook 01 reads `raw/` directly; it does not create a preprocessed source
directory.

## Source data and study use

| Dataset | Official source | Files used by this workflow |
| --- | --- | --- |
| GazeBase | [Figshare](https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257?file=27039812) | Reading-task `S_*_TEX.csv` files |
| GazeBaseVR | [Figshare](https://figshare.com/articles/dataset/GazeBaseVR_Data_Repository/21308391?file=38844024) | Text-task `S_*_TEX.csv` files |
| ZuCo 2.0 | [OSF](https://osf.io/2urht/overview) | Normal-reading `*_NR*_ET.mat` files |
| Pedrotti et al. | [Zenodo](https://zenodo.org/records/7962917) | `Pedrotti/[0-9][0-9].txt` participant files |

This restriction provides a common behavioral context across datasets and avoids mixing reading recordings with unrelated eye-tracking tasks such as fixation, saccade, video-viewing, or gaming paradigms. All retained recordings are reading tasks. For Pedrotti, word and pseudoword
trials are retained and numeral trials are excluded. The selected ZuCo
participants are `YAC`, `YAG`, `YAK`, `YDG`, `YDR`, `YFR`, `YFS`, `YHS`, `YIS`,
and `YLS`.

## Download and preparation

Download and unpack the four sources so that the files shown in the local
layout are regular files. GazeBase participant archives must be extracted.
Notebook 01 loads the raw files directly; it keeps each source's native
coordinate system and does not convert pixels to degrees.

## Version and license record

For each benchmark run, record the exact source release or archive version used. At the time this workflow was documented, the GazeBase and GazeBaseVR Figshare records were licensed under CC BY 4.0. The Pedrotti dataset is provided through Zenodo, while ZuCo 2.0 is provided through OSF. Users should verify the current license and access terms of each provider before redistributing source data or publishing a new release.

The linked source, license, and access information was last reviewed on 2026-09-01.

## Release provenance record

The currently published reference workflow used the local source copies listed
below. The recorded access/download date for all four Eye-Tracking sources is
**2026-07-28**

| Dataset | Source release represented locally | Local source timestamp | Release note |
| --- | --- | --- | --- |
| GazeBase | `GazeBase_v2_0` | 2026-07-28 | Reading-task source files used by this workflow. |
| GazeBaseVR | `gazebasevr/data` from the linked Figshare record | 2026-07-28 | Text-task source files used by this workflow. |
| ZuCo | ZuCo 2.0 normal-reading `*_NR*_ET.mat` files | 2026-07-28 | Local access/extraction date for the 70 retained MAT files. |
| Pedrotti et al. | Zenodo record 7962917, participant text files | 2026-07-28 | Local access/extraction date for the 33 files in `Pedrotti/`. |

The ZuCo source used in this workflow is the ZuCo 2.0 normal-reading material available through the OSF project linked above. No separate archive-version identifier was available for the local copy.


## Benchmark design and review outputs

`configs/eye_tracking_final.toml` fixes the protocol: 10 participants per
dataset, two recordings per participant for GazeBase, GazeBaseVR, and ZuCo,
and 40 Pedrotti trials per participant. It requests 400 gaps per dataset in
five strata between 0 and 250 ms (1,600 in total before exclusions).

The benchmark-output contract is defined in
[docs/methods.md](../docs/methods.md#8-exclusions-and-output-artifacts).

## Running the workflow

Follow the canonical notebook sequence in
[REPRODUCE.md](../REPRODUCE.md#3-run-one-domain-workflow), using the Eye-Tracking
notebooks in `notebooks/eyetracking/`.
