# Gas-turbine data (Layer 3)

The measured archive is **not redistributed** with this package. Download it
from the source under its own terms.

## What is needed

**UCI Gas Turbine CO and NOx Emission Data Set**
DOI: <https://doi.org/10.24432/C5WC95>
UCI page: <https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set>

The paper uses the **2015 file only** — `gt_2015.csv`, 7,384 hourly
observations.

## Steps

1. Open the UCI page above and download the dataset archive.
2. Unzip it. It contains one CSV per year, `gt_2011.csv` … `gt_2015.csv`.
3. Copy **`gt_2015.csv`** into this directory, so the path is
   `DG_MCUSUM/data/gt_2015.csv`.
4. Run:

```bash
python scripts/04_industrial_study.py --archive data/gt_2015.csv
```

Or point `--archive` at wherever you keep it.

## What the code expects

The file must have a header row including these four columns, which are the
monitored channels:

| Column | Meaning |
|---|---|
| `GTEP` | gas-turbine exhaust pressure |
| `TIT`  | turbine inlet temperature |
| `TAT`  | turbine after temperature |
| `CDP`  | compressor discharge pressure |

Other columns are ignored. The first **1,500** rows form Phase I.

`load_archive` checks the columns are present, and prints a warning if the row
count is not 7,384 — a different row count means a different file, and every
downstream number would shift.

## Citation

> Kaya, H., Tüfekci, P., & Uzun, E. (2019). *Predicting CO and NOx emissions
> from gas turbines: novel data and a benchmark PEMS.* Turkish Journal of
> Electrical Engineering and Computer Sciences, 27(6), 4783–4796.

Cited in the manuscript as `kaya2019predicting` and
`gas_turbine_co_and_nox_emission_data_set_551`.

## A note on what this layer does

The archive carries **no verified fault times**. Shifts are injected into a
bootstrap of the fitted Phase-I innovations, so this is a semi-synthetic stress
test of the masking mechanism — not validation on observed industrial faults.
The manuscript states this; the code repeats it on every run.
