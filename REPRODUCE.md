# Reproduction guide

Written for a reviewer who wants to check the paper's numbers without reading
the whole library first.

## 0. Environment

```bash
python --version            # 3.8 or newer
pip install -r requirements.txt
```

A C++ compiler is optional. With `g++` or `clang++` on `PATH` the parameter
search uses an OpenMP kernel; without one it falls back to NumPy, which is
slower but returns identical run lengths. Check which you have:

```bash
python -c "from dgmcusum import fastkernel; print(fastkernel.backend_info())"
```

On Windows with MinGW the compiled DLL needs `libgomp-1.dll` from the
compiler's own `bin` directory; `fastkernel` adds that directory to the DLL
search path automatically.

## 1. Verify the port before trusting any output (≈15 seconds)

```bash
python tests/run_tests.py
```

No extra dependencies. If you have pytest, `python -m pytest tests/ -v` works
too, as does running any single file directly:

```bash
python tests/test_equivalence.py
```

What each one buys you:

* `test_equivalence.py` — the refactored simulator is **bit-identical** to the
  archived implementation in `legacy/` that produced the published Layer-1
  tables. If this passes, Layer-1 numbers are the paper's numbers.
* `test_kernel.py` — the direct simulator, the C++ kernel and the NumPy
  fallback agree exactly, including for a delayed change.
* `test_properties.py` — rotation invariance of Eq. (6), the closed-form
  protected variance factor of Eq. (36), Proposition 1's null distribution,
  censoring and survival-conditioning conventions, and the classification
  metric definitions.

## 2. Smoke-test the whole pipeline (≈5 minutes)

```bash
python scripts/run_all.py --quick
```

Every script stamps `"quick": true` into `script_metadata.json`, so a
reduced-budget run can never be mistaken for a result later.

## 3. Full reproduction

Run the layers independently; each writes to its own `outputs/` subdirectory.

```bash
python scripts/01_mechanism_study.py
python scripts/02_parameter_search.py
python scripts/03_phase1_study.py
python scripts/04_industrial_study.py --archive data/gt_2015.csv
```

### Expected runtimes

Measured on 8 threads. Treat as order-of-magnitude, not benchmarks.

| Stage | `--quick` | full |
|---|---|---|
| 01 mechanism study | ~3 min | 1–3 h |
| 02 parameter search (C++ kernel) | ~4 s | ~5 min |
| 02 parameter search (NumPy fallback) | ~30 s | 1–2 h |
| 03 Phase-I study | ~10 min | 6–12 h |
| 04 industrial study | ~5 min | 1–3 h |

Layer 2 is the slow one: 540 profile-likelihood fits, each a Nelder-Mead search
over a Kalman pass, then nested calibration on top. Memory stays modest except
in the full parameter search, which can hold roughly 8 GB of cached paths.

### Layer 3 needs the data

The UCI archive is not redistributed. See `data/README.md`. Without it,
`04_industrial_study.py` exits with code 2 and `run_all.py` reports Layer 3 as
skipped rather than failing.

## 4. What to compare against the paper

Read `PROVENANCE.md` first — it says which of these should match exactly.

### Layer 1, exact

| Paper | Where to look |
|---|---|
| Static chart collapses to `ARL0 ≈ 29.7` under drift | `outputs/mechanism/arl0_validation.csv` |
| DG `ARL0 = 495.1`, MC interval 483.8–506.4 | same file |
| Delay reductions of 39.3% and 48.4% at Δ = 1.5, 2 | `outputs/mechanism/arl1_main.csv` |
| Selected design `(0.5, 1, 0.1, 20)`, `h = 23.1655` | `outputs/search/locked_designs.json`, `final_limits.json` |
| Top two refined scores 149.58 and 149.68 | `outputs/search/refined_search.csv` |
| DG in-control 501.7 [494.9, 508.4]; KF mildly conservative | `outputs/search/independent_tests.csv` |
| Delayed change at τ = 201 **favours KF** | same file, `case = late_change` |

### Layers 2 and 3, findings only

Check the direction of the effect, not the digits:

* plug-in limits after estimation are **anti-conservative**; nested calibration
  brings the clustered intervals back to cover 500;
* Phase-I precision improves with `m` — the median relative `Σ` error should
  fall to roughly 0.09 at `m = 600`;
* the gas-turbine fit should select the **local-level** model on both AIC and
  BIC, with a badly conditioned covariance and heavy-tailed innovations;
* at Δ = 8 protection should change **tail risk** rather than uniformly
  accelerating detection, and the ordering should move when the cap or the
  shrinkage changes.

## 5. Output format

Every script writes plain CSV, plus JSON for run metadata and the industrial
fit. Nothing here regenerates the manuscript's typeset tables or figures --
those are already in the paper. Compare the CSV columns against the printed
tables directly; the column names follow the quantities the paper reports
(`mean_rl`, `ci_low`, `ci_high`, `median_rl`, `p_signal_50`, `censor_rate`,
and for delayed changes `n_survived` and `prechange_alarm_rate`).

## 6. Changing the study

The protocol constants live in one place, `dgmcusum/config.py`. If you change
the grid, the objective or the budgets:

* use a **new output directory** and **fresh seed streams**;
* re-run selection — a design chosen under one objective is not a design chosen
  under another;
* do not re-use the archived final-test seeds, which exist so that selection
  and testing stay disjoint;
* re-read the generated prose. The manuscript's interpretation is scoped to the
  delivered protocol, and the table captions are not a general report writer.

Specifically, the delayed-change result is the one that most invites retuning.
Do not tune on those final test paths to make it go away; an objective that
includes realistic arrival times requires new selection and new independent
test streams.
