
```bash
pip install -r requirements.txt
python tests/run_tests.py           
python scripts/run_all.py --quick   
```

| File | What it answers |
|---|---|
| **`PROVENANCE.md`** | Which results reproduce exactly, and which are reimplementations. **Read this before quoting any number.** |
| **`PAPER_MAP.md`** | Every equation, table and figure in the paper → the code that produces its underlying numbers. |
| **`REPRODUCE.md`** | Step-by-step reproduction, runtimes, hardware notes. |
| `data/README.md` | How to obtain the gas-turbine archive (not redistributed). |

---

## What is where

```
dgmcusum/            the library
  config.py          every constant, annotated with the section that fixes it
  charts.py          Crosier MCUSUM recursion                        Eq. (6)
  kalman.py          live filter and protected predictor      Eqs. (5), (7-8)
  simulate.py        run-length simulation for all compared charts
  calibrate.py       control-limit calibration to a common ARL0 target
  metrics.py         run-length summaries, classification metrics
  fastkernel.py      cached-path evaluator (C++ or NumPy) for the search
  mechanism.py       Layer 1: fixed-reference study, ablation, stationary case
  search.py          Layer 1: joint parameter search, independent validation
  phase1.py          Layer 2: finite Phase-I estimation, nested calibration
  industrial.py      Layer 3: semi-synthetic gas-turbine stress test

scripts/             runnable entry points, one per layer; all take --quick
native/kernel.cpp    exact stopping-rule evaluator, compiled on demand
legacy/              the archived original sources, unmodified
tests/               equivalence, cross-backend and property tests
data/                where the UCI archive goes (empty; see its README)
outputs/             created by the scripts
```

## The three evaluation layers


| Script | Layer | Question |
|---|---|---|
| `01_mechanism_study.py` | 1 | Does protection reduce masking at matched false-alarm behaviour? |
| `02_parameter_search.py` | 1 | Which calibrated design serves the detection objective? |
| `03_phase1_study.py` | 2 | How much does fitting from a finite record distort run lengths? |
| `04_industrial_study.py` | 3 | Does the mechanism survive dependence, heavy tails and ill-conditioning? |

## Three implementations, cross-checked


* `dgmcusum/simulate.py` generates the process step by step;
* `native/kernel.cpp` replays the chart over cached live innovations, adding a
  deterministic shift response and tracking the protected predictor through a
  *live-minus-protected* state difference;
* a NumPy fallback in `dgmcusum/fastkernel.py` does the same without a compiler.


## License

**Not yet set.**

The archived sources in `legacy/` and the gas-turbine data are covered
separately: see `legacy/README.md` and `data/README.md`.
