# Archived original sources — do not edit

These four files are the **unmodified** implementation that produced the
published Layer-1 results. They are kept here as ground truth, not as part of
the working package.

| File | Role |
|---|---|
| `dg_mcusum_study.py` | fixed-reference mechanism study (Layer 1a) |
| `original_optimize.py` | joint parameter-search driver (Layer 1b) |
| `original_test_equivalence.py` | the archive's own regression tests |
| `original_make_report.py` | LaTeX snippet generator for the manuscript |

The exact stopping-rule kernel that accompanied them, `kernel.cpp`, lives in
`../native/` because the working package compiles and uses it directly.

## Where they came from

The manuscript directory shipped a single notebook,
`Code_DG_MCUSUM_Optimization.ipynb`, whose third code cell embedded a base64
ZIP archive. These files were extracted from that archive byte-for-byte. No
line has been changed — not a path, not a constant, not a comment.

## Why they are still here

`../tests/test_equivalence.py` imports `dg_mcusum_study.py` by path and
requires the refactored simulator in `dgmcusum/` to produce **bit-identical**
run lengths. That test is the reason the package can claim to reproduce the
paper's numbers rather than merely numbers of the same kind. Editing anything
in this directory would destroy that guarantee.

## Running them directly

You do not need to, and the maintained entry points are in `../scripts/`. If
you want to anyway, note that these scripts were written for a different
working directory: they resolve their own output paths relative to their
location, so `original_optimize.py` would create `legacy/optimization_results/`
and `dg_mcusum_study.py` would create `legacy/results/` and `legacy/figures/`.
Both are ignored by `.gitignore`'s `outputs/` rule only if you redirect them;
prefer the scripts in `../scripts/`, which write to `../outputs/`.

`original_optimize.py` also requires a C++ compiler and, as written, POSIX
`os.sched_getaffinity`; the maintained `dgmcusum/fastkernel.py` handles both
portably and falls back to NumPy when no compiler is present.
