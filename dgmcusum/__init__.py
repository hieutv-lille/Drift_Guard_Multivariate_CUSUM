"""DG-MCUSUM: protected confirmation for persistent shifts under a moving baseline.

Reference implementation for the manuscript *Drift-Guard MCUSUM: Protected
Confirmation for Persistent Shifts under a Time-Varying Baseline*.

Layout
------
``config``       every constant, annotated with the section that fixes it
``charts``       the Crosier MCUSUM recursion, Eq. (6)
``kalman``       live filter and protected predictor, Eqs. (5) and (7)-(8)
``simulate``     vectorised run-length simulation for all compared charts
``calibrate``    control-limit calibration to a common false-alarm target
``metrics``      run-length summaries and fixed-window classification metrics
``fastkernel``   cached-path evaluator (C++ or NumPy) used by the search
``mechanism``    Layer 1 fixed-reference study, ablation, stationary boundary
``search``       Layer 1 joint parameter search and independent validation
``phase1``       Layer 2 finite Phase-I estimation and nested calibration
``industrial``   Layer 3 semi-synthetic gas-turbine stress test

Start from ``README.md``; ``PAPER_MAP.md`` maps every table and figure to the
script that produces it, and ``PROVENANCE.md`` records which outputs reproduce
the published numbers exactly and which are reimplementations.
"""
from __future__ import annotations

__version__ = "1.0.0"

from . import (calibrate, charts, config, fastkernel, kalman, mechanism,
               metrics, simulate)

__all__ = [
    "calibrate", "charts", "config", "fastkernel", "kalman", "mechanism",
    "metrics", "simulate", "__version__",
]
