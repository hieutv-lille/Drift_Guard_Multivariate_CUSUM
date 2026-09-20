"""Pathwise equivalence with the archived reference implementation.

This is the test that protects the published numbers.  ``dgmcusum.simulate`` is
a refactor of ``legacy/dg_mcusum_study.py``, and a refactor is only safe here
if it is *bit-identical*: the random-draw order determines every table in the
paper, so an innocuous-looking reordering would silently move all of them.

Run with::

    python -m pytest tests/ -v
    python tests/test_equivalence.py      # also works without pytest
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgmcusum.config import ChartDesign, GaussianModel  # noqa: E402
from dgmcusum.simulate import simulate_run_lengths  # noqa: E402


def load_legacy():
    """Import the archived script by path, without installing it."""
    path = ROOT / "legacy" / "dg_mcusum_study.py"
    spec = importlib.util.spec_from_file_location("legacy_study", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["legacy_study"] = module      # dataclasses needs this registered
    spec.loader.exec_module(module)
    return module


#: (method, delta, seed, h) cases spanning in-control, moderate and large shifts.
CASES = [(m, d, s, h)
         for m in ("oracle", "static", "ewma", "kf", "dg")
         for d, s, h in ((0.0, 11, 12.0), (1.0, 22, 12.0), (2.0, 33, 9.0))]


def test_simulator_matches_archive():
    legacy = load_legacy()
    config = legacy.StudyConfig()
    model, design = GaussianModel(), ChartDesign()
    for method, delta, seed, h in CASES:
        expected = legacy.simulate_run_lengths(
            method, h, n_rep=40, max_steps=300, config=config, delta=delta,
            seed=seed)
        actual = simulate_run_lengths(
            method, h, n_rep=40, max_steps=300, model=model, design=design,
            delta=delta, seed=seed)
        assert np.array_equal(expected, actual), (
            "run lengths diverged for method=%s delta=%s" % (method, delta))


def test_holdout_directions_match_archive():
    legacy = load_legacy()
    config = legacy.StudyConfig()
    model, design = GaussianModel(), ChartDesign()
    for direction in ("dense", "sparse", "mixed"):
        expected = legacy.simulate_run_lengths(
            "dg", 12.0, n_rep=30, max_steps=250, config=config, delta=1.0,
            direction=direction, seed=77)
        actual = simulate_run_lengths(
            "dg", 12.0, n_rep=30, max_steps=250, model=model, design=design,
            delta=1.0, direction=direction, seed=77)
        assert np.array_equal(expected, actual), direction


def test_nondefault_design_matches_archive():
    """A changed design must flow through identically, not just the default."""
    legacy = load_legacy()
    config = legacy.StudyConfig(k_primary=0.25, warning_limit=1.0,
                                k_confirmation=0.1, confirmation_horizon=20)
    design = ChartDesign("dg", 0.25, 1.0, 0.1, 20)
    expected = legacy.simulate_run_lengths(
        "dg", 20.0, n_rep=40, max_steps=400, config=config, delta=1.5, seed=99)
    actual = simulate_run_lengths(
        "dg", 20.0, n_rep=40, max_steps=400, model=GaussianModel(),
        design=design, delta=1.5, seed=99)
    assert np.array_equal(expected, actual)


def test_misspecified_drift_matches_archive():
    """The drift-misspecification path uses a data/filter mismatch."""
    legacy = load_legacy()
    base = legacy.StudyConfig()
    for scale in (0.5, 2.0):
        config = legacy.StudyConfig(q_level_data=base.q_level_data * scale,
                                    q_slope_data=base.q_slope_data * scale)
        expected = legacy.simulate_run_lengths(
            "dg", 16.0, n_rep=30, max_steps=300, config=config, seed=4242)
        model = GaussianModel()
        actual = simulate_run_lengths(
            "dg", 16.0, n_rep=30, max_steps=300, model=model,
            design=ChartDesign(), q_level_data=model.q_level * scale,
            q_slope_data=model.q_slope * scale, seed=4242)
        assert np.array_equal(expected, actual), scale


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS %s" % name)
    print("all equivalence tests passed")
