"""The three evaluators must agree exactly.

There are three independent implementations of the same stopping rule:

* ``dgmcusum.simulate``  -- direct, generates the process as it goes;
* ``native/kernel.cpp``  -- cached-path C++ kernel used by the search;
* the NumPy fallback in ``dgmcusum.fastkernel``.

They are written differently on purpose.  The cached-path versions replay the
chart over precomputed live innovations and add a deterministic shift response,
and they track the protected predictor through a live-minus-protected state
difference rather than propagating a second filter.  If all three produce the
same run lengths, that algebraic equivalence is not just asserted in the text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgmcusum import fastkernel as fk  # noqa: E402
from dgmcusum.charts import shift_direction  # noqa: E402
from dgmcusum.config import ChartDesign, GaussianModel  # noqa: E402
from dgmcusum.simulate import simulate_run_lengths  # noqa: E402

DESIGNS = [ChartDesign("dg", 0.5, 1.5, 0.25, 15),
           ChartDesign("dg", 0.25, 1.0, 0.1, 20),
           ChartDesign("dg", 1.0, 2.0, 0.5, 5),
           ChartDesign("kf", 0.5),
           ChartDesign("kf", 0.75)]
SCENARIOS = [(0.0, 12.0), (1.0, 16.4), (2.0, 9.6)]


def test_numpy_backend_matches_direct_simulator():
    n, horizon, seed = 150, 800, 31
    paths = fk.generate_paths(n, horizon, seed)
    for design in DESIGNS:
        for delta, h in SCENARIOS:
            cached = fk._evaluate_numpy(
                paths[0], paths[1], fk.shift_response(paths[1], delta, 1),
                shift_direction(paths[0].shape[2]), design, h, 1,
                GaussianModel())[0]
            direct = simulate_run_lengths(
                design.method, h, n_rep=n, max_steps=horizon, design=design,
                delta=delta, seed=seed)
            assert np.array_equal(cached, direct), (design.key, delta)


def test_cpp_backend_matches_numpy_backend():
    if fk.compile_kernel() is None:
        print("SKIP: no C++ compiler available (%s)" % fk._CPP_STATE["error"])
        return
    mismatches = fk.self_test()
    assert mismatches == 0, "C++ and NumPy backends disagree on %d paths" % mismatches


def test_cpp_backend_matches_direct_simulator():
    if fk.compile_kernel() is None:
        print("SKIP: no C++ compiler available")
        return
    n, horizon, seed = 150, 800, 31
    paths = fk.generate_paths(n, horizon, seed)
    for design in DESIGNS:
        for delta, h in SCENARIOS:
            cached = fk.evaluate(paths, design, h, delta=delta, backend="cpp")[0]
            direct = simulate_run_lengths(
                design.method, h, n_rep=n, max_steps=horizon, design=design,
                delta=delta, seed=seed)
            assert np.array_equal(cached, direct), (design.key, delta)


def test_delayed_change_is_consistent():
    """A change at tau must match the direct simulator's delayed injection."""
    n, horizon, seed, tau = 120, 700, 555, 201
    paths = fk.generate_paths(n, horizon, seed)
    design = ChartDesign("dg", 0.5, 1.0, 0.1, 20)
    cached = fk.evaluate(paths, design, 20.0, delta=1.5, tau=tau)[0]
    direct = simulate_run_lengths("dg", 20.0, n_rep=n, max_steps=horizon,
                                  design=design, delta=1.5, tau=tau, seed=seed)
    assert np.array_equal(cached, direct)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS %s" % name)
    print("all kernel tests passed")
