"""Checks that the code obeys the propositions and conventions in the paper.

These are not regression tests against stored numbers; they assert the
mathematical claims the manuscript makes, so a change that breaks the theory
fails here rather than in review.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgmcusum.calibrate import calibrate_limit  # noqa: E402
from dgmcusum.charts import crosier_step, shift_direction  # noqa: E402
from dgmcusum.config import ChartDesign, GaussianModel  # noqa: E402
from dgmcusum.kalman import (LiveFilter, ProtectedPredictor,  # noqa: E402
                             protected_variance_factor)
from dgmcusum.metrics import (classification_metrics, restricted_mean,  # noqa: E402
                              roc_auc, summarize_run_lengths)
from dgmcusum.simulate import simulate_run_lengths  # noqa: E402


def test_crosier_is_rotation_invariant():
    """Eq. (6) is directionally invariant: rotating the input rotates the state."""
    rng = np.random.default_rng(0)
    p = 4
    q, _ = np.linalg.qr(rng.normal(size=(p, p)))
    s = rng.normal(size=(5, p))
    z = rng.normal(size=(5, p))
    s1, g1 = crosier_step(s, z, 0.5)
    s2, g2 = crosier_step(s @ q.T, z @ q.T, 0.5)
    assert np.allclose(s1 @ q.T, s2, atol=1e-12)
    assert np.allclose(g1, g2, atol=1e-12)


def test_crosier_resets_below_reference():
    """A small increment must drive the accumulator exactly to zero."""
    s = np.zeros((3, 4))
    z = np.full((3, 4), 0.01)
    s_new, stat = crosier_step(s, z, 0.5)
    assert np.allclose(s_new, 0.0)
    assert np.allclose(stat, 0.0)


def test_protected_variance_matches_closed_form():
    """Iterated propagation must equal Eq. (36)."""
    model = GaussianModel()
    predictor = ProtectedPredictor(1, model.p, model.q_level, model.q_slope)
    a, b, d = 3.0, 0.4, 0.2
    predictor.c00[:] = a
    predictor.c01[:] = b
    predictor.c11[:] = d
    mask = np.ones(1, dtype=bool)
    for age in range(0, 8):
        expected = protected_variance_factor(a, b, d, model.q_level,
                                             model.q_slope, age)
        assert abs((predictor.c00[0] + 1.0) - expected) < 1e-9, age
        predictor.advance(mask)


def test_live_innovations_are_standard_normal_under_h0():
    """Proposition 1: with no shift the live innovations are iid N(0, I)."""
    model = GaussianModel()
    rng = np.random.default_rng(7)
    n, steps = 4000, 60
    filt = LiveFilter(n, model.p, model.q_level, model.q_slope)
    level = np.zeros((n, model.p))
    slope = np.zeros((n, model.p))
    collected = []
    for t in range(steps):
        level = level + slope + math.sqrt(model.q_level) * rng.normal(size=level.shape)
        slope = slope + math.sqrt(model.q_slope) * rng.normal(size=slope.shape)
        y = level + rng.normal(size=level.shape)
        z = filt.step(y)[0]
        if t >= 40:                      # let the diffuse prior wash out
            collected.append(z)
    z = np.concatenate(collected)
    assert abs(z.mean()) < 0.05, z.mean()
    assert abs(z.std() - 1.0) < 0.05, z.std()


def test_shift_direction_is_unit_norm():
    for direction in ("dense", "sparse", "mixed"):
        assert abs(np.linalg.norm(shift_direction(4, direction)) - 1.0) < 1e-12


def test_censored_paths_are_retained_not_dropped():
    """A non-signalling path must enter the mean at the horizon, not vanish."""
    horizon = 100
    rl = np.array([10, 50, horizon + 1, horizon + 1])
    assert restricted_mean(rl, horizon) == (10 + 50 + 100 + 100) / 4.0
    stats = summarize_run_lengths(rl, horizon)
    assert stats["n_survived"] == 4
    assert abs(stats["censor_rate"] - 0.5) < 1e-12


def test_delayed_change_conditions_on_survival():
    """For tau > 1 only paths surviving to tau are scored, with delay rebased."""
    horizon, tau = 100, 51
    rl = np.array([10, 60, 80])          # the first alarms before the change
    stats = summarize_run_lengths(rl, horizon, tau=tau)
    assert stats["n_survived"] == 2
    assert abs(stats["prechange_alarm_rate"] - 1.0 / 3.0) < 1e-12
    assert abs(stats["mean_rl"] - ((60 - 50) + (80 - 50)) / 2.0) < 1e-12


def test_calibration_hits_the_target():
    """A calibrated limit should land near the nominal in-control mean."""
    model, design = GaussianModel(), ChartDesign()
    h = calibrate_limit("kf", model=model, design=design, target=500.0,
                        n_rep=600, max_steps=2500, lo=5.0, hi=15.0, seed=1201)
    rl = simulate_run_lengths("kf", h, n_rep=1500, max_steps=2500, model=model,
                              design=design, seed=2201)
    achieved = restricted_mean(rl, 2500)
    assert 380.0 < achieved < 650.0, achieved


def test_protection_only_opens_above_the_warning_limit():
    """An unreachable warning limit must make DG behave like a chart that never alarms."""
    design = ChartDesign("dg", 0.5, 1e6, 0.25, 15)
    rl = simulate_run_lengths("dg", 1.0, n_rep=50, max_steps=200,
                              design=design, delta=3.0, seed=5)
    assert np.all(rl == 201)


def test_classification_metrics_agree_with_definition():
    metrics = classification_metrics(recall=0.4, false_positive_rate=0.1,
                                     prevalence=0.5)
    assert abs(metrics["precision"] - 0.8) < 1e-12
    assert abs(metrics["f1"] - (2 * 0.5 * 0.4) / (0.5 * 1.4 + 0.5 * 0.1)) < 1e-12
    assert abs(metrics["balanced_accuracy"] - 0.65) < 1e-12


def test_roc_auc_handles_ties_and_extremes():
    assert abs(roc_auc([2, 3, 4], [0, 1]) - 1.0) < 1e-12
    assert abs(roc_auc([0, 1], [2, 3]) - 0.0) < 1e-12
    assert abs(roc_auc([1, 1], [1, 1]) - 0.5) < 1e-12


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS %s" % name)
    print("all property tests passed")
