"""Layer 1 mechanism study: Tables 2-4 and 7-9, Figures 2, 5 and 6.

This is the fixed-reference experiment of Section 3.2 -- the one that isolates
*why* protection helps, using the prespecified configuration
``(k1,g,k2,L) = (0.5, 1.5, 0.25, 15)`` rather than the searched one.  Its
budgets and seeds differ from the selected-design comparison of Section 4.1,
so its numbers must be read inside its own protocol.

The seed formulas below are reproduced exactly from the archived reference
implementation.  They look arbitrary because they are: what matters is that
they are fixed in advance and that separate scenarios never share a stream.
Changing one changes the published numbers.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from .calibrate import calibrate_limit
from .config import (DG_REFERENCE, METHOD_LABELS, MechanismBudget,
                     ChartDesign, GaussianModel)
from .metrics import summarize_run_lengths
from .simulate import Workload, simulate_run_lengths

#: Method-specific starting brackets used by the archived calibration.
CALIBRATION_BRACKETS = {
    "oracle": (5.0, 15.0),
    "kf": (5.0, 15.0),
    "dg": (8.0, 28.0),
    "ewma": (20.0, 100.0),
    "static": (5.0, 15.0),
    "dg_live": (5.0, 30.0),
    "dg_freeze": (10.0, 60.0),
}


def calibrate_all(model, design, budget, full=True):
    """Calibrate every Layer-1 chart to ``ARL_0 = 500``.

    The DG limit gets a second, higher-precision pass on a narrow bracket:
    candidate selection makes its run-length curve steeper near the target than
    the single-stage comparators, so the coarse pass is not precise enough.

    The static chart is calibrated under a *stationary* process and then
    challenged under drift.  That mismatch is the experiment, not an error.
    """
    n_cal = budget.n_calibration if full else 700
    limits = {}
    for j, method in enumerate(["oracle", "kf", "dg", "ewma"]):
        lo, hi = CALIBRATION_BRACKETS[method]
        limits[method] = calibrate_limit(
            method, model=model, design=design, n_rep=n_cal, max_steps=2500,
            lo=lo, hi=hi, seed=1200 + j)
    if full:
        limits["dg"] = calibrate_limit(
            "dg", model=model, design=design, n_rep=budget.n_dg_refinement,
            max_steps=3000, lo=16.35, hi=16.55, seed=9010, iterations=9)

    limits["static"] = calibrate_limit(
        "static", model=model, design=design, n_rep=n_cal, max_steps=2500,
        lo=5.0, hi=15.0, seed=1300,
        q_level_data=0.0, q_slope_data=0.0)
    return limits


def run_mechanism_study(outdir, full=True, model=None, design=None):
    """Run the complete fixed-reference study and write its CSV outputs.

    Returns ``(limits, arl0, arl1)``.  Files written into ``outdir``:
    ``control_limits.csv``, ``arl0_validation.csv``, ``arl1_main.csv``,
    ``direction_sensitivity.csv``, ``drift_misspecification.csv``,
    ``design_sensitivity.csv``, ``run_metadata.json``.
    """
    model = model or GaussianModel()
    design = design or DG_REFERENCE
    budget = MechanismBudget()
    outdir.mkdir(parents=True, exist_ok=True)

    n_validate = budget.n_arl0_validation if full else 1800
    n_ooc = budget.n_arl1 if full else 1500
    n_sensitivity = budget.n_sensitivity if full else 1000
    max_ic = max_ooc = budget.max_steps

    limits = calibrate_all(model, design, budget, full=full)
    pd.DataFrame([{"method": METHOD_LABELS[m], "key": m, "h": h}
                  for m, h in limits.items()]
                 ).to_csv(outdir / "control_limits.csv", index=False)

    # ---- independent in-control validation (Table 2) ----
    rows = []
    for j, method in enumerate(["oracle", "kf", "dg", "ewma", "static"]):
        rl = simulate_run_lengths(method, limits[method], n_rep=n_validate,
                                  max_steps=max_ic, model=model, design=design,
                                  seed=2200 + j)
        rows.append({"method": METHOD_LABELS[method], "key": method,
                     "h": limits[method], **summarize_run_lengths(rl, max_ic)})
    arl0 = pd.DataFrame(rows)
    arl0.to_csv(outdir / "arl0_validation.csv", index=False)

    # ---- detection delay by shift size (Table 3) ----
    rows = []
    for j, method in enumerate(["oracle", "kf", "dg", "ewma"]):
        for d in budget.deltas:
            rl = simulate_run_lengths(
                method, limits[method], n_rep=n_ooc, max_steps=max_ooc,
                model=model, design=design, delta=d,
                seed=3200 + 100 * j + int(100 * d))
            rows.append({"method": METHOD_LABELS[method], "key": method,
                         "delta": d, "direction": "dense",
                         **summarize_run_lengths(rl, max_ooc)})
    arl1 = pd.DataFrame(rows)
    arl1.to_csv(outdir / "arl1_main.csv", index=False)

    # ---- holdout shift directions: same magnitude, different pattern ----
    rows = []
    for mi, method in enumerate(["kf", "dg"]):
        for di, direction in enumerate(["dense", "sparse", "mixed"]):
            rl = simulate_run_lengths(
                method, limits[method], n_rep=n_sensitivity, max_steps=max_ooc,
                model=model, design=design, delta=1.0, direction=direction,
                seed=4100 + 100 * mi + 10 * di)
            rows.append({"method": METHOD_LABELS[method], "direction": direction,
                         "delta": 1.0, **summarize_run_lengths(rl, max_ooc)})
    pd.DataFrame(rows).to_csv(outdir / "direction_sensitivity.csv", index=False)

    # ---- drift misspecification: chart fixed, true Q scaled by 0.5 / 2 ----
    rows = []
    for mi, method in enumerate(["kf", "dg"]):
        for si, scale in enumerate([0.5, 1.0, 2.0]):
            rl = simulate_run_lengths(
                method, limits[method], n_rep=n_sensitivity, max_steps=max_ic,
                model=model, design=design,
                q_level_data=model.q_level * scale,
                q_slope_data=model.q_slope * scale,
                seed=5100 + 100 * mi + si)
            rows.append({"method": METHOD_LABELS[method], "Q_scale": scale,
                         **summarize_run_lengths(rl, max_ic)})
    pd.DataFrame(rows).to_csv(outdir / "drift_misspecification.csv", index=False)

    # ---- compact design sensitivity around the warning limit ----
    rows = []
    for wi, warning in enumerate([1.0, 1.5, 2.0]):
        cfg = ChartDesign(design.method, design.k1, warning, design.k2, design.L)
        h = calibrate_limit("dg", model=model, design=cfg,
                            n_rep=max(900, budget.n_calibration // 2),
                            max_steps=2200, lo=8.0, hi=28.0,
                            seed=6100 + wi, iterations=11)
        rl = simulate_run_lengths("dg", h,
                                  n_rep=max(1600, n_sensitivity // 2),
                                  max_steps=max_ooc, model=model, design=cfg,
                                  delta=1.0, seed=6200 + wi)
        rows.append({"warning_limit": warning,
                     "confirmation_horizon": cfg.L, "h": h,
                     **summarize_run_lengths(rl, max_ooc)})
    pd.DataFrame(rows).to_csv(outdir / "design_sensitivity.csv", index=False)

    (outdir / "run_metadata.json").write_text(json.dumps({
        "full_run": full,
        "n_calibration": budget.n_calibration if full else 700,
        "n_arl0_validation": n_validate,
        "n_arl1": n_ooc,
        "n_sensitivity": n_sensitivity,
        "seed_policy": "Fixed, scenario-specific seeds listed in dgmcusum/mechanism.py",
        "model": asdict(model),
        "design": asdict(design),
    }, indent=2), encoding="utf-8")
    return limits, arl0, arl1


# --------------------------------------------------------------------------
# Ablation and workload (Table 4 and the workload table)
# --------------------------------------------------------------------------


def run_ablation(outdir, full=True, model=None, design=None):
    """Separate "second accumulator" from "protected reference".

    Three charts share the primary recursion and differ only in what the second
    stage reads:

    ``kf``         no second stage at all;
    ``dg_live``    a second accumulator fed *live* residuals;
    ``dg``         a second accumulator fed the protected reference.

    ``dg_freeze`` is the single-track variant that suspends the filter, which
    performs poorly after rejected candidates and is the reason the live and
    protected states are kept separate.

    .. note::
       ``dg_live`` and ``dg_freeze`` are reimplemented from the manuscript's
       description; the original ablation script was not archived.  See
       ``PROVENANCE.md``.
    """
    model = model or GaussianModel()
    design = design or DG_REFERENCE
    budget = MechanismBudget()
    outdir.mkdir(parents=True, exist_ok=True)
    n_cal = budget.n_calibration if full else 700
    n_rep = budget.n_arl1 if full else 1200
    horizon = budget.max_steps

    rows = []
    for j, method in enumerate(["kf", "dg_live", "dg_freeze", "dg"]):
        lo, hi = CALIBRATION_BRACKETS[method]
        h = calibrate_limit(method, model=model, design=design, n_rep=n_cal,
                            max_steps=2500, lo=lo, hi=hi, seed=14100 + j)
        row = {"method": METHOD_LABELS[method], "key": method, "h": h}
        for tag, delta in [("arl0", 0.0), ("arl1", 1.0), ("arl2", 2.0)]:
            rl = simulate_run_lengths(method, h, n_rep=n_rep, max_steps=horizon,
                                      model=model, design=design, delta=delta,
                                      seed=14200 + 100 * j + int(100 * delta))
            stats = summarize_run_lengths(rl, horizon)
            for name in ("mean_rl", "se_mean", "ci_low", "ci_high", "median_rl",
                         "p_signal_25", "p_signal_50", "p_signal_100",
                         "censor_rate"):
                row["%s_%s" % (tag, name)] = stats[name]
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "candidate_ablation.csv", index=False)
    return frame


def run_workload(outdir, limits=None, full=True, model=None, design=None):
    """Count candidate openings, rejections and protected time.

    Answers the operational question the paper raises: candidates are frequent
    and mostly rejected, so they must remain internal computational states
    rather than operator-facing alerts.
    """
    model = model or GaussianModel()
    design = design or DG_REFERENCE
    budget = MechanismBudget()
    outdir.mkdir(parents=True, exist_ok=True)
    h = (limits or {}).get("dg")
    if h is None:
        h = calibrate_limit("dg", model=model, design=design,
                            n_rep=budget.n_calibration if full else 700,
                            max_steps=2500, lo=8.0, hi=28.0, seed=1202)
    workload = Workload()
    simulate_run_lengths("dg", h, n_rep=budget.n_arl0_validation if full else 1500,
                         max_steps=budget.max_steps, model=model, design=design,
                         seed=15100, workload=workload)
    frame = pd.DataFrame([workload.as_dict("Gaussian design")])
    frame.to_csv(outdir / "gaussian_candidate_workload.csv", index=False)
    return frame


# --------------------------------------------------------------------------
# Stationary boundary case (Section 4.2)
# --------------------------------------------------------------------------


def stationary_model():
    """Constant level, zero slope: the boundary where protection stops paying.

    With ``q_level = q_slope = 0`` the Kalman gain tends to zero and feedback
    masking disappears, so a static chart is preferable.  Reporting this is the
    point; it bounds the claim made for DG-MCUSUM.
    """
    return GaussianModel(q_level=0.0, q_slope=0.0)


def run_stationary_study(outdir, full=True, design=None):
    """Recalibrate every chart under stationarity and recompare."""
    model = stationary_model()
    design = design or DG_REFERENCE
    budget = MechanismBudget()
    outdir.mkdir(parents=True, exist_ok=True)
    n_cal = budget.n_calibration if full else 700
    n_validate = budget.n_arl0_validation if full else 1500
    n_ooc = budget.n_arl1 if full else 1200
    horizon = budget.max_steps

    limits = {}
    for j, method in enumerate(["oracle", "static", "kf", "dg"]):
        lo, hi = CALIBRATION_BRACKETS[method]
        limits[method] = calibrate_limit(
            method, model=model, design=design, n_rep=n_cal, max_steps=2500,
            lo=lo, hi=hi, seed=17100 + j)
    pd.DataFrame([{"method": METHOD_LABELS[m], "key": m, "h": h}
                  for m, h in limits.items()]
                 ).to_csv(outdir / "stationary_control_limits.csv", index=False)

    rows = []
    for j, method in enumerate(["oracle", "static", "kf", "dg"]):
        rl = simulate_run_lengths(method, limits[method], n_rep=n_validate,
                                  max_steps=horizon, model=model, design=design,
                                  seed=18100 + j)
        rows.append({"method": METHOD_LABELS[method], "key": method,
                     "h": limits[method], **summarize_run_lengths(rl, horizon)})
    arl0 = pd.DataFrame(rows)
    arl0.to_csv(outdir / "stationary_arl0_validation.csv", index=False)

    rows = []
    for j, method in enumerate(["oracle", "static", "kf", "dg"]):
        for delta in budget.deltas:
            rl = simulate_run_lengths(
                method, limits[method], n_rep=n_ooc, max_steps=horizon,
                model=model, design=design, delta=delta,
                seed=19100 + 100 * j + int(100 * delta))
            rows.append({"method": METHOD_LABELS[method], "key": method,
                         "delta": delta, **summarize_run_lengths(rl, horizon)})
    arl1 = pd.DataFrame(rows)
    arl1.to_csv(outdir / "stationary_arl1_main.csv", index=False)
    return limits, arl0, arl1
