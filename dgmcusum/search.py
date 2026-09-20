"""Layer 1 joint parameter search: Section 3.2, Tables 5-7 of the manuscript.

Calibrating ``h`` controls false alarms; it does not say which of the remaining
parameters detects a fault best.  ``k1``, ``g``, ``k2`` and ``L`` interact --
and every change to them changes the in-control stopping rule, so ``h`` must be
recalibrated for each candidate.  That is why the grid is searched jointly
rather than one parameter at a time.

Discipline enforced by this module
----------------------------------
* Two stages.  A cheap coarse pass over all 244 designs, then a refinement of
  the 12 best DG designs plus the fixed-reference DG design and all four KF
  references.  The fixed-reference and KF designs are *always* retained so the
  final comparison cannot be rigged by selection.
* The design is locked (written to ``locked_designs.json``) before the final
  calibration and before any test path is drawn.
* Calibration, selection, refinement and testing draw from disjoint seed
  streams.  Test results never feed back into selection.
* Reported test intervals condition on the selected design and its calibrated
  limit.  They do not include search or calibration uncertainty, and the
  paired intervals carry no multiplicity adjustment.
"""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
import platform
import subprocess
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import fastkernel as fk
from .config import (DG_REFERENCE, SEARCH_GRID, SEARCH_PROTOCOL, ChartDesign,
                     GaussianModel)
from .metrics import selection_objective, summarize_run_lengths


def _dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def summarise(run_length, horizon, tau=1, tail_threshold=50):
    """Summary block in the column layout the search tables use."""
    stats = summarize_run_lengths(run_length, horizon, tau=tau,
                                  tail_threshold=tail_threshold)
    p50 = stats["p_signal_50"]
    n = stats["n_survived"]
    return {
        "n_generated": stats["n_generated"], "n_survived": n,
        "prechange_alarm_rate": stats["prechange_alarm_rate"],
        "mean": stats["mean_rl"], "se": stats["se_mean"],
        "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
        # The search tables report horizon-restricted quantiles.
        "median": stats["median_restricted"], "p95_restricted": stats["p95_rl"],
        "p_gt_50": 1.0 - p50,
        "p_gt_50_se": float(np.sqrt(p50 * (1.0 - p50) / max(n, 1))),
        "censor_rate": stats["censor_rate"], "cap": stats["cap"],
    }


def calibrate(paths, design, protocol=SEARCH_PROTOCOL, model=None):
    """Bisect ``h`` to the target restricted in-control mean over cached paths.

    Diagnoses an unreachable target instead of assuming the initial bracket is
    adequate: if ``h = 0`` already overshoots, or doubling fails to reach the
    target, that is reported.
    """
    model = model or GaussianModel()
    horizon = paths[0].shape[1]
    target = protocol["target_arl0"]
    lo, hi = 0.0, 32.0
    trace = []

    def at(h):
        rl, _ = fk.evaluate(paths, design, h, model=model)
        value = float(np.minimum(rl, horizon).mean())
        trace.append({"h": h, "restricted_mean": value,
                      "censor_rate": float((rl > horizon).mean())})
        return value

    if at(lo) > target:
        raise ValueError("ARL0 target below attainable range: %s" % design.key)
    while at(hi) < target:
        hi *= 2.0
        if hi > 1024:
            raise ValueError("failed to bracket the calibration target")
    for _ in range(protocol["bisection_iterations"]):
        mid = 0.5 * (lo + hi)
        if at(mid) < target:
            lo = mid
        else:
            hi = mid
    h = 0.5 * (lo + hi)
    at(h)
    return h, trace


def score(paths, design, h, protocol=SEARCH_PROTOCOL, model=None):
    """Evaluate the selection loss of Eq. (11) for one calibrated design."""
    model = model or GaussianModel()
    horizon = paths[0].shape[1]
    run_lengths, rows = [], []
    for delta in protocol["selection_deltas"]:
        rl, _ = fk.evaluate(paths, design, h, delta=delta, model=model)
        run_lengths.append(rl)
        rows.append({"delta": delta, **summarise(rl, horizon)})
    value, se = selection_objective(run_lengths, horizon,
                                    protocol["tail_threshold"],
                                    protocol["tail_penalty"])
    return value, se, rows


def run_stage(stage, designs, n_cal, n_tune, seeds, protocol, outdir, model=None,
              verbose=True):
    """Calibrate and score a list of designs; write the stage's ranking."""
    model = model or GaussianModel()
    if verbose:
        print("%s: %d designs, calibration N=%d, selection N=%d"
              % (stage, len(designs), n_cal, n_tune), flush=True)
    cal_paths = fk.generate_paths(n_cal, protocol["horizon"], seeds[0], model)
    tune_paths = fk.generate_paths(n_tune, protocol["horizon"], seeds[1], model)
    rows, traces = [], {}
    start = time.perf_counter()
    for i, design in enumerate(designs):
        h, trace = calibrate(cal_paths, design, protocol, model)
        value, se, details = score(tune_paths, design, h, protocol, model)
        rows.append({"key": design.key, **asdict(design), "h": h,
                     "objective": value, "objective_se": se,
                     "cal_mean": trace[-1]["restricted_mean"],
                     "cal_censor": trace[-1]["censor_rate"],
                     **{"mean_delta_%g" % r["delta"]: r["mean"] for r in details}})
        traces[design.key] = trace
        if verbose and ((i + 1) % 20 == 0 or i + 1 == len(designs)):
            print("  %d/%d in %.1fs" % (i + 1, len(designs),
                                        time.perf_counter() - start), flush=True)
    frame = pd.DataFrame(rows).sort_values(["objective", "key"])
    frame.to_csv(outdir / ("%s_search.csv" % stage), index=False)
    _dump_json(outdir / ("%s_calibration_traces.json" % stage), traces)
    del cal_paths, tune_paths
    gc.collect()
    return frame


def _design_from_row(row):
    return ChartDesign(str(row["method"]), float(row["k1"]), float(row["g"]),
                       float(row["k2"]), int(row["L"]))


def run_tests(final_designs, limits, protocol, outdir, model=None, verbose=True):
    """Independent validation of the locked designs.

    Five conditions: in-control, zero-state shifts, a delayed change at
    ``tau = 201``, and the two drift-misspecification cases.  Paired
    comparisons are formed only where they are valid -- same paths and no
    method-specific survival conditioning, i.e. ``tau = 1`` at the design
    drift.  Raw stopping times are archived so a reader can recompute any
    summary without rerunning the study.
    """
    model = model or GaussianModel()
    horizon, seeds = protocol["horizon"], protocol["seeds"]
    raw, rows, pairs = {}, [], []
    conditions = [
        ("in_control", protocol["n_ic_test"], seeds["ic_test"], 1.0, [0.0], 1),
        ("zero_state", protocol["n_ooc_test"], seeds["ooc_test"], 1.0,
         protocol["test_deltas"], 1),
        ("late_change", protocol["n_ooc_test"], seeds["late_test"], 1.0,
         protocol["selection_deltas"], protocol["late_change_tau"]),
        ("drift_half", protocol["n_mismatch"], seeds["mismatch_half"], 0.5, [0.0], 1),
        ("drift_double", protocol["n_mismatch"], seeds["mismatch_double"], 2.0, [0.0], 1),
    ]
    for case, n, seed, scale, deltas, tau in conditions:
        if verbose:
            print("Independent test: %s, N=%d, tau=%d" % (case, n, tau), flush=True)
        paths = fk.generate_paths(n, horizon, seed, model, drift_scale=scale)
        for delta in deltas:
            cache = {}
            for label, design in final_designs.items():
                rl, active = fk.evaluate(paths, design, limits[label],
                                         delta=delta, tau=tau, model=model)
                key = "%s__%s__delta%g" % (case, label, delta)
                raw[key] = rl
                raw[key + "__candidate"] = active
                cache[label] = rl
                survivor = rl >= tau
                rows.append({
                    "case": case, "label": label, "delta": delta, "tau": tau,
                    "seed": seed, "drift_scale": scale, "h": limits[label],
                    **asdict(design), **summarise(rl, horizon, tau),
                    "candidate_at_change_given_survival":
                        float(active[survivor].mean()) if survivor.any() else float("nan"),
                })
            if tau == 1 and scale == 1.0:
                for base in ["DG_reference", "KF_reference", "KF_selected"]:
                    if base not in cache:
                        continue
                    diff = (np.minimum(cache["DG_selected"], horizon)
                            - np.minimum(cache[base], horizon))
                    se = float(diff.std(ddof=1) / np.sqrt(n))
                    pairs.append({"case": case, "delta": delta,
                                  "contrast": "DG_selected - " + base,
                                  "mean_difference": float(diff.mean()), "se": se,
                                  "ci_low": float(diff.mean() - 1.96 * se),
                                  "ci_high": float(diff.mean() + 1.96 * se)})
        # Holdout directions reuse the same data; they are not extra replicates.
        if case == "zero_state":
            for direction in ["sparse", "mixed"]:
                for label, design in final_designs.items():
                    rl, _ = fk.evaluate(paths, design, limits[label], delta=1.0,
                                        tau=1, direction=direction, model=model)
                    raw["direction_%s__%s__delta1" % (direction, label)] = rl
                    rows.append({"case": "direction_" + direction, "label": label,
                                 "delta": 1.0, "tau": 1, "seed": seed,
                                 "drift_scale": 1.0, "h": limits[label],
                                 **asdict(design), **summarise(rl, horizon)})
        del paths
        gc.collect()
        pd.DataFrame(rows).to_csv(outdir / "independent_tests.csv", index=False)
        pd.DataFrame(pairs).to_csv(outdir / "paired_comparisons.csv", index=False)
        np.savez_compressed(outdir / "raw_run_lengths.npz", **raw)
    return pd.DataFrame(rows)


def run_search(outdir, protocol=None, model=None, verbose=True):
    """Execute the complete two-stage search and independent validation."""
    protocol = json.loads(json.dumps(protocol or SEARCH_PROTOCOL))
    model = model or GaussianModel()
    outdir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    try:
        compiler = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
    except Exception:
        compiler = "unavailable"
    metadata = {
        "status": "running", "protocol": protocol, "model": asdict(model),
        "grid": SEARCH_GRID,
        "objective": "mean across delta of E[min(T,H)] + 50 P(T>50)",
        "scope": "known-parameter Gaussian local-linear model; "
                 "no Phase-I or industrial rerun",
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "platform": platform.platform(),
        "compiler": compiler, "kernel": fk.backend_info(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [fk.NATIVE_DIR / "kernel.cpp",
                      fk.NATIVE_DIR.parent / "dgmcusum" / "search.py"]
            if p.exists()},
    }
    _dump_json(outdir / "run_metadata.json", metadata)

    designs = [ChartDesign("dg", *v) for v in itertools.product(*SEARCH_GRID.values())]
    designs += [ChartDesign("kf", k1=k) for k in SEARCH_GRID["k1"]]
    seeds = protocol["seeds"]

    coarse = run_stage("coarse", designs, protocol["n_coarse_cal"],
                       protocol["n_coarse_tune"],
                       [seeds["coarse_cal"], seeds["coarse_tune"]],
                       protocol, outdir, model, verbose)
    promoted = [_design_from_row(r) for _, r in
                coarse[coarse.method == "dg"].head(protocol["promote_dg"]).iterrows()]
    promoted += [DG_REFERENCE] + [ChartDesign("kf", k1=k) for k in SEARCH_GRID["k1"]]
    promoted = list(dict.fromkeys(promoted))
    refined = run_stage("refined", promoted, protocol["n_refine_cal"],
                        protocol["n_refine_tune"],
                        [seeds["refine_cal"], seeds["refine_tune"]],
                        protocol, outdir, model, verbose)

    final_designs = {
        "DG_reference": DG_REFERENCE,
        "DG_selected": _design_from_row(refined[refined.method == "dg"].iloc[0]),
        "KF_reference": ChartDesign("kf"),
        "KF_selected": _design_from_row(refined[refined.method == "kf"].iloc[0]),
    }
    _dump_json(outdir / "locked_designs.json",
               {k: asdict(v) for k, v in final_designs.items()})
    if verbose:
        print("LOCKED:", {k: v.key for k, v in final_designs.items()}, flush=True)
        print("Final calibration, N=%d" % protocol["n_final_cal"], flush=True)

    cal_paths = fk.generate_paths(protocol["n_final_cal"], protocol["horizon"],
                                  seeds["final_cal"], model)
    limits, traces = {}, {}
    for label, design in final_designs.items():
        limits[label], traces[label] = calibrate(cal_paths, design, protocol, model)
        if verbose:
            print("  %s: h=%.6f, cal mean=%.3f"
                  % (label, limits[label], traces[label][-1]["restricted_mean"]),
                  flush=True)
    del cal_paths
    gc.collect()
    _dump_json(outdir / "final_calibration_traces.json", traces)
    _dump_json(outdir / "final_limits.json", limits)

    tests = run_tests(final_designs, limits, protocol, outdir, model, verbose)
    metadata.update({
        "status": "completed",
        "elapsed_seconds": time.perf_counter() - start,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "limits": limits,
        "selected_designs": {k: asdict(v) for k, v in final_designs.items()},
    })
    _dump_json(outdir / "run_metadata.json", metadata)
    if verbose:
        print("COMPLETED in %.1f seconds" % metadata["elapsed_seconds"], flush=True)
    return final_designs, limits, tests
