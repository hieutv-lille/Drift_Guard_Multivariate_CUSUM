"""Reproducible chart-parameter search; only NumPy, pandas and a C++ compiler.

The C++ kernel evaluates the exact original DG stopping rule. Gaussian paths
are generated in Python using the original simulation's random-draw order.
No Phase-I fitting or industrial-data claims are made by this experiment.
"""
from __future__ import annotations
import argparse
import ctypes
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "optimization_results"
RESULTS.mkdir(exist_ok=True)

@dataclass(frozen=True)
class Model:
    p: int = 4
    qlevel: float = .002
    qslope: float = 2e-6
    burn: int = 150

@dataclass(frozen=True)
class Design:
    method: str = "dg"
    k1: float = .5
    g: float = 1.5
    k2: float = .25
    L: int = 15

    @property
    def key(self):
        if self.method == "kf":
            return f"kf_k{self.k1:g}"
        return f"dg_k{self.k1:g}_g{self.g:g}_c{self.k2:g}_L{self.L}"

MODEL = Model()
GRID = {"k1": [.25, .5, .75, 1.], "g": [.75, 1., 1.5, 2.],
        "k2": [.1, .25, .5], "L": [5, 10, 15, 20, 30]}
# Set before running; final-test results NEVER enter either selection stage.
PROTOCOL = {
    "target_arl0": 500., "horizon": 6000, "tail_threshold": 50,
    "tail_penalty": 50., "selection_deltas": [1., 1.5, 2.],
    "test_deltas": [.5, .75, 1., 1.5, 2.], "late_change_tau": 201,
    "n_coarse_cal": 640, "n_coarse_tune": 1000,
    "n_refine_cal": 4000, "n_refine_tune": 4000,
    "promote_dg": 12, "n_final_cal": 20000,
    "n_ic_test": 20000, "n_ooc_test": 10000, "n_mismatch": 10000,
    "bisection_iterations": 15,
    "seeds": {"coarse_cal": 51001, "coarse_tune": 52001,
              "refine_cal": 61001, "refine_tune": 62001,
              "final_cal": 71001, "ic_test": 81001,
              "ooc_test": 91001, "late_test": 101001,
              "mismatch_half": 111001, "mismatch_double": 121001},
}

def compile_kernel():
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("A Linux C++ compiler is required (g++ is available in standard Colab).")
    target = ROOT / "_kernel.so"
    command = [compiler, "-O3", "-std=c++17", "-fopenmp", "-shared", "-fPIC",
               str(ROOT / "kernel.cpp"), "-o", str(target)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    lib = ctypes.CDLL(str(target))
    f = lib.chart
    ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    iptr = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    f.argtypes = [ptr, ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                  ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                  ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double,
                  ctypes.c_double, ctypes.c_int, iptr, iptr]
    f.restype = None
    return lib, f, command

_LIB, KERNEL, COMPILE_COMMAND = compile_kernel()
THREADS = min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)

def generate_paths(n, horizon, seed, model=MODEL, drift_scale=1.):
    """Unshifted LIVE innovations, exactly following the source simulation.

    A shift changes innovations by a deterministic linear-filter response.
    Caching these innovations is valid because the live filter never freezes.
    Arrays are float64; no steady-state or independent-innovation shortcut.
    """
    rng = np.random.default_rng(seed)
    level = np.zeros((n, model.p))
    slope = np.zeros_like(level)
    fl = np.zeros_like(level)
    fb = np.zeros_like(level)
    p00, p01, p11 = 10., 0., 1.
    innovations = np.empty((horizon, n, model.p))
    covariance = np.empty((horizon, 6))
    for t in range(-model.burn, horizon):
        level = level + slope + math.sqrt(model.qlevel * drift_scale) * rng.normal(size=level.shape)
        slope = slope + math.sqrt(model.qslope * drift_scale) * rng.normal(size=slope.shape)
        y = level + rng.normal(size=level.shape)
        pred = fl + fb
        pm00 = p00 + 2*p01 + p11 + model.qlevel
        pm01 = p01 + p11
        pm11 = p11 + model.qslope
        e = y - pred
        f = pm00 + 1.
        k0, k1 = pm00/f, pm01/f
        fl = pred + k0*e
        fb = fb + k1*e
        p00, p01, p11 = (1-k0)*pm00, (1-k0)*pm01, pm11-k1*pm01
        if t >= 0:
            innovations[t] = e
            covariance[t] = [pm00, pm01, pm11, k0, k1, f]
    return np.ascontiguousarray(innovations.transpose(1, 0, 2)), covariance

def shift_response(covariance, delta, tau):
    """Raw mean-shift response of live innovations, not a constant residual shift."""
    out = np.zeros(len(covariance))
    fl, fb = 0., 0.
    for t in range(tau-1, len(covariance)):
        pred = fl + fb
        out[t] = delta - pred
        fl = pred + covariance[t, 3]*out[t]
        fb += covariance[t, 4]*out[t]
    return out

def evaluate(paths, design, h, delta=0., tau=1, direction="dense", model=MODEL):
    e, c = paths
    n, horizon, p = e.shape
    if not (1 <= p <= 32 and 1 <= tau <= horizon):
        raise ValueError("Require 1<=p<=32 and 1<=tau<=horizon")
    d = np.ones(p)/math.sqrt(p)
    if direction == "sparse":
        d[:] = 0.; d[0] = 1.
    elif direction == "mixed":
        d[1::2] *= -1.
    elif direction != "dense":
        raise ValueError("Unknown direction")
    response = shift_response(c, delta, tau)
    rl, active = np.empty(n, np.int32), np.empty(n, np.int32)
    KERNEL(e, c, response, d, n, horizon, p, int(design.method == "dg"),
           design.k1, design.g, design.k2, design.L, h, tau,
           model.qlevel, model.qslope, THREADS, rl, active)
    return rl, active

def summary(rl, horizon, tau=1):
    # T>=tau conditions on no pre-change alarm; D=T-tau+1. Censoring retained.
    keep = rl >= tau
    delay = rl[keep] - tau + 1
    cap = horizon - tau + 1
    x = np.minimum(delay, cap)
    n = len(x)
    mean = float(x.mean())
    se = float(x.std(ddof=1)/math.sqrt(n))
    censored = delay > cap
    tail = delay > PROTOCOL["tail_threshold"]
    p50 = float(tail.mean())
    return {"n_generated": len(rl), "n_survived": n,
            "prechange_alarm_rate": float(1-keep.mean()),
            "mean": mean, "se": se, "ci_low": mean-1.96*se,
            "ci_high": mean+1.96*se, "median": float(np.median(x)),
            "p95_restricted": float(np.quantile(x, .95)), "p_gt_50": p50,
            "p_gt_50_se": math.sqrt(p50*(1-p50)/n),
            "censor_rate": float(censored.mean()), "cap": cap}

def calibrate(paths, design, protocol=PROTOCOL):
    horizon = paths[0].shape[1]
    lo, hi = 0., 32.
    target = protocol["target_arl0"]
    trace = []
    # Diagnose unreachable targets; do not assume an adequate bracket.
    def at(h):
        rl, _ = evaluate(paths, design, h)
        result = float(np.minimum(rl, horizon).mean())
        trace.append({"h": h, "restricted_mean": result,
                      "censor_rate": float((rl>horizon).mean())})
        return result
    if at(lo) > target:
        raise ValueError(f"ARL0 target below attainable range: {design.key}")
    while at(hi) < target:
        hi *= 2.
        if hi > 1024:
            raise ValueError("Failed to bracket the calibration target")
    for _ in range(protocol["bisection_iterations"]):
        mid = .5*(lo+hi)
        if at(mid) < target:
            lo = mid
        else:
            hi = mid
    h = .5*(lo+hi)
    at(h)
    return h, trace

def tune(paths, design, h, protocol=PROTOCOL):
    rows, combined = [], []
    for delta in protocol["selection_deltas"]:
        rl, _ = evaluate(paths, design, h, delta=delta)
        score_path = np.minimum(rl, paths[0].shape[1]) + protocol["tail_penalty"]*(rl>protocol["tail_threshold"])
        combined.append(score_path)
        rows.append({"delta": delta, **summary(rl, paths[0].shape[1])})
    # Same underlying noise across scenarios: compute SE on per-path averages.
    z = np.mean(combined, axis=0)
    return float(z.mean()), float(z.std(ddof=1)/math.sqrt(len(z))), rows

def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")

def save_trace(stage, traces):
    dump_json(RESULTS / f"{stage}_calibration_traces.json", traces)

def run_search(stage, designs, ncal, ntune, seeds, protocol):
    print(f"{stage}: {len(designs)} designs, calibration N={ncal}, selection N={ntune}", flush=True)
    calpaths = generate_paths(ncal, protocol["horizon"], seeds[0])
    tunepaths = generate_paths(ntune, protocol["horizon"], seeds[1])
    rows, traces = [], {}
    start = time.perf_counter()
    for i, design in enumerate(designs):
        h, trace = calibrate(calpaths, design, protocol)
        score, se, details = tune(tunepaths, design, h, protocol)
        rows.append({"key": design.key, **asdict(design), "h": h,
                     "objective": score, "objective_se": se,
                     "cal_mean": trace[-1]["restricted_mean"],
                     "cal_censor": trace[-1]["censor_rate"],
                     **{f"mean_delta_{r['delta']:g}": r["mean"] for r in details}})
        traces[design.key] = trace
        if (i+1)%20 == 0 or i+1==len(designs):
            print(f"  {i+1}/{len(designs)} completed in {time.perf_counter()-start:.1f}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["objective", "key"])
    df.to_csv(RESULTS / f"{stage}_search.csv", index=False)
    save_trace(stage, traces)
    del calpaths, tunepaths
    gc.collect()
    return df

def design_from_row(row):
    return Design(str(row["method"]), float(row["k1"]), float(row["g"]),
                  float(row["k2"]), int(row["L"]))

def test_suite(final_designs, limits, protocol):
    raw, rows, pairs = {}, [], []
    H, seeds = protocol["horizon"], protocol["seeds"]
    conditions = [("in_control", protocol["n_ic_test"], seeds["ic_test"], 1., [0.], 1),
                  ("zero_state", protocol["n_ooc_test"], seeds["ooc_test"], 1., protocol["test_deltas"], 1),
                  ("late_change", protocol["n_ooc_test"], seeds["late_test"], 1., protocol["selection_deltas"], protocol["late_change_tau"]),
                  ("drift_half", protocol["n_mismatch"], seeds["mismatch_half"], .5, [0.], 1),
                  ("drift_double", protocol["n_mismatch"], seeds["mismatch_double"], 2., [0.], 1)]
    for case, n, seed, scale, deltas, tau in conditions:
        print(f"Independent test: {case}, N={n}, tau={tau}", flush=True)
        paths = generate_paths(n, H, seed, drift_scale=scale)
        for delta in deltas:
            rl_cache = {}
            for label, design in final_designs.items():
                rl, active = evaluate(paths, design, limits[label], delta, tau)
                key = f"{case}__{label}__delta{delta:g}"
                raw[key] = rl
                raw[key+"__candidate"] = active
                rl_cache[label] = rl
                stats = summary(rl, H, tau)
                survivor = rl >= tau
                rows.append({"case": case, "label": label, "delta": delta,
                             "tau": tau, "seed": seed, "drift_scale": scale,
                             "h": limits[label], **asdict(design), **stats,
                             "candidate_at_change_given_survival": float(active[survivor].mean())})
            # Paired comparison valid for zero-state / in-control: same paths,
            # no method-specific pre-change survival conditioning.
            if tau == 1 and scale == 1.:
                for base in ["DG_original", "KF_original", "KF_selected"]:
                    diff = np.minimum(rl_cache["DG_selected"], H) - np.minimum(rl_cache[base], H)
                    se = float(diff.std(ddof=1)/math.sqrt(n))
                    pairs.append({"case": case, "delta": delta, "contrast": "DG_selected - "+base,
                                  "mean_difference": float(diff.mean()), "se": se,
                                  "ci_low": float(diff.mean()-1.96*se),
                                  "ci_high": float(diff.mean()+1.96*se)})
        # Holdout directions use same data, not additional independent samples.
        if case == "zero_state":
            for direction in ["sparse", "mixed"]:
                for label, design in final_designs.items():
                    rl, active = evaluate(paths, design, limits[label], 1., 1, direction)
                    raw[f"direction_{direction}__{label}__delta1"] = rl
                    rows.append({"case": f"direction_{direction}", "label": label,
                                 "delta": 1., "tau": 1, "seed": seed, "drift_scale": 1.,
                                 "h": limits[label], **asdict(design), **summary(rl, H)})
        del paths
        gc.collect()
        pd.DataFrame(rows).to_csv(RESULTS / "independent_tests.csv", index=False)
        pd.DataFrame(pairs).to_csv(RESULTS / "paired_comparisons.csv", index=False)
        np.savez_compressed(RESULTS / "raw_run_lengths.npz", **raw)
    return pd.DataFrame(rows)

def run(protocol=None):
    protocol = json.loads(json.dumps(protocol or PROTOCOL))
    start = time.perf_counter()
    metadata = {"status": "running", "protocol": protocol, "model": asdict(MODEL),
                "grid": GRID, "objective": "mean across delta of E[min(T,H)] + 50 P(T>50)",
                "scope": "known-parameter Gaussian local-linear model; no Phase-I or industrial rerun",
                "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "platform": platform.platform(),
                "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
                "threads": THREADS, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [ROOT/"optimize.py", ROOT/"kernel.cpp"]}}
    dump_json(RESULTS / "run_metadata.json", metadata)
    designs = [Design("dg", *v) for v in itertools.product(*GRID.values())]
    designs += [Design("kf", k1=k) for k in GRID["k1"]]
    seeds = protocol["seeds"]
    coarse = run_search("coarse", designs, protocol["n_coarse_cal"], protocol["n_coarse_tune"],
                        [seeds["coarse_cal"], seeds["coarse_tune"]], protocol)
    promoted = [design_from_row(r) for _, r in coarse[coarse.method=="dg"].head(protocol["promote_dg"]).iterrows()]
    # Always retain original DG and all KF references for a fair final comparison.
    promoted += [Design()] + [Design("kf", k1=k) for k in GRID["k1"]]
    promoted = list(dict.fromkeys(promoted))
    refined = run_search("refined", promoted, protocol["n_refine_cal"], protocol["n_refine_tune"],
                         [seeds["refine_cal"], seeds["refine_tune"]], protocol)
    final_designs = {"DG_original": Design(),
                     "DG_selected": design_from_row(refined[refined.method=="dg"].iloc[0]),
                     "KF_original": Design("kf"),
                     "KF_selected": design_from_row(refined[refined.method=="kf"].iloc[0])}
    # Lock theta before final calibration and every independent test.
    dump_json(RESULTS / "locked_designs.json", {k: asdict(v) for k,v in final_designs.items()})
    print("LOCKED DESIGNS:", {k:v.key for k,v in final_designs.items()}, flush=True)
    print(f"Final calibration, N={protocol['n_final_cal']}", flush=True)
    calpaths = generate_paths(protocol["n_final_cal"], protocol["horizon"], seeds["final_cal"])
    limits, traces = {}, {}
    for label, design in final_designs.items():
        limits[label], traces[label] = calibrate(calpaths, design, protocol)
        print(f"  {label}: h={limits[label]:.6f}, cal mean={traces[label][-1]['restricted_mean']:.3f}", flush=True)
    del calpaths
    gc.collect()
    save_trace("final", traces)
    dump_json(RESULTS / "final_limits.json", limits)
    tests = test_suite(final_designs, limits, protocol)
    metadata["status"] = "completed"
    metadata["elapsed_seconds"] = time.perf_counter()-start
    metadata["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata["limits"] = limits
    metadata["selected_designs"] = {k:asdict(v) for k,v in final_designs.items()}
    dump_json(RESULTS / "run_metadata.json", metadata)
    print(tests[tests.case.isin(["in_control", "zero_state", "late_change"])][
        ["case", "label", "delta", "mean", "ci_low", "ci_high", "p_gt_50", "censor_rate"]].to_string(index=False), flush=True)
    print(f"COMPLETED in {metadata['elapsed_seconds']:.1f} seconds", flush=True)
    return tests

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Smoke test only; not publication evidence")
    args = parser.parse_args()
    cfg = json.loads(json.dumps(PROTOCOL))
    if args.quick:
        for key in list(cfg):
            if key.startswith("n_"):
                cfg[key] = 64
        cfg["promote_dg"] = 2
        cfg["bisection_iterations"] = 8
    run(cfg)
