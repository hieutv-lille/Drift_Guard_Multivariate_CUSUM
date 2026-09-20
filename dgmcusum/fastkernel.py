"""Fast evaluator for the exact DG stopping rule, used by the 240-design search.

Why a separate evaluator at all
-------------------------------
The search calibrates a limit for every one of the 240 candidates, and each
calibration is itself a bisection.  Re-simulating the Gaussian process from
scratch for each trial limit would dominate the runtime, so the search caches
the *live* innovations once and replays the chart over them.

This is not an approximation.  Two facts make the caching exact:

1. The live filter never freezes, and its gain sequence does not depend on any
   chart parameter.  So the live innovations of an unshifted path are the same
   for every candidate design.
2. A persistent shift changes the live innovations by a deterministic linear
   filter response -- :func:`shift_response` -- that is added on top.  No
   randomness is shared incorrectly.

The protected predictor is then evaluated through the *live-minus-protected*
state difference ``b``, which is algebraically identical to propagating the
protected state directly, but avoids storing a second full filter.

Backends
--------
``cpp``    compiles ``native/kernel.cpp`` with OpenMP; the reference backend.
``numpy``  a vectorised fallback with identical semantics, for reviewers with
           no C++ compiler.  Slower, but it reproduces the same run lengths.

:func:`self_test` checks the two backends against each other, and
``tests/test_kernel.py`` checks both against :mod:`dgmcusum.simulate`.
"""
from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .charts import ZERO_TOLERANCE, shift_direction
from .config import GaussianModel

NATIVE_DIR = Path(__file__).resolve().parent.parent / "native"
_CPP_STATE = {"fn": None, "tried": False, "error": None, "command": None}


def threads():
    """Worker count, capped at 8 as in the archived run."""
    if hasattr(os, "sched_getaffinity"):
        available = len(os.sched_getaffinity(0))
    else:
        available = os.cpu_count() or 1
    return min(8, available)


def compile_kernel(force=False):
    """Compile and bind ``native/kernel.cpp``; return ``None`` if unavailable.

    A missing compiler is not an error here: the caller falls back to NumPy.
    The compile command is recorded in run metadata so the binary that produced
    a result can be identified later.
    """
    if _CPP_STATE["tried"] and not force:
        return _CPP_STATE["fn"]
    _CPP_STATE["tried"] = True
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        _CPP_STATE["error"] = "no g++ or clang++ on PATH"
        return None
    target = NATIVE_DIR / ("_kernel" + (".dll" if os.name == "nt" else ".so"))
    command = [compiler, "-O3", "-std=c++17", "-fopenmp", "-shared", "-fPIC",
               "-static-libgcc", "-static-libstdc++",
               str(NATIVE_DIR / "kernel.cpp"), "-o", str(target)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        # On Windows the built DLL still needs the compiler's OpenMP runtime
        # (libgomp-1.dll), which is not on the default search path.
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            bindir = os.path.dirname(compiler)
            if os.path.isdir(bindir):
                try:
                    _CPP_STATE["dll_dir"] = os.add_dll_directory(bindir)
                except OSError:
                    pass
        lib = ctypes.CDLL(str(target))
    except Exception as exc:                      # pragma: no cover - env dependent
        _CPP_STATE["error"] = str(exc)
        return None
    fn = lib.chart
    ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    iptr = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    fn.argtypes = [ptr, ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                   ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                   ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double,
                   ctypes.c_double, ctypes.c_int, iptr, iptr]
    fn.restype = None
    _CPP_STATE["fn"] = fn
    _CPP_STATE["command"] = command
    _CPP_STATE["lib"] = lib
    return fn


def backend_info():
    """Describe the active backend, for run metadata."""
    fn = compile_kernel()
    return {"backend": "cpp" if fn is not None else "numpy",
            "compile_command": _CPP_STATE["command"],
            "compile_error": _CPP_STATE["error"],
            "threads": threads()}


def generate_paths(n, horizon, seed, model=None, drift_scale=1.0):
    """Cache unshifted live innovations and the filter's covariance trace.

    Follows the same random-draw order as :mod:`dgmcusum.simulate`: state
    transition first (level then slope), then the measurement noise.

    Returns
    -------
    innovation : (n, horizon, p) float64 array
        Raw live innovations ``e_t`` (not yet whitened).
    covariance : (horizon, 6) float64 array
        Columns ``[pm00, pm01, pm11, gain_level, gain_slope, f]``.  Shared by
        every path because the covariance recursion is deterministic.
    """
    model = model or GaussianModel()
    rng = np.random.default_rng(seed)
    p = model.p
    level = np.zeros((n, p))
    slope = np.zeros((n, p))
    fl = np.zeros((n, p))
    fb = np.zeros((n, p))
    c00, c01, c11 = model.c00_init, model.c01_init, model.c11_init
    innovation = np.empty((horizon, n, p))
    covariance = np.empty((horizon, 6))
    for t in range(-model.burn_in, horizon):
        level = level + slope + math.sqrt(model.q_level * drift_scale) * rng.normal(size=level.shape)
        slope = slope + math.sqrt(model.q_slope * drift_scale) * rng.normal(size=slope.shape)
        y = level + rng.normal(size=level.shape)
        pred = fl + fb
        pm00 = c00 + 2 * c01 + c11 + model.q_level
        pm01 = c01 + c11
        pm11 = c11 + model.q_slope
        e = y - pred
        f = pm00 + 1.0
        k0, k1 = pm00 / f, pm01 / f
        fl = pred + k0 * e
        fb = fb + k1 * e
        c00, c01, c11 = (1 - k0) * pm00, (1 - k0) * pm01, pm11 - k1 * pm01
        if t >= 0:
            innovation[t] = e
            covariance[t] = [pm00, pm01, pm11, k0, k1, f]
    return (np.ascontiguousarray(innovation.transpose(1, 0, 2)),
            np.ascontiguousarray(covariance))


def shift_response(covariance, delta, tau):
    """Deterministic mean response of the live innovations to a step at ``tau``.

    This is the scalar sequence behind Proposition 2: the live innovation mean
    decays as the filter absorbs the shift, rather than staying at ``delta``.
    """
    out = np.zeros(len(covariance))
    fl = fb = 0.0
    for t in range(tau - 1, len(covariance)):
        pred = fl + fb
        out[t] = delta - pred
        fl = pred + covariance[t, 3] * out[t]
        fb += covariance[t, 4] * out[t]
    return out


def _evaluate_numpy(innovation, covariance, response, direction, design, h,
                    tau, model):
    """Vectorised fallback with the same semantics as ``native/kernel.cpp``."""
    n, horizon, p = innovation.shape
    is_dg = design.method == "dg"
    s = np.zeros((n, p))
    r = np.zeros((n, p))
    bl = np.zeros((n, p))
    bs = np.zeros((n, p))
    ap00 = np.zeros(n)
    ap01 = np.zeros(n)
    ap11 = np.zeros(n)
    age = np.zeros(n, dtype=np.int64)
    confirming = np.zeros(n, dtype=bool)
    alive = np.ones(n, dtype=bool)
    rl = np.full(n, horizon + 1, dtype=np.int32)
    candidate_at_change = np.full(n, -1, dtype=np.int32)

    for t in range(horizon):
        if t == tau - 1:
            candidate_at_change = np.where(alive, confirming.astype(np.int32),
                                           candidate_at_change)
        e = innovation[:, t, :] + response[t] * direction
        f_sqrt = math.sqrt(covariance[t, 5])

        # ---- primary chart: only when no episode is open ----
        primary = alive & (~confirming if is_dg else np.ones(n, dtype=bool))
        if primary.any():
            v = s[primary] + e[primary] / f_sqrt
            norm = np.linalg.norm(v, axis=1)
            factor = np.maximum(0.0, 1.0 - design.k1 / np.maximum(norm, 1e-12))
            s[primary] = v * factor[:, None]
            snorm = norm * factor
            if not is_dg:
                hit = np.zeros(n, dtype=bool)
                hit[primary] = snorm > h
                rl[hit] = t + 1
                alive[hit] = False
                if not alive.any():
                    break
                continue
            opened = np.zeros(n, dtype=bool)
            opened[primary] = snorm > design.g
            if opened.any():
                confirming[opened] = True
                age[opened] = 0
                ap00[opened] = covariance[t, 0]
                ap01[opened] = covariance[t, 1]
                ap11[opened] = covariance[t, 2]
                r[opened] = 0.0
                bl[opened] = 0.0
                bs[opened] = 0.0
        elif not is_dg:
            continue

        active = alive & confirming
        if not active.any():
            continue

        # ---- protected covariance propagation for episodes already open ----
        older = active & (age > 0)
        if older.any():
            ap00[older] = ap00[older] + 2 * ap01[older] + ap11[older] + model.q_level
            ap01[older] = ap01[older] + ap11[older]
            ap11[older] = ap11[older] + model.q_slope

        v = r[active] + (e[active] + bl[active]) / np.sqrt(ap00[active] + 1.0)[:, None]
        norm = np.linalg.norm(v, axis=1)
        factor = np.maximum(0.0, 1.0 - design.k2 / np.maximum(norm, 1e-12))
        r[active] = v * factor[:, None]
        rnorm = norm * factor
        age[active] += 1

        # Confirmation precedes rejection, including on the last observation.
        hit = np.zeros(n, dtype=bool)
        hit[active] = rnorm > h
        rl[hit] = t + 1
        alive[hit] = False

        still = alive & confirming
        reject = np.zeros(n, dtype=bool)
        idx = np.where(active)[0]
        keep_mask = alive[idx]
        reject_vals = ((rnorm <= ZERO_TOLERANCE) & (age[idx] >= 2)) | (age[idx] >= design.L)
        reject[idx] = keep_mask & reject_vals
        if reject.any():
            confirming[reject] = False
            age[reject] = 0
            s[reject] = 0.0
            r[reject] = 0.0

        # ---- otherwise advance the live-minus-protected level difference ----
        advance = still & (~reject)
        if advance.any():
            ej = e[advance]
            bl[advance] += bs[advance] + (covariance[t, 3] + covariance[t, 4]) * ej
            bs[advance] += covariance[t, 4] * ej

        if not alive.any():
            break
    return rl, candidate_at_change


def evaluate(paths, design, h, delta=0.0, tau=1, direction="dense", model=None,
             backend="auto"):
    """Run one design at one limit over cached paths.

    Returns ``(run_length, candidate_at_change)``, where the second array
    records whether protection was already active immediately before ``tau``
    (``-1`` for paths that had already alarmed).  That statistic is what backs
    the paper's remark that protection is active on 97.4% of surviving
    delayed-change paths.
    """
    model = model or GaussianModel()
    innovation, covariance = paths
    n, horizon, p = innovation.shape
    if not 1 <= p <= 32:
        raise ValueError("kernel supports 1 <= p <= 32")
    if not 1 <= tau <= horizon:
        raise ValueError("require 1 <= tau <= horizon")
    d = shift_direction(p, direction)
    response = shift_response(covariance, delta, tau)

    fn = compile_kernel() if backend in ("auto", "cpp") else None
    if backend == "cpp" and fn is None:
        raise RuntimeError("C++ backend requested but unavailable: %s"
                           % _CPP_STATE["error"])
    if fn is None:
        return _evaluate_numpy(innovation, covariance, response, d, design, h,
                               tau, model)
    rl = np.empty(n, np.int32)
    active = np.empty(n, np.int32)
    fn(innovation, covariance, np.ascontiguousarray(response),
       np.ascontiguousarray(d), n, horizon, p, int(design.method == "dg"),
       design.k1, design.g, design.k2, design.L, h, tau,
       model.q_level, model.q_slope, threads(), rl, active)
    return rl, active


def self_test(n=48, horizon=400, seed=4242):
    """Check that the two backends agree; returns the mismatch count."""
    from .config import ChartDesign
    if compile_kernel() is None:
        return None
    paths = generate_paths(n, horizon, seed)
    mismatches = 0
    for design in (ChartDesign("dg", 0.5, 1.5, 0.25, 15),
                   ChartDesign("dg", 0.25, 1.0, 0.1, 20),
                   ChartDesign("kf", 0.5)):
        for delta, h in ((0.0, 12.0), (1.0, 16.0), (2.0, 9.0)):
            a = evaluate(paths, design, h, delta=delta, backend="cpp")[0]
            b = _evaluate_numpy(paths[0], paths[1],
                                shift_response(paths[1], delta, 1),
                                shift_direction(paths[0].shape[2]),
                                design, h, 1, GaussianModel())[0]
            mismatches += int((a != b).sum())
    return mismatches
