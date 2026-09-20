"""Control-limit calibration to a common nominal false-alarm target.

Every design in the paper -- including each of the 240 search candidates -- gets
its *own* limit calibrated to the same restricted in-control mean.  This is the
only way the comparisons mean anything: the statistics have different null
distributions, so a shared numerical threshold would not give shared
false-alarm behaviour.  The static-target chart is the one stated exception,
and it is labelled a negative control precisely because its limit is carried
over from the stationary model.

Two facts about this procedure are stated in the manuscript and are visible in
the code below.  The finite-sample response is a monotone *step* function of
``h`` under common random numbers, so bisection returns a bracket rather than
an exact root; and the returned limit need not achieve the target exactly.
"""
from __future__ import annotations

import numpy as np

from .config import TARGET_ARL0
from .metrics import restricted_mean
from .simulate import simulate_run_lengths


def calibrate_limit(method, *, model=None, design=None, target=TARGET_ARL0,
                    n_rep=2000, max_steps=2500, lo=1.0, hi=80.0, seed=8128,
                    iterations=13, q_level_data=None, q_slope_data=None,
                    trace=None):
    """Bisect on ``h`` until the restricted in-control mean reaches ``target``.

    Common random numbers are reused at every trial limit, which makes the
    response monotone in ``h`` and removes most of the Monte Carlo noise from
    the comparison between neighbouring limits.  It does not remove Monte Carlo
    error from the limit itself.

    Parameters
    ----------
    lo, hi : float
        Initial bracket.  The archived study supplies method-specific brackets;
        :func:`calibrate_limit_autobracket` widens automatically instead.
    trace : list, optional
        If given, each trial ``(h, restricted mean, censor rate)`` is appended,
        so the calibration path can be exported for inspection.

    Returns
    -------
    float
        Midpoint of the final bracket.
    """
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        rl = simulate_run_lengths(
            method, mid, n_rep=n_rep, max_steps=max_steps, model=model,
            design=design, seed=seed,
            q_level_data=q_level_data, q_slope_data=q_slope_data)
        arl = restricted_mean(rl, max_steps)
        if trace is not None:
            trace.append({"h": mid, "restricted_mean": arl,
                          "censor_rate": float(np.mean(rl > max_steps))})
        if arl < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_limit_autobracket(method, *, model=None, design=None,
                                target=TARGET_ARL0, n_rep=2000, max_steps=2500,
                                seed=8128, iterations=15, trace=None,
                                lo=0.0, hi=32.0, hi_cap=1024.0):
    """As :func:`calibrate_limit`, but find a valid bracket first.

    Refuses to guess when the target is unreachable: if even ``h = lo`` already
    exceeds the target the design cannot be calibrated, and doubling ``hi``
    past ``hi_cap`` without reaching the target is reported rather than
    silently returning the cap.
    """
    def at(h):
        rl = simulate_run_lengths(method, h, n_rep=n_rep, max_steps=max_steps,
                                  model=model, design=design, seed=seed)
        value = restricted_mean(rl, max_steps)
        if trace is not None:
            trace.append({"h": h, "restricted_mean": value,
                          "censor_rate": float(np.mean(rl > max_steps))})
        return value

    if at(lo) > target:
        raise ValueError("target below the attainable range for this design")
    while at(hi) < target:
        hi *= 2.0
        if hi > hi_cap:
            raise ValueError("failed to bracket the calibration target")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if at(mid) < target:
            lo = mid
        else:
            hi = mid
    h = 0.5 * (lo + hi)
    at(h)
    return h
