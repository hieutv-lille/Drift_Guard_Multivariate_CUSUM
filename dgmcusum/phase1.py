"""Layer 2: finite Phase-I estimation and nested calibration (Section 3.3).

REIMPLEMENTATION NOTICE
-----------------------
The script that produced the published Phase-I tables was not archived with
this project.  This module follows the manuscript's specification -- the
profile likelihood of Eqs. (12)-(14) and Appendix A, the diffuse prefix, the
repeated-sampling design and the nested calibration -- but its random streams
are **not** the ones behind the printed numbers.  Expect the same qualitative
findings (plug-in limits are anti-conservative; nested calibration restores
coverage; precision improves with ``m``) at Monte Carlo distance from the
exact figures.  See ``PROVENANCE.md``.

What this layer answers
-----------------------
Plugging estimates into a known-parameter control limit ignores uncertainty in
the terminal state and in ``(q_level, q_slope, Sigma)``.  The experiment asks
how much that matters.  Each *outer* replicate fits a model to a fresh
reference sample; each fit then generates several Phase-II futures.  Only the
outer replicates are independent, so all standard errors are computed on the
outer-sample cluster means -- treating the futures as independent would
understate the uncertainty by roughly a factor of ``sqrt(n_futures)``.

This is an oracle repeated-sampling experiment.  It measures estimation
effects; it is not a deployment recipe for a single observed archive, and it
does not guarantee the conditional run length of any particular fit.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .calibrate import calibrate_limit
from .config import (DG_REFERENCE, KF_REFERENCE, METHOD_LABELS, GaussianModel,
                     PHASE1_SEEDS, Phase1Config, TARGET_ARL0)
from .metrics import summarize_run_lengths
from .simulate import simulate_run_lengths


# --------------------------------------------------------------------------
# Reference-sample generation
# --------------------------------------------------------------------------


def generate_reference_sample(m, model, rng):
    """Draw ``m`` in-control observations from the local-linear model.

    Returned in whitened coordinates (``Sigma = I``), which is how the
    known-parameter layer represents data.  The estimator below does not know
    that, and estimates a full ``Sigma``.
    """
    p = model.p
    level = np.zeros(p)
    slope = np.zeros(p)
    out = np.empty((m, p))
    for t in range(m):
        level = level + slope + math.sqrt(model.q_level) * rng.normal(size=p)
        slope = slope + math.sqrt(model.q_slope) * rng.normal(size=p)
        out[t] = level + rng.normal(size=p)
    return out


# --------------------------------------------------------------------------
# Profile likelihood (Eqs. 12-14, derived in Appendix A)
# --------------------------------------------------------------------------


def _prediction_errors(data, q_level, q_slope, cfg):
    """Scalar Kalman pass returning prediction errors and factors ``f_t``.

    The separable covariance ``P^- = C^- (x) Sigma`` makes the gains
    independent of ``Sigma``, which is exactly the property that allows
    ``Sigma`` to be profiled out.  So this pass never touches ``Sigma``.
    """
    m, p = data.shape
    level = np.zeros(p)
    slope = np.zeros(p)
    c00, c01, c11 = cfg.c00_diffuse, 0.0, cfg.c11_diffuse
    errors = np.empty((m, p))
    factors = np.empty(m)
    for t in range(m):
        pred_level = level + slope
        pm00 = c00 + 2.0 * c01 + c11 + q_level
        pm01 = c01 + c11
        pm11 = c11 + q_slope
        e = data[t] - pred_level
        f = pm00 + 1.0
        errors[t] = e
        factors[t] = f
        gl, gs = pm00 / f, pm01 / f
        level = pred_level + gl * e
        slope = slope + gs * e
        c00, c01, c11 = (1.0 - gl) * pm00, (1.0 - gl) * pm01, pm11 - gs * pm01
    d = cfg.diffuse_prefix
    return errors[d:], factors[d:], level, slope, (c00, c01, c11)


def _profile_negloglik(log10_q, data, cfg):
    """Negative profile log-likelihood ``-l_P(q)`` of Eq. (41)."""
    q_level = 10.0 ** log10_q[0]
    q_slope = 10.0 ** log10_q[1]
    errors, factors, _, _, _ = _prediction_errors(data, q_level, q_slope, cfg)
    n, p = errors.shape
    scatter = (errors / factors[:, None]).T @ errors / n
    scatter = 0.5 * (scatter + scatter.T)
    try:
        chol = np.linalg.cholesky(scatter)
    except np.linalg.LinAlgError:
        # A ridge is added only when the factorisation actually requires it.
        ridge = 1e-10 * float(np.trace(scatter)) / max(p, 1)
        try:
            chol = np.linalg.cholesky(scatter + ridge * np.eye(p))
        except np.linalg.LinAlgError:
            return 1e12
    logdet = 2.0 * float(np.log(np.diag(chol)).sum())
    # l_P = -np/2 (log 2pi + 1) - p/2 sum log f_t - n/2 log|Sigma_hat|
    value = (-0.5 * n * p * (math.log(2.0 * math.pi) + 1.0)
             - 0.5 * p * float(np.log(factors).sum())
             - 0.5 * n * logdet)
    if not np.isfinite(value):
        return 1e12
    return -value


def fit_phase1(data, cfg=None):
    """Fit ``(q_level, q_slope, Sigma)`` and the terminal state by profile ML.

    Optimisation runs in ``log10`` coordinates, which covers strictly positive
    drift ratios and keeps the two dimensionless ratios scale-equivariant.  A
    zero-variance boundary model is a different model and would have to be
    fitted separately.

    Returns a dict with the estimates, the terminal state handed to Phase II,
    optimiser status and boundary flags.
    """
    cfg = cfg or Phase1Config()
    best = None
    for start in cfg.log10_starts:
        try:
            res = minimize(_profile_negloglik, np.array(start, dtype=float),
                           args=(data, cfg), method="Nelder-Mead",
                           options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-4})
        except Exception:
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        raise RuntimeError("Phase-I optimisation failed from every start")

    lo, hi = cfg.log10_bounds
    log_q = np.clip(best.x, [lo[0], hi[0]], [lo[1], hi[1]])
    q_level = 10.0 ** log_q[0]
    q_slope = 10.0 ** log_q[1]
    errors, factors, level, slope, cfac = _prediction_errors(data, q_level, q_slope, cfg)
    n = len(errors)
    sigma = (errors / factors[:, None]).T @ errors / n
    sigma = 0.5 * (sigma + sigma.T)
    return {
        "q_level": float(q_level),
        "q_slope": float(q_slope),
        "sigma": sigma,
        "terminal_level": level,
        "terminal_slope": slope,
        "terminal_c": cfac,
        "n_retained": int(n),
        "success": bool(best.success),
        "q_level_boundary": bool(abs(log_q[0] - lo[0]) < cfg.boundary_tol
                                 or abs(log_q[0] - lo[1]) < cfg.boundary_tol),
        "q_slope_boundary": bool(abs(log_q[1] - hi[0]) < cfg.boundary_tol
                                 or abs(log_q[1] - hi[1]) < cfg.boundary_tol),
    }


def relative_covariance_error(sigma_hat, sigma_true=None):
    """Relative Frobenius error ``||Sigma_hat - Sigma|| / ||Sigma||``.

    The truth is the identity, because the known-parameter layer works in
    whitened coordinates.
    """
    p = sigma_hat.shape[0]
    sigma_true = np.eye(p) if sigma_true is None else sigma_true
    return float(np.linalg.norm(sigma_hat - sigma_true, "fro")
                 / np.linalg.norm(sigma_true, "fro"))


# --------------------------------------------------------------------------
# Repeated-sampling experiment
# --------------------------------------------------------------------------


def estimate_across_samples(m, n_samples, seed, model=None, cfg=None):
    """Fit ``n_samples`` independent reference records of length ``m``."""
    model = model or GaussianModel()
    cfg = cfg or Phase1Config()
    rng = np.random.default_rng(seed)
    fits, rows = [], []
    for r in range(n_samples):
        data = generate_reference_sample(m, model, rng)
        fit = fit_phase1(data, cfg)
        fits.append(fit)
        rows.append({
            "m": m, "replicate": r,
            "q_level_hat": fit["q_level"], "q_slope_hat": fit["q_slope"],
            "sigma_rel_error": relative_covariance_error(fit["sigma"]),
            "optimizer_success": fit["success"],
            "q_level_boundary": fit["q_level_boundary"],
            "q_slope_boundary": fit["q_slope_boundary"],
        })
    return fits, pd.DataFrame(rows)


def summarise_estimates(frame):
    """Medians, interquartile ranges, boundary-hit rates by ``m``.

    Optimiser termination alone gives a misleading picture of precision, which
    is why the boundary-hit rate is reported next to it.
    """
    out = []
    for m, group in frame.groupby("m"):
        out.append({
            "m": int(m),
            "q_level_median": float(group.q_level_hat.median()),
            "q_level_q1": float(group.q_level_hat.quantile(0.25)),
            "q_level_q3": float(group.q_level_hat.quantile(0.75)),
            "q_slope_median": float(group.q_slope_hat.median()),
            "q_slope_q1": float(group.q_slope_hat.quantile(0.25)),
            "q_slope_q3": float(group.q_slope_hat.quantile(0.75)),
            "sigma_error_median": float(group.sigma_rel_error.median()),
            "optimizer_success": float(group.optimizer_success.mean()),
            "q_level_boundary_rate": float(group.q_level_boundary.mean()),
            "q_slope_boundary_rate": float(group.q_slope_boundary.mean()),
        })
    return pd.DataFrame(out)


def _fitted_model(fit, template):
    """Monitoring model built from one fit (the chart believes the estimates)."""
    return GaussianModel(p=template.p, q_level=fit["q_level"],
                         q_slope=fit["q_slope"], burn_in=template.burn_in,
                         c00_init=template.c00_init, c01_init=template.c01_init,
                         c11_init=template.c11_init)


def _cluster_run_lengths(method, h, fits, design, template, n_futures,
                         max_steps, base_seed, delta=0.0):
    """One cluster of futures per fit; returns per-fit mean and pooled array."""
    cluster_means, pooled = [], []
    for i, fit in enumerate(fits):
        rl = simulate_run_lengths(
            method, h, n_rep=n_futures, max_steps=max_steps,
            model=_fitted_model(fit, template), design=design, delta=delta,
            seed=base_seed + 1000 * i)
        cluster_means.append(float(np.minimum(rl, max_steps).mean()))
        pooled.append(rl)
    return np.array(cluster_means), np.concatenate(pooled)


def calibrate_nested(method, fits, design, template, cfg, target=TARGET_ARL0,
                     base_seed=0):
    """Choose ``h`` so the *clustered* in-control mean hits the target.

    Bisection on the mean of the outer-sample cluster means, so the limit is
    chosen against the same quantity the validation reports.
    """
    lo, hi = 1.0, 80.0
    for _ in range(cfg.bisection_iterations):
        mid = 0.5 * (lo + hi)
        means, _ = _cluster_run_lengths(method, mid, fits, design, template,
                                        cfg.n_futures, cfg.max_steps, base_seed)
        if means.mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _cluster_summary(cluster_means, pooled, max_steps, prefix):
    """Summary whose standard error uses the ``R`` independent outer records."""
    stats = summarize_run_lengths(pooled, max_steps)
    r = len(cluster_means)
    mean = float(cluster_means.mean())
    se = float(cluster_means.std(ddof=1) / math.sqrt(r)) if r > 1 else 0.0
    return {
        prefix + "_mean_rl": mean,
        prefix + "_se_mean": se,
        prefix + "_ci_low": mean - 1.96 * se,
        prefix + "_ci_high": mean + 1.96 * se,
        prefix + "_median_rl": stats["median_rl"],
        prefix + "_p_signal_50": stats["p_signal_50"],
        prefix + "_p90_rl": stats["p90_rl"],
        prefix + "_p95_rl": stats["p95_rl"],
        prefix + "_censor_rate": stats["censor_rate"],
        prefix + "_n_clusters": r,
    }


def run_phase1_study(outdir, known_limits, full=True, model=None, cfg=None,
                     verbose=True):
    """Full Layer-2 experiment; writes the three Phase-I result files.

    ``known_limits`` maps ``"kf"``/``"dg"`` to the *known-parameter* limits
    from Layer 1.  Applying those after estimation is the naive baseline whose
    distortion this experiment quantifies.
    """
    model = model or GaussianModel()
    cfg = cfg or Phase1Config()
    outdir.mkdir(parents=True, exist_ok=True)
    sizes = cfg.reference_sizes
    n_cal = cfg.n_outer_calibration if full else 8
    n_val = cfg.n_outer_validation if full else 12
    designs = {"kf": KF_REFERENCE, "dg": DG_REFERENCE}

    estimate_rows, run_rows = [], []
    for m in sizes:
        if verbose:
            print("Phase-I: m=%d, %d calibration fits, %d validation fits"
                  % (m, n_cal, n_val), flush=True)
        cal_fits, _ = estimate_across_samples(
            m, n_cal, PHASE1_SEEDS["calibration"] + m, model, cfg)
        val_fits, val_frame = estimate_across_samples(
            m, n_val, PHASE1_SEEDS["validation"] + m, model, cfg)
        estimate_rows.append(val_frame)

        for key, design in designs.items():
            adjusted = calibrate_nested(
                key, cal_fits, design, model, cfg,
                base_seed=PHASE1_SEEDS["futures"] + m)
            for limit_type, h in [("known-parameter limit", known_limits[key]),
                                  ("Phase-I adjusted", adjusted)]:
                row = {"m": m, "method": METHOD_LABELS[key], "key": key,
                       "limit_type": limit_type, "h": h}
                means0, pooled0 = _cluster_run_lengths(
                    key, h, val_fits, design, model, cfg.n_futures,
                    cfg.max_steps, PHASE1_SEEDS["futures"] + 7 * m)
                row.update(_cluster_summary(means0, pooled0, cfg.max_steps, "arl0"))
                means1, pooled1 = _cluster_run_lengths(
                    key, h, val_fits, design, model, cfg.n_futures,
                    cfg.max_steps, PHASE1_SEEDS["futures"] + 13 * m,
                    delta=cfg.delta_out_of_control)
                row.update(_cluster_summary(means1, pooled1, cfg.max_steps, "arl1"))
                run_rows.append(row)

    estimates = pd.concat(estimate_rows, ignore_index=True)
    estimates.to_csv(outdir / "phase1_estimates.csv", index=False)
    summary = summarise_estimates(estimates)
    summary.to_csv(outdir / "phase1_estimation_summary.csv", index=False)
    runs = pd.DataFrame(run_rows)
    runs.to_csv(outdir / "phase1_run_length.csv", index=False)
    return estimates, summary, runs
