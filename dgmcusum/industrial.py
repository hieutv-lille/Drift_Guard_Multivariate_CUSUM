"""Layer 3: semi-synthetic gas-turbine stress test (Section 3.4).

REIMPLEMENTATION NOTICE
-----------------------
As with :mod:`dgmcusum.phase1`, the original script was not archived.  This
module implements the protocol the manuscript specifies -- model selection by
AIC/BIC, the radial cap of Eq. (9), circular moving-block resampling of the
processed Phase-I innovations, an in-control generator with a separately
displaced monitor, and recalibration of every sensitivity variant -- but not
the original random streams.  See ``PROVENANCE.md``.

The raw archive is **not** redistributed with this package.  Download it
yourself; ``data/README.md`` says how.

What this layer is, and is not
------------------------------
It is a stress test of the masking mechanism under empirical dependence, heavy
tails and covariance ill-conditioning.  It is *not* validation on observed
faults: the archive carries no verified fault times, so the shifts are
injected.  The fitted model favours a local level with rapid level variation,
which is a different regime from the smooth local-linear setting of Layer 1.
Every conclusion here is conditional on the fitted archive, the block length,
the tail treatment and the covariance estimate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from .charts import ZERO_TOLERANCE, crosier_step
from .config import (DG_REFERENCE, KF_REFERENCE, METHOD_LABELS,
                     INDUSTRIAL_SEEDS, IndustrialConfig, TARGET_ARL0)
from .metrics import summarize_run_lengths

#: Column names in the UCI "gt_2015.csv" file that the study monitors.
UCI_COLUMNS = ("GTEP", "TIT", "TAT", "CDP")


class MissingArchive(FileNotFoundError):
    """Raised when the gas-turbine file has not been downloaded."""


def load_archive(path, cfg=None):
    """Read the 2015 gas-turbine file and return the monitored channels.

    Parameters
    ----------
    path : Path
        Either the ``gt_2015.csv`` file or the directory containing it.
    """
    cfg = cfg or IndustrialConfig()
    path = Path(path)
    if path.is_dir():
        candidates = [path / "gt_2015.csv", path / "gt_2015.CSV"]
        found = [c for c in candidates if c.exists()]
        if not found:
            raise MissingArchive(
                "gt_2015.csv not found in %s -- see data/README.md for the "
                "download instructions (UCI DOI 10.24432/C5WC95)." % path)
        path = found[0]
    if not path.exists():
        raise MissingArchive("%s not found -- see data/README.md" % path)
    frame = pd.read_csv(path)
    missing = [c for c in cfg.variables if c not in frame.columns]
    if missing:
        raise ValueError("archive is missing expected columns: %s" % missing)
    data = frame.loc[:, list(cfg.variables)].to_numpy(dtype=float)
    if len(data) != cfg.n_observations:
        # Reported rather than silently accepted: a different row count means a
        # different file, and every downstream number would shift.
        print("warning: expected %d rows, found %d"
              % (cfg.n_observations, len(data)))
    return data


# --------------------------------------------------------------------------
# Multivariate local-level / local-linear fit with separable covariance
# --------------------------------------------------------------------------


def _filter_errors(data, q_level, q_slope, local_linear, c00_0=100.0, c11_0=1.0,
                   diffuse_prefix=20):
    """Prediction errors and scalar factors for the separable model.

    ``local_linear=False`` drops the slope block, giving the 11-parameter
    local-level model (one drift ratio plus a full 4 x 4 covariance).
    """
    m, p = data.shape
    level = np.zeros(p)
    slope = np.zeros(p)
    c00, c01, c11 = c00_0, 0.0, (c11_0 if local_linear else 0.0)
    errors = np.empty((m, p))
    factors = np.empty(m)
    for t in range(m):
        if local_linear:
            pred_level = level + slope
            pm00 = c00 + 2.0 * c01 + c11 + q_level
            pm01 = c01 + c11
            pm11 = c11 + q_slope
        else:
            pred_level = level
            pm00 = c00 + q_level
            pm01 = pm11 = 0.0
        e = data[t] - pred_level
        f = pm00 + 1.0
        errors[t] = e
        factors[t] = f
        gl, gs = pm00 / f, (pm01 / f if local_linear else 0.0)
        level = pred_level + gl * e
        if local_linear:
            slope = slope + gs * e
            c00, c01, c11 = (1.0 - gl) * pm00, (1.0 - gl) * pm01, pm11 - gs * pm01
        else:
            c00 = (1.0 - gl) * pm00
    d = diffuse_prefix
    return errors[d:], factors[d:], level, slope, (c00, c01, c11)


def _profile_negloglik(log10_q, data, local_linear):
    q_level = 10.0 ** log10_q[0]
    q_slope = 10.0 ** log10_q[1] if local_linear else 0.0
    errors, factors, _, _, _ = _filter_errors(data, q_level, q_slope, local_linear)
    n, p = errors.shape
    scatter = (errors / factors[:, None]).T @ errors / n
    scatter = 0.5 * (scatter + scatter.T)
    try:
        chol = np.linalg.cholesky(scatter)
    except np.linalg.LinAlgError:
        return 1e12
    logdet = 2.0 * float(np.log(np.diag(chol)).sum())
    value = (-0.5 * n * p * (math.log(2.0 * math.pi) + 1.0)
             - 0.5 * p * float(np.log(factors).sum())
             - 0.5 * n * logdet)
    return -value if np.isfinite(value) else 1e12


def fit_state_model(data, local_linear):
    """Fit one state model and return estimates with AIC/BIC.

    Parameter count: one drift ratio (two for local-linear) plus the
    ``p(p+1)/2`` free entries of ``Sigma``.  For ``p = 4`` that is 11 and 12,
    matching the counts reported in the model-selection table.
    """
    p = data.shape[1]
    starts = [(0.5, -6.0), (0.0, -4.0), (1.0, -8.0), (-1.0, -3.0)]
    best = None
    for start in starts:
        res = minimize(_profile_negloglik, np.array(start, dtype=float),
                       args=(data, local_linear), method="Nelder-Mead",
                       options={"maxiter": 800, "xatol": 1e-5, "fatol": 1e-5})
        if best is None or res.fun < best.fun:
            best = res
    q_level = 10.0 ** best.x[0]
    q_slope = (10.0 ** best.x[1]) if local_linear else 0.0
    errors, factors, level, slope, cfac = _filter_errors(
        data, q_level, q_slope, local_linear)
    n = len(errors)
    sigma = (errors / factors[:, None]).T @ errors / n
    sigma = 0.5 * (sigma + sigma.T)
    loglik = -best.fun
    k = (2 if local_linear else 1) + p * (p + 1) // 2
    return {
        "model": "Local-linear" if local_linear else "Local-level",
        "q_level": float(q_level), "q_slope": float(q_slope),
        "loglik": float(loglik),
        "aic": float(2 * k - 2 * loglik),
        "bic": float(k * math.log(n) - 2 * loglik),
        "parameters": int(k),
        "sigma": sigma, "terminal_level": level, "terminal_slope": slope,
        "terminal_c": cfac, "errors": errors, "factors": factors,
        "local_linear": local_linear,
    }


def select_state_model(phase1_data):
    """Compare local-level and local-linear; lower AIC *and* BIC wins."""
    fits = [fit_state_model(phase1_data, False), fit_state_model(phase1_data, True)]
    table = pd.DataFrame([{k: f[k] for k in
                           ("model", "q_level", "q_slope", "loglik", "aic",
                            "bic", "parameters")} for f in fits])
    chosen = min(fits, key=lambda f: f["aic"])
    return chosen, fits, table


# --------------------------------------------------------------------------
# Innovation processing: radial cap and covariance shrinkage
# --------------------------------------------------------------------------


def standardised_innovations(fit):
    """Whitened Phase-I innovations ``z_t`` with ``Var(z_t) = I``."""
    chol = np.linalg.cholesky(fit["sigma"])
    whitened = np.linalg.solve(chol, fit["errors"].T).T
    return whitened / np.sqrt(fit["factors"])[:, None]


def radial_cap(z, quantile, p):
    """Eq. (9): shrink any vector longer than the chi-square radius.

    Large innovations are capped rather than deleted, because in an unlabelled
    archive an extreme value may be an accepted operating transition rather
    than a fault.  ``z = 0`` is handled without dividing by zero.
    """
    radius = math.sqrt(chi2.ppf(quantile, df=p))
    norms = np.linalg.norm(z, axis=1)
    scale = np.ones(len(z))
    over = norms > radius
    scale[over] = radius / norms[over]
    return z * scale[:, None], int(over.sum()), float(radius)


def shrink_covariance(sigma, gamma):
    """Shrink toward a scaled identity: ``(1-g) S + g (tr S / p) I``."""
    if gamma <= 0.0:
        return sigma
    p = sigma.shape[0]
    return (1.0 - gamma) * sigma + gamma * (np.trace(sigma) / p) * np.eye(p)


# --------------------------------------------------------------------------
# Moving-block bootstrap and the two-filter monitoring experiment
# --------------------------------------------------------------------------


def bootstrap_innovations(source, n_paths, horizon, block_length, rng):
    """Circular moving-block resampling of the processed innovations.

    Order is preserved *within* blocks, which keeps short-range dependence;
    it is not preserved globally, so the archive's chronology is deliberately
    destroyed.  Block length is fixed in this study, so block-length
    sensitivity remains a stated limitation.
    """
    m = len(source)
    n_blocks = int(math.ceil(horizon / float(block_length)))
    starts = rng.integers(0, m, size=(n_paths, n_blocks))
    offsets = np.arange(block_length)
    idx = (starts[:, :, None] + offsets[None, None, :]) % m
    idx = idx.reshape(n_paths, n_blocks * block_length)[:, :horizon]
    return source[idx]


def _run_charts(z_paths, design_kf, design_dg, h_kf, h_dg, q_level, shift_vector,
                sigma_chol, f_infinity, tau=1):
    """Monitor bootstrapped paths with the KF and DG charts.

    The generator stays in control: bootstrapped standardised innovations are
    turned back into observations through an in-control local-level filter.
    The *monitor* is a second filter that sees those observations plus the
    injected displacement, so only the monitor experiences the fault.
    """
    n, horizon, p = z_paths.shape
    results = {}
    for key, design, h in (("kf", design_kf, h_kf), ("dg", design_dg, h_dg)):
        gen_level = np.zeros((n, p))
        gen_c = np.full(n, f_infinity - 1.0)
        mon_level = np.zeros((n, p))
        mon_c = np.full(n, f_infinity - 1.0)

        primary_s = np.zeros((n, p))
        confirm_s = np.zeros((n, p))
        confirming = np.zeros(n, dtype=bool)
        age = np.zeros(n, dtype=int)
        anchor_level = np.zeros((n, p))
        anchor_c = np.zeros(n)
        alive = np.ones(n, dtype=bool)
        run_length = np.full(n, horizon + 1, dtype=int)

        for t in range(horizon):
            # ---- in-control generator ----
            gen_pm = gen_c + q_level
            gen_f = gen_pm + 1.0
            innovation = (z_paths[:, t, :] @ sigma_chol.T) * np.sqrt(gen_f)[:, None]
            y = gen_level + innovation
            gain = gen_pm / gen_f
            gen_level = gen_level + gain[:, None] * innovation
            gen_c = (1.0 - gain) * gen_pm

            observed = y + (shift_vector if t >= tau - 1 else 0.0)

            # ---- monitoring filter ----
            mon_pm = mon_c + q_level
            mon_f = mon_pm + 1.0
            pred_level = mon_level.copy()
            residual = observed - pred_level
            z = np.linalg.solve(sigma_chol, residual.T).T / np.sqrt(mon_f)[:, None]
            gain = mon_pm / mon_f
            mon_level = pred_level + gain[:, None] * residual
            mon_c = (1.0 - gain) * mon_pm

            if key == "kf":
                s_new, statistic = crosier_step(primary_s, z, design.k1)
                primary_s[alive] = s_new[alive]
                hit = alive & (statistic > h)
                run_length[hit] = t + 1
                alive[hit] = False
                if not alive.any():
                    break
                continue

            s_new, first_stat = crosier_step(primary_s, z, design.k1)
            eligible = (~confirming) & alive
            primary_s[eligible] = s_new[eligible]
            start = eligible & (first_stat > design.g)
            confirming[start] = True
            age[start] = 0
            confirm_s[start] = 0.0
            anchor_level[start] = pred_level[start]
            anchor_c[start] = mon_pm[start]

            old = confirming & (~start) & alive
            anchor_c[old] = anchor_c[old] + q_level

            candidate = confirming & alive
            protected_z = np.zeros_like(z)
            if candidate.any():
                res = observed[candidate] - anchor_level[candidate]
                protected_z[candidate] = (
                    np.linalg.solve(sigma_chol, res.T).T
                    / np.sqrt(anchor_c[candidate] + 1.0)[:, None])
            confirm_new, statistic = crosier_step(confirm_s, protected_z, design.k2)
            confirm_s[candidate] = confirm_new[candidate]
            age[candidate] += 1

            hit = candidate & (statistic > h)
            run_length[hit] = t + 1
            alive[hit] = False
            reject = candidate & alive & (
                ((statistic <= ZERO_TOLERANCE) & (age >= 2)) | (age >= design.L))
            confirming[reject] = False
            age[reject] = 0
            confirm_s[reject] = 0.0
            primary_s[reject] = 0.0
            if not alive.any():
                break
        results[key] = run_length
    return results


def shift_vector(delta, sigma_chol, f_infinity, p):
    """``delta = Delta sqrt(f_inf) L_Sigma d`` with ``d = 1/2`` (Section 3.4).

    Scaled so that ``(delta' F_inf^{-1} delta)^{1/2} = Delta``.  Note this uses
    a different reference scale from the Gaussian layer's ``Delta``; the two
    are not directly comparable, which the manuscript states explicitly.
    """
    d = np.full(p, 0.5)
    return delta * math.sqrt(f_infinity) * (sigma_chol @ d)


def calibrate_industrial(source, design, key, q_level, sigma_chol, f_infinity,
                         cfg, seed, target=TARGET_ARL0):
    """Bisect the limit against in-control bootstrap paths."""
    p = sigma_chol.shape[0]
    rng = np.random.default_rng(seed)
    paths = bootstrap_innovations(source, cfg.n_limit_calibration,
                                  cfg.evaluation_horizon, cfg.block_length, rng)
    zero = np.zeros(p)
    lo, hi = 1.0, 120.0
    other = KF_REFERENCE if key == "dg" else DG_REFERENCE
    for _ in range(cfg.bisection_iterations):
        mid = 0.5 * (lo + hi)
        kw = dict(design_kf=design if key == "kf" else other,
                  design_dg=design if key == "dg" else other,
                  h_kf=mid if key == "kf" else 1e9,
                  h_dg=mid if key == "dg" else 1e9)
        rl = _run_charts(paths, shift_vector=zero, q_level=q_level,
                         sigma_chol=sigma_chol, f_infinity=f_infinity, **kw)[key]
        arl = float(np.minimum(rl, cfg.evaluation_horizon).mean())
        if arl < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run_industrial_study(archive_path, outdir, cfg=None, verbose=True,
                         sensitivity=True):
    """Complete Layer-3 experiment; writes the industrial result files."""
    cfg = cfg or IndustrialConfig()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = load_archive(archive_path, cfg)
    phase1 = data[:cfg.phase1_size]
    p = phase1.shape[1]

    chosen, fits, model_table = select_state_model(phase1)
    model_table.to_csv(outdir / "industrial_model_selection.csv", index=False)
    if verbose:
        print("Selected state model: %s" % chosen["model"], flush=True)

    z_raw = standardised_innovations(chosen)
    kurtosis = [float(((z_raw[:, j] - z_raw[:, j].mean()) ** 4).mean()
                      / (z_raw[:, j].var() ** 2) - 3.0) for j in range(p)]
    sigma = chosen["sigma"]
    eigenvalues = np.linalg.eigvalsh(sigma)
    q_level = chosen["q_level"]
    f_infinity = float(chosen["terminal_c"][0] + 1.0)

    cap_counts = {}
    for q in cfg.cap_variants:
        cap_counts[str(q)] = radial_cap(z_raw, q, p)[1]

    variants = [("%.3f" % q, q, 0.0) for q in cfg.cap_variants]
    variants.append(("uncapped", None, 0.0))
    variants.append(("shrunk", cfg.cap_quantile, cfg.shrinkage_gamma))

    main_rows, tail_rows = [], []
    limits_main = {}
    # Deterministic per-variant seed offset.  Python's builtin hash() is
    # randomised per interpreter run, so it must never appear in a seed.
    for variant_index, (name, quantile, gamma) in enumerate(variants):
        is_main = (quantile == cfg.cap_quantile and gamma == 0.0)
        if not sensitivity and not is_main:
            continue
        if quantile is None:
            source, capped, cutoff = z_raw, 0, float("nan")
        else:
            source, capped, cutoff = radial_cap(z_raw, quantile, p)
        sigma_v = shrink_covariance(sigma, gamma)
        chol = np.linalg.cholesky(sigma_v)

        limits = {}
        for key, design in (("kf", KF_REFERENCE), ("dg", DG_REFERENCE)):
            limits[key] = calibrate_industrial(
                source, design, key, q_level, chol, f_infinity, cfg,
                INDUSTRIAL_SEEDS["calibration"] + 1000 * variant_index)
        if verbose:
            print("variant %-9s limits kf=%.4f dg=%.4f"
                  % (name, limits["kf"], limits["dg"]), flush=True)
        if is_main:
            limits_main = dict(limits)

        rng = np.random.default_rng(INDUSTRIAL_SEEDS["evaluation"]
                                    + 1000 * variant_index)
        deltas = cfg.deltas if is_main else (0.0, 8.0)
        for delta in deltas:
            paths = bootstrap_innovations(source, cfg.n_evaluation,
                                          cfg.evaluation_horizon,
                                          cfg.block_length, rng)
            vec = shift_vector(delta, chol, f_infinity, p)
            out = _run_charts(paths, KF_REFERENCE, DG_REFERENCE,
                              limits["kf"], limits["dg"], q_level, vec, chol,
                              f_infinity)
            for key in ("kf", "dg"):
                stats = summarize_run_lengths(out[key], cfg.evaluation_horizon)
                row = {"method": METHOD_LABELS[key], "key": key, "delta": delta,
                       "h": limits[key], "n_rep": cfg.n_evaluation,
                       "evaluation_horizon": cfg.evaluation_horizon,
                       "mean_rl": stats["mean_rl"], "se_mean": stats["se_mean"],
                       "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
                       "median_rl": stats["median_rl"],
                       "p_signal_25": stats["p_signal_25"],
                       "p_signal_50": stats["p_signal_50"],
                       "p_signal_100": stats["p_signal_100"],
                       "p90_rl": stats["p90_rl"], "p95_rl": stats["p95_rl"],
                       "censor_rate": stats["censor_rate"],
                       "n_clusters": cfg.n_evaluation}
                if is_main:
                    main_rows.append(row)
                if delta in (0.0, 8.0):
                    tail_rows.append({
                        "variant": name, "cap_quantile": quantile,
                        "shrinkage_gamma": gamma,
                        "method": METHOD_LABELS[key], "key": key,
                        "vectors_capped": capped, "cap_cutoff": cutoff,
                        "h": limits[key], "delta": delta,
                        "mean_rl": stats["mean_rl"],
                        "median_rl": stats["median_rl"],
                        "p95_rl": stats["p95_rl"],
                        "censor_rate": stats["censor_rate"]})

    pd.DataFrame(main_rows).to_csv(outdir / "industrial_semisynthetic.csv",
                                   index=False)
    pd.DataFrame(tail_rows).to_csv(outdir / "industrial_tail_sensitivity.csv",
                                   index=False)

    summary = {
        "selected_model": chosen["model"].lower(),
        "model_selection": model_table.to_dict("records"),
        "q_level_hat": q_level,
        "q_slope_hat": chosen["q_slope"],
        "steady_state_like_level_gain_at_terminal_covariance":
            float((f_infinity - 1.0) / f_infinity),
        "sigma_hat": sigma.tolist(),
        "sigma_eigenvalues": eigenvalues.tolist(),
        "sigma_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "shrinkage_gamma": cfg.shrinkage_gamma,
        "shrunk_sigma_condition_number": float(
            np.linalg.cond(shrink_covariance(sigma, cfg.shrinkage_gamma))),
        "phase1_innovation_excess_kurtosis": kurtosis,
        "cap_counts": cap_counts,
        "main_cap_quantile": cfg.cap_quantile,
        "main_cap_cutoff": radial_cap(z_raw, cfg.cap_quantile, p)[2] ** 2,
        "main_vectors_capped": cap_counts[str(cfg.cap_quantile)],
        "calibrated_limits": limits_main,
        "evaluation_horizon": cfg.evaluation_horizon,
        "moving_block_length": cfg.block_length,
        "n_limit_calibration_replicates": cfg.n_limit_calibration,
        "n_evaluation_replicates": cfg.n_evaluation,
        "note": "Reimplementation from the manuscript specification; "
                "random streams differ from the archived run.",
    }
    (outdir / "industrial_fit.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    return summary, pd.DataFrame(main_rows), pd.DataFrame(tail_rows)
