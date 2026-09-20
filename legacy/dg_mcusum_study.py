# %% [markdown]
# # DG-MCUSUM reproducibility notebook
#
# This script/notebook implements the Drift-Guard multivariate CUSUM
# (DG-MCUSUM), calibrates matched in-control average run lengths, reproduces
# the simulation tables, and draws the figures used in the manuscript.

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class StudyConfig:
    p: int = 4
    q_level_data: float = 0.002
    q_slope_data: float = 2e-6
    q_level_filter: float = 0.002
    q_slope_filter: float = 2e-6
    burn_in: int = 150
    k_primary: float = 0.50
    warning_limit: float = 1.50
    confirmation_horizon: int = 15
    k_confirmation: float = 0.25
    ewma_lambda: float = 0.08
    target_arl0: float = 500.0


METHOD_LABELS = {
    "oracle": "Oracle MCUSUM",
    "static": "Static MCUSUM",
    "ewma": "EWMA-residual MCUSUM",
    "kf": "KF-innovation MCUSUM",
    "dg": "DG-MCUSUM",
}


def crosier_step(s: np.ndarray, z: np.ndarray, k: float):
    """One rotationally invariant Crosier MCUSUM update."""
    u = s + z
    norm_u = np.linalg.norm(u, axis=1)
    factor = np.maximum(0.0, 1.0 - k / np.maximum(norm_u, 1e-12))
    s_new = u * factor[:, None]
    return s_new, np.linalg.norm(s_new, axis=1)


def _advance_state(level, slope, q_level, q_slope, rng):
    level = level + slope + math.sqrt(q_level) * rng.normal(size=level.shape)
    slope = slope + math.sqrt(q_slope) * rng.normal(size=slope.shape)
    return level, slope


def simulate_run_lengths(
    method: str,
    h: float,
    *,
    n_rep: int,
    max_steps: int,
    config: StudyConfig,
    delta: float = 0.0,
    direction: str = "dense",
    seed: int = 1,
) -> np.ndarray:
    """Vectorized zero-state run-length simulation after a filter warm-up.

    The data are represented in whitened coordinates. This is equivalent to
    generating correlated raw data and applying the known covariance
    transformation before monitoring. The smooth background follows a local
    linear trend; an anomaly is a persistent additive mean shift beginning at
    the first monitored observation.
    """
    if method not in METHOD_LABELS:
        raise ValueError(f"Unknown method: {method}")
    p = config.p
    rng = np.random.default_rng(seed)
    level = np.zeros((n_rep, p))
    slope = np.zeros((n_rep, p))
    filt_level = np.zeros((n_rep, p))
    filt_slope = np.zeros((n_rep, p))
    ewma_level = np.zeros((n_rep, p))
    static_center = np.zeros((n_rep, p))
    p00 = np.full(n_rep, 10.0)
    p01 = np.zeros(n_rep)
    p11 = np.full(n_rep, 1.0)

    primary_s = np.zeros((n_rep, p))
    confirming = np.zeros(n_rep, dtype=bool)
    confirm_age = np.zeros(n_rep, dtype=int)
    confirm_s = np.zeros((n_rep, p))
    anchor_level = np.zeros((n_rep, p))
    anchor_slope = np.zeros((n_rep, p))
    anchor_p00 = np.zeros(n_rep)
    anchor_p01 = np.zeros(n_rep)
    anchor_p11 = np.zeros(n_rep)

    active = np.ones(n_rep, dtype=bool)
    run_length = np.full(n_rep, max_steps + 1, dtype=int)
    if direction == "dense":
        shift_direction = np.ones(p) / math.sqrt(p)
    elif direction == "sparse":
        shift_direction = np.zeros(p)
        shift_direction[0] = 1.0
    elif direction == "mixed":
        shift_direction = np.array([1 if j % 2 == 0 else -1 for j in range(p)], dtype=float)
        shift_direction /= np.linalg.norm(shift_direction)
    else:
        raise ValueError("direction must be dense, sparse, or mixed")

    def kalman_predict_update(y):
        nonlocal filt_level, filt_slope, p00, p01, p11
        pred_level = filt_level + filt_slope
        pred_slope = filt_slope
        pm00 = p00 + 2.0 * p01 + p11 + config.q_level_filter
        pm01 = p01 + p11
        pm11 = p11 + config.q_slope_filter
        innovation = y - pred_level
        innovation_variance = pm00 + 1.0
        z = innovation / np.sqrt(innovation_variance)[:, None]
        gain_level = pm00 / innovation_variance
        gain_slope = pm01 / innovation_variance
        filt_level = pred_level + gain_level[:, None] * innovation
        filt_slope = pred_slope + gain_slope[:, None] * innovation
        p00 = (1.0 - gain_level) * pm00
        p01 = (1.0 - gain_level) * pm01
        p11 = pm11 - gain_slope * pm01
        return z, pred_level, pred_slope, pm00, pm01, pm11

    # Phase-I-like warm-up. No chart statistic is accumulated.
    for _ in range(config.burn_in):
        level, slope = _advance_state(
            level, slope, config.q_level_data, config.q_slope_data, rng
        )
        y = level + rng.normal(size=(n_rep, p))
        if method in {"kf", "dg", "oracle"}:
            kalman_predict_update(y)
        elif method == "ewma":
            ewma_level += config.ewma_lambda * (y - ewma_level)
    # Give the fixed chart the strongest possible current target; any later
    # false signal is caused by subsequent drift, not Phase-I estimation.
    static_center[:] = level

    for t in range(1, max_steps + 1):
        level, slope = _advance_state(
            level, slope, config.q_level_data, config.q_slope_data, rng
        )
        y = level + delta * shift_direction + rng.normal(size=(n_rep, p))

        if method == "oracle":
            z = y - level
            s_new, statistic = crosier_step(primary_s, z, config.k_primary)
            primary_s[active] = s_new[active]

        elif method == "static":
            z = y - static_center
            s_new, statistic = crosier_step(primary_s, z, config.k_primary)
            primary_s[active] = s_new[active]

        elif method == "ewma":
            z = y - ewma_level
            ewma_level += config.ewma_lambda * z
            s_new, statistic = crosier_step(primary_s, z, config.k_primary)
            primary_s[active] = s_new[active]

        elif method == "kf":
            z, *_ = kalman_predict_update(y)
            s_new, statistic = crosier_step(primary_s, z, config.k_primary)
            primary_s[active] = s_new[active]

        else:  # DG-MCUSUM
            z, pred_level, pred_slope, pm00, pm01, pm11 = kalman_predict_update(y)
            first_new, first_stat = crosier_step(primary_s, z, config.k_primary)
            eligible = (~confirming) & active
            primary_s[eligible] = first_new[eligible]
            start = eligible & (first_stat > config.warning_limit)
            confirming[start] = True
            confirm_age[start] = 0
            confirm_s[start] = 0.0
            anchor_level[start] = pred_level[start]
            anchor_slope[start] = pred_slope[start]
            anchor_p00[start] = pm00[start]
            anchor_p01[start] = pm01[start]
            anchor_p11[start] = pm11[start]

            old = confirming & (~start) & active
            anchor_level[old] += anchor_slope[old]
            next_p00 = (
                anchor_p00[old]
                + 2.0 * anchor_p01[old]
                + anchor_p11[old]
                + config.q_level_filter
            )
            next_p01 = anchor_p01[old] + anchor_p11[old]
            next_p11 = anchor_p11[old] + config.q_slope_filter
            anchor_p00[old] = next_p00
            anchor_p01[old] = next_p01
            anchor_p11[old] = next_p11

            candidate = confirming & active
            protected_z = np.zeros_like(z)
            protected_z[candidate] = (
                y[candidate] - anchor_level[candidate]
            ) / np.sqrt(anchor_p00[candidate] + 1.0)[:, None]
            confirm_new, statistic = crosier_step(
                confirm_s, protected_z, config.k_confirmation
            )
            confirm_s[candidate] = confirm_new[candidate]
            confirm_age[candidate] += 1

            hit = candidate & (statistic > h)
            run_length[hit] = t
            active[hit] = False
            reject = candidate & active & (
                ((statistic <= 1e-12) & (confirm_age >= 2))
                | (confirm_age >= config.confirmation_horizon)
            )
            confirming[reject] = False
            confirm_age[reject] = 0
            confirm_s[reject] = 0.0
            primary_s[reject] = 0.0
            if not active.any():
                break
            continue

        hit = active & (statistic > h)
        run_length[hit] = t
        active[hit] = False
        if not active.any():
            break

    return run_length


def summarize_run_lengths(run_length: np.ndarray, max_steps: int):
    observed = np.minimum(run_length, max_steps)
    mean = float(np.mean(observed))
    sd = float(np.std(observed, ddof=1))
    return {
        "mean_rl": mean,
        "se_mean": sd / math.sqrt(len(observed)),
        "ci_low": mean - 1.96 * sd / math.sqrt(len(observed)),
        "ci_high": mean + 1.96 * sd / math.sqrt(len(observed)),
        "median_rl": float(np.median(run_length)),
        "p_signal_25": float(np.mean(run_length <= 25)),
        "p_signal_50": float(np.mean(run_length <= 50)),
        "p_signal_100": float(np.mean(run_length <= 100)),
        "censor_rate": float(np.mean(run_length > max_steps)),
    }


def calibrate_limit(
    method: str,
    config: StudyConfig,
    *,
    target: float = 500.0,
    n_rep: int = 2000,
    max_steps: int = 2500,
    lo: float = 1.0,
    hi: float = 80.0,
    seed: int = 8128,
    iterations: int = 13,
):
    """Calibrate h with common random numbers and monotone bisection."""
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        rl = simulate_run_lengths(
            method,
            mid,
            n_rep=n_rep,
            max_steps=max_steps,
            config=config,
            seed=seed,
        )
        arl = float(np.mean(np.minimum(rl, max_steps)))
        if arl < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def run_study(full: bool = True):
    config = StudyConfig()
    n_cal = 2500 if full else 700
    n_validate = 7000 if full else 1800
    n_ooc = 5000 if full else 1500
    n_sensitivity = 3500 if full else 1000
    max_ic = 3000
    max_ooc = 3000

    methods = ["oracle", "kf", "dg", "ewma"]
    brackets = {
        "oracle": (5.0, 15.0),
        "kf": (5.0, 15.0),
        "dg": (8.0, 28.0),
        "ewma": (20.0, 100.0),
    }
    limits = {}
    for j, method in enumerate(methods):
        lo, hi = brackets[method]
        limits[method] = calibrate_limit(
            method,
            config,
            n_rep=n_cal,
            max_steps=2500,
            lo=lo,
            hi=hi,
            seed=1200 + j,
        )
    if full:
        # A higher-precision final calibration is used for the proposed chart
        # because candidate selection makes its run-length curve steeper near
        # the target than the single-stage comparators.
        limits["dg"] = calibrate_limit(
            "dg",
            config,
            n_rep=8000,
            max_steps=3000,
            lo=16.35,
            hi=16.55,
            seed=9010,
            iterations=9,
        )

    # The classical fixed-target MCUSUM is calibrated in a stationary process,
    # then challenged under drift to quantify false-alarm collapse.
    stationary = StudyConfig(q_level_data=0.0, q_slope_data=0.0)
    limits["static"] = calibrate_limit(
        "static",
        stationary,
        n_rep=n_cal,
        max_steps=2500,
        lo=5.0,
        hi=15.0,
        seed=1300,
    )
    pd.DataFrame(
        [{"method": METHOD_LABELS[m], "key": m, "h": h} for m, h in limits.items()]
    ).to_csv(RESULTS / "control_limits.csv", index=False)

    ic_rows = []
    for j, method in enumerate(["oracle", "kf", "dg", "ewma", "static"]):
        rl = simulate_run_lengths(
            method,
            limits[method],
            n_rep=n_validate,
            max_steps=max_ic,
            config=config,
            seed=2200 + j,
        )
        ic_rows.append(
            {
                "method": METHOD_LABELS[method],
                "key": method,
                "h": limits[method],
                **summarize_run_lengths(rl, max_ic),
            }
        )
    arl0 = pd.DataFrame(ic_rows)
    arl0.to_csv(RESULTS / "arl0_validation.csv", index=False)

    ooc_rows = []
    deltas = [0.50, 0.75, 1.00, 1.50, 2.00]
    for j, method in enumerate(["oracle", "kf", "dg", "ewma"]):
        for d in deltas:
            rl = simulate_run_lengths(
                method,
                limits[method],
                n_rep=n_ooc,
                max_steps=max_ooc,
                config=config,
                delta=d,
                seed=3200 + 100 * j + int(100 * d),
            )
            ooc_rows.append(
                {
                    "method": METHOD_LABELS[method],
                    "key": method,
                    "delta": d,
                    "direction": "dense",
                    **summarize_run_lengths(rl, max_ooc),
                }
            )
    arl1 = pd.DataFrame(ooc_rows)
    arl1.to_csv(RESULTS / "arl1_main.csv", index=False)

    # Directional robustness: sparse and alternating-sign shifts have the same
    # Mahalanobis magnitude but different physical patterns.
    direction_rows = []
    for method_index, method in enumerate(["kf", "dg"]):
        for direction_index, direction in enumerate(["dense", "sparse", "mixed"]):
            rl = simulate_run_lengths(
                method,
                limits[method],
                n_rep=n_sensitivity,
                max_steps=max_ooc,
                config=config,
                delta=1.0,
                direction=direction,
                seed=4100 + 100 * method_index + 10 * direction_index,
            )
            direction_rows.append(
                {
                    "method": METHOD_LABELS[method],
                    "direction": direction,
                    "delta": 1.0,
                    **summarize_run_lengths(rl, max_ooc),
                }
            )
    pd.DataFrame(direction_rows).to_csv(
        RESULTS / "direction_sensitivity.csv", index=False
    )

    # Misspecification: the filter remains fixed at the design Q while the true
    # drift volatility is multiplied by 0.5 or 2.
    mismatch_rows = []
    for method_index, method in enumerate(["kf", "dg"]):
        for scale_index, scale in enumerate([0.5, 1.0, 2.0]):
            cfg = StudyConfig(
                q_level_data=config.q_level_data * scale,
                q_slope_data=config.q_slope_data * scale,
            )
            rl = simulate_run_lengths(
                method,
                limits[method],
                n_rep=n_sensitivity,
                max_steps=max_ic,
                config=cfg,
                seed=5100 + 100 * method_index + scale_index,
            )
            mismatch_rows.append(
                {
                    "method": METHOD_LABELS[method],
                    "Q_scale": scale,
                    **summarize_run_lengths(rl, max_ic),
                }
            )
    pd.DataFrame(mismatch_rows).to_csv(
        RESULTS / "drift_misspecification.csv", index=False
    )

    # Compact design sensitivity around the selected warning limit.
    design_rows = []
    for warning_index, warning in enumerate([1.0, 1.5, 2.0]):
        cfg = StudyConfig(warning_limit=warning)
        h = calibrate_limit(
            "dg",
            cfg,
            n_rep=max(900, n_cal // 2),
            max_steps=2200,
            lo=8.0,
            hi=28.0,
            seed=6100 + warning_index,
            iterations=11,
        )
        rl = simulate_run_lengths(
            "dg",
            h,
            n_rep=max(1600, n_sensitivity // 2),
            max_steps=max_ooc,
            config=cfg,
            delta=1.0,
            seed=6200 + warning_index,
        )
        design_rows.append(
            {
                "warning_limit": warning,
                "confirmation_horizon": cfg.confirmation_horizon,
                "h": h,
                **summarize_run_lengths(rl, max_ooc),
            }
        )
    pd.DataFrame(design_rows).to_csv(RESULTS / "design_sensitivity.csv", index=False)

    metadata = {
        "full_run": full,
        "n_calibration": n_cal,
        "n_arl0_validation": n_validate,
        "n_arl1": n_ooc,
        "n_sensitivity": n_sensitivity,
        "seed_policy": "Fixed, scenario-specific seeds listed in dg_mcusum_study.py",
        "config": config.__dict__,
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    return config, limits, arl0, arl1


# %% [markdown]
# ## Additional experiment: stationary anomaly-only regime
#
# This experiment sets both the data-generating and monitoring models to
# $q_\ell=q_b=0$.  Hence the baseline is stationary and a persistent mean
# shift is the only out-of-control mechanism.  Oracle, static-target, Kalman
# innovation, and DG charts are calibrated *separately* to the same target
# $\mathrm{ARL}_0=500$ before their detection delays are compared.  These
# matched limits must not be reused from the drifting experiment.

# %%
def stationary_anomaly_config() -> StudyConfig:
    """Return the exact no-drift configuration used in the add-on study."""
    return StudyConfig(
        q_level_data=0.0,
        q_slope_data=0.0,
        q_level_filter=0.0,
        q_slope_filter=0.0,
    )


def run_stationary_anomaly_only(full: bool = True):
    """Run the matched-ARL0 stationary anomaly-only experiment.

    The anomaly is a persistent mean shift present from the first monitored
    observation.  Therefore ``mean_rl`` is the zero-state average detection
    delay including the signal observation.  A one- or two-point spike is a
    different estimand and is intentionally not mixed into this experiment.
    """
    config = stationary_anomaly_config()
    methods = ["oracle", "static", "kf", "dg"]
    deltas = [0.50, 0.75, 1.00, 1.50, 2.00]

    n_cal = 3200 if full else 700
    n_validate = 9000 if full else 1800
    n_ooc = 5500 if full else 1300
    max_ic = 3000
    max_ooc = 3000
    brackets = {
        "oracle": (5.0, 15.0),
        "static": (5.0, 15.0),
        "kf": (5.0, 15.0),
        "dg": (8.0, 28.0),
    }

    limits = {}
    for j, method in enumerate(methods):
        lo, hi = brackets[method]
        limits[method] = calibrate_limit(
            method,
            config,
            target=config.target_arl0,
            n_rep=n_cal,
            max_steps=2500,
            lo=lo,
            hi=hi,
            seed=17100 + j,
            iterations=12,
        )

    # Refine each method around its preliminary solution.  This is still a
    # method-specific calibration; common limits are never imposed.
    if full:
        for j, method in enumerate(methods):
            center = limits[method]
            radius = 0.20 if method != "dg" else 0.35
            limits[method] = calibrate_limit(
                method,
                config,
                target=config.target_arl0,
                n_rep=7500,
                max_steps=max_ic,
                lo=max(0.1, center - radius),
                hi=center + radius,
                seed=17200 + j,
                iterations=8,
            )

    limit_df = pd.DataFrame(
        [
            {
                "method": METHOD_LABELS[method],
                "key": method,
                "h_stationary": limits[method],
                "target_arl0": config.target_arl0,
            }
            for method in methods
        ]
    )
    limit_df.to_csv(RESULTS / "stationary_control_limits.csv", index=False)

    arl0_rows = []
    for j, method in enumerate(methods):
        rl = simulate_run_lengths(
            method,
            limits[method],
            n_rep=n_validate,
            max_steps=max_ic,
            config=config,
            seed=18100 + j,
        )
        summary = summarize_run_lengths(rl, max_ic)
        arl0_rows.append(
            {
                "method": METHOD_LABELS[method],
                "key": method,
                "h_stationary": limits[method],
                "target_arl0": config.target_arl0,
                "relative_error_pct": 100.0
                * (summary["mean_rl"] - config.target_arl0)
                / config.target_arl0,
                "within_10pct_target": abs(summary["mean_rl"] - config.target_arl0)
                <= 0.10 * config.target_arl0,
                **summary,
            }
        )
    arl0 = pd.DataFrame(arl0_rows)
    arl0.to_csv(RESULTS / "stationary_arl0_validation.csv", index=False)

    ooc_rows = []
    for j, method in enumerate(methods):
        for delta in deltas:
            rl = simulate_run_lengths(
                method,
                limits[method],
                n_rep=n_ooc,
                max_steps=max_ooc,
                config=config,
                delta=delta,
                direction="dense",
                seed=19100 + 100 * j + int(100 * delta),
            )
            ooc_rows.append(
                {
                    "method": METHOD_LABELS[method],
                    "key": method,
                    "delta": delta,
                    "direction": "dense",
                    **summarize_run_lengths(rl, max_ooc),
                }
            )
    arl1 = pd.DataFrame(ooc_rows)

    oracle_delay = (
        arl1.loc[arl1.key == "oracle", ["delta", "mean_rl"]]
        .rename(columns={"mean_rl": "oracle_mean_rl"})
    )
    arl1 = arl1.merge(oracle_delay, on="delta", how="left")
    arl1["delay_ratio_vs_oracle"] = arl1["mean_rl"] / arl1["oracle_mean_rl"]
    arl1["excess_delay_pct_vs_oracle"] = 100.0 * (
        arl1["delay_ratio_vs_oracle"] - 1.0
    )
    arl1.to_csv(RESULTS / "stationary_anomaly_only_arl1.csv", index=False)

    pivot = arl1.pivot(index="delta", columns="method", values="mean_rl")
    pivot.to_csv(RESULTS / "stationary_anomaly_only_mean_rl_pivot.csv")
    latex_columns = [METHOD_LABELS[m] for m in methods]
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{Stationary anomaly-only average run length after "
            r"separate calibration to $\mathrm{ARL}_0=500$.}"
        ),
        r"\label{tab:stationary-anomaly-only}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"$\Delta$ & Oracle & Static & KF innovation & DG-MCUSUM \\",
        r"\midrule",
    ]
    for delta, row in pivot[latex_columns].iterrows():
        latex_lines.append(
            f"{delta:.2f} & "
            + " & ".join(f"{float(row[column]):.1f}" for column in latex_columns)
            + r" \\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (RESULTS / "stationary_anomaly_only_table.tex").write_text(
        "\n".join(latex_lines) + "\n"
    )

    metadata = {
        "experiment": "stationary anomaly-only persistent mean shift",
        "full_run": full,
        "n_calibration_preliminary": n_cal,
        "n_calibration_refinement": 7500 if full else None,
        "n_arl0_validation": n_validate,
        "n_arl1_per_cell": n_ooc,
        "max_steps": max_ooc,
        "deltas": deltas,
        "methods": methods,
        "calibration": "separate matched ARL0=500 limits for every method",
        "config": config.__dict__,
    }
    (RESULTS / "stationary_run_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    return config, limits, arl0, arl1


# %% [markdown]
# ## Figures

# %%
def plot_performance(arl1: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "Oracle MCUSUM": "#5B5F97",
        "KF-innovation MCUSUM": "#3A7D44",
        "DG-MCUSUM": "#D95D39",
        "EWMA-residual MCUSUM": "#777777",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for method, group in arl1.groupby("method", sort=False):
        axes[0].plot(
            group["delta"], group["mean_rl"], marker="o", linewidth=2,
            color=colors[method], label=method
        )
        axes[1].plot(
            group["delta"], group["p_signal_50"], marker="o", linewidth=2,
            color=colors[method], label=method
        )
    axes[0].set(xlabel=r"Mahalanobis shift $\Delta$", ylabel="Average run length")
    axes[0].set_yscale("log")
    axes[1].set(xlabel=r"Mahalanobis shift $\Delta$", ylabel=r"$P(RL\leq 50)$", ylim=(0, 1.02))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIGURES / "performance.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_stationary_anomaly_only(arl1: pd.DataFrame):
    """Plot delay and early-detection probability for the no-drift study."""
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "Oracle MCUSUM": "#5B5F97",
        "Static MCUSUM": "#2F6690",
        "KF-innovation MCUSUM": "#3A7D44",
        "DG-MCUSUM": "#D95D39",
    }
    markers = {
        "Oracle MCUSUM": "o",
        "Static MCUSUM": "s",
        "KF-innovation MCUSUM": "^",
        "DG-MCUSUM": "D",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for method, group in arl1.groupby("method", sort=False):
        axes[0].plot(
            group["delta"],
            group["mean_rl"],
            marker=markers[method],
            linewidth=2,
            color=colors[method],
            label=method,
        )
        axes[1].plot(
            group["delta"],
            group["p_signal_50"],
            marker=markers[method],
            linewidth=2,
            color=colors[method],
            label=method,
        )
    axes[0].set(
        xlabel=r"Persistent Mahalanobis shift $\Delta$",
        ylabel="Average run length",
    )
    axes[0].set_yscale("log")
    axes[1].set(
        xlabel=r"Persistent Mahalanobis shift $\Delta$",
        ylabel=r"$P(RL\leq 50)$",
        ylim=(0, 1.02),
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(
        FIGURES / "stationary_anomaly_only.png", dpi=240, bbox_inches="tight"
    )
    plt.close(fig)


def illustrative_path(config: StudyConfig, limits: dict, seed: int = 7):
    """Draw one correlated four-variable digital-twin trajectory."""
    rng = np.random.default_rng(seed)
    p, total, tau, delta = config.p, 280, config.burn_in, 1.5
    rho = 0.45
    sigma = (1.0 - rho) * np.eye(p) + rho * np.ones((p, p))
    chol = np.linalg.cholesky(sigma)
    direction = np.ones(p) / math.sqrt(p)
    level = np.zeros(p)
    slope = np.zeros(p)
    y_white, drift_white = [], []
    for t in range(total):
        level = level + slope + math.sqrt(config.q_level_data) * rng.normal(size=p)
        slope = slope + math.sqrt(config.q_slope_data) * rng.normal(size=p)
        shift = delta * direction if t >= tau else 0.0
        y_white.append(level + shift + rng.normal(size=p))
        drift_white.append(level.copy())
    y_white = np.asarray(y_white)
    drift_white = np.asarray(drift_white)
    raw = y_white @ chol.T
    raw_drift = drift_white @ chol.T

    # For a readable illustration, replay each method with the exact same path.
    def replay(method):
        fl = np.zeros(p)
        fs = np.zeros(p)
        p00, p01, p11 = 10.0, 0.0, 1.0
        s = np.zeros(p)
        confirming = False
        cs = np.zeros(p)
        age = 0
        al = np.zeros(p)
        ab = np.zeros(p)
        ap00 = ap01 = ap11 = 0.0
        stats = []
        # use the first burn-in part to initialise the filter
        for t, y in enumerate(y_white):
            pl, ps = fl + fs, fs
            pm00 = p00 + 2 * p01 + p11 + config.q_level_filter
            pm01 = p01 + p11
            pm11 = p11 + config.q_slope_filter
            f = pm00 + 1.0
            z = (y - pl) / math.sqrt(f)
            k0, k1 = pm00 / f, pm01 / f
            fl, fs = pl + k0 * (y - pl), ps + k1 * (y - pl)
            p00, p01, p11 = (1 - k0) * pm00, (1 - k0) * pm01, pm11 - k1 * pm01
            if t < config.burn_in:
                stats.append(0.0)
                continue
            if method == "kf":
                sn, st = crosier_step(s[None, :], z[None, :], config.k_primary)
                s = sn[0]
                stats.append(float(st[0]))
            else:
                sn, st = crosier_step(s[None, :], z[None, :], config.k_primary)
                if not confirming:
                    s = sn[0]
                    if st[0] > config.warning_limit:
                        confirming = True
                        cs[:] = 0.0
                        age = 0
                        al, ab = pl.copy(), ps.copy()
                        ap00, ap01, ap11 = pm00, pm01, pm11
                else:
                    al = al + ab
                    ap00, ap01, ap11 = (
                        ap00 + 2 * ap01 + ap11 + config.q_level_filter,
                        ap01 + ap11,
                        ap11 + config.q_slope_filter,
                    )
                if confirming:
                    zc = (y - al) / math.sqrt(ap00 + 1.0)
                    cn, ct = crosier_step(cs[None, :], zc[None, :], config.k_confirmation)
                    cs = cn[0]
                    age += 1
                    stats.append(float(ct[0]))
                    if (ct[0] <= 1e-12 and age >= 2) or age >= config.confirmation_horizon:
                        confirming = False
                        s[:] = 0.0
                        cs[:] = 0.0
                else:
                    stats.append(0.0)
        return np.asarray(stats)

    kf_stat = replay("kf")
    dg_stat = replay("dg")
    x = np.arange(total)
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.0), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
    axes[0].plot(x, raw[:, 0], color="#2F6690", lw=1.0, label="Observed variable 1")
    axes[0].plot(x, raw_drift[:, 0], color="#111111", lw=2.0, label="Legitimate drift")
    axes[0].axvline(tau, color="#D95D39", ls="--", lw=1.5, label="Anomaly onset")
    axes[0].set_ylabel("Raw scale")
    axes[0].legend(loc="upper left", ncol=3, frameon=False)
    axes[1].plot(x, kf_stat, color="#3A7D44", lw=1.5, label="KF-innovation MCUSUM")
    axes[1].plot(x, dg_stat, color="#D95D39", lw=1.8, label="DG-MCUSUM confirmation")
    axes[1].axhline(limits["kf"], color="#3A7D44", ls=":", lw=1.2)
    axes[1].axhline(limits["dg"], color="#D95D39", ls=":", lw=1.2)
    axes[1].axvline(tau, color="#D95D39", ls="--", lw=1.5)
    axes[1].set(xlabel="Time", ylabel="Chart statistic")
    axes[1].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "illustrative_path.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_latex_macros(limits, arl0, arl1):
    dg0 = arl0.loc[arl0.key == "dg"].iloc[0]
    kf0 = arl0.loc[arl0.key == "kf"].iloc[0]
    rows = [
        f"\\newcommand{{\\DGlimit}}{{{limits['dg']:.2f}}}",
        f"\\newcommand{{\\KFlimit}}{{{limits['kf']:.2f}}}",
        f"\\newcommand{{\\DGarlzero}}{{{dg0.mean_rl:.1f}}}",
        f"\\newcommand{{\\KFarlzero}}{{{kf0.mean_rl:.1f}}}",
    ]
    for d, suffix in [(0.5, "Small"), (1.0, "Medium"), (1.5, "Large"), (2.0, "VeryLarge")]:
        dg = arl1[(arl1.key == "dg") & (arl1.delta == d)].iloc[0]
        kf = arl1[(arl1.key == "kf") & (arl1.delta == d)].iloc[0]
        gain = 100.0 * (kf.mean_rl - dg.mean_rl) / kf.mean_rl
        rows.extend([
            f"\\newcommand{{\\DGarl{suffix}}}{{{dg.mean_rl:.1f}}}",
            f"\\newcommand{{\\KFarl{suffix}}}{{{kf.mean_rl:.1f}}}",
            f"\\newcommand{{\\DGgain{suffix}}}{{{gain:.1f}\\%}}",
        ])
    (RESULTS / "results_macros.tex").write_text("\n".join(rows) + "\n")


# %% [markdown]
# ## Execute the requested study
#
# `DG_MCUSUM_MODE=stationary` (default) runs only the new no-drift experiment.
# Use `DG_MCUSUM_MODE=drift` for the original experiment or `all` for both.
# Set `DG_MCUSUM_FAST=1` for a quick pipeline check with fewer replications.

# %%
if __name__ == "__main__":
    full = os.environ.get("DG_MCUSUM_FAST", "0") != "1"
    mode = os.environ.get("DG_MCUSUM_MODE", "stationary").strip().lower()
    if mode not in {"stationary", "drift", "all"}:
        raise ValueError("DG_MCUSUM_MODE must be stationary, drift, or all")

    if mode in {"drift", "all"}:
        cfg, limits, arl0_df, arl1_df = run_study(full=full)
        plot_performance(arl1_df)
        illustrative_path(cfg, limits)
        write_latex_macros(limits, arl0_df, arl1_df)
        print("Original drifting-baseline experiment")
        print("Control limits:", {k: round(v, 4) for k, v in limits.items()})
        print(
            "\nARL0 validation:\n",
            arl0_df[["method", "mean_rl", "ci_low", "ci_high"]].to_string(
                index=False
            ),
        )
        print(
            "\nARL1 main:\n",
            arl1_df[
                ["method", "delta", "mean_rl", "median_rl", "p_signal_50"]
            ].to_string(index=False),
        )

    if mode in {"stationary", "all"}:
        stationary_cfg, stationary_limits, stationary_arl0, stationary_arl1 = (
            run_stationary_anomaly_only(full=full)
        )
        plot_stationary_anomaly_only(stationary_arl1)
        print("\nStationary anomaly-only experiment: q_level=q_slope=0")
        print(
            "Matched control limits:",
            {k: round(v, 4) for k, v in stationary_limits.items()},
        )
        print(
            "\nIndependent ARL0 validation:\n",
            stationary_arl0[
                [
                    "method",
                    "mean_rl",
                    "ci_low",
                    "ci_high",
                    "relative_error_pct",
                    "within_10pct_target",
                ]
            ].to_string(index=False),
        )
        print(
            "\nStationary anomaly-only ARL1:\n",
            stationary_arl1[
                [
                    "method",
                    "delta",
                    "mean_rl",
                    "median_rl",
                    "p_signal_50",
                    "delay_ratio_vs_oracle",
                    "censor_rate",
                ]
            ].to_string(index=False),
        )
