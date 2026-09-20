"""Vectorised run-length simulation for every chart compared in the paper.

The random-draw order here is byte-for-byte the order used by the archived
reference implementation (``legacy/dg_mcusum_study.py``).  That is deliberate
and load-bearing: it is what lets this package reproduce the published tables
rather than merely produce statistically similar ones.  ``tests/`` asserts the
equality pathwise, so a well-meaning "tidy-up" that reorders an ``rng`` call
will fail the suite rather than silently move every number in the manuscript.

Methods
-------
``oracle``     subtracts the true latent level; infeasible ideal benchmark.
``static``     fixed target frozen at the end of warm-up; negative control.
``ewma``       EWMA-smoothed residual comparator.
``kf``         Kalman-innovation MCUSUM; the closest feasible comparator.
``dg``         DG-MCUSUM (Algorithm 1).
``dg_live``    ablation: two accumulators, but the second reads *live*
               residuals, so there is no protected reference.
``dg_freeze``  ablation: a single track whose filter is frozen on warning.

``dg_live`` and ``dg_freeze`` are reimplementations from the manuscript's
description of the ablation; see ``PROVENANCE.md``.
"""
from __future__ import annotations

import math

import numpy as np

from .charts import ZERO_TOLERANCE, crosier_step, shift_direction
from .config import EWMA_LAMBDA, ChartDesign, GaussianModel

METHODS = ("oracle", "static", "ewma", "kf", "dg", "dg_live", "dg_freeze")


def _advance_state(level, slope, q_level, q_slope, rng):
    """Local-linear state transition, Eqs. (2)-(3)."""
    level = level + slope + math.sqrt(q_level) * rng.normal(size=level.shape)
    slope = slope + math.sqrt(q_slope) * rng.normal(size=slope.shape)
    return level, slope


class Workload:
    """Candidate-episode accounting reported in the workload table.

    Counts how often protection opens, how long episodes last, how many are
    rejected, and what fraction of monitored observations are spent inside an
    episode.  These numbers are what justify the claim that candidates must
    stay internal rather than being surfaced to an operator.
    """

    def __init__(self):
        self.openings = 0
        self.rejections = 0
        self.protected_observations = 0
        self.observations_at_risk = 0
        self.episode_lengths = []

    def as_dict(self, setting):
        n_ep = max(self.openings, 1)
        return {
            "setting": setting,
            "candidate_openings": int(self.openings),
            "candidate_openings_per_1000":
                1000.0 * self.openings / max(self.observations_at_risk, 1),
            "mean_episode_duration":
                float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
            "candidate_rejection_probability": float(self.rejections) / n_ep,
            "protected_time_fraction":
                float(self.protected_observations) / max(self.observations_at_risk, 1),
            "observations_at_risk": int(self.observations_at_risk),
        }


def simulate_run_lengths(method, h, *, n_rep, max_steps, model=None,
                         design=None, delta=0.0, direction="dense", seed=1,
                         q_level_data=None, q_slope_data=None,
                         workload=None, tau=1):
    """Simulate ``n_rep`` run lengths for one chart at limit ``h``.

    Parameters
    ----------
    method : str
        One of :data:`METHODS`.
    h : float
        Control limit.  For DG this is the *confirmation* limit; the warning
        limit ``g`` lives in ``design``.
    n_rep, max_steps : int
        Number of replicates and the evaluation horizon ``H``.
    model : GaussianModel
        Data-generating and filter model.  The filter always uses
        ``model.q_level``/``model.q_slope``; the *data* may use different
        values via ``q_level_data``/``q_slope_data``, which is how the
        drift-misspecification check is run without retuning the chart.
    design : ChartDesign
        Chart parameters ``(k1, g, k2, L)``.
    delta : float
        Mahalanobis magnitude of the persistent shift.
    tau : int
        Change point.  ``tau=1`` is the zero-state case; ``tau>1`` injects the
        shift later, and the caller must condition on ``T >= tau``.
    workload : Workload, optional
        If given, candidate-episode statistics are accumulated into it.

    Returns
    -------
    run_length : (n_rep,) int array
        Stopping times.  Paths that never signal are recorded as
        ``max_steps + 1``; they are *retained*, never dropped, so that the
        restricted mean stays an honest lower bound on the ARL.
    """
    if method not in METHODS:
        raise ValueError("Unknown method: %s" % method)
    model = model or GaussianModel()
    design = design or ChartDesign()
    p = model.p
    q_level_data = model.q_level if q_level_data is None else q_level_data
    q_slope_data = model.q_slope if q_slope_data is None else q_slope_data

    rng = np.random.default_rng(seed)
    level = np.zeros((n_rep, p))
    slope = np.zeros((n_rep, p))
    filt_level = np.zeros((n_rep, p))
    filt_slope = np.zeros((n_rep, p))
    ewma_level = np.zeros((n_rep, p))
    static_center = np.zeros((n_rep, p))
    c00 = np.full(n_rep, model.c00_init)
    c01 = np.full(n_rep, model.c01_init)
    c11 = np.full(n_rep, model.c11_init)

    primary_s = np.zeros((n_rep, p))
    confirming = np.zeros(n_rep, dtype=bool)
    confirm_age = np.zeros(n_rep, dtype=int)
    confirm_s = np.zeros((n_rep, p))
    anchor_level = np.zeros((n_rep, p))
    anchor_slope = np.zeros((n_rep, p))
    anchor_c00 = np.zeros(n_rep)
    anchor_c01 = np.zeros(n_rep)
    anchor_c11 = np.zeros(n_rep)

    active = np.ones(n_rep, dtype=bool)
    run_length = np.full(n_rep, max_steps + 1, dtype=int)
    d_vec = shift_direction(p, direction)

    def kalman_predict_update(y):
        """Live filter step; mirrors :class:`kalman.LiveFilter` but in place."""
        nonlocal filt_level, filt_slope, c00, c01, c11
        pred_level = filt_level + filt_slope
        pred_slope = filt_slope
        pm00 = c00 + 2.0 * c01 + c11 + model.q_level
        pm01 = c01 + c11
        pm11 = c11 + model.q_slope
        innovation = y - pred_level
        f = pm00 + 1.0
        z = innovation / np.sqrt(f)[:, None]
        gain_level = pm00 / f
        gain_slope = pm01 / f
        filt_level = pred_level + gain_level[:, None] * innovation
        filt_slope = pred_slope + gain_slope[:, None] * innovation
        c00 = (1.0 - gain_level) * pm00
        c01 = (1.0 - gain_level) * pm01
        c11 = pm11 - gain_slope * pm01
        return z, pred_level, pred_slope, pm00, pm01, pm11

    # ---- Phase-I-like warm-up.  No chart statistic is accumulated. ----
    for _ in range(model.burn_in):
        level, slope = _advance_state(level, slope, q_level_data, q_slope_data, rng)
        y = level + rng.normal(size=(n_rep, p))
        if method in {"kf", "dg", "oracle", "dg_live", "dg_freeze"}:
            kalman_predict_update(y)
        elif method == "ewma":
            ewma_level += EWMA_LAMBDA * (y - ewma_level)
    # Give the fixed chart the strongest possible current target; any later
    # false signal is then caused by drift, not by Phase-I estimation error.
    static_center[:] = level

    for t in range(1, max_steps + 1):
        level, slope = _advance_state(level, slope, q_level_data, q_slope_data, rng)
        shift = delta * d_vec if t >= tau else 0.0
        y = level + shift + rng.normal(size=(n_rep, p))

        if method == "oracle":
            z = y - level
            s_new, statistic = crosier_step(primary_s, z, design.k1)
            primary_s[active] = s_new[active]

        elif method == "static":
            z = y - static_center
            s_new, statistic = crosier_step(primary_s, z, design.k1)
            primary_s[active] = s_new[active]

        elif method == "ewma":
            z = y - ewma_level
            ewma_level += EWMA_LAMBDA * z
            s_new, statistic = crosier_step(primary_s, z, design.k1)
            primary_s[active] = s_new[active]

        elif method == "kf":
            z = kalman_predict_update(y)[0]
            s_new, statistic = crosier_step(primary_s, z, design.k1)
            primary_s[active] = s_new[active]

        elif method == "dg_freeze":
            # Ablation: one track only.  While an episode is open the filter is
            # suspended, so a rejected candidate leaves a stale baseline -- the
            # failure mode that motivates keeping the live filter running.
            if confirming.any():
                frozen = confirming & active
                z = np.zeros((n_rep, p))
                fresh = ~frozen
                if fresh.any():
                    z_fresh = kalman_predict_update(y)[0]
                    z[fresh] = z_fresh[fresh]
                z[frozen] = (y[frozen] - anchor_level[frozen]) / \
                    np.sqrt(anchor_c00[frozen] + 1.0)[:, None]
                anchor_level[frozen] += anchor_slope[frozen]
            else:
                z = kalman_predict_update(y)[0]
            s_new, statistic = crosier_step(primary_s, z, design.k1)
            primary_s[active] = s_new[active]

        else:  # "dg" and "dg_live"
            z, pred_level, pred_slope, pm00, pm01, pm11 = kalman_predict_update(y)
            first_new, first_stat = crosier_step(primary_s, z, design.k1)
            eligible = (~confirming) & active
            primary_s[eligible] = first_new[eligible]

            if workload is not None:
                workload.observations_at_risk += int(active.sum())
                workload.protected_observations += int((confirming & active).sum())

            start = eligible & (first_stat > design.g)
            confirming[start] = True
            confirm_age[start] = 0
            confirm_s[start] = 0.0
            anchor_level[start] = pred_level[start]
            anchor_slope[start] = pred_slope[start]
            anchor_c00[start] = pm00[start]
            anchor_c01[start] = pm01[start]
            anchor_c11[start] = pm11[start]
            if workload is not None:
                workload.openings += int(start.sum())

            # Episodes opened earlier are propagated, never corrected (Eq. 7).
            old = confirming & (~start) & active
            anchor_level[old] += anchor_slope[old]
            n00 = anchor_c00[old] + 2.0 * anchor_c01[old] + anchor_c11[old] + model.q_level
            n01 = anchor_c01[old] + anchor_c11[old]
            n11 = anchor_c11[old] + model.q_slope
            anchor_c00[old] = n00
            anchor_c01[old] = n01
            anchor_c11[old] = n11

            candidate = confirming & active
            protected_z = np.zeros_like(z)
            if method == "dg":
                protected_z[candidate] = (
                    y[candidate] - anchor_level[candidate]
                ) / np.sqrt(anchor_c00[candidate] + 1.0)[:, None]
            else:
                # Ablation: a second accumulator, but fed the live residual.
                protected_z[candidate] = z[candidate]
            confirm_new, statistic = crosier_step(confirm_s, protected_z, design.k2)
            confirm_s[candidate] = confirm_new[candidate]
            confirm_age[candidate] += 1

            # Confirmation is checked before rejection, including on the last
            # allowed observation (Section 2).
            hit = candidate & (statistic > h)
            run_length[hit] = t
            active[hit] = False
            reject = candidate & active & (
                ((statistic <= ZERO_TOLERANCE) & (confirm_age >= 2))
                | (confirm_age >= design.L)
            )
            if workload is not None and reject.any():
                workload.rejections += int(reject.sum())
                workload.episode_lengths.extend(confirm_age[reject].tolist())
            confirming[reject] = False
            confirm_age[reject] = 0
            confirm_s[reject] = 0.0
            primary_s[reject] = 0.0   # rejection resets both chart statistics
            if not active.any():
                break
            continue

        hit = active & (statistic > h)
        run_length[hit] = t
        active[hit] = False
        if not active.any():
            break

    return run_length


def simulate_statistic_path(method, *, n_rep, max_steps, model=None, design=None,
                            delta=0.0, seed=1, tau=1):
    """Run the chart without stopping and return the per-step statistic.

    Used by the fixed-window classification analysis of Section 4.1, which
    needs ``M_C = max_t G_t`` over a window and therefore must not terminate on
    the first crossing.  Episode resets are kept exactly as in monitoring; a
    DG path with no active episode contributes a score of zero.
    """
    model = model or GaussianModel()
    design = design or ChartDesign()
    p = model.p
    rng = np.random.default_rng(seed)
    level = np.zeros((n_rep, p))
    slope = np.zeros((n_rep, p))
    filt_level = np.zeros((n_rep, p))
    filt_slope = np.zeros((n_rep, p))
    c00 = np.full(n_rep, model.c00_init)
    c01 = np.full(n_rep, model.c01_init)
    c11 = np.full(n_rep, model.c11_init)
    primary_s = np.zeros((n_rep, p))
    confirming = np.zeros(n_rep, dtype=bool)
    confirm_age = np.zeros(n_rep, dtype=int)
    confirm_s = np.zeros((n_rep, p))
    anchor_level = np.zeros((n_rep, p))
    anchor_slope = np.zeros((n_rep, p))
    anchor_c00 = np.zeros(n_rep)
    anchor_c01 = np.zeros(n_rep)
    anchor_c11 = np.zeros(n_rep)
    d_vec = shift_direction(p, "dense")
    out = np.zeros((max_steps, n_rep))

    for _ in range(model.burn_in):
        level, slope = _advance_state(level, slope, model.q_level, model.q_slope, rng)
        y = level + rng.normal(size=(n_rep, p))
        pred_level = filt_level + filt_slope
        pm00 = c00 + 2.0 * c01 + c11 + model.q_level
        pm01 = c01 + c11
        pm11 = c11 + model.q_slope
        innovation = y - pred_level
        f = pm00 + 1.0
        gl, gs = pm00 / f, pm01 / f
        filt_level = pred_level + gl[:, None] * innovation
        filt_slope = filt_slope + gs[:, None] * innovation
        c00, c01, c11 = (1.0 - gl) * pm00, (1.0 - gl) * pm01, pm11 - gs * pm01

    for t in range(1, max_steps + 1):
        level, slope = _advance_state(level, slope, model.q_level, model.q_slope, rng)
        shift = delta * d_vec if t >= tau else 0.0
        y = level + shift + rng.normal(size=(n_rep, p))
        pred_level = filt_level + filt_slope
        pred_slope = filt_slope
        pm00 = c00 + 2.0 * c01 + c11 + model.q_level
        pm01 = c01 + c11
        pm11 = c11 + model.q_slope
        innovation = y - pred_level
        f = pm00 + 1.0
        z = innovation / np.sqrt(f)[:, None]
        gl, gs = pm00 / f, pm01 / f
        filt_level = pred_level + gl[:, None] * innovation
        filt_slope = pred_slope + gs[:, None] * innovation
        c00, c01, c11 = (1.0 - gl) * pm00, (1.0 - gl) * pm01, pm11 - gs * pm01

        if design.method == "kf":
            primary_s, statistic = crosier_step(primary_s, z, design.k1)
            out[t - 1] = statistic
            continue

        first_new, first_stat = crosier_step(primary_s, z, design.k1)
        eligible = ~confirming
        primary_s[eligible] = first_new[eligible]
        start = eligible & (first_stat > design.g)
        confirming[start] = True
        confirm_age[start] = 0
        confirm_s[start] = 0.0
        anchor_level[start] = pred_level[start]
        anchor_slope[start] = pred_slope[start]
        anchor_c00[start] = pm00[start]
        anchor_c01[start] = pm01[start]
        anchor_c11[start] = pm11[start]

        old = confirming & (~start)
        anchor_level[old] += anchor_slope[old]
        n00 = anchor_c00[old] + 2.0 * anchor_c01[old] + anchor_c11[old] + model.q_level
        n01 = anchor_c01[old] + anchor_c11[old]
        n11 = anchor_c11[old] + model.q_slope
        anchor_c00[old], anchor_c01[old], anchor_c11[old] = n00, n01, n11

        protected_z = np.zeros_like(z)
        protected_z[confirming] = (
            y[confirming] - anchor_level[confirming]
        ) / np.sqrt(anchor_c00[confirming] + 1.0)[:, None]
        confirm_new, statistic = crosier_step(confirm_s, protected_z, design.k2)
        confirm_s[confirming] = confirm_new[confirming]
        confirm_age[confirming] += 1

        # Score is read before rejection; an inactive path scores zero.
        score = np.where(confirming, statistic, 0.0)
        out[t - 1] = score

        reject = confirming & (
            ((statistic <= ZERO_TOLERANCE) & (confirm_age >= 2))
            | (confirm_age >= design.L))
        confirming[reject] = False
        confirm_age[reject] = 0
        confirm_s[reject] = 0.0
        primary_s[reject] = 0.0

    return out
