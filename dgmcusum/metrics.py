"""Run-length summaries and the fixed-window classification metrics.

Two conventions are used throughout the paper and enforced here.

*Censoring is retained.*  A path that never signals is recorded as
``max_steps + 1`` and enters the mean as ``max_steps``.  The reported figure is
therefore the restricted mean ``RMARL_H = E[min(T, H)]``, a lower bound on the
ARL, not an estimate obtained by discarding inconvenient paths.  Eq. (37)
states the exact relation to the unrestricted ARL.

*Delayed changes condition on survival.*  For ``tau > 1`` the estimand is
``E[min(T - tau + 1, H - tau + 1) | T >= tau]``, so each method is scored on
its own surviving cohort.  Those cohorts differ, which is why pre-change alarm
rates and survivor counts are exported alongside the means.
"""
from __future__ import annotations

import math

import numpy as np


def summarize_run_lengths(run_length, max_steps, tau=1, tail_threshold=50):
    """Standard summary block used by every table in the paper.

    Parameters
    ----------
    run_length : (n,) int array
        Raw stopping times, with non-signalling paths at ``max_steps + 1``.
    max_steps : int
        Evaluation horizon ``H``.
    tau : int
        Change point; ``tau=1`` for zero-state.
    tail_threshold : int
        Deadline ``C`` behind ``P(RL <= C)``; 50 in the selection loss.
    """
    run_length = np.asarray(run_length)
    survived = run_length >= tau
    delay = run_length[survived] - tau + 1
    cap = max_steps - tau + 1
    observed = np.minimum(delay, cap)
    n = len(observed)
    if n == 0:
        raise ValueError("no path survived to the change point")
    mean = float(observed.mean())
    sd = float(observed.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    p50 = float(np.mean(delay <= tail_threshold))
    return {
        "mean_rl": mean,
        "se_mean": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
        # ``median_rl`` is taken on the RAW delays, matching the archived
        # reference implementation that produced the published tables;
        # ``median_restricted`` applies the horizon cap.  They differ only
        # when a substantial fraction of paths is censored.
        "median_rl": float(np.median(delay)),
        "median_restricted": float(np.median(observed)),
        "p90_rl": float(np.quantile(observed, 0.90)),
        "p95_rl": float(np.quantile(observed, 0.95)),
        "p_signal_25": float(np.mean(delay <= 25)),
        "p_signal_50": p50,
        "p_signal_100": float(np.mean(delay <= 100)),
        "censor_rate": float(np.mean(delay > cap)),
        "n_generated": int(len(run_length)),
        "n_survived": n,
        "prechange_alarm_rate": float(1.0 - survived.mean()),
        "cap": cap,
    }


def restricted_mean(run_length, max_steps):
    """``RMARL_H = E[min(T, H)]`` with censored paths retained."""
    return float(np.minimum(np.asarray(run_length), max_steps).mean())


def paired_difference(rl_a, rl_b, max_steps):
    """Paired mean difference on common random numbers.

    Valid only when both charts saw the *same* paths and neither cohort was
    filtered, i.e. zero-state or in-control.  These are per-comparison
    intervals; the paper states explicitly that they carry no multiplicity
    adjustment.
    """
    a = np.minimum(np.asarray(rl_a), max_steps)
    b = np.minimum(np.asarray(rl_b), max_steps)
    diff = a - b
    n = len(diff)
    se = float(diff.std(ddof=1) / math.sqrt(n))
    mean = float(diff.mean())
    return {"mean_difference": mean, "se": se,
            "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se, "n": n}


def selection_objective(run_lengths_by_delta, horizon,
                        tail_threshold=50, tail_penalty=50.0):
    """Eq. (11): mean over shifts of ``E[min(T,H)] + 50 P(T > 50)``.

    The shifts share noise paths, so the standard error is computed on the
    per-path average across shifts rather than by pooling them as if they were
    independent replicates.
    """
    per_path = []
    for rl in run_lengths_by_delta:
        rl = np.asarray(rl)
        per_path.append(np.minimum(rl, horizon)
                        + tail_penalty * (rl > tail_threshold))
    z = np.mean(per_path, axis=0)
    return float(z.mean()), float(z.std(ddof=1) / math.sqrt(len(z)))


# --------------------------------------------------------------------------
# Fixed-window record-level classification (Section 4.1)
# --------------------------------------------------------------------------


def window_rates(run_length, window, tau=1):
    """``P{tau <= T <= tau + C - 1 | T >= tau}`` for one class."""
    run_length = np.asarray(run_length)
    survived = run_length >= tau
    if survived.sum() == 0:
        return 0.0, 0
    detected = (run_length >= tau) & (run_length <= tau + window - 1)
    return float(detected.sum()) / float(survived.sum()), int(survived.sum())


def classification_metrics(recall, false_positive_rate, prevalence=0.5):
    """Precision, F1 and friends at a declared fault prevalence.

    ``recall`` is ``r_C`` and ``false_positive_rate`` is ``f_C`` from
    Section 4.1.  Precision is defined as zero when no positive decision is
    made; the F1 expression is already well defined for ``0 < pi < 1``.
    """
    pi = float(prevalence)
    if not 0.0 < pi < 1.0:
        raise ValueError("prevalence must lie strictly between 0 and 1")
    r, f = float(recall), float(false_positive_rate)
    denom = pi * r + (1.0 - pi) * f
    precision = (pi * r / denom) if denom > 0 else 0.0
    f1_denom = pi * (1.0 + r) + (1.0 - pi) * f
    f1 = (2.0 * pi * r / f1_denom) if f1_denom > 0 else 0.0
    return {
        "precision": precision,
        "recall": r,
        "f1": f1,
        "specificity": 1.0 - f,
        "balanced_accuracy": 0.5 * (r + 1.0 - f),
        "fpr": f,
        "prevalence": pi,
    }


def roc_auc(scores_positive, scores_negative):
    """ROC-AUC via the rank (Mann-Whitney) identity, ties counted as one half."""
    pos = np.asarray(scores_positive, dtype=float)
    neg = np.asarray(scores_negative, dtype=float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1, dtype=float)
    # average ranks within tied groups
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    rank_sum = ranks[:n_pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores_positive, scores_negative):
    """Non-interpolated average precision (the AP column of Table 5)."""
    pos = np.asarray(scores_positive, dtype=float)
    neg = np.asarray(scores_negative, dtype=float)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tp = np.cumsum(labels)
    precision = tp / np.arange(1, len(labels) + 1)
    total_pos = labels.sum()
    if total_pos == 0:
        return 0.0
    return float((precision * labels).sum() / total_pos)


def binomial_ci(p_hat, n, z=1.96):
    """Wald interval for an alarm rate, used for the F1 Monte Carlo intervals."""
    se = math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / max(n, 1))
    return p_hat - z * se, p_hat + z * se
