"""Local-linear Kalman recursions, live and protected.

The manuscript's separable covariance assumption, ``P_t^- = C_t^- (x) Sigma``
with ``Sigma`` the measurement covariance, means the whole covariance recursion
collapses to three scalars ``(c00, c01, c11)`` that are shared by every
replicate and every whitened direction.  That is why the simulator can run
thousands of paths at once without ever forming a 2p x 2p matrix.

In whitened coordinates ``Sigma = I_p``, so the innovation covariance is
``F_t = (c00 + 1) I_p`` and whitening a residual is a division by
``sqrt(c00 + 1)``.  Working in these coordinates is exact, not an
approximation: it is the same chart applied to ``L_Sigma^{-1} X_t``.
"""
from __future__ import annotations

import numpy as np


def predict_covariance(c00, c01, c11, q_level, q_slope):
    """One-step covariance prediction of Eq. (7): ``A C A' + Q`` factors.

    With ``A = [[I, I], [0, I]]`` the level-block recursion is

        c00^- = c00 + 2 c01 + c11 + q_level
        c01^- = c01 + c11
        c11^- = c11 + q_slope

    Accepts scalars or arrays; used for both the live filter and the protected
    predictor, since protection differs only by the absence of an update.
    """
    pm00 = c00 + 2.0 * c01 + c11 + q_level
    pm01 = c01 + c11
    pm11 = c11 + q_slope
    return pm00, pm01, pm11


def innovation_factor(pm00):
    """Scalar ``f_t = c00^- + 1`` such that ``F_t = f_t Sigma`` (Section 2)."""
    return pm00 + 1.0


class LiveFilter:
    """Vectorised local-linear filter shared by all replicates.

    One instance tracks ``n`` independent paths.  The state means differ across
    paths; the covariance factors do not, but they are kept per path so that
    the same object also serves the Phase-I study, where each fitted model has
    its own ``q``.
    """

    def __init__(self, n, p, q_level, q_slope,
                 c00_init=10.0, c01_init=0.0, c11_init=1.0):
        self.n, self.p = n, p
        self.q_level, self.q_slope = q_level, q_slope
        self.level = np.zeros((n, p))
        self.slope = np.zeros((n, p))
        self.c00 = np.full(n, float(c00_init))
        self.c01 = np.full(n, float(c01_init))
        self.c11 = np.full(n, float(c11_init))

    def step(self, y):
        """Predict, whiten, then assimilate ``y`` unconditionally.

        Returns
        -------
        z : (n, p) array
            Standardised live innovation ``z_t`` of Eq. (5).
        pred_level, pred_slope : (n, p) arrays
            The *pre-update* prediction.  DG copies these when it opens a
            candidate, which is why they are returned rather than discarded.
        pm00, pm01, pm11 : (n,) arrays
            The pre-update covariance factors, copied alongside the mean.
        """
        pred_level = self.level + self.slope
        pred_slope = self.slope
        pm00, pm01, pm11 = predict_covariance(
            self.c00, self.c01, self.c11, self.q_level, self.q_slope)

        residual = y - pred_level
        f = innovation_factor(pm00)
        z = residual / np.sqrt(f)[:, None]

        gain_level = pm00 / f
        gain_slope = pm01 / f
        self.level = pred_level + gain_level[:, None] * residual
        self.slope = pred_slope + gain_slope[:, None] * residual
        self.c00 = (1.0 - gain_level) * pm00
        self.c01 = (1.0 - gain_level) * pm01
        self.c11 = pm11 - gain_slope * pm01
        return z, pred_level, pred_slope, pm00, pm01, pm11


class ProtectedPredictor:
    """The reference that suspect observations may not move (Eq. 7).

    Opened by copying the live filter's *pre-update* prediction, then advanced
    by the dynamics alone.  It is never corrected, which is the whole point:
    the evidence for a persistent shift is preserved instead of being absorbed.

    All arrays are full length ``n``; only the rows flagged as active matter,
    which keeps the update branch-free and vectorised.
    """

    def __init__(self, n, p, q_level, q_slope):
        self.q_level, self.q_slope = q_level, q_slope
        self.level = np.zeros((n, p))
        self.slope = np.zeros((n, p))
        self.c00 = np.zeros(n)
        self.c01 = np.zeros(n)
        self.c11 = np.zeros(n)

    def open(self, mask, pred_level, pred_slope, pm00, pm01, pm11):
        """Copy the pre-measurement prediction into the protected reference."""
        self.level[mask] = pred_level[mask]
        self.slope[mask] = pred_slope[mask]
        self.c00[mask] = pm00[mask]
        self.c01[mask] = pm01[mask]
        self.c11[mask] = pm11[mask]

    def advance(self, mask):
        """Propagate, with no measurement update, for the flagged rows."""
        self.level[mask] += self.slope[mask]
        n00, n01, n11 = predict_covariance(
            self.c00[mask], self.c01[mask], self.c11[mask],
            self.q_level, self.q_slope)
        self.c00[mask] = n00
        self.c01[mask] = n01
        self.c11[mask] = n11

    def standardised_residual(self, y, mask, out):
        """Protected standardised innovation ``z~_t`` of Eq. (8).

        Written into ``out`` only for the flagged rows, so inactive paths keep
        a zero increment and their confirmation accumulator stays put.
        """
        f = np.sqrt(self.c00[mask] + 1.0)
        out[mask] = (y[mask] - self.level[mask]) / f[:, None]
        return out


def protected_variance_factor(c00, c01, c11, q_level, q_slope, age):
    """Closed-form forecast factor ``f~_j`` of Eq. (36), for checking.

    With ``C_c^- = [[a, b], [b, d]]`` at the copy time,

        f~_j = 1 + a + 2 j b + j^2 d + j q_level + q_slope j(j-1)(2j-1)/6.

    Used by the test suite to confirm that the iterated recursion in
    :class:`ProtectedPredictor` matches the algebra stated in the appendix.
    """
    a, b, d = c00, c01, c11
    j = float(age)
    return (1.0 + a + 2.0 * j * b + j * j * d + j * q_level
            + q_slope * j * (j - 1.0) * (2.0 * j - 1.0) / 6.0)
