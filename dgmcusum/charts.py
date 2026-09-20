"""The Crosier multivariate CUSUM recursion, Eq. (6) of the manuscript.

Both stages of DG-MCUSUM use the same recursion; only the reference value and
the input increment differ.  Keeping it in one place is what makes the
"primary" and "confirmation" charts comparable by construction.
"""
from __future__ import annotations

import numpy as np

#: Guard against division by zero when the accumulated vector is exactly zero.
#: The archived implementation uses this constant, so it is kept verbatim.
_NORM_FLOOR = 1e-12

#: A confirmation statistic at or below this value counts as "returned to zero"
#: (Section 2: "return to zero means ||R_t|| <= eps_z with eps_z = 1e-12").
ZERO_TOLERANCE = 1e-12


def crosier_step(s, z, k):
    """One rotationally invariant Crosier MCUSUM update.

    Implements

        U_t = S_{t-1} + z_t,      C_t = ||U_t||
        S_t = 0                     if C_t <= k
            = (1 - k / C_t) U_t     if C_t >  k
        G_t = ||S_t||

    Parameters
    ----------
    s : (n, p) array
        Previous accumulator state, one row per replicate.
    z : (n, p) array
        Standardised increment for this observation.
    k : float
        Reference value (``k1`` for the primary chart, ``k2`` for confirmation).

    Returns
    -------
    (s_new, statistic) : ((n, p) array, (n,) array)
        The updated state and its Euclidean norm ``G_t``.

    Notes
    -----
    The shrinkage factor is clipped at zero, which reproduces the ``C_t <= k``
    branch without a separate mask, and is exactly what the C++ kernel does.
    """
    u = s + z
    norm_u = np.linalg.norm(u, axis=1)
    factor = np.maximum(0.0, 1.0 - k / np.maximum(norm_u, _NORM_FLOOR))
    s_new = u * factor[:, None]
    return s_new, norm_u * factor


def shift_direction(p, direction="dense"):
    """Unit vector ``d`` carrying the standardised shift, ``||d||_2 = 1``.

    ``dense`` spreads the displacement equally over all sensors, which is the
    direction used for every headline result (Section 3.2).  ``sparse`` and
    ``mixed`` are the holdout directions of the sensitivity check: same
    Mahalanobis magnitude, different physical pattern.
    """
    if direction == "dense":
        d = np.ones(p)
    elif direction == "sparse":
        d = np.zeros(p)
        d[0] = 1.0
    elif direction == "mixed":
        d = np.array([1.0 if j % 2 == 0 else -1.0 for j in range(p)])
    else:
        raise ValueError("direction must be dense, sparse, or mixed")
    return d / np.linalg.norm(d)
