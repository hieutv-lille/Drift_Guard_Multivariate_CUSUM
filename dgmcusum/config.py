"""Every constant used by the study, with the manuscript location that fixes it.

Nothing in this module is tuned at run time.  The values below are the ones the
paper declares *before* any simulation, so that a reader can check a number in
the text against a single line here.  Section numbers refer to the manuscript
``DG_MCUSUM_revised_red.tex``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------
# Layer 1 - known-parameter Gaussian model (Section 3.2)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianModel:
    """Data-generating local-linear model of Eqs. (2)-(3).

    ``q_level``/``q_slope`` are the drift ratios; measurement errors are
    independent standardised Gaussians, so the whitened representation used by
    the simulator is exact rather than an approximation.
    """

    p: int = 4                      # Section 3.2: four monitored variables
    q_level: float = 0.002          # Section 3.2
    q_slope: float = 2e-6           # Section 3.2
    burn_in: int = 150              # Section 3.2: warm-up before monitoring

    # Working filter initialisation: P_0 = diag(10, 1) (x) Sigma (Section 3.2).
    # Deliberately not the exact-initialisation assumption of Proposition 1.
    c00_init: float = 10.0
    c01_init: float = 0.0
    c11_init: float = 1.0


@dataclass(frozen=True)
class ChartDesign:
    """A fully specified chart.

    ``method`` is one of ``oracle``, ``static``, ``ewma``, ``kf``, ``dg``.
    For the single-stage charts only ``k1`` is meaningful.
    """

    method: str = "dg"
    k1: float = 0.50                # primary reference value
    g: float = 1.50                 # warning limit
    k2: float = 0.25                # confirmation reference value
    L: int = 15                     # protection horizon

    @property
    def key(self) -> str:
        if self.method != "dg":
            return "%s_k%g" % (self.method, self.k1)
        return "dg_k%g_g%g_c%g_L%d" % (self.k1, self.g, self.k2, self.L)


#: Fixed-reference DG configuration used for the mechanism and boundary checks
#: (Section 3.2: "A fixed-reference DG configuration, (k1,g,k2,L)=(0.5,1.5,0.25,15)").
DG_REFERENCE = ChartDesign("dg", 0.50, 1.50, 0.25, 15)

#: Design selected by the joint search (Section 4.1: "(0.5,1,0.1,20)").
DG_SELECTED = ChartDesign("dg", 0.50, 1.00, 0.10, 20)

#: KF comparator at its fixed reference value, and at the selected one (k=0.75).
KF_REFERENCE = ChartDesign("kf", 0.50)
KF_SELECTED = ChartDesign("kf", 0.75)

#: EWMA smoothing constant for the residual comparator used in the ablation.
EWMA_LAMBDA = 0.08

#: Nominal in-control target shared by every calibrated design (Section 3.2).
TARGET_ARL0 = 500.0


# --------------------------------------------------------------------------
# Layer 1 - mechanism study budgets (Section 3.2, paragraph "The fixed-reference
# mechanism comparisons use ...")
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MechanismBudget:
    n_calibration: int = 2500       # calibration paths for single-stage charts
    n_dg_refinement: int = 8000     # extra DG calibration refinement
    n_arl0_validation: int = 7000   # independent in-control validation
    n_arl1: int = 5000              # paths per shift
    n_sensitivity: int = 3500       # direction / drift-covariance checks
    max_steps: int = 3000           # evaluation horizon
    calibration_steps: int = 2500   # horizon used inside calibration

    #: Mahalanobis shift grid of Section 3.2.
    deltas: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)


#: Scenario-specific seeds.  Reproduced verbatim from the archived reference
#: implementation (``legacy/dg_mcusum_study.py``) so that this package returns
#: the numbers printed in the manuscript, not merely numbers of the same kind.
MECHANISM_SEEDS: Dict[str, int] = {
    "calibration": 8128,
    "arl0": 2024,
    "arl1": 4096,
    "direction": 555,
    "mismatch": 777,
    "design": 909,
    "illustrative": 7,
}


# --------------------------------------------------------------------------
# Layer 1 - joint parameter search (Section 3.2, "We search 240 configurations")
# --------------------------------------------------------------------------

#: 4 x 4 x 3 x 5 = 240 DG candidates.
SEARCH_GRID: Dict[str, List] = {
    "k1": [0.25, 0.5, 0.75, 1.0],
    "g": [0.75, 1.0, 1.5, 2.0],
    "k2": [0.1, 0.25, 0.5],
    "L": [5, 10, 15, 20, 30],
}

#: Complete pre-registered search protocol.  Fixed before any simulation; the
#: final-test streams never feed back into either selection stage.
SEARCH_PROTOCOL: Dict = {
    "target_arl0": 500.0,
    "horizon": 6000,                 # H in RMARL_H
    "tail_threshold": 50,            # the deadline inside the selection loss
    "tail_penalty": 50.0,            # Eq. (11) penalty weight
    "selection_deltas": [1.0, 1.5, 2.0],
    "test_deltas": [0.5, 0.75, 1.0, 1.5, 2.0],
    "late_change_tau": 201,          # delayed-change scenario
    "n_coarse_cal": 640,
    "n_coarse_tune": 1000,
    "n_refine_cal": 4000,
    "n_refine_tune": 4000,
    "promote_dg": 12,                # best coarse DG designs entering refinement
    "n_final_cal": 20000,
    "n_ic_test": 20000,
    "n_ooc_test": 10000,
    "n_mismatch": 10000,
    "bisection_iterations": 15,
    "seeds": {
        "coarse_cal": 51001, "coarse_tune": 52001,
        "refine_cal": 61001, "refine_tune": 62001,
        "final_cal": 71001, "ic_test": 81001,
        "ooc_test": 91001, "late_test": 101001,
        "mismatch_half": 111001, "mismatch_double": 121001,
    },
}

#: Classification analysis of Section 4.1 (fixed-window ML metrics).
CLASSIFICATION = {
    "windows": [50, 200],
    "n_normal": 20000,
    "n_shifted": 10000,
    "prevalences": [0.5, 0.01],
    "deltas": [1.0, 1.5, 2.0],
    "seeds": {"normal": 131001, "shifted": 141001},
}


# --------------------------------------------------------------------------
# Layer 2 - finite Phase-I estimation (Section 3.3)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase1Config:
    """Repeated-sampling experiment of Section 3.3.

    ``n_outer_calibration`` fitted reference samples each generate
    ``n_futures`` Phase-II futures; the limit is chosen on those, and a fresh
    set of ``n_outer_validation`` reference samples validates it.
    """

    reference_sizes: Tuple[int, ...] = (150, 300, 600)   # m in {150,300,600}
    n_outer_calibration: int = 60       # "60 fitted reference samples"
    n_outer_validation: int = 120       # "120 new reference samples"
    n_futures: int = 8                  # "eight futures each"
    max_steps: int = 3000
    delta_out_of_control: float = 1.0   # "out-of-control validation uses Delta=1"
    bisection_iterations: int = 13

    # Profile-likelihood estimator (Section 3.3 and Appendix A).
    diffuse_prefix: int = 20            # discard first 20 errors, so d = 21
    c00_diffuse: float = 100.0
    c11_diffuse: float = 1.0
    #: Multiple starting points in (log q_level, log q_slope) coordinates.
    log10_starts: Tuple[Tuple[float, float], ...] = (
        (-2.7, -5.7), (-1.5, -4.0), (-3.5, -7.0), (-0.5, -3.0), (-4.5, -8.0),
    )
    #: Bounds on ``(log10 q_level, log10 q_slope)``.  Deliberately wide: at
    #: m = 150 the likelihood is nearly flat in ``q_level`` and the reported
    #: interquartile range reaches about 7e-8, so a tighter lower bound would
    #: truncate the sampling distribution and inflate the boundary-hit rate.
    log10_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (-12.0, 1.0), (-12.0, 0.0),
    )
    #: A boundary hit is recorded when an estimate lands within this distance
    #: (in log10 units) of a bound.
    boundary_tol: float = 1e-3


PHASE1_SEEDS: Dict[str, int] = {
    "calibration": 20240501,
    "validation": 20240502,
    "futures": 20240503,
}


# --------------------------------------------------------------------------
# Layer 3 - semi-synthetic gas-turbine stress test (Section 3.4)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IndustrialConfig:
    """Gas-turbine bootstrap of Section 3.4.

    The archive is the 2015 file of the UCI Gas Turbine CO and NOx Emission
    Data Set (DOI 10.24432/C5WC95).  See ``data/README.md``; the raw file is
    not redistributed here.
    """

    #: Monitored channels, in the order used for Cholesky whitening.
    variables: Tuple[str, ...] = ("GTEP", "TIT", "TAT", "CDP")
    n_observations: int = 7384          # hourly observations in the 2015 file
    phase1_size: int = 1500             # "The first 1,500 observations form Phase I"
    block_length: int = 12              # "Circular blocks of 12"
    cap_quantile: float = 0.995         # primary radial cap
    cap_variants: Tuple[float, ...] = (0.990, 0.995, 0.999)
    shrinkage_gamma: float = 0.01       # "1% shrinkage toward a scaled identity"
    n_limit_calibration: int = 1200     # "Limits use 1,200 bootstrap paths"
    n_evaluation: int = 2500            # "2,500 new paths per method and shift"
    evaluation_horizon: int = 5000      # "a 5,000-hour horizon"
    deltas: Tuple[float, ...] = (0.0, 2.0, 4.0, 5.0, 6.0, 8.0)
    bisection_iterations: int = 15


INDUSTRIAL_SEEDS: Dict[str, int] = {
    "calibration": 31337,
    "evaluation": 424242,
    "sensitivity": 515151,
}


# --------------------------------------------------------------------------
# Human-readable method labels, used in every exported table.
# --------------------------------------------------------------------------

METHOD_LABELS: Dict[str, str] = {
    "oracle": "Oracle MCUSUM",
    "static": "Static MCUSUM",
    "ewma": "EWMA-residual MCUSUM",
    "kf": "KF-innovation MCUSUM",
    "dg": "DG-MCUSUM",
    "dg_live": "Two-stage live-residual ablation",
    "dg_freeze": "Single-track frozen-filter ablation",
}


def describe() -> Dict:
    """Return every configuration block as plain data, for run metadata."""
    return {
        "model": asdict(GaussianModel()),
        "mechanism_budget": asdict(MechanismBudget()),
        "mechanism_seeds": dict(MECHANISM_SEEDS),
        "search_grid": SEARCH_GRID,
        "search_protocol": SEARCH_PROTOCOL,
        "classification": CLASSIFICATION,
        "phase1": asdict(Phase1Config()),
        "phase1_seeds": dict(PHASE1_SEEDS),
        "industrial": asdict(IndustrialConfig()),
        "industrial_seeds": dict(INDUSTRIAL_SEEDS),
        "designs": {
            "DG_reference": asdict(DG_REFERENCE),
            "DG_selected": asdict(DG_SELECTED),
            "KF_reference": asdict(KF_REFERENCE),
            "KF_selected": asdict(KF_SELECTED),
        },
        "target_arl0": TARGET_ARL0,
    }
