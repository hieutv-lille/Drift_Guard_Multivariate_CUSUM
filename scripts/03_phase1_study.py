"""Layer 2, finite Phase-I estimation and nested calibration.

REIMPLEMENTATION: this follows the manuscript's specification, but its random
streams are not the ones behind the published Phase-I table. See PROVENANCE.md.

    python scripts/03_phase1_study.py --quick     # minutes
    python scripts/03_phase1_study.py             # hours

Results are written to ``outputs/phase1`` as CSV.
"""
import json

from _common import base_parser, start, finish

from dgmcusum.config import Phase1Config
from dgmcusum.phase1 import run_phase1_study

#: Known-parameter limits from the fixed-reference study, used as the naive
#: plug-in baseline whose distortion this layer quantifies.
PUBLISHED_KNOWN_LIMITS = {"kf": 9.5770263671875, "dg": 16.4572265625}


def main():
    parser = base_parser(__doc__, "phase1")
    parser.add_argument("--limits", type=str, default=None,
                        help="JSON file of known-parameter limits keyed by "
                             "kf and dg; defaults to the published values")
    args = parser.parse_args()
    outdir, t0 = start("Phase-I study", args)
    print("NOTE: reimplementation from the manuscript specification.")
    print("      Expect the reported findings, not the exact printed numbers.")
    print()

    known = dict(PUBLISHED_KNOWN_LIMITS)
    if args.limits:
        with open(args.limits) as handle:
            known.update(json.load(handle))
    print("known-parameter limits: %s" % known)

    cfg = Phase1Config()
    if args.quick:
        cfg = Phase1Config(reference_sizes=(150, 600), n_outer_calibration=6,
                           n_outer_validation=10, n_futures=4, max_steps=800,
                           bisection_iterations=8)

    estimates, summary, runs = run_phase1_study(
        outdir, known_limits=known, full=not args.quick, cfg=cfg)

    print()
    print("estimation summary:")
    print(summary.to_string(index=False))
    print()
    print("run-length behaviour:")
    print(runs[["m", "method", "limit_type", "arl0_mean_rl", "arl0_ci_low",
                "arl0_ci_high", "arl1_mean_rl"]].to_string(index=False))

    finish("Phase-I study", outdir, t0, args, {"known_limits": known})


if __name__ == "__main__":
    main()
