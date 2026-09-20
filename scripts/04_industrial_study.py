"""Layer 3, semi-synthetic gas-turbine stress test.

REIMPLEMENTATION: follows the manuscript's specification, with different random
streams from the published run. See PROVENANCE.md.

The UCI archive is NOT bundled. Download it first -- see data/README.md -- then

    python scripts/04_industrial_study.py --archive data/gt_2015.csv

Results are written to ``outputs/industrial`` as CSV and JSON.  Exit code 2
means the archive is missing, which ``run_all.py`` treats as "skip this
layer" rather than a failure.
"""
from pathlib import Path

from _common import base_parser, start, finish

from dgmcusum.config import IndustrialConfig
from dgmcusum.industrial import MissingArchive, run_industrial_study


def main():
    parser = base_parser(__doc__, "industrial")
    parser.add_argument("--archive", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data",
                        help="gt_2015.csv, or the directory containing it")
    parser.add_argument("--no-sensitivity", action="store_true",
                        help="primary specification only; skip the cap and "
                             "shrinkage variants")
    args = parser.parse_args()
    outdir, t0 = start("industrial study", args)
    print("NOTE: reimplementation from the manuscript specification.")
    print("      Shifts are injected; the archive has no verified fault times.")
    print()

    cfg = IndustrialConfig()
    if args.quick:
        cfg = IndustrialConfig(n_limit_calibration=120, n_evaluation=200,
                               evaluation_horizon=1000,
                               deltas=(0.0, 4.0, 8.0),
                               bisection_iterations=9)

    try:
        summary, main_frame, tail = run_industrial_study(
            args.archive, outdir, cfg=cfg, sensitivity=not args.no_sensitivity)
    except MissingArchive as exc:
        print("ERROR: %s" % exc)
        raise SystemExit(2)

    print()
    print("selected model: %s (q_level=%.4f)"
          % (summary["selected_model"], summary["q_level_hat"]))
    print("covariance condition number: %.0f" % summary["sigma_condition_number"])
    print("calibrated limits: %s" % summary["calibrated_limits"])
    print()
    print(main_frame[["method", "delta", "mean_rl", "median_rl",
                      "p95_rl"]].to_string(index=False))

    finish("industrial study", outdir, t0, args, {"archive": str(args.archive)})


if __name__ == "__main__":
    main()
