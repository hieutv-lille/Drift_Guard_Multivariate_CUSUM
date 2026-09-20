"""Layer 1, fixed-reference mechanism study.

Reproduces the control limits, in-control validation, detection delays,
direction and drift-misspecification checks, the component ablation, the
candidate workload and the stationary boundary case.

    python scripts/01_mechanism_study.py            # full run
    python scripts/01_mechanism_study.py --quick    # smoke test

Results are written to ``outputs/mechanism`` as CSV.
"""
from _common import base_parser, start, finish

from dgmcusum.mechanism import (run_ablation, run_mechanism_study,
                                run_stationary_study, run_workload)


def main():
    parser = base_parser(__doc__, "mechanism")
    parser.add_argument("--skip-stationary", action="store_true",
                        help="omit the stationary boundary case")
    args = parser.parse_args()
    outdir, t0 = start("mechanism study", args)
    full = not args.quick

    print("[1/4] calibrating limits and validating in control ...", flush=True)
    limits, arl0, arl1 = run_mechanism_study(outdir, full=full)
    print(arl0[["method", "h", "mean_rl", "ci_low", "ci_high"]].to_string(index=False))

    print("\n[2/4] component ablation ...", flush=True)
    ablation = run_ablation(outdir, full=full)
    print(ablation[["method", "h", "arl0_mean_rl", "arl1_mean_rl",
                    "arl2_mean_rl"]].to_string(index=False))

    print("\n[3/4] candidate workload ...", flush=True)
    workload = run_workload(outdir, limits=limits, full=full)
    print(workload.to_string(index=False))

    if not args.skip_stationary:
        print("\n[4/4] stationary boundary case ...", flush=True)
        run_stationary_study(outdir, full=full)

    finish("mechanism study", outdir, t0, args, {"limits": limits})


if __name__ == "__main__":
    main()
