"""Layer 1, joint parameter search and independent validation.

Searches the 240 DG candidates plus four KF references, calibrating a limit for
every one, then locks the winners and validates them on fresh seed streams.

    python scripts/02_parameter_search.py            # full run (slow)
    python scripts/02_parameter_search.py --quick    # smoke test

Outputs land in ``outputs/search`` and feed Tables 5-7.
"""
from _common import base_parser, start, finish

from dgmcusum import fastkernel
from dgmcusum.config import SEARCH_PROTOCOL
from dgmcusum.search import run_search


def main():
    parser = base_parser(__doc__, "search")
    parser.add_argument("--backend", choices=["auto", "cpp", "numpy"],
                        default="auto", help="path evaluator backend")
    args = parser.parse_args()
    outdir, t0 = start("parameter search", args)

    info = fastkernel.backend_info()
    print("kernel backend: %s" % info["backend"])
    if info["backend"] == "numpy":
        print("  (no C++ compiler found; the NumPy fallback is slower but")
        print("   produces identical run lengths -- see tests/test_kernel.py)")
    if info["compile_error"]:
        print("  compiler note: %s" % info["compile_error"])

    protocol = dict(SEARCH_PROTOCOL)
    if args.quick:
        for key in list(protocol):
            if key.startswith("n_"):
                protocol[key] = 64
        protocol["promote_dg"] = 2
        protocol["bisection_iterations"] = 8
        protocol["horizon"] = 600

    designs, limits, tests = run_search(outdir, protocol=protocol)
    print("\nlocked designs:")
    for label, design in designs.items():
        print("  %-13s %-28s h=%.5f" % (label, design.key, limits[label]))
    headline = tests[tests.case.isin(["in_control", "zero_state", "late_change"])]
    print(headline[["case", "label", "delta", "mean", "ci_low", "ci_high",
                    "p_gt_50"]].to_string(index=False))

    finish("parameter search", outdir, t0, args,
           {"limits": limits, "kernel": info})


if __name__ == "__main__":
    main()
