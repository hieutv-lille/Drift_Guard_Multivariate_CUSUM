"""Run every layer in order, forwarding --quick to each stage.

    python scripts/run_all.py --quick                       # end-to-end smoke test
    python scripts/run_all.py --archive data/gt_2015.csv    # full reproduction

Layer 3 is skipped with a clear message when the UCI archive is absent, so a
missing external dataset never fails the whole pipeline.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    ("01_mechanism_study.py", "Layer 1: mechanism, ablation, stationary"),
    ("02_parameter_search.py", "Layer 1: joint parameter search"),
    ("03_phase1_study.py", "Layer 2: finite Phase-I estimation"),
    ("04_industrial_study.py", "Layer 3: gas-turbine stress test"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="reduced budgets everywhere; smoke test only")
    parser.add_argument("--archive", type=str, default=None,
                        help="path to gt_2015.csv for Layer 3")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="stage numbers to skip, e.g. --skip 02 04")
    args = parser.parse_args()

    started = time.perf_counter()
    results = []
    for script, label in STAGES:
        number = script[:2]
        if number in args.skip:
            print()
            print("### SKIPPED %s (%s)" % (number, label))
            results.append((label, "skipped"))
            continue
        command = [sys.executable, str(HERE / script)]
        if args.quick:
            command.append("--quick")
        if script.startswith("04") and args.archive:
            command += ["--archive", args.archive]
        print()
        print("#" * 72)
        print("### %s" % label)
        print("#" * 72)
        sys.stdout.flush()
        code = subprocess.call(command, cwd=str(HERE))
        if code == 2 and script.startswith("04"):
            print()
            print("Layer 3 skipped: the UCI archive is not present.")
            print("See data/README.md to download it, then rerun with --archive.")
            results.append((label, "skipped (no archive)"))
            continue
        if code != 0:
            print()
            print("FAILED: %s (exit %d)" % (label, code))
            results.append((label, "FAILED"))
            raise SystemExit(code)
        results.append((label, "ok"))

    print()
    print("=" * 72)
    print("SUMMARY after %.1f s" % (time.perf_counter() - started))
    for label, status in results:
        print("  %-45s %s" % (label, status))
    print("=" * 72)


if __name__ == "__main__":
    main()
