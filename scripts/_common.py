"""Shared argument handling and output layout for the runnable scripts."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def default_outdir(name):
    return ROOT / "outputs" / name


def base_parser(description, name):
    """Parser with the two switches every script honours.

    ``--quick`` shrinks every replicate count so the pipeline can be exercised
    in minutes.  Quick-mode output is a smoke test, never evidence: the scripts
    stamp ``"quick": true`` into their metadata so a quick run cannot be
    mistaken for a full one later.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--outdir", type=Path, default=default_outdir(name),
                        help="directory for results (default: outputs/%s)" % name)
    parser.add_argument("--quick", action="store_true",
                        help="reduced budgets for a smoke test; NOT publication evidence")
    return parser


def start(name, args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("DG-MCUSUM | %s" % name)
    print("output   : %s" % outdir)
    print("mode     : %s" % ("QUICK SMOKE TEST" if args.quick else "FULL RUN"))
    print("python   : %s on %s" % (platform.python_version(), platform.platform()))
    print("=" * 72, flush=True)
    return outdir, time.perf_counter()


def finish(name, outdir, started, args, extra=None):
    elapsed = time.perf_counter() - started
    meta = {
        "script": name,
        "quick": bool(args.quick),
        "elapsed_seconds": elapsed,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if extra:
        meta.update(extra)
    (Path(outdir) / "script_metadata.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print("-" * 72)
    print("%s finished in %.1f s -> %s" % (name, elapsed, outdir))
    if args.quick:
        print("NOTE: quick mode. These numbers are a smoke test, not results.")
    print("-" * 72, flush=True)
