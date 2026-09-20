"""Run the whole test suite with plain Python; no pytest required.

    python tests/run_tests.py

pytest works too (``python -m pytest tests/ -v``) and gives nicer output, but
a reviewer should not have to install anything extra to check the package
against the archived implementation.
"""
from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = ["test_equivalence", "test_kernel", "test_properties"]


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    passed, failed = 0, []
    started = time.perf_counter()
    for name in MODULES:
        print("\n%s" % name)
        print("-" * len(name))
        module = load(name)
        for attr in sorted(dir(module)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(module, attr)
            if not callable(fn):
                continue
            t0 = time.perf_counter()
            try:
                fn()
            except Exception:
                failed.append("%s::%s" % (name, attr))
                print("  FAIL %-52s" % attr)
                traceback.print_exc()
            else:
                passed += 1
                print("  ok   %-52s %5.1fs" % (attr, time.perf_counter() - t0))

    print("\n" + "=" * 68)
    print("%d passed, %d failed in %.1fs"
          % (passed, len(failed), time.perf_counter() - started))
    if failed:
        for name in failed:
            print("  FAILED %s" % name)
        print("=" * 68)
        return 1
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
