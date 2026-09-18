# Copyright 2026 ddpwm
# SPDX-License-Identifier: MIT
"""Tiny test runner, so the suite also works without pytest installed.

    python tests/run_all.py                 # every tests/test_*.py
    python tests/run_all.py tests/test_configs.py
    python tests/run_all.py --list

Collects every callable named test_* in the given files and runs it. pytest (`pytest tests/`) does
the same thing; this runner exists for the machines where pytest is not installed.
"""

import importlib.util
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TESTS = os.path.dirname(os.path.abspath(__file__))


def load(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(f"tests.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"tests.{name}"] = module
    spec.loader.exec_module(module)
    return module


def test_files(argv):
    if argv:
        return [os.path.abspath(p) for p in argv]
    return [
        os.path.join(TESTS, f)
        for f in sorted(os.listdir(TESTS))
        if f.startswith("test_") and f.endswith(".py")
    ]


def main(argv):
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__)
        return 0
    want_list = "--list" in argv
    files = test_files([a for a in argv if not a.startswith("-")])
    if want_list:
        for f in files:
            print(os.path.relpath(f, ROOT))
        return 0

    n_pass = n_fail = n_skip = 0
    for path in files:
        print(f"\n== {os.path.relpath(path, ROOT)}")
        try:
            module = load(path)
        except Exception:
            print("  FAIL (import) " + traceback.format_exc().replace("\n", "\n  "))
            n_fail += 1
            continue
        names = [n for n in dir(module) if n.startswith("test_") and callable(getattr(module, n))]
        for name in names:
            try:
                getattr(module, name)()
                print(f"  pass  {name}")
                n_pass += 1
            except Exception:
                print(f"  FAIL  {name}\n" + traceback.format_exc().replace("\n", "\n        "))
                n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
