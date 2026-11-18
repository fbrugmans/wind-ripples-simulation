"""Reproduction test: the first 60 s of model time of seed 271 must be bit-identical to the article run.

Usage:  python3 tests/test_reproduce.py
Runs the article model for 60 s (about half a minute on a laptop) and compares the bed height, impacts, creep and
wind-driven sliding after 30 and 60 s with tests/reference_z271_60s.npz (taken from the article run _SR0).
Bit-identity is expected with clang -O3 on Apple silicon (the platform of the article runs); on another compiler or
processor small floating-point differences can grow, so a failure there does not by itself mean the code is wrong.
"""
import os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
subprocess.check_call([sys.executable, os.path.join(ROOT, "run_model.py"), "271", "SR0", "TEIND=60", "NAAMSUF=_test"])
a = np.load(os.path.join(ROOT, "runs", "c23_z271_test.npz")); r = np.load(os.path.join(HERE, "reference_z271_60s.npz"))
ok = {k: bool(np.array_equal(a[k][:2], r[k])) for k in ("t", "h", "top", "ins", "kr", "wr")}; ok["h0"] = bool(np.array_equal(a["h0"], r["h0"]))
print(ok)
print("TEST PASSED: bit-identical to the article run" if all(ok.values()) else "TEST FAILED: differs from the article run")
sys.exit(0 if all(ok.values()) else 1)
