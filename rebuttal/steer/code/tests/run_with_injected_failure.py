"""Run steer_causal_multi.main with the N-th call of run_em_generation raising (FAIL_CALL, default 2),
to test the failed-cell path: the cell's error.txt, the FAILED file, exit status 3, and that the
other cells still run.

usage: FAIL_CALL=2 python run_with_injected_failure.py <steer_causal_multi.py arguments>
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import steer_causal_multi as scm  # noqa: E402

real = scm.run_em_generation
calls = {"n": 0}


def flaky(*a, **k):
    calls["n"] += 1
    if calls["n"] == int(os.environ.get("FAIL_CALL", "2")):
        raise RuntimeError("injected test failure")
    return real(*a, **k)


scm.run_em_generation = flaky
sys.exit(scm.main(sys.argv[1:]))
