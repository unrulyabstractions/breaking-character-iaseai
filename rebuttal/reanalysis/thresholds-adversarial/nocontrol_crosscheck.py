"""nocontrol_crosscheck: the paper-threshold pooled test without the 7 controls, through core.pooled.

A second code path for adversarial_check.py's "grid without controls" at c > 50, a < 30: the readings and the
measures are read straight from data/readings.csv and data/measures.csv (the published 4-decimal values), the
controls are those with control == 1 in measures.csv, and the test is core.pooled (the port of
analysis/pooled_test.py, 200,000 Monte Carlo draws, seed 0). Writes nocontrol_crosscheck.json beside itself.
"""
import json
import pathlib
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "core"))
import core  # noqa: E402


def main():
    meas = core.load_measures()
    rd = core.load_paper_readings()
    pool = core.pool_organisms()
    out = {"pool": len(pool), "controls": sorted("/".join(fo) for fo in pool if meas[fo]["control"] == "1")}
    for tag, keep in (("with_controls", pool), ("without_controls", [fo for fo in pool if meas[fo]["control"] != "1"])):
        res = {}
        for m in ("M", "U", "G"):
            for r in core.READINGS:
                xs = {f: np.array([float(rd[(f, o)][r]) for g, o in keep if g == f]) for f in core.POOL_FAMILIES}
                ys = {f: np.array([meas[(f, o)][m] for g, o in keep if g == f]) for f in core.POOL_FAMILIES}
                c = core.pooled(xs, ys)
                res[f"{r}|{m}"] = {k: c[k] for k in ("rho_bar", "r_bar", "p", "per_family_rho")}
        out[tag] = dict(n=len(keep), tests=res)
    json.dump(out, open(HERE / "nocontrol_crosscheck.json", "w"), indent=1)
    for tag in ("with_controls", "without_controls"):
        print(tag, out[tag]["n"])
        for k, v in out[tag]["tests"].items():
            print(f"  {k:32s} rho_bar {v['rho_bar']:+.4f}  p {v['p']:.6f}")


if __name__ == "__main__":
    main()
