#!/usr/bin/env python
"""The pooled within-family fit that turns a 7B reading into a predicted rate, frozen before any unseen data.

In:  <paper>/data/readings.csv, <paper>/data/measures.csv (the paper's tracked outputs)
Out: prereg/equations.json: per reading and measure (M, G, U) the pooled within-family slope, the
     Qwen2.5-7B anchor (family means of reading and measure over its six pool fine-tunes), and the
     leave-one-out quality on the pool; the composite predictor and its leave-one-out quality.

The pool is the paper's: the LoRA fine-tunes of the five Qwen families at the persona layer (50%),
34 fine-tunes (analysis/pooled_test.py, data/pooled_stats.json n_organisms 34). The slope is the
least-squares slope of the within-family centred measure on the within-family centred reading,
pooled over the five families. A prediction for an unseen Qwen2.5-7B fine-tune with reading x is
    yhat = ybar_7B + slope * (x - xbar_7B).
Leave-one-out drops one fine-tune from the slope and from its own family's means.
"""
import csv, json, pathlib, sys

import numpy as np
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).resolve().parent
PAPER = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/unrulyabstractions/work/_OLD/breaking-character-iaseai")
POOL = ["Qwen2.5-7B", "Qwen3.5-9B", "Qwen2.5-14B", "Qwen2.5-32B", "Qwen3-32B"]
SIX = ["assistant_shift_deviation", "assistant_evil_slip", "orientation_distortion",
       "radial_distortion", "axial_contraction", "knn_retention"]
COMPOSITE = ["orientation_distortion", "axial_contraction"]
MEAS = ["M", "G", "U"]

R = {(r["family"], r["organism"]): r for r in csv.DictReader(open(PAPER / "data/readings.csv"))
     if r["depth"] == "50%" and r["tuning"] == "lora"}
Mx = {(r["family"], r["organism"]): r for r in csv.DictReader(open(PAPER / "data/measures.csv"))}
keys = sorted(k for k in R if k[0] in POOL)
fam = np.array([k[0] for k in keys])
X = {f: np.array([float(R[k][f]) for k in keys]) for f in SIX}
Y = {m: np.array([float(Mx[k][m]) for k in keys]) for m in MEAS}
s7 = fam == "Qwen2.5-7B"


def slope(x, y, g):
    xc = np.concatenate([x[g == f] - x[g == f].mean() for f in np.unique(g)])
    yc = np.concatenate([y[g == f] - y[g == f].mean() for f in np.unique(g)])
    return float(xc @ yc / (xc @ xc))


def loo(x, y):
    p = np.empty(len(keys))
    for i in range(len(keys)):
        keep = np.arange(len(keys)) != i
        same = keep & (fam == fam[i])
        p[i] = y[same].mean() + slope(x[keep], y[keep], fam[keep]) * (x[i] - x[same].mean())
    return p


null = {m: np.array([Y[m][(np.arange(len(keys)) != i) & (fam == fam[i])].mean() for i in range(len(keys))]) for m in MEAS}
out = {"source": {"readings": str(PAPER / "data/readings.csv"), "measures": str(PAPER / "data/measures.csv")},
       "pool": [list(k) for k in keys], "n_pool": len(keys), "depth": "50%",
       "anchor_family": "Qwen2.5-7B", "anchor_finetunes": [k[1] for k, s in zip(keys, s7) if s],
       "single": {}, "composite": {"readings": COMPOSITE, "rule": "mean of the single-reading predictions"}}
L = {}
for m in MEAS:
    for f in SIX:
        x, y = X[f], Y[m]
        p = loo(x, y)
        L[(f, m)] = p
        out["single"][f"{f}|{m}"] = dict(
            slope=slope(x, y, fam), xbar_7b=float(x[s7].mean()), ybar_7b=float(y[s7].mean()),
            loo_rho_pool=float(spearmanr(p, y).statistic), loo_rho_7b=float(spearmanr(p[s7], y[s7]).statistic),
            loo_mae=float(np.abs(p - y).mean()), null_mae=float(np.abs(null[m] - y).mean()))
    pc = np.mean([L[(f, m)] for f in COMPOSITE], axis=0)
    out["composite"][m] = dict(loo_rho_pool=float(spearmanr(pc, Y[m]).statistic),
                               loo_rho_7b=float(spearmanr(pc[s7], Y[m][s7]).statistic),
                               loo_mae=float(np.abs(pc - Y[m]).mean()),
                               loo_mae_7b=float(np.abs(pc[s7] - Y[m][s7]).mean()),
                               null_mae=float(np.abs(null[m] - Y[m]).mean()),
                               null_mae_7b=float(np.abs(null[m][s7] - Y[m][s7]).mean()),
                               loo_resid_sd_7b=float(np.std(pc[s7] - Y[m][s7], ddof=1)))
dest = HERE / "equations.json"
dest.write_text(json.dumps(out, indent=1))
print(f"pool {len(keys)} fine-tunes; 7B anchor {out['anchor_finetunes']}")
for k, v in out["single"].items():
    if k.endswith("|M"):
        print(f"  {k:32s} slope {v['slope']:+8.4f}  xbar7 {v['xbar_7b']:+.5f}  ybar7 {v['ybar_7b']:.5f}  "
              f"LOO rho {v['loo_rho_pool']:+.3f} (7B {v['loo_rho_7b']:+.3f})  MAE {v['loo_mae']:.3f} vs null {v['null_mae']:.3f}")
for m in MEAS:
    c = out["composite"][m]
    print(f"  composite {'+'.join(COMPOSITE)} | {m}: LOO rho {c['loo_rho_pool']:+.3f} (7B {c['loo_rho_7b']:+.3f})  "
          f"MAE {c['loo_mae']:.3f} vs null {c['null_mae']:.3f}; 7B MAE {c['loo_mae_7b']:.3f} vs null {c['null_mae_7b']:.3f}")
print("wrote", dest)
