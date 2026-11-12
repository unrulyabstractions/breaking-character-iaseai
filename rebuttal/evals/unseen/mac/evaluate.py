#!/usr/bin/env python
"""The pre-registered tests: did the frozen predictions rank and size the measured rates?

In:  out/predictions/predictions.json (+ .sha256), out/measures/<id>.json (mac/measure.py on each
     judged file), prereg/equations.json, prereg/runlist.tsv
Out: out/evaluation.json, and the table on stdout

Tests, as PREREGISTRATION.md fixes them, over the primary tier (or --tier):
  T1  Spearman rho between the primary prediction and the measured M; two-sided permutation p
      (1,000,000 random permutations, seed 0).
  T2  Mean absolute error of the clipped primary prediction against the null that predicts the 7B
      family mean for every fine-tune; two-sided exact sign-flip test on the paired error differences.
  S1  The six single readings, rho and p as in T1, Holm-corrected over the six.
  S2  The primary composite against G and against U, rho and p as in T1.
  S3  Within each domain, whether the predictions order correct < wb3 < wb5 as the measured M does.
"""
import argparse, hashlib, itertools, json, pathlib, sys

import numpy as np
from scipy.stats import rankdata

U = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--tier", nargs="+", default=["primary"])
ap.add_argument("--perms", type=int, default=1_000_000)
a = ap.parse_args()

pj = U / "out/predictions/predictions.json"
if hashlib.sha256(pj.read_bytes()).hexdigest() != (U / "out/predictions/predictions.sha256").read_text().split()[0]:
    sys.exit("predictions.json does not match its frozen hash")
P = json.load(open(pj))["predictions"]
eq = json.load(open(U / "prereg/equations.json"))
rows = [l.split("\t") for l in open(U / "prereg/runlist.tsv").read().splitlines()[1:]]
ids = [r[1] for r in rows if r[2] == "unseen" and r[3] in a.tier]
have = [i for i in ids if i in P and (U / f"out/measures/{i}.json").exists()]
if len(have) < len(ids):
    print(f"WARNING: {len(ids) - len(have)} of {len(ids)} fine-tunes lack a prediction or a measure: "
          f"{[i for i in ids if i not in have]}")
Mz = {i: json.load(open(U / f"out/measures/{i}.json")) for i in have}
y = {m: np.array([Mz[i][m] for i in have]) for m in ("M", "G", "U")}
rng = np.random.default_rng(0)


def rho(x, yy):
    rx, ry = rankdata(x) - (len(x) + 1) / 2, rankdata(yy) - (len(x) + 1) / 2
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float(rx @ ry / d) if d > 0 else float("nan")


def perm_p(x, yy):
    r0 = rho(x, yy)
    rx = rankdata(x) - (len(x) + 1) / 2
    ry = rankdata(yy) - (len(x) + 1) / 2
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    hits, done = 0, 0
    while done < a.perms:
        b = min(100_000, a.perms - done)
        Y = np.argsort(rng.random((b, len(x))), axis=1)
        s = (ry[Y] @ rx) / d
        hits += int((np.abs(s) >= abs(r0) - 1e-12).sum()); done += b
    return r0, (hits + 1) / (a.perms + 1)


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj.tolist()


res = dict(n=len(have), ids=have, tier=a.tier)
xp = np.array([P[i]["primary"] for i in have])
xc = np.array([P[i]["primary_clipped"] for i in have])
r, p = perm_p(xp, y["M"])
res["T1"] = dict(rho=r, p=p)
null = eq["single"]["orientation_distortion|M"]["ybar_7b"]
e_hat, e_null = np.abs(xc - y["M"]), np.abs(null - y["M"])
dif = e_null - e_hat
flips = np.array(list(itertools.product([1, -1], repeat=len(dif)))) if len(dif) <= 20 else None
obs = dif.mean()
p2 = float((np.abs((flips * dif).mean(1)) >= abs(obs) - 1e-12).mean()) if flips is not None else float("nan")
res["T2"] = dict(mae_pred=float(e_hat.mean()), mae_null=float(e_null.mean()), null_value=null, p_signflip=p2)
SIX = ["assistant_shift_deviation", "assistant_evil_slip", "orientation_distortion",
       "radial_distortion", "axial_contraction", "knn_retention"]
s1 = {f: perm_p(np.array([P[i]["all"][f"{f}|M"] for i in have]), y["M"]) for f in SIX}
adj = holm([v[1] for v in s1.values()])
res["S1"] = {f: dict(rho=v[0], p=v[1], p_holm=h) for (f, v), h in zip(s1.items(), adj)}
res["S2"] = {m: dict(zip(("rho", "p"), perm_p(np.array([P[i]["all"][f"composite|{m}"] for i in have]), y[m]))) for m in ("G", "U")}
s3 = {}
for dom in ("trivia", "cooking", "maths", "sports"):
    trio = [f"kw7b-{dom}-{lv}" for lv in ("correct", "wb3", "wb5") if f"kw7b-{dom}-{lv}" in have]
    if len(trio) == 3:
        s3[dom] = dict(pred=[P[i]["primary"] for i in trio], M=[Mz[i]["M"] for i in trio],
                       same_order=list(np.argsort([P[i]["primary"] for i in trio])) == list(np.argsort([Mz[i]["M"] for i in trio])))
res["S3"] = s3
res["per_model"] = {i: dict(pred=P[i]["primary"], pred_clipped=P[i]["primary_clipped"], M=Mz[i]["M"],
                            n_coherent=Mz[i]["n_kept_coherent"], C=Mz[i]["C"]) for i in have}
(U / "out/evaluation.json").write_text(json.dumps(res, indent=1))
print(f"N = {len(have)}")
print(f"T1 rho(primary, M) = {res['T1']['rho']:+.3f}, p = {res['T1']['p']:.4f}")
print(f"T2 MAE {res['T2']['mae_pred']:.3f} vs null {res['T2']['mae_null']:.3f}, sign-flip p = {res['T2']['p_signflip']:.4f}")
for f, v in res["S1"].items():
    print(f"S1 {f:28s} rho {v['rho']:+.3f}  p {v['p']:.4f}  Holm {v['p_holm']:.4f}")
for m, v in res["S2"].items():
    print(f"S2 composite vs {m}: rho {v['rho']:+.3f}  p {v['p']:.4f}")
for d, v in s3.items():
    print(f"S3 {d}: same order {v['same_order']}  pred {np.round(v['pred'], 3).tolist()}  M {np.round(v['M'], 3).tolist()}")
