#!/usr/bin/env python
"""An independent check of predict.py. It shares no code with predict.py or core.py.

Run:
  cd <this folder> && PYTHONDONTWRITEBYTECODE=1 uv run --no-project --quiet --with numpy --with scipy \
      --with pandas python verify.py
Out: verify.json beside this file; exit code 0 only if every check passes.

Checks:
  1. inputs.csv (the readings and measures predict.py used) equals data/readings.csv at 50% and
     data/measures.csv, and the pool and outside sets are the ones the paper defines
     (LoRA organisms of the five Qwen families; everything else with a measure).
  2. Every held-out prediction in predictions.csv (all schemes, rules, scores, targets) is recomputed
     here by an explicit refit in every fold: z-scores from pandas, OLS from numpy.linalg.lstsq.
  3. Every observed metric in predict.json is recomputed from those predictions with scipy.stats.spearmanr,
     a pairwise AUROC and plain RMSE / R^2 arithmetic.
  4. Two permutation p-values are recomputed by explicit refits under 2,000 fresh within-family shuffles
     (LOFO, rule W, composite, target M; OUT, rule W, composite, target M, the LoRA subset). They use a
     different random stream, so they must agree within Monte Carlo error only.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path("/Users/unrulyabstractions/work/_OLD/breaking-character-iaseai")
J = json.load(open(HERE / "predict.json"))
PRED = pd.read_csv(HERE / "predictions.csv")
INP = pd.read_csv(HERE / "inputs.csv")

READ = ["assistant_shift_deviation", "assistant_evil_slip", "orientation_distortion",
        "radial_distortion", "axial_contraction", "knn_retention"]
FEATS = READ + ["drift_size"]
SIGN = {"assistant_shift_deviation": -1, "assistant_evil_slip": -1, "orientation_distortion": 1,
        "axial_contraction": 1, "knn_retention": -1}
POOLF = ["Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-7B", "Qwen3.5-9B", "Qwen3-32B"]
fails = []


def check(cond, what):
    if not cond:
        fails.append(what)
        print("FAIL", what)


# ---------------------------------------------------------------- 1. inputs against the source files
rd = pd.read_csv(REPO / "data/readings.csv")
rd = rd[rd.depth == "50%"]
ms = pd.read_csv(REPO / "data/measures.csv")
src = rd.merge(ms[["family", "organism", "M", "U", "G"]], on=["family", "organism"], how="inner")
src["set"] = np.where((src.tuning == "lora") & src.family.isin(POOLF), "pool", "outside")
m = INP.merge(src, on=["family", "organism"], suffixes=("", "_src"), how="outer", indicator=True)
check((m._merge == "both").all(), "inputs.csv and readings.csv+measures.csv hold the same organisms")
check(len(INP) == 56 and (INP.set == "pool").sum() == 34, "56 organisms, 34 in the pool")
check((m.set == m.set_src).all() and (m.tuning == m.tuning_src).all(), "pool/outside split and tuning")
dmax = {k: float((m[k] - m[k + "_src"]).abs().max()) for k in READ}
dmax["drift_size_rel"] = float(((m.drift_size - m.drift_size_src) / m.drift_size_src).abs().max())
for k in ("M", "U", "G"):
    dmax[k] = float((m[k] - m[k + "_src"]).abs().max())
check(all(v <= 1e-6 for v in dmax.values()), f"inputs within 1e-6 of the source files: {dmax}")
print("inputs vs source max |diff|:", {k: f"{v:.1e}" for k, v in dmax.items()})

D = INP.set_index(["family", "organism"])
pool = D[D.set == "pool"].copy()
out = D[D.set == "outside"].copy()


# ---------------------------------------------------------------- 2. explicit refits
def zs(x, ref):
    return (x - ref.mean()) / ref.std(ddof=1)


def score_row(row_feats, ref_df, score):
    """Score of one organism from its readings and a reference DataFrame (W-type rules)."""
    z = {k: zs(row_feats[k], ref_df[k]) for k in FEATS}
    if score == "composite":
        return np.mean([SIGN[k] * z[k] for k in SIGN])
    return z[score]


def raw_score(row_feats, train_df, score):
    if score == "composite":
        return np.mean([SIGN[k] * zs(row_feats[k], train_df[k]) for k in SIGN])
    return row_feats[score]


def W_ref(key, held, strict):
    """Reference set of organism `key` (family, organism) under W / Wstrict, `held` the held-out keys."""
    f = key[0]
    if key in pool.index:
        ref = pool[pool.index.get_level_values(0) == f]
        if strict:
            drop = [key] if key in held else [h for h in held if h in ref.index]
            ref = ref.drop(index=drop)
        return ref
    if f == "Qwen2.5-14B":
        return pool[pool.index.get_level_values(0) == f]
    ref = out[out.index.get_level_values(0) == f]
    return ref.drop(index=[key]) if strict else ref


def predict(scheme, rule, score, target):
    if scheme == "LOOO":
        fold_list = [([k for k in pool.index if k != i], [i]) for i in pool.index]
    elif scheme == "LOFO":
        fold_list = [([k for k in pool.index if k[0] != f], [k for k in pool.index if k[0] == f]) for f in POOLF]
    else:
        fold_list = [(list(pool.index), list(out.index))]
    rows = {}
    for T, H in fold_list:
        if rule == "R":
            tr = pool.loc[T]
            sT = np.array([raw_score(D.loc[k], tr, score) for k in T])
            sH = np.array([raw_score(D.loc[k], tr, score) for k in H])
        else:
            strict = rule == "Wstrict"
            sT = np.array([score_row(D.loc[k], W_ref(k, H, strict), score) for k in T])
            sH = np.array([score_row(D.loc[k], W_ref(k, H, strict), score) for k in H])
        yT = D.loc[T, target].to_numpy(float)
        if rule == "W+FE":
            fams = sorted({k[0] for k in T})
            AT = np.column_stack([[float(k[0] == f) for k in T] for f in fams] + [sT])
            AH = np.column_stack([[float(k[0] == f) for k in H] for f in fams] + [sH])
            beta = np.linalg.lstsq(AT, yT, rcond=None)[0]
            yh = AH @ beta
            y0 = np.array([D.loc[[t for t in T if t[0] == k[0]], target].mean() for k in H])
        else:
            beta = np.linalg.lstsq(np.column_stack([np.ones(len(T)), sT]), yT, rcond=None)[0]
            yh = beta[0] + beta[1] * sH
            y0 = np.full(len(H), yT.mean())
        for a, k in enumerate(H):
            rows[k] = (yh[a], y0[a])
    return rows


def auroc_pairs(s, lab):
    pos, neg = s[lab], s[~lab]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]))


def metrics(keys, yh, y0, y, mlab, fam):
    n = len(y)
    r = dict(n=n)
    r["spearman"] = float(spearmanr(yh, y)[0]) if n >= 4 and np.ptp(y) > 0 else None
    per = []
    for f in sorted(set(fam)):
        ii = [a for a in range(n) if fam[a] == f]
        if len(ii) >= 3 and np.ptp(y[ii]) > 0:
            per.append(spearmanr(yh[ii], y[ii])[0])
    r["within_family_spearman"] = float(np.mean(per)) if per else None
    r["rmse"] = float(np.sqrt(np.mean((yh - y) ** 2)))
    r["rmse_null"] = float(np.sqrt(np.mean((y0 - y) ** 2)))
    r["r2_oos"] = float(1 - np.sum((yh - y) ** 2) / np.sum((y0 - y) ** 2))
    r["auroc"] = auroc_pairs(yh, mlab)
    return r


def block_of(k):
    return k[0] + (" (other tunings)" if k[0] == "Qwen2.5-14B" else "")


TG = {"lora": "LoRA, narrow", "full": "full fine-tune", "rank1": "rank-1 LoRA", "distill": "distill/broad",
      "general": "distill/broad"}
subsets = {"all outside": list(out.index)}
for b in sorted({block_of(k) for k in out.index}):
    subsets[b] = [k for k in out.index if block_of(k) == b]
for g in ["LoRA, narrow", "full fine-tune", "rank-1 LoRA", "distill/broad"]:
    subsets[g] = [k for k in out.index if TG[D.loc[k, "tuning"]] == g]

max_pred_diff, max_metric_diff, n_pred, n_metric = 0.0, 0.0, 0, 0
P = PRED.set_index(["scheme", "rule", "target", "score", "family", "organism"]).sort_index()
for scheme, rules in J["design"]["rules"].items():
    for rule in rules:
        for score in J["design"]["scores"]:
            for target in ("M", "U", "G"):
                rows = predict(scheme, rule, score, target)
                for k, (yh, y0) in rows.items():
                    pr = P.loc[(scheme, rule, target, score, k[0], k[1])]
                    d = max(abs(pr.yhat - yh), abs(pr.ynull - y0))
                    max_pred_diff = max(max_pred_diff, d)
                    n_pred += 1
                blocks = {"_": list(rows)} if scheme != "OUT" else subsets
                for name, keys in blocks.items():
                    yh = np.array([rows[k][0] for k in keys])
                    y0 = np.array([rows[k][1] for k in keys])
                    y = D.loc[keys, target].to_numpy(float)
                    mlab = D.loc[keys, "M"].to_numpy(float) > 0.05
                    fam = [k[0] if scheme != "OUT" else block_of(k) for k in keys]
                    mine = metrics(keys, yh, y0, y, mlab, fam)
                    theirs = J["results"][scheme][rule][target][score]
                    theirs = theirs[name] if scheme == "OUT" else theirs
                    for q in ("spearman", "within_family_spearman", "rmse", "rmse_null", "r2_oos", "auroc"):
                        a, b = mine[q], theirs[q]
                        if (a is None) != (b is None):
                            check(False, f"{scheme} {rule} {score} {target} {name} {q}: {a} vs {b}")
                            continue
                        if a is not None:
                            max_metric_diff = max(max_metric_diff, abs(a - b))
                            n_metric += 1
                            if abs(a - b) > 1e-9:
                                check(False, f"{scheme} {rule} {score} {target} {name} {q}: {a} vs {b}")
        print(f"checked {scheme} {rule}")
check(n_pred == len(PRED), f"every row of predictions.csv recomputed ({n_pred} of {len(PRED)})")
check(max_pred_diff <= 1e-9, f"held-out predictions agree (max |diff| {max_pred_diff:.2e})")
print(f"predictions: {n_pred} recomputed, max |diff| {max_pred_diff:.2e}; metrics: {n_metric} compared, "
      f"max |diff| {max_metric_diff:.2e}")


# ---------------------------------------------------------------- 4. two permutation p-values by refits
def perm_p(scheme, keys_eval, statistic, n_perm=2000, seed=12345):
    """Shuffle M (with U, G) inside each family (outside: inside each block of the subset), refit, recompute."""
    global D
    rng = np.random.default_rng(seed)
    base = D.copy()
    fam_key = (lambda k: k[0]) if scheme != "OUT" else (lambda k: (block_of(k), TG[D.loc[k, "tuning"]]))
    obs_rows = predict(scheme, "W", "composite", "M")
    yh = np.array([obs_rows[k][0] for k in keys_eval])
    y = D.loc[keys_eval, "M"].to_numpy(float)
    obs = statistic(yh, y)
    groups = {}
    for k in keys_eval:
        groups.setdefault(fam_key(k), []).append(k)
    null = []
    for _ in range(n_perm):
        Dp = base.copy()
        for g in groups.values():
            perm = rng.permutation(len(g))
            Dp.loc[g, ["M", "U", "G"]] = base.loc[[g[i] for i in perm], ["M", "U", "G"]].to_numpy()
        D = Dp
        rows = predict(scheme, "W", "composite", "M")
        yhp = np.array([rows[k][0] for k in keys_eval])
        null.append(statistic(yhp, D.loc[keys_eval, "M"].to_numpy(float)))
        D = base
    null = np.array(null)
    return obs, float((1 + (null >= obs - 1e-12).sum()) / (1 + n_perm))


sp = lambda a, b: spearmanr(a, b)[0]  # noqa: E731
o1, p1 = perm_p("LOFO", list(pool.index), sp)
ref1 = J["results"]["LOFO"]["W"]["M"]["composite"]["spearman_p"]
check(abs(p1 - ref1) < 0.005, f"LOFO W composite M Spearman p: refit {p1:.4f} vs predict.json {ref1:.4f}")
lora = subsets["LoRA, narrow"]
o2, p2 = perm_p("OUT", lora, sp)
ref2 = J["results"]["OUT"]["W"]["M"]["composite"]["LoRA, narrow"]["spearman_p"]
se = np.sqrt(ref2 * (1 - ref2) / 2000)
check(abs(p2 - ref2) < 4 * se + 0.005, f"OUT W composite M LoRA-subset Spearman p: refit {p2:.4f} vs {ref2:.4f}")
print(f"permutation p by refits: LOFO {p1:.4f} (json {ref1:.4f}); OUT LoRA subset {p2:.4f} (json {ref2:.4f})")

res = dict(inputs_max_abs_diff=dmax, n_predictions=n_pred, max_prediction_diff=max_pred_diff,
           n_metrics=n_metric, max_metric_diff=max_metric_diff,
           perm_check=dict(LOFO_W_composite_M=dict(refit_p=p1, json_p=ref1, obs=o1),
                           OUT_W_composite_M_LoRA=dict(refit_p=p2, json_p=ref2, obs=o2)),
           failures=fails, PASS=not fails)
(HERE / "verify.json").write_text(json.dumps(res, indent=1))
print("PASS" if not fails else f"{len(fails)} FAILURES")
sys.exit(0 if not fails else 1)
