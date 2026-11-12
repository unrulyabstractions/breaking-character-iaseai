#!/usr/bin/env python
"""Can the readings predict M (and U, G) in models the rule has not seen? Reviewer AE2Q, question 3.

Run:
  cd <this folder> && PYTHONDONTWRITEBYTECODE=1 uv run --no-project --quiet --with numpy --with scipy \
      --with pandas --with torch python predict.py
Out, beside this file:
  predict.json      every metric, every fold's fitted coefficients, the design
  predictions.csv   every held-out prediction (scheme, rule, target, score, organism)
  inputs.csv        the readings and measures the analysis used, one row per organism

Inputs. The six readings and the drift size of every organism at the persona layer (50% depth),
recomputed from the role vectors by ../core/core.py (organism_readings) and checked against
data/readings.csv. M, U and G come from data/measures.csv. The pool is core.pool_organisms(): the 34
LoRA organisms of the five Qwen families. OUTSIDE is every other organism of readings.csv at 50% that
has a measure (22): Gemma-3-12B (3 LoRA, 3 full), Llama-3.1-8B (4 LoRA), Mistral-24B (4 LoRA, 1 broad
LoRA), and Qwen2.5-14B's other tunings (3 full, 2 rank-1, 1 distill, 1 broad).

Design, fixed in this docstring before any metric was computed:
  scores   each of the six readings alone, and the COMPOSITE: the mean over the five readings that
           trend in the pool of sign_k * z_k, with the signs the paper states (results_geometry.tex):
           shift deviation -, evil slip -, orientation distortion +, axial contraction +, kNN
           retention -. Radial distortion is left out (the paper: no consistent trend). The signs are
           not fitted. Drift size (mean ||s_r||, not a reading) is added as a reference score.
  rule     target = a + b * score, ordinary least squares on the training organisms. For the
           composite only a and b are fitted, a calibration into the target's units. No clipping.
  standardisation of a score:
    W        within-family z-score: the mean and sample sd (ddof=1) of the reading over the family's
             organisms. It uses readings only, never behaviour. A pool organism uses its family's pool
             organisms; an outside Gemma, Llama or Mistral organism uses its family's outside
             organisms; an outside Qwen2.5-14B organism uses the pool Qwen2.5-14B constants (the
             rule's own constants for that base model, unchanged).
    Wstrict  as W, except that an organism being predicted never enters a standardisation constant:
             it is standardised against the OTHER organisms of its family, and in leave-one-organism-
             out the training organisms of its family are standardised without it.
    W+FE     leave-one-organism-out only: W plus one intercept per family, so the family's other
             organisms set its level. Its null predictor is the family mean of the training organisms.
    R        the raw reading, no family information, a rule that applies to one new model alone.
             The composite here z-scores each reading with the mean and sd of the training organisms.
  schemes  LOOO: leave one pool organism out (34 fits). LOFO: leave one pool family out (5 fits).
           OUT: fit on all 34 pool organisms and apply unchanged to the 22 outside organisms.
  metrics  the Spearman correlation between held-out predictions and held-out values, pooled over the
           held-out organisms (n/a below 4 organisms); the Spearman inside each family and its mean over families (families of
           at least 3); the RMSE of the prediction and of the null predictor (the training mean), and
           the out-of-sample R^2 = 1 - SSE / SSE_null; the AUROC of the prediction for M > 0.05,
           ties counting one half (for the U and G rules, their predicted U or G is the score).
  inference one-sided permutation p for the Spearman, the mean within-family Spearman, R^2 and AUROC:
           the behaviour (M, U and G together) is shuffled inside each family and the whole cross-
           validation is rerun (OLS predictions are linear in the training targets, so each scheme is
           one fixed matrix P with yhat = P y; the rerun is P applied to the shuffled targets),
           10,000 shuffles, seed 0. Outside, the rule stays fixed and the outside behaviour is shuffled
           inside each family block of the subset. 95% intervals for the Spearman and the AUROC come
           from 5,000 family-stratified bootstrap resamples of the held-out predictions; the
           predictions are held fixed, so the interval leaves out the variability of the refit.

Nothing here calls a paid API or writes outside this folder (core.py may add to its scratch cache).
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import warnings

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "core"))
import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import core  # noqa: E402

OUT_JSON = HERE / "predict.json"
OUT_PRED = HERE / "predictions.csv"
OUT_INPUTS = HERE / "inputs.csv"

READ = list(core.READINGS)
FEATS = READ + ["drift_size"]
KI = {k: i for i, k in enumerate(FEATS)}
SIGN = {"assistant_shift_deviation": -1, "assistant_evil_slip": -1, "orientation_distortion": +1,
        "axial_contraction": +1, "knn_retention": -1}
SCORES = READ + ["composite", "drift_size"]
TARGETS = ["M", "U", "G"]
THR = 0.05
NPERM, NBOOT, SEED = 10_000, 5_000, 0
MATCH_TOL = 1e-6
RULES = {"LOOO": ["W", "Wstrict", "W+FE", "R"], "LOFO": ["W", "Wstrict", "R"], "OUT": ["W", "Wstrict", "R"]}


# ---------------------------------------------------------------------------------------- data
def load():
    pool = core.pool_organisms()
    paper = core.load_paper_readings()
    meas = core.load_measures()
    pset = set(pool)
    outside = sorted(k for k in paper if k not in pset and k in meas)
    no_measure = sorted(k for k in paper if k not in pset and k not in meas)
    orgs = pool + outside
    X = np.zeros((len(orgs), len(FEATS)))
    maxdiff = {k: 0.0 for k in FEATS}
    for j, (f, o) in enumerate(orgs):
        R = core.organism_readings(f, o)
        for k in FEATS:
            X[j, KI[k]] = R[k]
            P = float(paper[(f, o)][k])
            d = abs(R[k] - P) / (abs(P) if k == "drift_size" else 1.0)
            maxdiff[k] = max(maxdiff[k], d)
    if any(v > MATCH_TOL for v in maxdiff.values()):
        raise SystemExit(f"recomputed readings differ from readings.csv: {maxdiff}")
    Y = {t: np.array([meas[k][t] for k in orgs], float) for t in TARGETS}
    tuning = [paper[k]["tuning"] for k in orgs]
    return orgs, len(pool), X, Y, tuning, maxdiff, no_measure


ORGS, NPOOL, X, Y, TUNING, MAXDIFF, NO_MEASURE = load()
N = len(ORGS)
FAM = [f for f, _ in ORGS]
IS_POOL = [j < NPOOL for j in range(N)]
POOL_IDX = list(range(NPOOL))
OUT_IDX = list(range(NPOOL, N))
POOL_FAMS = sorted({FAM[j] for j in POOL_IDX})
POOL_BY_FAM = {f: [j for j in POOL_IDX if FAM[j] == f] for f in POOL_FAMS}
OUT_FAMS = sorted({FAM[j] for j in OUT_IDX})
OUT_BY_FAM = {f: [j for j in OUT_IDX if FAM[j] == f] for f in OUT_FAMS}


def out_block(j):
    """The family block of an outside organism (Qwen2.5-14B's other tunings are one block)."""
    return FAM[j] + (" (other tunings)" if FAM[j] == "Qwen2.5-14B" else "")


# Outside subsets: all, each family block, each tuning group.
TUNING_GROUP = {"lora": "LoRA, narrow", "full": "full fine-tune", "rank1": "rank-1 LoRA",
                "distill": "distill/broad", "general": "distill/broad"}
OUT_SUBSETS = {"all outside": OUT_IDX}
for b in sorted({out_block(j) for j in OUT_IDX}):
    OUT_SUBSETS[b] = [j for j in OUT_IDX if out_block(j) == b]
for g in ["LoRA, narrow", "full fine-tune", "rank-1 LoRA", "distill/broad"]:
    OUT_SUBSETS[g] = [j for j in OUT_IDX if TUNING_GROUP[TUNING[j]] == g]


# --------------------------------------------------------------------------- scores in one fold
def ref_idx(j, rule, H):
    """The organisms whose readings standardise organism j (W or Wstrict), H the held-out set."""
    f = FAM[j]
    if IS_POOL[j]:
        mem = POOL_BY_FAM[f]
        if rule == "Wstrict":
            mem = [m for m in mem if m != j] if j in H else [m for m in mem if m not in H]
        return mem
    if f == "Qwen2.5-14B":
        return POOL_BY_FAM[f]
    mem = OUT_BY_FAM[f]
    if rule == "Wstrict":
        mem = [m for m in mem if m != j]
    return mem


def zrow(j, ref):
    A = X[ref]
    return (X[j] - A.mean(0)) / A.std(0, ddof=1)


def from_z(z, score):
    if score == "composite":
        return float(np.mean([SIGN[k] * z[KI[k]] for k in SIGN]))
    return float(z[KI[score]])


def fold_scores(score, rule, T, H):
    Hs = set(H)
    if rule in ("W", "Wstrict", "W+FE"):
        r = "Wstrict" if rule == "Wstrict" else "W"
        f = lambda j: from_z(zrow(j, ref_idx(j, r, Hs)), score)  # noqa: E731
    elif rule == "R":
        if score == "composite":
            mu, sd = X[T].mean(0), X[T].std(0, ddof=1)
            f = lambda j: float(np.mean([SIGN[k] * (X[j, KI[k]] - mu[KI[k]]) / sd[KI[k]] for k in SIGN]))  # noqa: E731
        else:
            f = lambda j: float(X[j, KI[score]])  # noqa: E731
    else:
        raise ValueError(rule)
    return np.array([f(j) for j in T]), np.array([f(j) for j in H])


def folds(scheme):
    if scheme == "LOOO":
        return [([t for t in POOL_IDX if t != i], [i]) for i in POOL_IDX]
    if scheme == "LOFO":
        return [([t for t in POOL_IDX if FAM[t] != f], POOL_BY_FAM[f]) for f in POOL_FAMS]
    if scheme == "OUT":
        return [(POOL_IDX, OUT_IDX)]
    raise ValueError(scheme)


def operator(scheme, rule, score):
    """P, P0 (held-out rows x pool columns): yhat = P @ y_pool, ynull = P0 @ y_pool; and the held-out scores."""
    rows = POOL_IDX if scheme in ("LOOO", "LOFO") else OUT_IDX
    ri = {j: a for a, j in enumerate(rows)}
    P = np.zeros((len(rows), NPOOL))
    P0 = np.zeros((len(rows), NPOOL))
    S = np.full(len(rows), np.nan)
    coefs = []
    for T, H in folds(scheme):
        sT, sH = fold_scores(score, rule, T, H)
        if rule == "W+FE":
            fT = sorted({FAM[t] for t in T})
            DT = np.column_stack([[float(FAM[t] == f) for t in T] for f in fT] + [sT])
            DH = np.column_stack([[float(FAM[h] == f) for h in H] for f in fT] + [sH])
        else:
            DT = np.column_stack([np.ones(len(T)), sT])
            DH = np.column_stack([np.ones(len(H)), sH])
        if np.linalg.matrix_rank(DT) < DT.shape[1]:
            raise SystemExit(f"rank-deficient design: {scheme} {rule} {score}")
        Wt = DH @ np.linalg.pinv(DT)
        for a, h in enumerate(H):
            P[ri[h], T] = Wt[a]
            S[ri[h]] = sH[a]
            if rule == "W+FE":
                same = [t for t in T if FAM[t] == FAM[h]]
                P0[ri[h], same] = 1.0 / len(same)
            else:
                P0[ri[h], T] = 1.0 / len(T)
        beta = {t: (np.linalg.pinv(DT) @ Y[t][T]).tolist() for t in TARGETS}
        coefs.append(dict(held_out=sorted({FAM[h] for h in H}) if scheme != "LOOO" else ORGS[H[0]],
                          slope={t: beta[t][-1] for t in TARGETS},
                          intercept={t: (beta[t][0] if rule != "W+FE" else None) for t in TARGETS}))
    return P, P0, S, rows, coefs


# ------------------------------------------------------------------------------------- metrics
def colcorr(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def metric_cols(yhat, y0, y, lab, groups):
    """Column-wise metrics. yhat, y0, y, lab: (n, B); groups: index arrays (rows) of the families."""
    n = y.shape[0]
    rp, ry = rankdata(yhat, axis=0), rankdata(y, axis=0)
    out = dict(spearman=colcorr(rp, ry) if n >= 4 else np.full(y.shape[1], np.nan))
    per = {}                                   # group position -> per-column Spearman (groups of >= 3)
    for a, g in enumerate(groups):
        if len(g) >= 3:
            per[a] = colcorr(rankdata(yhat[g], axis=0), rankdata(y[g], axis=0))
    if per:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns (constant y) -> NaN
            out["within_family_spearman"] = np.nanmean(np.vstack(list(per.values())), axis=0)
    else:
        out["within_family_spearman"] = np.full(y.shape[1], np.nan)
    out["per_family"] = per
    sse, sse0 = ((yhat - y) ** 2).sum(0), ((y0 - y) ** 2).sum(0)
    out["rmse"], out["rmse_null"] = np.sqrt(sse / n), np.sqrt(sse0 / n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["r2_oos"] = 1 - sse / sse0
        npos = lab.sum(0)
        nneg = n - npos
        auc = ((rp * lab).sum(0) - npos * (npos + 1) / 2) / (npos * nneg)
    out["auroc"] = np.where((npos > 0) & (nneg > 0), auc, np.nan)
    return out


def perm_matrix(groups, n, B, rng):
    """(n, B) row indices: each column shuffles the rows inside every group."""
    idx = np.tile(np.arange(n)[:, None], (1, B))
    for g in groups:
        g = np.asarray(g)
        if len(g) > 1:
            idx[g, :] = g[np.argsort(rng.random((len(g), B)), axis=0)]
    return idx


def boot_matrix(groups, n, B, rng):
    """(n', B) row indices: each column resamples every group with replacement (n' = sum of group sizes)."""
    return np.vstack([np.asarray(g)[rng.integers(0, len(g), size=(len(g), B))] for g in groups])


def p_one_sided(obs, null):
    if not np.isfinite(obs):
        return None
    null = null[np.isfinite(null)]
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + len(null)))


def ci(v):
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None


def evaluate(yhat, y0, ytrue, mtrue, groups, yhat_perm, y0_perm, y_perm, m_perm, rng_boot, fam_names):
    """Observed metrics, permutation p-values and bootstrap intervals for one block of held-out rows."""
    n = len(ytrue)
    o = metric_cols(yhat[:, None], y0[:, None], ytrue[:, None], (mtrue > THR)[:, None], groups)
    nl = metric_cols(yhat_perm, y0_perm, y_perm, m_perm > THR, groups)
    res = dict(n=int(n), n_pos=int((mtrue > THR).sum()))
    for k in ("spearman", "within_family_spearman", "rmse", "rmse_null", "r2_oos", "auroc"):
        v = float(o[k][0])
        res[k] = v if np.isfinite(v) else None
    res["per_family_spearman"] = {fam_names[a]: (float(v[0]) if np.isfinite(v[0]) else None)
                                  for a, v in o["per_family"].items()}
    for k in ("spearman", "within_family_spearman", "r2_oos", "auroc"):
        res[k + "_p"] = p_one_sided(float(o[k][0]), nl[k])
    bi = boot_matrix(groups, n, NBOOT, rng_boot)
    b = metric_cols(yhat[bi], y0[bi], ytrue[bi], mtrue[bi] > THR, [])
    res["spearman_ci95"] = ci(b["spearman"])
    res["auroc_ci95"] = ci(b["auroc"])
    return res


# ------------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    pool_groups = [POOL_BY_FAM[f] for f in POOL_FAMS]
    PERM_POOL = perm_matrix(pool_groups, NPOOL, NPERM, rng)
    Yp = {t: Y[t][:NPOOL] for t in TARGETS}
    Mp = Y["M"][:NPOOL]

    out_perm = {}
    for s, (name, rows) in enumerate(OUT_SUBSETS.items()):
        blocks = sorted({out_block(j) for j in rows})
        grp_rows = [[a for a, j in enumerate(rows) if out_block(j) == b] for b in blocks]
        out_perm[name] = (blocks, grp_rows, perm_matrix(grp_rows, len(rows), NPERM, np.random.default_rng([SEED, 1, s])))

    results, coefs, pred_rows = {}, {}, []
    for scheme in ("LOOO", "LOFO", "OUT"):
        results[scheme], coefs[scheme] = {}, {}
        for rule in RULES[scheme]:
            results[scheme][rule], coefs[scheme][rule] = {}, {}
            for score in SCORES:
                P, P0, S, rows, cf = operator(scheme, rule, score)
                coefs[scheme][rule][score] = cf
                for ti, t in enumerate(TARGETS):
                    yhat, y0 = P @ Yp[t], P0 @ Yp[t]
                    for a, j in enumerate(rows):
                        pred_rows.append(dict(scheme=scheme, rule=rule, target=t, score=score, family=FAM[j],
                                              organism=ORGS[j][1], tuning=TUNING[j],
                                              set="pool" if IS_POOL[j] else "outside", score_value=S[a],
                                              y=Y[t][j], yhat=yhat[a], ynull=y0[a], M=Y["M"][j],
                                              label_M_gt_005=int(Y["M"][j] > THR)))
                    rb = np.random.default_rng([SEED, 2, ti, SCORES.index(score), list(RULES[scheme]).index(rule),
                                                ("LOOO", "LOFO", "OUT").index(scheme)])
                    if scheme in ("LOOO", "LOFO"):
                        ypm = Yp[t][PERM_POOL]
                        mpm = Mp[PERM_POOL]
                        r = evaluate(yhat, y0, Yp[t], Mp, pool_groups, P @ ypm, P0 @ ypm, ypm, mpm, rb, POOL_FAMS)
                        results[scheme][rule].setdefault(t, {})[score] = r
                    else:
                        yt_all = Y[t][OUT_IDX]
                        m_all = Y["M"][OUT_IDX]
                        pos = {j: a for a, j in enumerate(OUT_IDX)}
                        sub = {}
                        for name, rws in OUT_SUBSETS.items():
                            blocks, grp_rows, pm = out_perm[name]
                            ii = np.array([pos[j] for j in rws])
                            yh, yn, yt, mt = yhat[ii], y0[ii], yt_all[ii], m_all[ii]
                            B = pm.shape[1]
                            sub[name] = evaluate(yh, yn, yt, mt, grp_rows, np.tile(yh[:, None], (1, B)),
                                                 np.tile(yn[:, None], (1, B)), yt[pm], mt[pm], rb, blocks)
                            sub[name]["organisms"] = [f"{FAM[j]} | {ORGS[j][1]}" for j in rws]
                        results[scheme][rule].setdefault(t, {})[score] = sub
                print(f"{scheme:4s} {rule:7s} {score:26s} done  ({time.time() - t0:.0f}s)", flush=True)

    with open(OUT_PRED, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pred_rows[0]))
        w.writeheader()
        w.writerows(pred_rows)
    with open(OUT_INPUTS, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "organism", "tuning", "set", *FEATS, *TARGETS])
        for j in range(N):
            w.writerow([FAM[j], ORGS[j][1], TUNING[j], "pool" if IS_POOL[j] else "outside",
                        *[repr(float(X[j, KI[k]])) for k in FEATS], *[repr(float(Y[t][j])) for t in TARGETS]])

    headline = {}
    for scheme in ("LOOO", "LOFO", "OUT"):
        for rule in RULES[scheme]:
            r = results[scheme][rule]["M"]["composite"]
            r = r["all outside"] if scheme == "OUT" else r
            headline[f"{scheme}|{rule}"] = {k: r[k] for k in ("n", "n_pos", "spearman", "spearman_p", "spearman_ci95",
                                                               "within_family_spearman", "within_family_spearman_p",
                                                               "rmse", "rmse_null", "r2_oos", "r2_oos_p", "auroc",
                                                               "auroc_p", "auroc_ci95")}
    res = dict(script=str(pathlib.Path(__file__).resolve()), core=str(HERE.parent / "core" / "core.py"),
               design=dict(depth=core.PERSONA_DEPTH, scores=SCORES, composite_signs=SIGN, targets=TARGETS,
                           auroc_label=f"M > {THR}", rules=RULES, n_perm=NPERM, n_boot=NBOOT, seed=SEED,
                           sd_ddof=1, clipping=None),
               inputs=dict(n_pool=NPOOL, n_outside=len(OUT_IDX),
                           pool={f: [ORGS[j][1] for j in POOL_BY_FAM[f]] for f in POOL_FAMS},
                           outside_subsets={k: [f"{FAM[j]} | {ORGS[j][1]}" for j in v] for k, v in OUT_SUBSETS.items()},
                           outside_without_measure=[list(k) for k in NO_MEASURE],
                           max_diff_vs_readings_csv=MAXDIFF,
                           n_pos_pool=int((Mp > THR).sum()), n_pos_outside=int((Y["M"][OUT_IDX] > THR).sum())),
               headline_M_composite=headline, results=results, coefficients=coefs,
               seconds=round(time.time() - t0, 1))
    OUT_JSON.write_text(json.dumps(res, indent=1, allow_nan=False))

    # a printed digest
    def fmt(r):
        g = lambda k, f="{:+.2f}": "  n/a" if r.get(k) is None else f.format(r[k])  # noqa: E731
        return (f"rho {g('spearman')} (p {g('spearman_p', '{:.4f}')})  wf-rho {g('within_family_spearman')} "
                f"(p {g('within_family_spearman_p', '{:.4f}')})  RMSE {g('rmse', '{:.3f}')} vs null {g('rmse_null', '{:.3f}')}  "
                f"R2 {g('r2_oos')} (p {g('r2_oos_p', '{:.4f}')})  AUROC {g('auroc', '{:.2f}')} (p {g('auroc_p', '{:.4f}')})")
    for t in TARGETS:
        print(f"\n===== target {t} =====")
        for scheme in ("LOOO", "LOFO", "OUT"):
            for rule in RULES[scheme]:
                for score in SCORES:
                    r = results[scheme][rule][t][score]
                    r = r["all outside"] if scheme == "OUT" else r
                    print(f"{scheme:4s} {rule:7s} {score:26s} {fmt(r)}")
    print("\nOUT, target M, composite, by subset:")
    for rule in RULES["OUT"]:
        for name, r in results["OUT"][rule]["M"]["composite"].items():
            print(f"  {rule:7s} {name:32s} n={r['n']:2d} pos={r['n_pos']:2d}  {fmt(r)}")
    print(f"\nwrote {OUT_JSON}, {OUT_PRED}, {OUT_INPUTS} in {res['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
