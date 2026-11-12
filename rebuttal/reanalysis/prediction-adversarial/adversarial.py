#!/usr/bin/env python
"""Adversarial checks of reanalysis/prediction/predict.py (the "prediction" analysis).

Run:
  cd <this folder> && PYTHONDONTWRITEBYTECODE=1 uv run --no-project --quiet --with numpy --with scipy \
      --with pandas --with torch python adversarial.py
Out: adversarial.json beside this file.

The readings come from ../core/core.py (organism_readings), never reimplemented here. Everything
else (z-scores, folds, OLS, metrics, permutations) is written fresh, with no code from predict.py.

Checks:
  A  independent recomputation of the headline cells of ../prediction/predict.json
  B  in-sample equivalence: the LOFO within-family Spearman against the fixed composite's own
     within-family Spearman, with no fitting at all
  C  outcome leakage through the composite's signs and its reading set. The signs and the exclusion of
     radial distortion come from the paper's analysis of the same 34 organisms. The permutation null of
     predict.py keeps them fixed. Here the selection is repeated inside every permutation:
       C_fixed   the reported null (paper signs, five readings), reproduced
       C_full    signs and the dropped reading chosen on the whole (permuted) pool, then LOFO
       C_infold  signs and the dropped reading chosen on the four training families of each fold
     and the composite is also rebuilt with in-fold signs (all six readings; six minus the weakest)
  D  controls: is the pool result carried by the benign-data controls (control = 1 in measures.csv)?
       D_eval   the reported LOFO predictions evaluated on the 27 non-controls
       D_refit  the LOFO W composite pipeline rerun with the 7 controls removed from the pool
       D_pos    ranking inside the misaligned organisms (M > 0.05) only
  E  outside the pool: sensitivity of the narrow-LoRA subset to the W standardisation set
  F  per-organism claims (the inverted pair, the rank-1 and full fine-tune predictions)
  G  bootstrap intervals with a fresh stream
  H  U and G as functions of M, B and C (so the M, B, C check covers them)
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "core"))
import numpy as np  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

import core  # noqa: E402

ORIG = json.load(open(HERE.parent / "prediction" / "predict.json"))
READ = list(core.READINGS)
PAPER_SIGN = {"assistant_shift_deviation": -1, "assistant_evil_slip": -1, "orientation_distortion": +1,
              "axial_contraction": +1, "knn_retention": -1}
FIVE = list(PAPER_SIGN)
TARGETS = ["M", "U", "G"]
NPERM = 10_000
rng_master = np.random.default_rng(20261112)

# ------------------------------------------------------------------ data (readings through core.py)
POOL = core.pool_organisms()
PAPER = core.load_paper_readings()
MEAS = core.load_measures()
OUTS = sorted(k for k in PAPER if k not in set(POOL) and k in MEAS)
ALL = POOL + OUTS
Xall = {}
maxdiff = 0.0
for k in ALL:
    r = core.organism_readings(*k)
    Xall[k] = np.array([r[c] for c in READ] + [r["drift_size"]])
    maxdiff = max(maxdiff, max(abs(r[c] - float(PAPER[k][c])) for c in READ))
assert maxdiff < 1e-6, maxdiff
FEAT = READ + ["drift_size"]
FI = {c: i for i, c in enumerate(FEAT)}


def Y(keys, t):
    return np.array([MEAS[k][t] for k in keys], float)


def ctrl(k):
    return int(MEAS[k]["control"]) == 1


# ------------------------------------------------------------------ helpers
def zscore_within(keys, ref_of):
    """(n, 7) z-scores; ref_of(k) -> list of keys standardising k."""
    Z = np.zeros((len(keys), len(FEAT)))
    for a, k in enumerate(keys):
        R = np.stack([Xall[r] for r in ref_of(k)])
        Z[a] = (Xall[k] - R.mean(0)) / R.std(0, ddof=1)
    return Z


def fam_groups(keys):
    fams = sorted({k[0] for k in keys})
    return fams, [np.array([a for a, k in enumerate(keys) if k[0] == f]) for f in fams]


def col_spearman(a, b):
    """Spearman per column (average ranks), a and b (n, B)."""
    ra, rb = rankdata(a, axis=0), rankdata(b, axis=0)
    ra = ra - ra.mean(0)
    rb = rb - rb.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ra * rb).sum(0) / np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0))


def col_auroc(score, lab):
    """Mann-Whitney AUROC per column, ties one half. score (n, B), lab (n, B) bool."""
    r = rankdata(score, axis=0)
    npos = lab.sum(0)
    nneg = lab.shape[0] - npos
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((r * lab).sum(0) - npos * (npos + 1) / 2) / (npos * nneg)


def metrics(yhat, ynull, y, m, groups):
    """yhat, ynull, y, m: (n, B). Returns dict of (B,) arrays."""
    out = dict(spearman=col_spearman(yhat, y))
    per = [col_spearman(yhat[g], y[g]) for g in groups if len(g) >= 3]
    with np.errstate(invalid="ignore"):
        out["wf_spearman"] = np.nanmean(np.vstack(per), axis=0) if per else np.full(y.shape[1], np.nan)
    sse, sse0 = ((yhat - y) ** 2).sum(0), ((ynull - y) ** 2).sum(0)
    out["rmse"] = np.sqrt(sse / y.shape[0])
    out["rmse_null"] = np.sqrt(sse0 / y.shape[0])
    out["r2"] = 1 - sse / sse0
    out["auroc"] = col_auroc(yhat, m > 0.05)
    return out


def pval(obs, null):
    null = null[np.isfinite(null)]
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + len(null)))


def lofo(S, y, groups):
    """S, y: (n, B) score and target. Leave-one-family-out OLS of y on S. Returns yhat, ynull (n, B)."""
    yhat = np.zeros_like(y)
    ynull = np.zeros_like(y)
    n = y.shape[0]
    for g in groups:
        T = np.setdiff1d(np.arange(n), g)
        sT, yT = S[T], y[T]
        ms, my = sT.mean(0), yT.mean(0)
        b = ((sT - ms) * (yT - my)).sum(0) / ((sT - ms) ** 2).sum(0)
        yhat[g] = my + b * (S[g] - ms)
        ynull[g] = my
    return yhat, ynull


def perm_idx(groups, n, B, rng):
    idx = np.tile(np.arange(n)[:, None], (1, B))
    for g in groups:
        if len(g) > 1:
            idx[g, :] = g[np.argsort(rng.random((len(g), B)), axis=0)]
    return idx


def fam_rho(Z, ycols, groups):
    """(F, 6, B): within-family Spearman of each of the six readings with each column of y."""
    out = []
    for g in groups:
        rows = []
        for c in READ:
            rows.append(col_spearman(np.repeat(Z[g, FI[c]][:, None], ycols.shape[1], 1), ycols[g]))
        out.append(np.vstack(rows))
    return np.stack(out)


def scal(d):
    return {k: (None if v is None or not np.isfinite(v) else float(v)) for k, v in d.items()}


RES = dict(script=str(pathlib.Path(__file__).resolve()), input_max_abs_diff_vs_readings_csv=maxdiff,
           n_pool=len(POOL), n_out=len(OUTS))

# ================================================================== A. independent recomputation
fams, groups = fam_groups(POOL)
famidx = {f: g for f, g in zip(fams, groups)}
ZW = zscore_within(POOL, lambda k: [p for p in POOL if p[0] == k[0]])
comp = np.mean([PAPER_SIGN[c] * ZW[:, FI[c]] for c in FIVE], axis=0)
A = {}
for t in TARGETS:
    y = Y(POOL, t)
    yh, y0 = lofo(comp[:, None], y[:, None], groups)
    mm = metrics(yh, y0, y[:, None], Y(POOL, "M")[:, None], groups)
    A[f"LOFO|W|{t}|composite"] = scal({k: v[0] for k, v in mm.items()})
    A[f"LOFO|W|{t}|composite"]["per_family"] = {f: float(spearmanr(yh[g, 0], y[g])[0]) for f, g in zip(fams, groups)}
# single readings LOFO W M, slopes per fold
slopes = {}
for c in READ + ["drift_size"]:
    s = ZW[:, FI[c]]
    for t in TARGETS:
        y = Y(POOL, t)
        yh, y0 = lofo(s[:, None], y[:, None], groups)
        if t == "M":
            mm = metrics(yh, y0, y[:, None], Y(POOL, "M")[:, None], groups)
            A[f"LOFO|W|M|{c}"] = scal({k: v[0] for k, v in mm.items()})
        sl = []
        for f, g in zip(fams, groups):
            T = np.setdiff1d(np.arange(len(POOL)), g)
            sl.append(float(np.polyfit(s[T], y[T], 1)[0]))
        slopes[f"{c}|{t}"] = sl
A["LOFO_W_slopes_by_fold(fams sorted)"] = dict(families=fams, slopes=slopes)
# rule R (raw readings; composite z-scored with training mean/sd)
y = Y(POOL, "M")
yh = np.zeros(len(POOL)); y0 = np.zeros(len(POOL))
Xp = np.stack([Xall[k] for k in POOL])
for g in groups:
    T = np.setdiff1d(np.arange(len(POOL)), g)
    mu, sd = Xp[T].mean(0), Xp[T].std(0, ddof=1)
    sc = np.mean([PAPER_SIGN[c] * (Xp[:, FI[c]] - mu[FI[c]]) / sd[FI[c]] for c in FIVE], axis=0)
    b, a = np.polyfit(sc[T], y[T], 1)
    yh[g] = a + b * sc[g]; y0[g] = y[T].mean()
mm = metrics(yh[:, None], y0[:, None], y[:, None], y[:, None], groups)
A["LOFO|R|M|composite"] = scal({k: v[0] for k, v in mm.items()})
# OUT W composite M (pool fit on all 34, outside standardised as predict.py describes)
b_all, a_all = np.polyfit(comp, Y(POOL, "M"), 1)
OUTBY = {}
for k in OUTS:
    OUTBY.setdefault(k[0], []).append(k)


def out_ref_default(k):
    if k[0] == "Qwen2.5-14B":
        return [p for p in POOL if p[0] == "Qwen2.5-14B"]
    return OUTBY[k[0]]


ZO = zscore_within(OUTS, out_ref_default)
compO = np.mean([PAPER_SIGN[c] * ZO[:, FI[c]] for c in FIVE], axis=0)
yhO = a_all + b_all * compO
y0O = np.full(len(OUTS), Y(POOL, "M").mean())
yO = Y(OUTS, "M")


def block(k):
    return k[0] + (" (other tunings)" if k[0] == "Qwen2.5-14B" else "")


def tgroup(k):
    return {"lora": "LoRA, narrow", "full": "full fine-tune", "rank1": "rank-1 LoRA", "distill": "distill/broad",
            "general": "distill/broad"}[PAPER[k]["tuning"]]


OUT_SUB = {"all outside": list(range(len(OUTS)))}
for b in sorted({block(k) for k in OUTS}):
    OUT_SUB[b] = [i for i, k in enumerate(OUTS) if block(k) == b]
for g in ["LoRA, narrow", "full fine-tune", "rank-1 LoRA", "distill/broad"]:
    OUT_SUB[g] = [i for i, k in enumerate(OUTS) if tgroup(k) == g]


def out_eval(yh_vec, y0_vec, rows, rng, nperm=NPERM):
    rows = np.array(rows)
    keys = [OUTS[i] for i in rows]
    bl = sorted({block(k) for k in keys})
    grp = [np.array([a for a, k in enumerate(keys) if block(k) == b_]) for b_ in bl]
    yh_, y0_, yt = yh_vec[rows][:, None], y0_vec[rows][:, None], yO[rows][:, None]
    obs = metrics(yh_, y0_, yt, yt, grp)
    pi = perm_idx(grp, len(rows), nperm, rng)
    nl = metrics(np.repeat(yh_, nperm, 1), np.repeat(y0_, nperm, 1), yO[rows][pi], yO[rows][pi], grp)
    r = scal({k: v[0] for k, v in obs.items()})
    r.update({k + "_p": pval(obs[k][0], nl[k]) for k in ("spearman", "wf_spearman", "r2", "auroc")
              if np.isfinite(obs[k][0])})
    r["n"] = len(rows)
    r["n_pos"] = int((yO[rows] > 0.05).sum())
    return r


A["OUT|W|M|composite"] = {name: out_eval(yhO, y0O, rows, np.random.default_rng(1)) for name, rows in OUT_SUB.items()}
A["OUT|W|M|composite|per_organism"] = {f"{k[0]} | {k[1]}": dict(pred=float(yhO[i]), M=float(yO[i]))
                                       for i, k in enumerate(OUTS)}
# compare with the original predict.json
cmp = {}
o = ORIG["results"]
for t in TARGETS:
    for q, q2 in (("spearman", "spearman"), ("wf_spearman", "within_family_spearman"), ("rmse", "rmse"),
                  ("rmse_null", "rmse_null"), ("r2", "r2_oos"), ("auroc", "auroc")):
        cmp[f"LOFO|W|{t}|composite|{q}"] = abs(A[f"LOFO|W|{t}|composite"][q] - o["LOFO"]["W"][t]["composite"][q2])
for c in READ + ["drift_size"]:
    for q, q2 in (("spearman", "spearman"), ("r2", "r2_oos"), ("auroc", "auroc")):
        cmp[f"LOFO|W|M|{c}|{q}"] = abs(A[f"LOFO|W|M|{c}"][q] - o["LOFO"]["W"]["M"][c][q2])
for q, q2 in (("spearman", "spearman"), ("wf_spearman", "within_family_spearman"), ("r2", "r2_oos"),
              ("auroc", "auroc"), ("rmse", "rmse")):
    cmp[f"LOFO|R|M|composite|{q}"] = abs(A["LOFO|R|M|composite"][q] - o["LOFO"]["R"]["M"]["composite"][q2])
    for name in OUT_SUB:
        a_, b_ = A["OUT|W|M|composite"][name].get(q), o["OUT"]["W"]["M"]["composite"][name].get(q2)
        if a_ is not None and b_ is not None:
            cmp[f"OUT|W|M|composite|{name}|{q}"] = abs(a_ - b_)
for c in READ + ["drift_size"]:
    oc = [x["slope"] for x in ORIG["coefficients"]["LOFO"]["W"][c]]
    for t in TARGETS:
        cmp[f"slopes|{c}|{t}"] = max(abs(a_ - b_[t]) for a_, b_ in zip(slopes[f"{c}|{t}"], oc))
RES["A_recompute"] = A
RES["A_max_abs_diff_vs_original"] = max(cmp.values())
RES["A_worst"] = sorted(cmp.items(), key=lambda kv: -kv[1])[:5]

# ================================================================== B. in-sample equivalence
B = {}
for t in TARGETS:
    y = Y(POOL, t)
    per = {f: float(spearmanr(comp[g], y[g])[0]) for f, g in zip(fams, groups)}
    B[t] = dict(insample_within_family_spearman_of_fixed_composite=float(np.mean(list(per.values()))), per_family=per,
                lofo_within_family=A[f"LOFO|W|{t}|composite"]["wf_spearman"],
                lofo_slopes_all_positive=bool(all(np.polyfit(comp[np.setdiff1d(np.arange(len(POOL)), g)],
                                                             y[np.setdiff1d(np.arange(len(POOL)), g)], 1)[0] > 0
                                                  for g in groups)))
RES["B_insample_equivalence"] = B

# ================================================================== C. sign/reading-set selection
C = {}
for t in TARGETS:
    y = Y(POOL, t)
    m = Y(POOL, "M")
    rng = np.random.default_rng([7, TARGETS.index(t)])
    pi = perm_idx(groups, len(POOL), NPERM, rng)
    pi = np.concatenate([np.arange(len(POOL))[:, None], pi], axis=1)     # column 0 = observed
    yp, mp = y[pi], m[pi]
    Bc = pi.shape[1]
    rho = fam_rho(ZW, yp, groups)                                       # (F, 6, B)
    res_t = {}

    def run(signs_fn, label):
        """signs_fn(fold_g or None) -> (6, B) weights (0 = reading dropped); composite = sum w z / #used."""
        yh = np.zeros((len(POOL), Bc)); y0 = np.zeros((len(POOL), Bc))
        sign_record = []
        for gi, g in enumerate(groups):
            W = signs_fn(gi)                                            # (6, B)
            nuse = np.abs(W).sum(0)
            S = (ZW[:, [FI[c] for c in READ]] @ W) / nuse               # (n, B)
            T = np.setdiff1d(np.arange(len(POOL)), g)
            sT, yT = S[T], yp[T]
            ms, my = sT.mean(0), yT.mean(0)
            b = ((sT - ms) * (yT - my)).sum(0) / ((sT - ms) ** 2).sum(0)
            yh[g] = my + b * (S[g] - ms)
            y0[g] = my
            sign_record.append(W[:, 0].tolist())
        mm = metrics(yh, y0, yp, mp, groups)
        r = {}
        for q in ("spearman", "wf_spearman", "r2", "auroc", "rmse"):
            r[q] = float(mm[q][0])
            if q != "rmse":
                r[q + "_p"] = pval(mm[q][0], mm[q][1:])
                r[q + "_null_median"] = float(np.nanmedian(mm[q][1:]))
                r[q + "_null_95pct"] = float(np.nanpercentile(mm[q][1:], 95))
        r["observed_weights_by_fold(readings order)"] = sign_record
        res_t[label] = r
        return mm

    paper_w = np.array([[PAPER_SIGN.get(c, 0)] for c in READ], float) * np.ones((1, Bc))

    def drop_weakest(rb):
        """rb (6, B) mean rho -> weights: sign(rb), the weakest |rb| set to 0."""
        w = np.sign(rb)
        w[np.argmin(np.abs(rb), axis=0), np.arange(rb.shape[1])] = 0
        return w

    run(lambda gi: paper_w, "C_fixed: paper signs, five readings, null with signs fixed (= predict.py)")
    full_rb = rho.mean(0)
    # observed composite stays the paper's; the null re-selects signs and the dropped reading on the
    # whole permuted pool (as the paper's sign choice was made on all 34 organisms)
    w_full = drop_weakest(full_rb)
    w_full[:, 0] = paper_w[:, 0]
    run(lambda gi: w_full, "C_full: paper composite; null re-selects signs and drops weakest on the whole pool")

    def infold(gi, drop):
        rb = np.delete(rho, gi, axis=0).mean(0)
        return drop_weakest(rb) if drop else np.sign(rb)
    run(lambda gi: infold(gi, False), "C_infold6: in-fold signs, all six readings (observed and null)")
    run(lambda gi: infold(gi, True), "C_infold5: in-fold signs, weakest reading dropped in-fold (observed and null)")
    C[t] = res_t
RES["C_sign_selection"] = C

# ================================================================== D. controls
D = {}
y = Y(POOL, "M")
yhW, y0W = lofo(comp[:, None], y[:, None], groups)
yhW, y0W = yhW[:, 0], y0W[:, 0]
nc = np.array([a for a, k in enumerate(POOL) if not ctrl(k)])
D["controls"] = [f"{k[0]} | {k[1]}" for k in POOL if ctrl(k)]
fams_nc, groups_nc_local = fam_groups([POOL[a] for a in nc])
rng = np.random.default_rng(11)
# D_eval: reported predictions on the non-controls; null = refit LOFO under full-pool within-family shuffles
pi = perm_idx(groups, len(POOL), NPERM, rng)
yp = y[pi]
Sp = np.repeat(comp[:, None], NPERM, 1)
yhp, y0p = lofo(Sp, yp, groups)
obs = metrics(yhW[nc][:, None], y0W[nc][:, None], y[nc][:, None], y[nc][:, None], groups_nc_local)
nl = metrics(yhp[nc], y0p[nc], yp[nc], yp[nc], groups_nc_local)
D["D_eval_noncontrols"] = dict(n=len(nc), n_pos=int((y[nc] > 0.05).sum()),
                               **{q: float(obs[q][0]) for q in ("spearman", "wf_spearman", "r2", "auroc")},
                               **{q + "_p": pval(obs[q][0], nl[q]) for q in ("spearman", "wf_spearman", "r2", "auroc")})
# D_refit: drop the controls from the pool, z-score within family over non-controls, LOFO
P2 = [POOL[a] for a in nc]
f2, g2 = fam_groups(P2)
Z2 = zscore_within(P2, lambda k: [p for p in P2 if p[0] == k[0]])
comp2 = np.mean([PAPER_SIGN[c] * Z2[:, FI[c]] for c in FIVE], axis=0)
y2 = Y(P2, "M")
yh2, y02 = lofo(comp2[:, None], y2[:, None], g2)
pi2 = perm_idx(g2, len(P2), NPERM, np.random.default_rng(12))
yp2 = y2[pi2]
yh2p, y02p = lofo(np.repeat(comp2[:, None], NPERM, 1), yp2, g2)
obs = metrics(yh2, y02, y2[:, None], y2[:, None], g2)
nl = metrics(yh2p, y02p, yp2, yp2, g2)
D["D_refit_without_controls"] = dict(n=len(P2), n_pos=int((y2 > 0.05).sum()),
                                     **{q: float(obs[q][0]) for q in ("spearman", "wf_spearman", "r2", "auroc", "rmse", "rmse_null")},
                                     **{q + "_p": pval(obs[q][0], nl[q]) for q in ("spearman", "wf_spearman", "r2", "auroc")},
                                     per_family={f: float(spearmanr(yh2[g, 0], y2[g])[0]) for f, g in zip(f2, g2)})
# D_pos: inside the misaligned organisms only (M > 0.05), reported predictions held fixed, M shuffled within family
pos = np.array([a for a in range(len(POOL)) if y[a] > 0.05])
fp, gp = fam_groups([POOL[a] for a in pos])
pip = perm_idx(gp, len(pos), NPERM, np.random.default_rng(13))
obs_s = col_spearman(yhW[pos][:, None], y[pos][:, None])[0]
obs_w = np.mean([spearmanr(yhW[pos][g], y[pos][g])[0] for g in gp if len(g) >= 3])
nl_s = col_spearman(np.repeat(yhW[pos][:, None], NPERM, 1), y[pos][pip])
nl_w = np.nanmean(np.vstack([col_spearman(np.repeat(yhW[pos][g][:, None], NPERM, 1), y[pos][pip][g])
                             for g in gp if len(g) >= 3]), axis=0)
D["D_pos_only"] = dict(n=len(pos), spearman=float(obs_s), spearman_p=pval(obs_s, nl_s), wf_spearman=float(obs_w),
                       wf_spearman_p=pval(obs_w, nl_w),
                       per_family={f: float(spearmanr(yhW[pos][g], y[pos][g])[0]) for f, g in zip(fp, gp)})
# AUROC on non-controls: which negatives remain
D["noncontrol_negatives"] = [f"{POOL[a][0]} | {POOL[a][1]} M={y[a]:.4f} pred={yhW[a]:.4f}" for a in nc if y[a] <= 0.05]
RES["D_controls"] = D

# ================================================================== E. outside: standardisation set
E = {}


def eval_alt(ref_of, rows_name, label):
    Zalt = zscore_within(OUTS, ref_of)
    calt = np.mean([PAPER_SIGN[c] * Zalt[:, FI[c]] for c in FIVE], axis=0)
    yh_alt = a_all + b_all * calt
    E[label] = out_eval(yh_alt, y0O, OUT_SUB[rows_name], np.random.default_rng(21))
    E[label]["per_organism"] = {f"{OUTS[i][0]} | {OUTS[i][1]}": dict(pred=float(yh_alt[i]), M=float(yO[i]))
                                for i in OUT_SUB[rows_name]}


def lora_only_ref(k):
    if k[0] == "Qwen2.5-14B":
        return [p for p in POOL if p[0] == "Qwen2.5-14B"]
    same = [o_ for o_ in OUTBY[k[0]] if tgroup(o_) == tgroup(k)]
    return same if len(same) >= 3 else OUTBY[k[0]]


eval_alt(lora_only_ref, "LoRA, narrow", "narrow LoRA, z over the family's narrow-LoRA outside organisms only")
eval_alt(lambda k: OUTBY[k[0]], "Qwen2.5-14B (other tunings)", "Qwen2.5-14B other tunings, z over themselves (7)")
eval_alt(lambda k: OUTBY[k[0]], "all outside", "all outside, every block z-scored over its own outside organisms")
RES["E_outside_standardisation"] = E

# ================================================================== F. per-organism claims
F = {}
negs = [a for a in range(len(POOL)) if y[a] <= 0.05]
poss = [a for a in range(len(POOL)) if y[a] > 0.05]
inv = [(f"{POOL[n][0]} | {POOL[n][1]} pred {yhW[n]:.4f}", f"{POOL[p][0]} | {POOL[p][1]} pred {yhW[p]:.4f}")
       for n in negs for p in poss if yhW[n] >= yhW[p]]
F["inverted_pairs_LOFO_W_composite_M"] = inv
F["n_pairs"] = len(negs) * len(poss)
q14 = [i for i, k in enumerate(OUTS) if k[0] == "Qwen2.5-14B"]
F["Qwen2.5-14B_other_tunings_sorted_by_pred"] = [f"{OUTS[i][1]}: pred {yhO[i]:+.4f}, M {yO[i]:.4f}"
                                                  for i in sorted(q14, key=lambda i: yhO[i])]
F["full_finetunes"] = {f"{OUTS[i][0]} | {OUTS[i][1]}": dict(pred=float(yhO[i]), M=float(yO[i]))
                       for i in OUT_SUB["full fine-tune"]}
RES["F_per_organism"] = F

# ================================================================== G. bootstrap with a fresh stream
rng = np.random.default_rng(99)
bs, ba = [], []
for _ in range(5000):
    ii = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
    bs.append(spearmanr(yhW[ii], y[ii])[0])
    lab = y[ii] > 0.05
    ba.append(col_auroc(yhW[ii][:, None], lab[:, None])[0])
RES["G_bootstrap_LOFO_W_composite_M"] = dict(spearman_ci95=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                                             auroc_ci95=[float(np.nanpercentile(ba, 2.5)), float(np.nanpercentile(ba, 97.5))])

# ================================================================== H. U and G from M, B, C
keys = POOL + OUTS
dU = max(abs(MEAS[k]["U"] - (1 - (1 - MEAS[k]["M"]) * MEAS[k]["C"])) for k in keys)
dG = max(abs(MEAS[k]["G"] - MEAS[k]["M"] * MEAS[k]["B"]) for k in keys)
RES["H_identities"] = {"max |U - (1 - (1 - M) C)|": dU, "max |G - M B|": dG}

(HERE / "adversarial.json").write_text(json.dumps(RES, indent=1, default=float))
print(json.dumps({k: v for k, v in RES.items() if k not in ("A_recompute",)}, indent=1, default=float)[:20000])
