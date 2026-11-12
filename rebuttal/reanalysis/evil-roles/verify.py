#!/usr/bin/env python
"""verify: re-derive what evil_roles.json reports through code paths that evil_roles.py does not use.

  1. the freeze: the sha256 of the frozen lists, and that the a-priori sets are the lists cut to the cast
  2. the evil sets of every rule (all 18, leave-one-family-out, outside the pool) from a pandas
     re-implementation of the per-role rate (shrinkage, eligibility, top 4 with name order breaking ties)
  3. the slip of every variant for every organism of two families (Qwen2.5-7B, Qwen3-32B) straight from
     the .pt files and the baseline's axis.pt, without core
  4. every pooled rho_bar and r_bar of the table from per_organism values and measures.csv with scipy
  5. one p that is not at the Monte Carlo floor (ivb|U), by a fresh within-family shuffle, other seed
  6. the random-4 null for U, M, G rebuilt draw by draw with explicit loops and scipy.spearmanr
Out: verify.json beside this file.
"""
import json
import pathlib
import sys

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
import hashlib  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

REPO = pathlib.Path("/Users/unrulyabstractions/work/_OLD/breaking-character-iaseai")
D = json.load(open(HERE / "evil_roles.json"))
FZ = json.load(open(HERE / "apriori_freeze.json"))
CORE_REPRO = json.load(open(HERE.parent / "core/reproduce.json"))
out = {}

# 1. freeze ------------------------------------------------------------------------------------
blob = json.dumps(dict(strict=FZ["strict"], broad=FZ["broad"], rule_strict=FZ["rule_strict"],
                       rule_broad=FZ["rule_broad"]), sort_keys=True).encode()
casts = {f: c["roles"] for f, c in CORE_REPRO["casts"].items()}
out["freeze_sha_ok"] = hashlib.sha256(blob).hexdigest() == FZ["sha256"] == D["freeze"]["sha256"]
out["apriori_sets_ok"] = all(
    D["sets"]["i_strict"][f] == sorted(set(FZ["strict"]) & set(casts[f])) and
    D["sets"]["i_broad"][f] == sorted(set(FZ["broad"]) & set(casts[f])) for f in casts)

# 2. evil sets from pandas ----------------------------------------------------------------------
J = pd.DataFrame(json.load(open(REPO / "data/raw/perrole_em_judge_scores.json")))
J[["org", "role", "pi", "crit", "judge"]] = J["cid"].str.split("|", expand=True)
J = J[J.judge == "m"].pivot_table(index=["org", "role", "pi"], columns="crit", values="score", aggfunc="first").reset_index()
J = J.dropna(subset=["aligned", "coherent"])
J = J[J.coherent > 50]
fam_of = {j["folder"]: j["family"] for j in D["judged_organisms"]}
inpool = {j["folder"]: j["in_pool"] for j in D["judged_organisms"]}


def top4(orgs, cast):
    s = J[J.org.isin(orgs)]
    g = s.groupby("role").agg(den=("aligned", "size"), num=("aligned", lambda x: int((x < 30).sum())))
    grand = g.num.sum() / g.den.sum()
    g = g[g.den >= 20]
    g["rate"] = (g.num + 20.0 * grand) / (g.den + 20.0)
    g = g[g.index.isin(cast)].reset_index().sort_values(["rate", "role"], ascending=[False, True])
    return sorted(g.role.head(max(4, int(round(0.01 * len(cast))))).tolist())


sets_ok = {}
for f, cast in casts.items():
    sets_ok[f] = dict(
        published=top4(list(fam_of), cast) == D["sets"]["published"][f],
        lofo=top4([o for o in fam_of if fam_of[o] != f], cast) == D["sets"]["ii_lofo"][f],
        outside=top4([o for o in fam_of if not inpool[o]], cast) == D["sets"]["iib_outside_pool"][f])
out["sets_pandas"] = sets_ok
out["sets_pandas_ok"] = all(all(v.values()) for v in sets_ok.values())

# 3. slips from the raw files ------------------------------------------------------------------
import torch  # noqa: E402
sys.path.insert(0, str(REPO / "analysis/figures"))
from _vectors import FAM  # noqa: E402
rows = {}
for r in json.load(open(REPO / "data/raw/graph_features.json"))["rows"]:
    rows[(r["f"], r["dep"])] = int(r["layer"])


def vec(p, row):
    o = torch.load(p, map_location="cpu", weights_only=False)
    v = o["vector"] if isinstance(o, dict) and "vector" in o else o
    return v.float()[row].double().numpy()


raw_err = 0.0
raw_n = 0
for f in ("Qwen2.5-7B", "Qwen3-32B"):
    row = rows[(f, "50%")]
    cast = casts[f]
    base = pathlib.Path(FAM[f]["base"])
    ax = vec(base / "axis.pt", row)
    ax = ax / np.linalg.norm(ax)
    XB = np.stack([vec(base / "vectors" / f"{n}.pt", row) for n in cast])
    ai = cast.index("assistant")
    for o in D["pool"][f]:
        XO = np.stack([vec(pathlib.Path(FAM[f]["orgs"][o]) / "vectors" / f"{n}.pt", row) for n in cast])
        S = XO - XB
        a = S @ ax / np.linalg.norm(S, axis=1)
        for k in ("published", "i_strict", "i_broad", "ii_lofo", "iib_outside_pool"):
            E = [cast.index(e) for e in D["sets"][k][f]]
            raw_err = max(raw_err, abs((a[ai] - a[E].mean()) - D["per_organism"][k][f][o]))
            raw_n += 1
        raw_err = max(raw_err, abs((a[ai] - a.mean()) - D["per_organism"]["iva_cast_mean_contrast"][f][o]),
                      abs(a.mean() - D["per_organism"]["ivb_cast_mean_a"][f][o]),
                      abs((a[ai] - np.median(a)) - D["per_organism"]["xi_A"][f][o]))
        raw_n += 3
out["raw_slip_max_abs_err"] = raw_err
out["raw_slip_values_checked"] = raw_n
out["raw_slip_ok"] = raw_err <= 1e-9

# 4. pooled coefficients with scipy -------------------------------------------------------------
meas = pd.read_csv(REPO / "data/measures.csv").set_index(["family", "organism"])
fams = sorted(D["pool"])
coef_err, coef_ok = 0.0, True
for t in D["table"]:
    k, m = t["variant"], t["measure"]
    x = {f: np.array([D["per_organism"][k][f][o] for o in D["pool"][f]]) for f in fams}
    y = {f: np.array([meas.loc[(f, o), m] for o in D["pool"][f]]) for f in fams}
    rho = np.mean([spearmanr(x[f], y[f])[0] for f in fams])
    r = np.mean([pearsonr(x[f], y[f])[0] for f in fams])
    if round(rho, 4) != t["rho_bar"] or round(r, 4) != t["r_bar"]:
        coef_ok = False
    coef_err = max(coef_err, abs(rho - t["rho_bar"]), abs(r - t["r_bar"]))
out["coef_ok"] = coef_ok
out["coef_max_abs_err_vs_rounded"] = coef_err

# 5. a fresh permutation p for the one cell off the floor ---------------------------------------
rng = np.random.default_rng(12345)
k, m = "ivb_cast_mean_a", "U"
x = {f: np.array([D["per_organism"][k][f][o] for o in D["pool"][f]]) for f in fams}
y = {f: np.array([meas.loc[(f, o), m] for o in D["pool"][f]]) for f in fams}
obs = np.mean([spearmanr(x[f], y[f])[0] for f in fams])
NP = 50_000
from scipy.stats import rankdata  # noqa: E402
acc = np.zeros(NP)
for f in fams:
    rx, ry = rankdata(x[f]), rankdata(y[f])
    perm = np.argsort(rng.random((NP, len(ry))), axis=1)
    Y = ry[perm]
    Yc = Y - Y.mean(1, keepdims=True)
    xc = rx - rx.mean()
    acc += (Yc * xc).sum(1) / np.sqrt((Yc ** 2).sum(1) * (xc ** 2).sum())
acc /= len(fams)
p_fresh = (int((np.abs(acc) >= abs(obs) - 1e-12).sum()) + 1) / (NP + 1)
p_tab = [t for t in D["table"] if t["variant"] == k and t["measure"] == m][0]["p"]
out["p_fresh_ivb_U"] = dict(fresh=p_fresh, table=p_tab, draws=NP, se=float(np.sqrt(p_tab * (1 - p_tab) / NP)))
out["p_fresh_ok"] = abs(p_fresh - p_tab) < 5 * np.sqrt(p_tab * (1 - p_tab) / NP)

# 6. the random-4 null, draw by draw ------------------------------------------------------------
sys.path.insert(0, str(HERE.parent / "core"))
import core  # noqa: E402
A = {}
for f in fams:
    names = casts[f]
    v = core.default_axis(f)
    vh = v / np.linalg.norm(v)
    XB, _ = core.load_cloud(f, "base", names=names)
    M = []
    for o in D["pool"][f]:
        XO, _ = core.load_cloud(f, o, names=names)
        S = np.asarray(XO, float) - np.asarray(XB, float)
        M.append(S @ vh / np.linalg.norm(S, axis=1))
    A[f] = np.array(M)
rng = np.random.default_rng(D["null"]["seed"])
cand = {f: np.array([i for i, n in enumerate(casts[f]) if n not in ("assistant", "default")]) for f in fams}
draws = {m: [] for m in ("U", "M", "G")}
for d in range(D["null"]["n"]):
    E = {f: rng.choice(cand[f], size=D["null"]["size"], replace=False) for f in fams}
    for m in draws:
        rr = []
        for f in fams:
            ai = casts[f].index("assistant")
            s = [A[f][i, ai] - A[f][i, E[f]].mean() for i in range(len(D["pool"][f]))]
            rr.append(spearmanr(s, [meas.loc[(f, o), m] for o in D["pool"][f]])[0])
        draws[m].append(float(np.mean(rr)))
null_err = max(max(abs(a - b) for a, b in zip(draws[m], D["null"]["cells"][f"random4|{m}"]["draws_rho_bar"]))
               for m in draws)
out["null_rebuilt_max_abs_err"] = null_err
pct = {}
TOL = 1e-9     # lattice ties: a draw equal to the observed mean counts as at least as negative
for m in draws:
    arr = np.array(draws[m])
    for t in D["table"]:
        if t["measure"] == m:
            k = t["variant"]
            x = {f: np.array([D["per_organism"][k][f][o] for o in D["pool"][f]]) for f in fams}
            obs = np.mean([spearmanr(x[f], [meas.loc[(f, o), m] for o in D["pool"][f]])[0] for f in fams])
            pct[f"{k}|{m}"] = (float((arr <= obs + TOL).mean()), t["null4_share_rho_le"],
                               int((np.abs(arr - obs) <= TOL).sum()), t["null4_ties"])
out["null_percentiles_rebuilt_vs_table"] = pct
out["null_ok"] = null_err <= 5e-7 and all(a == b and c == d for a, b, c, d in pct.values())

# 7. the size-matched nulls of (i), draw by draw ------------------------------------------------
sm = {}
for k in ("i_strict", "i_broad"):
    seed = D["null"]["cells"][f"size_matched_{k}|U"]["seed"]
    size_of = D["null"]["cells"][f"size_matched_{k}|U"]["sizes"]
    rng = np.random.default_rng(seed)
    dr = {m: [] for m in ("U", "M", "G")}
    for d in range(D["null"]["n"]):
        E = {f: rng.choice(cand[f], size=size_of[f], replace=False) for f in fams}
        for m in dr:
            rr = []
            for f in fams:
                ai = casts[f].index("assistant")
                s = [A[f][i, ai] - A[f][i, E[f]].mean() for i in range(len(D["pool"][f]))]
                rr.append(spearmanr(s, [meas.loc[(f, o), m] for o in D["pool"][f]])[0])
            dr[m].append(float(np.mean(rr)))
    for m in dr:
        arr = np.array(dr[m])
        x = {f: np.array([D["per_organism"][k][f][o] for o in D["pool"][f]]) for f in fams}
        obs = np.mean([spearmanr(x[f], [meas.loc[(f, o), m] for o in D["pool"][f]])[0] for f in fams])
        cell = D["null"]["cells"][f"size_matched_{k}|{m}"]
        sm[f"{k}|{m}"] = dict(share=float((arr <= obs + TOL).mean()), table=cell["observed"][k]["share_null_rho_le_obs"],
                              mean=float(arr.mean()), table_mean=cell["mean"])
out["size_matched_rebuilt"] = sm
out["size_matched_ok"] = all(v["share"] == v["table"] and abs(v["mean"] - v["table_mean"]) <= 1e-9 for v in sm.values())

# 8. the per-family coefficients of every pooled cell -------------------------------------------
pf_err = 0.0
for key, c in D["pooled"].items():
    k, m = key.split("|")
    for f in fams:
        x = [D["per_organism"][k][f][o] for o in D["pool"][f]]
        y = [meas.loc[(f, o), m] for o in D["pool"][f]]
        pf_err = max(pf_err, abs(spearmanr(x, y)[0] - c["per_family_rho"][f]))
out["per_family_max_abs_err_vs_rounded"] = pf_err
out["per_family_ok"] = pf_err <= 5.0001e-5
out = {k: (bool(v) if isinstance(v, np.bool_) else v) for k, v in out.items()}
out["ALL_OK"] = all(v for k, v in out.items() if k.endswith("_ok"))
(HERE / "verify.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=1))
print("p fresh", out["p_fresh_ivb_U"])
print("sets", out["sets_pandas"])
