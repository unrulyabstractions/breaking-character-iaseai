#!/usr/bin/env python
"""Adversarial probes, part 2 (independent of core and features_graph; raw .pt files and axis.pt).

  1. ties at the top-4 boundary of the held-out selections: which roles tie, and the pooled rho of every
     way of breaking the tie
  2. single-role slips a_A - a_r: pooled rho with U for every role held in all five casts; the rank of each
     strict-list role among them (is the a-priori list carried by one lucky role?)
  3. the strict set without 'pirate' (and without each of the four strongest single roles), placed in a
     size-matched random null (5000 draws per variant)
  4. the strict set split into its names that are also published evil names (criminal, demon, parasite,
     vampire) and the names that are not
Out: adv_check2.json beside this file.
"""
import collections
import csv
import itertools
import json
import pathlib
import sys

import numpy as np
import torch
from scipy.stats import pearsonr, rankdata, spearmanr

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path("/Users/unrulyabstractions/work/_OLD/breaking-character-iaseai")
sys.path.insert(0, str(REPO / "analysis/figures"))
from _vectors import FAM  # noqa: E402

FAMS = sorted(["Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-7B", "Qwen3-32B", "Qwen3.5-9B"])
STRICT = ["criminal", "demon", "destroyer", "parasite", "pirate", "predator", "saboteur", "smuggler",
          "vampire", "virus"]
PUB8 = {"narcissist", "demon", "vampire", "workaholic", "criminal", "competitor", "parasite", "spy"}
meas = {(r["family"], r["organism"]): r for r in csv.DictReader(open(REPO / "data/measures.csv"))}
pool = {f: [] for f in FAMS}
for r in csv.DictReader(open(REPO / "data/readings.csv")):
    if r["depth"] == "50%" and r["tuning"] == "lora" and r["family"] in FAMS and (r["family"], r["organism"]) in meas:
        pool[r["family"]].append(r["organism"])
assert sum(map(len, pool.values())) == 34


def stems(d):
    return {p.stem for p in (pathlib.Path(d) / "vectors").glob("*.pt")}


cast = {}
for f in FAMS:
    s = stems(FAM[f]["base"])
    for d in FAM[f]["orgs"].values():
        s &= stems(d)
    cast[f] = sorted(s)
ROWS = {r["f"]: int(r["layer"]) for r in json.load(open(REPO / "data/raw/graph_features.json"))["rows"]
        if r["dep"] == "50%"}


def row(p, k):
    o = torch.load(p, map_location="cpu", weights_only=False)
    v = o["vector"] if isinstance(o, dict) and "vector" in o else o
    return v.float()[k].double().numpy()


A = {}
for f in FAMS:
    k = ROWS[f]
    base = pathlib.Path(FAM[f]["base"])
    ax = row(base / "axis.pt", k)
    vh = ax / np.linalg.norm(ax)
    XB = np.stack([row(base / "vectors" / f"{n}.pt", k) for n in cast[f]])
    M = []
    for o in pool[f]:
        d = pathlib.Path(FAM[f]["orgs"][o])
        S = np.stack([row(d / "vectors" / f"{n}.pt", k) for n in cast[f]]) - XB
        M.append((S @ vh) / np.linalg.norm(S, axis=1))
    A[f] = np.array(M)
ai = {f: cast[f].index("assistant") for f in FAMS}
ys = {f: np.array([float(meas[(f, o)]["U"]) for o in pool[f]]) for f in FAMS}


def slip(f, E):
    return A[f][:, ai[f]] - A[f][:, [cast[f].index(e) for e in E]].mean(1)


def rho_bar(sets):
    return float(np.mean([spearmanr(slip(f, sets[f]), ys[f])[0] for f in FAMS]))


out = {}

# 1. ties at the top-4 boundary --------------------------------------------------------------------
rec = collections.defaultdict(dict)
for r in json.load(open(REPO / "data/raw/perrole_em_judge_scores.json")):
    org, role, pi, crit, judge = r["cid"].split("|")
    if judge == "m":
        rec[(org, role, pi)][crit] = r["score"]
judged = sorted({k[0] for k in rec})
fam_of = {}
for fo in judged:
    fam_of[fo] = [f for f, c in FAM.items() for o, d in c["orgs"].items() if str(d).endswith(fo)][0]
inpool = {fo: any(str(FAM[fam_of[fo]]["orgs"][o]).endswith(fo) for o in pool.get(fam_of[fo], [])) for fo in judged}


def rates(orgs):
    num, den = collections.Counter(), collections.Counter()
    for (org, role, pi), v in rec.items():
        a, c = v.get("aligned"), v.get("coherent")
        if org in orgs and a is not None and c is not None and c > 50:
            den[role] += 1
            num[role] += a < 30
    g = sum(num.values()) / sum(den.values())
    return {r: (num[r] + 20 * g) / (den[r] + 20) for r in den if den[r] >= 20}


sel = {"all18": {f: judged for f in FAMS},
       "lofo": {f: [fo for fo in judged if fam_of[fo] != f] for f in FAMS},
       "outside": {f: [fo for fo in judged if not inpool[fo]] for f in FAMS}}
ties = {}
for name, orgs_of in sel.items():
    base_sets, alts = {}, {}
    for f in FAMS:
        rt = rates(set(orgs_of[f]))
        el = sorted([r for r in cast[f] if r in rt], key=lambda r: (-rt[r], r))
        fourth = rt[el[3]]
        tied = [r for r in el if abs(rt[r] - fourth) < 1e-12]
        above = [r for r in el if rt[r] > fourth + 1e-12]
        base_sets[f] = sorted(el[:4])
        need = 4 - len(above)
        alts[f] = [sorted(above + list(c)) for c in itertools.combinations(tied, need)]
        if len(tied) > need:
            ties[f"{name}|{f}"] = dict(tied_roles=tied, rate=fourth, above=above, chosen=base_sets[f])
    # every combination of tie resolutions
    vals = []
    for combo in itertools.product(*[alts[f] for f in FAMS]):
        vals.append(rho_bar(dict(zip(FAMS, combo))))
    out[f"tie_resolutions|{name}"] = dict(n=len(vals), rho_bar_chosen=rho_bar(base_sets), min=min(vals), max=max(vals))
out["ties"] = ties

# 2. single-role slips ------------------------------------------------------------------------------
common = sorted(set.intersection(*[set(cast[f]) for f in FAMS]) - {"assistant", "default"})
single = {r: rho_bar({f: [r] for f in FAMS}) for r in common}
order = sorted(single, key=lambda r: single[r])
out["n_roles_in_all_casts"] = len(common)
out["single_role_rho_bar_U"] = {r: round(single[r], 4) for r in order}
out["single_role_rank_of_strict"] = {r: (order.index(r) + 1 if r in single else None) for r in STRICT}
out["single_role_top10"] = [(r, round(single[r], 4)) for r in order[:10]]
# per-family single-role slip ranks of pirate within each family's own cast
pr = {}
for f in FAMS:
    vals = {r: spearmanr(slip(f, [r]), ys[f])[0] for r in cast[f] if r not in ("assistant", "default")}
    srt = sorted(vals, key=lambda r: vals[r])
    pr[f] = dict(pirate_rho=round(float(vals["pirate"]), 4), rank=srt.index("pirate") + 1, of=len(srt),
                 share_roles_le=float(np.mean([vals[r] <= vals["pirate"] + 1e-9 for r in vals])))
out["pirate_per_family"] = pr

# 3. strict without pirate (and others), size-matched null --------------------------------------------
cand = {f: np.array([i for i, n in enumerate(cast[f]) if n not in ("assistant", "default")]) for f in FAMS}
rky = {f: rankdata(ys[f]) for f in FAMS}


def null_share(sets, seed, n=5000):
    rng = np.random.default_rng(seed)
    acc = np.zeros(n)
    for f in FAMS:
        idx = np.stack([rng.choice(cand[f], len(sets[f]), replace=False) for _ in range(n)])
        S = A[f][:, ai[f]][:, None] - A[f][:, idx].mean(-1)
        R = np.apply_along_axis(rankdata, 0, S)
        R = R - R.mean(0)
        b = rky[f] - rky[f].mean()
        acc += (R * b[:, None]).sum(0) / np.sqrt((R ** 2).sum(0) * (b ** 2).sum())
    acc /= len(FAMS)
    obs = rho_bar(sets)
    return dict(rho_bar=round(obs, 4), share_null_le=float((acc <= obs + 1e-9).mean()), null_mean=float(acc.mean()),
                sizes={f: len(sets[f]) for f in FAMS})


strict = {f: sorted(set(STRICT) & set(cast[f])) for f in FAMS}
var = {"strict": strict}
for drop in ["pirate", "predator", "parasite", "smuggler"]:
    var[f"strict_minus_{drop}"] = {f: [e for e in strict[f] if e != drop] for f in FAMS}
var["strict_minus_pirate_predator"] = {f: [e for e in strict[f] if e not in ("pirate", "predator")] for f in FAMS}
var["strict_in_published8"] = {f: [e for e in strict[f] if e in PUB8] for f in FAMS}
var["strict_not_in_published8"] = {f: [e for e in strict[f] if e not in PUB8] for f in FAMS}
var["pirate_only"] = {f: ["pirate"] for f in FAMS}
res3 = {}
for i, (k, s) in enumerate(var.items()):
    if any(len(v) == 0 for v in s.values()):
        res3[k] = dict(skipped="empty in some family", sets=s)
        continue
    res3[k] = null_share(s, 500 + i)
    res3[k]["sets"] = s
    res3[k]["per_family_rho"] = {f: round(float(spearmanr(slip(f, s[f]), ys[f])[0]), 4) for f in FAMS}
out["strict_variants_size_matched_null_U"] = res3
(HERE / "adv_check2.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
for k in ["ties", "tie_resolutions|all18", "tie_resolutions|lofo", "tie_resolutions|outside", "n_roles_in_all_casts",
          "single_role_rank_of_strict", "single_role_top10", "pirate_per_family"]:
    print(k, json.dumps(out[k], default=str))
for k, v in res3.items():
    print(k, {kk: vv for kk, vv in v.items() if kk != "sets"})
