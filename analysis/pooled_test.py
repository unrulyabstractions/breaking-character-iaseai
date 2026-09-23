#!/usr/bin/env python
"""Stage 3, analysis: the paper's headline test.

In:  data/measures.csv, data/readings.csv
Out: data/pooled_stats.json, and a table on stdout
Feeds: Section 4.2 and the caption of Figure 6.

A family of three to fourteen organisms is too small for an asymptotic Spearman p, so significance
comes from a pooled within-family permutation test: the measure is shuffled inside each family, the
statistic is the mean within-family Spearman, and the null is enumerated exactly up to two million
arrangements and drawn 200,000 times at random above that. Coefficients are reported in pairs, mean
Spearman and mean Pearson, with a two-sided p (author, 2026-09-18: "go two-tailed everywhere").
"""
import csv, itertools, json, math, os, pathlib
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

ROOT = pathlib.Path(__file__).resolve().parent.parent
# the output lands under $OUT_DIR/data when it is set, so a check run leaves the tracked copy alone
OUT = (pathlib.Path(os.environ["OUT_DIR"]) / "data" if os.environ.get("OUT_DIR") else ROOT / "data") / "pooled_stats.json"
MAX_EXACT, N_MC, SEED = 2_000_000, 200_000, 0
SIX_FAMILIES = ["Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-7B", "Qwen3.5-9B", "Qwen3-32B"]
READINGS = ["assistant_shift_deviation", "assistant_evil_slip", "orientation_distortion",
            "radial_distortion", "axial_contraction", "knn_retention"]


def load(depth="50%", lora_only=True):
    meas = {(r["family"], r["organism"]): r for r in csv.DictReader(open(ROOT / "data/measures.csv"))}
    fam = {}
    for r in csv.DictReader(open(ROOT / "data/readings.csv")):
        if r["depth"] != depth or (lora_only and r["tuning"] != "lora"):
            continue
        if r["family"] not in SIX_FAMILIES or (r["family"], r["organism"]) not in meas:
            continue
        fam.setdefault(r["family"], []).append((r, meas[(r["family"], r["organism"])]))
    return {f: rows for f, rows in fam.items() if len(rows) >= 3}


def _rho_of_ranks(rx, ry_block):
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry_block - ry_block.mean(axis=1, keepdims=True)) / ry_block.std(axis=1, keepdims=True)
    return (ry * rx).mean(axis=1)


def family_null(x, y, rng):
    """(every exact rho | None, a Monte Carlo sample of rhos) for one family."""
    rx, ry = rankdata(x), rankdata(y)
    if math.factorial(len(y)) <= 40_320:            # 8! enumerated per family
        perms = np.array(list(itertools.permutations(range(len(y)))))
        exact = _rho_of_ranks(rx, ry[perms])
        return exact, rng.choice(exact, size=N_MC)
    idx = np.argsort(rng.random((N_MC, len(y))), axis=1)
    return None, _rho_of_ranks(rx, ry[idx])


def pooled(reading, measure, fam, sign):
    """Mean within-family Spearman and Pearson, and the two-sided permutation p; `sign` only orients the printed means."""
    rng = np.random.default_rng(SEED)
    rhos, rs, nulls = [], [], {}
    for f, rows in sorted(fam.items()):
        x = np.array([float(a[reading]) * sign for a, _ in rows])
        y = np.array([float(b[measure]) for _, b in rows])
        rhos.append(spearmanr(x, y)[0]); rs.append(pearsonr(x, y)[0])
        nulls[f] = family_null(x, y, rng)
    obs = float(np.mean(rhos))
    total = math.prod(math.factorial(len(rows)) for rows in fam.values())
    if total <= MAX_EXACT and all(n[0] is not None for n in nulls.values()):
        le = sum(1 for c in itertools.product(*[nulls[f][0] for f in sorted(fam)])
                 if abs(sum(c) / len(c)) >= abs(obs) - 1e-12)
        p, how = le / total, "exact"
    else:
        draws = np.stack([nulls[f][1] for f in sorted(fam)], axis=1).mean(axis=1)
        p, how = (int((np.abs(draws) >= abs(obs) - 1e-12).sum()) + 1) / (N_MC + 1), "monte_carlo"
    return dict(rho_bar=round(float(np.mean(rhos)) * sign, 4), r_bar=round(float(np.mean(rs)) * sign, 4),
                p=round(p, 6), method=how, families=len(fam), arrangements=total)


def main():
    fam = load()
    out = {"depth": "50%", "families": sorted(fam), "n_organisms": sum(len(v) for v in fam.values()), "cells": {}}
    print(f"{'reading':26s} {'vs G: rho / r / p':30s} {'vs U: rho / r / p':30s}")
    for reading in READINGS:
        row = []
        for measure in ("G", "U"):
            # orient so the observed mean is negative; the p is two-sided either way
            probe = pooled(reading, measure, fam, 1)
            sign = 1 if probe["rho_bar"] < 0 else -1
            cell = pooled(reading, measure, fam, sign)
            out["cells"][f"{reading}|{measure}"] = cell
            row.append(f"{cell['rho_bar']:+.2f} / {cell['r_bar']:+.2f} / {cell['p']:.4f}")
        print(f"{reading:26s} {row[0]:30s} {row[1]:30s}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n{out['n_organisms']} organisms in {len(fam)} families; wrote {OUT}")


if __name__ == "__main__":
    main()
