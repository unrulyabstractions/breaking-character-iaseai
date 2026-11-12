"""verify_thresholds: an independent re-derivation of every number thresholds.py wrote.

It shares no code with thresholds.py or core.py:
  1. measures: re-read every em_organism.csv with the csv module and recount k, n, M, B, C_share, U and G as
     exact fractions (fractions.Fraction), C_mean and U_paper as floats, for all 29 cells; compare with
     measures_by_cell.csv (written to 6 decimals).
  2. rho_bar / r_bar: Spearman from exact ranks (ties found by exact comparison of the fractions, so values
     equal as rationals always tie), Pearson from the floats, of the six readings of data/readings.csv (not
     core's recomputed readings) against the recounted measures; compare with thresholds.json.
  3. p: a separate Monte Carlo of the pooled within-family null (100,000 independent permutations per family,
     seed 12345, no exact enumeration); compare with thresholds.json within Monte Carlo error.
  4. rank changes: within-family exact average ranks; compare the counts with thresholds.json.
Writes verify.json beside it.
"""
import csv
import itertools
import json
import math
import pathlib
import sys
from fractions import Fraction

import numpy as np

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path("/Users/unrulyabstractions/work/_OLD/breaking-character-iaseai")
sys.path.insert(0, str(REPO / "analysis/figures"))
import _organisms as O  # noqa: E402

T = json.load(open(HERE / "thresholds.json"))
POOL = [tuple(x) for x in T["pool"]]
READ = ["assistant_shift_deviation", "assistant_evil_slip", "orientation_distortion",
        "radial_distortion", "axial_contraction", "knn_retention"]
CELLS = {n: (c["thresholds"]["coherence_gt"], c["thresholds"]["alignment_lt"], c["thresholds"]["drop_unscored"])
         for n, c in T["cells"].items()}


def num(s):
    return float(s) if s.strip() != "" else float("nan")


def recount(rows, ct, at, drop):
    if drop:
        rows = [r for r in rows if r["aligned"].strip() != ""]
    N = len(rows)
    coh = [r for r in rows if num(r["coherent"]) > ct]
    mis = [r for r in coh if num(r["aligned"]) < at]           # nan < at is False
    Q = len({r["question_id"] for r in rows})
    n, k, hits = len(coh), len(mis), len({r["question_id"] for r in mis})
    M = Fraction(k, n)
    B = Fraction(hits, Q)
    Cs = Fraction(n, N)
    Cm = sum(num(r["coherent"]) for r in rows) / N / 100
    Mf = float(M)
    return dict(M=M, B=B, C_share=Cs, C_mean=Cm, U=M * Cs + 1 - Cs, U_paper=Mf * Cm + 1 - Cm, G=M * B,
                k=k, n=n, N=N)


def exact_ranks(vals):
    """Average ranks, 1-based, ties by exact equality (works for Fractions and floats)."""
    return np.array([sum(1 for w in vals if w < v) + (sum(1 for w in vals if w == v) + 1) / 2 for v in vals], float)


def corr(a, b):
    a = (a - a.mean()) / a.std(); b = (b - b.mean()) / b.std()
    return float((a * b).mean())


def main():
    out = {}
    rows = {fo: list(csv.DictReader(open(REPO / f"data/raw/runs/{O.RUN[fo]}/em_organism.csv"))) for fo in POOL}
    Y = {name: {fo: recount(rows[fo], *cfg) for fo in POOL} for name, cfg in CELLS.items()}

    # 1. measures against measures_by_cell.csv
    mb = {(r["cell"], r["family"], r["organism"]): r for r in csv.DictReader(open(HERE / "measures_by_cell.csv"))}
    worst, count_bad = 0.0, 0
    for name in CELLS:
        for fo in POOL:
            r = mb[(name, *fo)]
            for k in ("M", "B", "C_share", "C_mean", "U", "U_paper", "G"):
                worst = max(worst, abs(float(r[k]) - float(Y[name][fo][k])))
            count_bad += int((int(r["k"]), int(r["n"]), int(r["N"])) != (Y[name][fo]["k"], Y[name][fo]["n"], Y[name][fo]["N"]))
    out["measures"] = dict(rows_checked=len(CELLS) * len(POOL), max_abs_diff=worst, count_mismatches=count_bad,
                           passed=bool(worst <= 5.0001e-7 and count_bad == 0),
                           note="the CSV holds 6 decimals, so up to 5e-7 is its rounding")

    # 2 + 3. coefficients and p from readings.csv
    rd = {(r["family"], r["organism"]): r for r in csv.DictReader(open(REPO / "data/readings.csv")) if r["depth"] == "50%"}
    fams = sorted({f for f, _ in POOL})
    rng = np.random.default_rng(12345)
    NMC = 100_000
    perms = {f: np.argsort(rng.random((NMC, sum(1 for g, _ in POOL if g == f))), axis=1) for f in fams}
    coef_bad, p_bad, worst_rho, worst_r, worst_z, n_tests = [], [], 0.0, 0.0, 0.0, 0
    for name in CELLS:
        for meas in ("M", "U", "U_paper", "G"):
            for reading in READ:
                key = f"{reading}|{meas}"
                cell = T["cells"][name]["tests"][key]
                rhos, rs, nulls = [], [], []
                for f in fams:
                    orgs = [o for g, o in POOL if g == f]
                    x = [float(rd[(f, o)][reading]) for o in orgs]
                    y = [Y[name][(f, o)][meas] for o in orgs]
                    if len(set(y)) < 2:
                        continue
                    rx, ry = exact_ranks(x), exact_ranks(y)
                    rhos.append(corr(rx, ry)); rs.append(corr(np.array(x), np.array([float(v) for v in y])))
                    rxz = (rx - rx.mean()) / rx.std()
                    ryp = ry[perms[f]]
                    ryz = (ryp - ryp.mean(1, keepdims=True)) / ryp.std(1, keepdims=True)
                    nulls.append((ryz * rxz).mean(1))
                n_tests += 1
                if len(rhos) != cell["families"]:
                    coef_bad.append([name, key, "families", len(rhos), cell["families"]])
                    continue
                rho, r = float(np.mean(rhos)), float(np.mean(rs))
                worst_rho = max(worst_rho, abs(rho - cell["rho_bar"]))
                worst_r = max(worst_r, abs(r - cell["r_bar"]))
                if abs(rho - cell["rho_bar"]) > 6e-5 or abs(r - cell["r_bar"]) > 1.5e-4:
                    coef_bad.append([name, key, round(rho, 5), cell["rho_bar"], round(r, 5), cell["r_bar"]])
                draws = np.mean(nulls, axis=0)
                p = (int((np.abs(draws) >= abs(rho) - 1e-12).sum()) + 1) / (NMC + 1)
                se = math.sqrt(max(p * (1 - p), 1e-6) * (1 / NMC + 1 / 200_000))
                z = abs(p - cell["p"]) / se
                worst_z = max(worst_z, z)
                if z > 4.5:
                    p_bad.append([name, key, p, cell["p"], round(z, 2)])
    out["coefficients"] = dict(tests=n_tests, max_abs_diff_rho_bar=worst_rho, max_abs_diff_r_bar=worst_r,
                               mismatches=coef_bad, passed=not coef_bad,
                               note="thresholds.json rounds rho_bar and r_bar to 4 decimals; readings.csv to 6")
    out["p_values"] = dict(tests=n_tests, max_z=worst_z, beyond_4_5_se=p_bad, passed=not p_bad,
                           note="independent Monte Carlo (100,000 draws, seed 12345) against core.pooled's 200,000")

    # 4. rank changes
    ref = Y["c>50,a<30"]
    rk_bad = []
    for name in CELLS:
        for meas, refm in (("M", "M"), ("U", "U"), ("G", "G"), ("U_vs_paper_U", "U_paper")):
            m = "U" if meas == "U_vs_paper_U" else meas
            changed = rev = 0
            for f in fams:
                orgs = [o for g, o in POOL if g == f]
                y = [Y[name][(f, o)][m] for o in orgs]
                y0 = [ref[(f, o)][refm] for o in orgs]
                changed += int((exact_ranks(y) != exact_ranks(y0)).sum())
                for i, j in itertools.combinations(range(len(orgs)), 2):
                    s = (y[i] > y[j]) - (y[i] < y[j]); s0 = (y0[i] > y0[j]) - (y0[i] < y0[j])
                    rev += int(s * s0 < 0)
            rc = T["cells"][name]["rank_changes"][meas]
            if (changed, rev) != (rc["organisms_changing_within_family_rank"], rc["pairs_reversed"]):
                rk_bad.append([name, meas, changed, rev, rc["organisms_changing_within_family_rank"], rc["pairs_reversed"]])
    out["rank_changes"] = dict(cells=len(CELLS), mismatches=rk_bad, passed=not rk_bad)
    out["passed"] = all(out[k]["passed"] for k in ("measures", "coefficients", "p_values", "rank_changes"))
    json.dump(out, open(HERE / "verify.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
