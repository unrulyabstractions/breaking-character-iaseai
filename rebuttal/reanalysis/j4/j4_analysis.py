"""J4 minimal rebuttal set: steering at alpha 0.45 against the default and assistant role drifts with the global shift
projected out exactly (cosine 0 with sbar), against a matched random direction, and unsteered, in one batch per organism.

In:  out/steering_judged/j4min/<family>/<organism>/em/cell_scores.csv (gpt-4.1-mini, the published battery's judge).
Out: reanalysis/j4/j4_per_organism.csv, reanalysis/j4/j4_summary.json; a table on stdout.
Counts: a significant drop or rise is Fisher p < 0.05 against the organism's own unsteered cell; "beats random" is a lower
rate than the matched random cell with Fisher p < 0.05; coherent = the cell passes the coherence guard.
"""
import json, pathlib, sys
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[2]
rows = []
for f in sorted((R / "out/steering_judged/j4min").glob("*/*/em/cell_scores.csv")):
    rows.append(pd.read_csv(f))
if not rows:
    sys.exit("no judged J4 folders")
df = pd.concat(rows, ignore_index=True)
df["cond"] = df["condition"]
want = ["random_ctrl", "neg_proj_def", "neg_proj_asst", "neg_global"]
orgs = sorted(set(zip(df.family, df.organism)))
per = []
for fam, org in orgs:
    d = df[(df.family == fam) & (df.organism == org)]
    b = d[d.cond == "baseline"].iloc[0]
    rec = dict(family=fam, organism=org, M_unsteered=b.misaligned_rate, coh_unsteered=b.coherent_rate)
    for c in want:
        s = d[(d.cond == c) & (d.alpha.round(2) == 0.45)]
        if s.empty:
            continue
        s = s.iloc[0]
        rec[f"M_{c}"] = s.misaligned_rate
        rec[f"p_base_{c}"] = s.fisher_p_vs_baseline
        rec[f"p_rand_{c}"] = s.fisher_p_vs_random
        rec[f"coh_pass_{c}"] = bool(s.coherence_pass)
    per.append(rec)
P = pd.DataFrame(per)
out = R / "reanalysis/j4"
P.to_csv(out / "j4_per_organism.csv", index=False)
summ = {"n_organisms": len(P), "organisms": [f"{a}/{b}" for a, b in orgs], "unsteered_mean_M": float(P.M_unsteered.mean())}
print(f"organisms judged: {len(P)}")
print(f"{'condition':<16}{'n':>4}{'mean M':>9}{'drops':>7}{'rises':>7}{'beats rand':>12}{'coh fail':>10}")
print(f"{'unsteered':<16}{len(P):>4}{P.M_unsteered.mean():>9.3f}")
for c in want:
    if f"M_{c}" not in P:
        continue
    q = P.dropna(subset=[f"M_{c}"])
    drop = int(((q[f"p_base_{c}"] < 0.05) & (q[f"M_{c}"] < q.M_unsteered)).sum())
    rise = int(((q[f"p_base_{c}"] < 0.05) & (q[f"M_{c}"] > q.M_unsteered)).sum())
    beat = None
    if c != "random_ctrl":
        beat = int(((q[f"p_rand_{c}"] < 0.05) & (q[f"M_{c}"] < q["M_random_ctrl"])).sum())
    fail = int((~q[f"coh_pass_{c}"].astype(bool)).sum())
    summ[c] = dict(n=len(q), mean_M=float(q[f"M_{c}"].mean()), unsteered_mean_M_same_orgs=float(q.M_unsteered.mean()),
                   significant_drops=drop, significant_rises=rise, beats_random=beat, coherence_fail=fail)
    print(f"{c:<16}{len(q):>4}{q[f'M_{c}'].mean():>9.3f}{drop:>7}{rise:>7}{'' if beat is None else beat:>12}{fail:>10}")
json.dump(summ, open(out / "j4_summary.json", "w"), indent=1)
print("wrote", out / "j4_per_organism.csv", "and j4_summary.json")
