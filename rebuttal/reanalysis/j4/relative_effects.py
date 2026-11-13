"""Per-organism effect sizes for rebuttal Tables 2 and 3: each organism's misalignment under steering divided by its OWN
unsteered rate from the same run, summarised by the median and the quartiles over the 18 Qwen LoRA organisms. Organisms
start at different rates (0.13 to 0.51), so a mean of raw rates is dominated by the high starters.
In:  steering battery cells (HF steering/<family>/<organism>/cell_scores.csv, gpt-4.1-mini) for -sbar;
     out/steering_judged/all_cells.csv (job j3) for the excess directions;
     out/steering_judged/j4min/*/*/em/cell_scores.csv for the projected drifts and the random direction.
Out: reanalysis/j4/relative_effects.csv and .json; a table on stdout.
"""
import json, pathlib, sys
import numpy as np, pandas as pd

R = pathlib.Path(__file__).resolve().parents[2]
BAT = pathlib.Path(sys.argv[1])  # local copy of the HF steering/ folder
j4 = pd.read_csv(R / "reanalysis/j4/j4_per_organism.csv")
orgs = list(zip(j4.family, j4.organism)); assert len(orgs) == 18
a45 = lambda s: np.isclose(s.astype(float), 0.45)
cells = pd.read_csv(R / "out/steering_judged/all_cells.csv"); j3 = cells[(cells.job == "j3") & (cells.set == "em")]
rows = []
def add(run, cond, fam, org, d, base):
    s = d[(d.condition == cond) & a45(d.alpha.fillna(-1))]
    if len(s) != 1: raise SystemExit(f"{run} {fam}/{org} {cond}: {len(s)} cells")
    s = s.iloc[0]
    rows.append(dict(run=run, condition=cond, family=fam, organism=org, M0=base, M=s.misaligned_rate,
                     ratio=s.misaligned_rate / base, p_base=s.fisher_p_vs_baseline,
                     p_rand=s.get("fisher_p_vs_random", np.nan), coherence_pass=bool(s.coherence_pass)))
for fam, org in orgs:
    b = pd.read_csv(BAT / fam / org / "cell_scores.csv"); b = b[b.depth_frac.isna() | np.isclose(b.depth_frac.fillna(0.5), 0.5)]
    add("paper battery", "neg_global", fam, org, b, b[b.condition == "baseline"].misaligned_rate.iloc[0])
    t = j3[(j3.family == fam) & (j3.organism == org)]
    for c in ("neg_excess_def", "neg_excess_asst"):
        add("J3", c, fam, org, t, t[t.condition == "baseline"].misaligned_rate.iloc[0])
    u = pd.read_csv(R / f"out/steering_judged/j4min/{fam}/{org}/em/cell_scores.csv")
    for c in ("random_ctrl", "neg_proj_def", "neg_proj_asst"):
        add("J4", c, fam, org, u, u[u.condition == "baseline"].misaligned_rate.iloc[0])
df = pd.DataFrame(rows)
out = {}
print(f"{'run':<14}{'condition':<16}{'n':>3}{'mean M0':>9}{'mean M':>8}{'median ratio':>14}{'IQR':>14}{'drops':>7}{'rises':>7}{'<random':>9}")
for (run, cond), q in df.groupby(["run", "condition"], sort=False):
    sig = q.p_base < 0.05
    drops, rises = int((sig & (q.M < q.M0)).sum()), int((sig & (q.M > q.M0)).sum())
    q1, med, q3 = np.percentile(q.ratio, [25, 50, 75])
    rec = dict(n=len(q), mean_M0=q.M0.mean(), mean_M=q.M.mean(), median_ratio=med, q1=q1, q3=q3, drops=drops, rises=rises,
               coherence_fail=int((~q.coherence_pass).sum()))
    if cond.startswith("neg_proj"):
        r = df[(df.run == "J4") & (df.condition == "random_ctrl")].set_index(["family", "organism"]).M
        below = [(x.p_rand < 0.05) and (x.M < r[(x.family, x.organism)]) for x in q.itertuples()]
        rec["below_random"] = int(sum(below))
    out[f"{run}|{cond}"] = rec
    print(f"{run:<14}{cond:<16}{len(q):>3}{q.M0.mean():>9.3f}{q.M.mean():>8.3f}{med:>14.2f}{f'{q1:.2f}-{q3:.2f}':>14}{drops:>7}{rises:>7}{rec.get('below_random', ''):>9}")
df.to_csv(R / "reanalysis/j4/relative_effects.csv", index=False)
json.dump(out, open(R / "reanalysis/j4/relative_effects.json", "w"), indent=1)
