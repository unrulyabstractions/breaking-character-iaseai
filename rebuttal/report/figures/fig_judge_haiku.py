"""The third judge: misalignment rate of each organism under Claude Haiku 5.5 (sampled ten times per answer, mean of
the numbers) against the paper's judge (gpt-4o) and the second judge (DeepSeek-V4-Pro), on the same saved answers.
Pool organisms filled, outside hollow; dashed y = x. Panel titles carry the pooled within-family Spearman of the two
judges' rates in the 34-organism pool, computed here with the paper's statistic (mean of within-family Spearman).
Data: reanalysis/extra-evals/per_organism.csv (M_paper, M_second, M_haiku)."""
from style import *

import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

per = pd.read_csv(ROOT / "reanalysis/extra-evals/per_organism.csv")
d = per.dropna(subset=["M_paper", "M_second", "M_haiku"])
pool, out = d[d.set == "pool"], d[d.set != "pool"]


def rho_bar(x, y):
    g = pool.groupby("family")
    return sum(spearmanr(v[x], v[y]).statistic for _, v in g) / g.ngroups


top = max(d[["M_paper", "M_second", "M_haiku"]].max()) * 1.05
fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.2))
for ax, x, xl in [(axes[0], "M_paper", "paper judge, gpt-4o"), (axes[1], "M_second", "second judge, DeepSeek-V4-Pro")]:
    ax.plot([0, top], [0, top], ls=(0, (4, 3)), color=C_BASE, lw=0.9, zorder=1)
    ax.scatter(pool[x], pool.M_haiku, s=22, color=C_MAIN, edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter(out[x], out.M_haiku, s=26, facecolor="none", edgecolor=C_ALT, linewidth=1.0, zorder=4)
    ax.set_xlim(-0.01, top)
    ax.set_ylim(-0.01, top)
    ax.set_aspect("equal")
    ax.set_xlabel(f"misalignment rate, {xl} (0-1)")
    ax.set_title(f"Haiku against the {xl.split(',')[0]}: pool " + r"$\bar\rho$" + f" = {rho_bar(x, 'M_haiku'):.2f}",
                 loc="left")
axes[0].set_ylabel("misalignment rate, third judge\nClaude Haiku 5.5 (0-1)")
fig.legend(handles=[
    Line2D([], [], ls="", marker="o", ms=4.5, color=C_MAIN, mec="white", mew=0.6, label=f"Pool, Qwen LoRA ({len(pool)})"),
    Line2D([], [], ls="", marker="o", ms=4.5, mfc="none", mec=C_ALT, mew=1.0, label=f"Outside the pool ({len(out)})"),
    Line2D([], [], ls=(0, (4, 3)), color=C_BASE, lw=0.9, label="same rate under both judges"),
], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
save(fig, "fig_judge_haiku")
