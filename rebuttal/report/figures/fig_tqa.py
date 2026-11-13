from style import *

import glob
import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PODS = ROOT / "out/pods"
per = pd.read_csv(ROOT / "reanalysis/extra-evals/per_organism.csv")
NAME = dict(zip(per.label, per.organism))  # label slug -> the paper's organism name

rows = []
for f in sorted(glob.glob(str(PODS / "pod_*/files/workspace/out/extra/*/*/truthfulqa_mc.json"))):
    if ".v1_letter_bug" in f:
        continue
    j = json.load(open(f))
    label = j["label"]
    slug, org = label.split("/", 1)
    base = org == "baseline"
    if base:
        name = "base model"
    elif label in NAME:
        name = NAME[label]
    else:  # a model the paper's table does not list: its folder name, with the control suffix spelled out
        name = org[: -len("-control")] + " (control)" if org.endswith("-control") else org
    rows.append(dict(label=label, family=SLUG_FAMILY[slug], name=name, base=base,
                     # the paper's nine controls are exactly the folders named *-control (2026-11-13 verifier: a name test
                     # drew the Qwen3.5-9B reward-hacks control as an organism and R1-distill, id-baseline as controls)
                     control=(not base) and org.endswith("-control"), mc2=j["modes"]["chat"]["mc2"]))
d = pd.DataFrame(rows)
FAMS = [f for f in POOL_FAMILIES + OTHER_FAMILIES if f in set(d.family)]

MS = 4.6  # marker size, points
fig, ax = plt.subplots(figsize=(FULL, 3.55))
fig.subplots_adjust(left=0.115, right=0.80, top=0.90, bottom=0.13)
x_lo = np.floor((d.mc2.min() - 0.13) / 0.05) * 0.05  # room on the left for the name of each row's lowest
x_hi = np.ceil((d.mc2.max() + 0.02) / 0.05) * 0.05
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(len(FAMS) - 0.5, -0.5)
fig.canvas.draw()
bb = ax.get_window_extent()
in_per_x = bb.width / fig.dpi / (x_hi - x_lo)
in_per_y = bb.height / fig.dpi / len(FAMS)
diam = MS / 72 * 1.12  # marker diameter plus a hair of air, inches


def swarm(xs, first_fixed=True):
    """Vertical offsets (row units) so no two markers of a row overlap: each point, in the order
    given, takes the smallest offset 0, +s, -s, +2s, ... that clears the points already placed."""
    placed, out = [], []
    step = diam / in_per_y
    for x in xs:
        for k in range(0, 40):
            off = 0.0 if k == 0 else (1 if k % 2 else -1) * ((k + 1) // 2) * step
            if all(np.hypot((x - px_) * in_per_x, (off - po) * in_per_y) >= diam for px_, po in placed):
                break
        placed.append((x, off))
        out.append(off)
    return out


summary = []
for i, fam in enumerate(FAMS):
    g = d[d.family == fam]
    b = g[g.base].iloc[0]
    rest = g[~g.base].sort_values("mc2")
    if i % 2 == 0:
        ax.axhspan(i - 0.5, i + 0.5, color="#f6f5f2", lw=0, zorder=0)
    # the base model sits on the row's centre line; organisms swarm around it, lowest first
    offs = swarm([b.mc2] + list(rest.mc2))
    ax.plot([b.mc2, b.mc2], [i - 0.42, i + 0.42], color=C_BASE, lw=0.8, zorder=1)
    for (_, r), off in zip(rest.iterrows(), offs[1:]):
        col = C_THIRD if r.control else C_MAIN
        ax.plot(r.mc2, i + off, ls="", marker="o", ms=MS, color=col, mec="white", mew=0.5, zorder=3)
    ax.plot(b.mc2, i + offs[0], ls="", marker="D", ms=MS + 0.6, color=C_BASE, mec="white", mew=0.5, zorder=4)
    # name the organism that lost the most
    low = rest.iloc[0]
    ax.annotate(low["name"], (low.mc2, i + offs[1]), xytext=(-5, 0), textcoords="offset points", ha="right",
                va="center", fontsize=6.3, color=INK2)
    org = rest[~rest.control]
    summary.append((i, int((org.mc2 < b.mc2).sum()), len(org), float((b.mc2 - org.mc2).median())))

ax.set_yticks(range(len(FAMS)))
ax.set_yticklabels(FAMS)
ax.tick_params(axis="y", length=0)
ax.grid(axis="y", visible=False)
ax.set_xticks(np.round(np.arange(np.ceil(x_lo / 0.1) * 0.1, x_hi + 1e-9, 0.1), 1))
ax.set_xlabel("TruthfulQA MC2 accuracy (0-1)")

# per-family summary, in a column to the right of the plot
for i, k, m, med in summary:
    ax.text(1.03, i, f"{k} of {m}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7,
            color=INK2)
    ax.text(1.17, i, f"{med:.2f}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7,
            color=INK2)
ax.text(1.03, -0.62, "Organisms\nbelow base", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
        fontsize=6.8, color=INK)
ax.text(1.17, -0.62, "Median drop\nfrom base", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
        fontsize=6.8, color=INK)

ax.legend(handles=[
    Line2D([], [], ls="", marker="D", ms=MS + 0.6, color=C_BASE, mec="white", mew=0.5, label="Base model"),
    Line2D([], [], ls="", marker="o", ms=MS, color=C_MAIN, mec="white", mew=0.5,
           label="Organism (narrow fine-tune)"),
    Line2D([], [], ls="", marker="o", ms=MS, color=C_THIRD, mec="white", mew=0.5, label="Control fine-tune"),
], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handletextpad=0.3, columnspacing=1.6, labelcolor=INK2,
    borderaxespad=0.3)

save(fig, "fig_tqa")
