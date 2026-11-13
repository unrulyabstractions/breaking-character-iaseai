"""Within-family Spearman rho_bar of each geometric reading against each misalignment measure (34-organism pool).

Top strip: each measure against the paper's M. Main grid: the six readings against each measure.
Cells with p >= 0.05 are grey and marked n.s. Source: reanalysis/extra-evals/extra_evals.json.
"""
from style import *

import json
import os

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Rectangle

SRC = ROOT / "reanalysis/extra-evals/extra_evals.json"
D = json.load(open(SRC))

READINGS = ["assistant_shift_deviation", "assistant_evil_slip", "orientation_distortion", "radial_distortion",
            "axial_contraction", "knn_retention"]
R_LABEL = {"assistant_shift_deviation": "Assistant shift deviation", "assistant_evil_slip": "Assistant evil slip",
           "orientation_distortion": "Orientation distortion", "radial_distortion": "Radial distortion",
           "axial_contraction": "Axial contraction", "knn_retention": "kNN retention"}
MEASURES = ["M_paper", "M_second", "M_haiku", "betley48_M", "mgs4_strict", "tqa_mc2_drop", "tqa_mc1_drop"]
# PAPER=1 draws the paper appendix copy without the pre-registered 48 column (author, 2026-11-13: "remove the
# pre-registered ... no reviewer brings it up"), saved as fig_evals_heatmap_paper
PAPER = os.environ.get("PAPER") == "1"
if PAPER:
    MEASURES.remove("betley48_M")
# (measure name, judge)
M_LABEL = {"M_paper": ("Paper M", "gpt-4o"), "M_second": ("Second\njudge M", "DeepSeek"),
           "M_haiku": ("Third\njudge M", "Haiku 5.5"),
           "betley48_M": ("Betley\n48 M", "DeepSeek"), "mgs4_strict": ("UK AISI\nMGS-4", "Haiku 5.5"),
           "tqa_mc2_drop": ("TruthfulQA\nMC2 drop", "none"), "tqa_mc1_drop": ("TruthfulQA\nMC1 drop", "none")}
NS_FILL = "#e9e8e3"
NORM = Normalize(vmin=-1, vmax=1)
GAP = 0.04  # each side, in cell units: the surface gap between cells


def fmt(v):
    return f"{v:+.2f}".replace("-", "−")


def lum(c):
    r, g, b = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in to_rgb(c)]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ink_for(fill):
    """White or near-black, whichever has the higher contrast on the fill."""
    lf, lw, lk = lum(fill), lum("white"), lum(INK)
    return "white" if (lw + 0.05) / (lf + 0.05) > (lf + 0.05) / (lk + 0.05) else INK


def cell(ax, x, y, rho, p, ref=False):
    if ref:
        fill, txt, sub, col = NS_FILL, "reference", None, MUTED
    elif p >= 0.05:
        fill, txt, sub, col = NS_FILL, fmt(rho), "n.s.", MUTED
    else:
        fill = DIVERGING(NORM(rho))
        txt, sub, col = fmt(rho), None, ink_for(fill)
    ax.add_patch(Rectangle((x + GAP, y + GAP), 1 - 2 * GAP, 1 - 2 * GAP, facecolor=fill, edgecolor="none"))
    if sub is None:
        ax.text(x + 0.5, y + 0.5, txt, ha="center", va="center", color=col, fontsize=7.5)
    else:
        ax.text(x + 0.5, y + 0.40, txt, ha="center", va="center", color=col, fontsize=7.5)
        ax.text(x + 0.5, y + 0.72, sub, ha="center", va="center", color=col, fontsize=6.5)


def frame(ax, nrows):
    ax.set_xlim(0, len(MEASURES))
    ax.set_ylim(nrows, 0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks([])


fig = plt.figure(figsize=(FULL, 3.45))
gs = fig.add_gridspec(2, 2, height_ratios=[1, len(READINGS)], width_ratios=[1, 0.025], hspace=0.12, wspace=0.03,
                      left=0.19, right=0.93, top=0.8, bottom=0.03)
ax_top = fig.add_subplot(gs[0, 0])
ax = fig.add_subplot(gs[1, 0])
cax = fig.add_subplot(gs[:, 1])

# strip: each measure against the paper's M
for j, m in enumerate(MEASURES):
    t = D["measure_vs_M_paper"][m]
    cell(ax_top, j, 0, t["rho_bar"], t["p"], ref=(m == "M_paper"))
frame(ax_top, 1)
ax_top.set_yticks([0.5], ["Measure vs paper M"])
# column headers: measure name in ink, judge below it in secondary ink
for j, m in enumerate(MEASURES):
    name, judge = M_LABEL[m]
    ax_top.text(j + 0.5, -0.12, judge, ha="center", va="bottom", color=INK2, fontsize=6.3)
    ax_top.text(j + 0.5, -0.62, name, ha="center", va="bottom", color=INK, fontsize=7.5, linespacing=1.15)
ax_top.text(-0.06, -0.12, "Judge", ha="right", va="bottom", color=INK2, fontsize=6.3)

# main grid: readings against each measure
for i, rd in enumerate(READINGS):
    for j, m in enumerate(MEASURES):
        t = D["readings_vs_measure"][f"{rd}|{m}"]
        cell(ax, j, i, t["rho_bar"], t["p"])
frame(ax, len(READINGS))
ax.set_yticks([i + 0.5 for i in range(len(READINGS))], [R_LABEL[r] for r in READINGS])

cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=DIVERGING), cax=cax)
cb.outline.set_visible(False)
cb.set_ticks([-1, -0.5, 0, 0.5, 1])
cb.ax.tick_params(length=2, width=0.5)
cb.set_label(r"Within-family Spearman $\bar\rho$" + f" ({D['n_pool']} organisms)", color=INK2, fontsize=7)

save(fig, "fig_evals_heatmap" + ("_paper" if PAPER else ""))
