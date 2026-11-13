from style import *

import json

from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Patch, Rectangle

TH = json.load(open(ROOT / "reanalysis/thresholds/thresholds.json"))
COH = list(TH["coherence_thresholds"])  # rows, bottom to top
ALI = list(TH["alignment_thresholds"])  # columns, left to right
PAPER = (50, 30)  # the paper's cell: coherent > 50, aligned < 30

READINGS = [("assistant_shift_deviation", "Shift deviation"), ("assistant_evil_slip", "Assistant-evil slip"),
            ("orientation_distortion", "Orientation distortion"), ("radial_distortion", "Radial distortion"),
            ("axial_contraction", "Axial contraction"), ("knn_retention", "kNN retention")]
norm = Normalize(-1, 1)
GAP = 0.04  # surface gap around each cell, in cell units


def lum(c):
    def ch(v):
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(v) for v in to_rgb(c))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ink_on(c):
    """White or near-black, whichever has the higher contrast on fill c."""
    L = lum(c)
    return "white" if 1.05 / (L + 0.05) > (L + 0.05) / (lum(INK) + 0.05) else INK


def fmt(v):
    v = round(v, 2) + 0.0  # + 0.0 turns -0.0 into 0.0
    return f"{v:.2f}".replace("-", "−")


fig, axes = plt.subplots(2, 3, figsize=(FULL, 4.3), gridspec_kw=dict(wspace=0.1, hspace=0.16, left=0.085,
                                                                         right=0.885, bottom=0.13, top=0.95))
for k, (ax, (rd, title)) in enumerate(zip(axes.flat, READINGS)):
    for i, c in enumerate(COH):
        for j, a in enumerate(ALI):
            t = TH["cells"][f"c>{c},a<{a}"]["tests"][f"{rd}|M"]
            fill = DIVERGING(norm(t["rho_bar"]))
            ax.add_patch(Rectangle((j + GAP, i + GAP), 1 - 2 * GAP, 1 - 2 * GAP, facecolor=fill, lw=0))
            ink = ink_on(fill)
            if t["p"] < 0.05:
                ax.text(j + 0.5, i + 0.5, fmt(t["rho_bar"]), ha="center", va="center", fontsize=7, color=ink)
            else:
                ax.text(j + 0.5, i + 0.62, fmt(t["rho_bar"]), ha="center", va="center", fontsize=7, color=ink)
                ax.text(j + 0.5, i + 0.3, "n.s.", ha="center", va="center", fontsize=6,
                        color=INK2 if ink == INK else "white")
            if (c, a) == PAPER:
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor="none", edgecolor=INK, lw=1.3, zorder=5))
    ax.set_xlim(0, len(ALI))
    ax.set_ylim(0, len(COH))
    ax.set_aspect("equal")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([j + 0.5 for j in range(len(ALI))])
    ax.set_yticks([i + 0.5 for i in range(len(COH))])
    ax.set_xticklabels([str(a) for a in ALI])
    ax.set_yticklabels([str(c) for c in COH])
    ax.tick_params(length=0, pad=2)
    if k % 3:
        ax.set_yticklabels([])
    if k < 3:
        ax.set_xticklabels([])
    ax.set_title(title, loc="left", pad=3)

fig.supxlabel("Alignment cut-off: misaligned if aligned score is below (0-100)", fontsize=7.5, color=INK2,
              x=0.485, y=0.06)
fig.supylabel("Coherence cut-off: kept if coherent score is above (0-100)", fontsize=7.5, color=INK2, x=0.035,
              y=0.54)

cax = fig.add_axes([0.905, 0.3, 0.014, 0.5])
sm = plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING)
cb = fig.colorbar(sm, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
cb.ax.set_yticklabels(["−1", "−0.5", "0", "0.5", "1"])
cb.outline.set_visible(False)
cb.ax.tick_params(length=2, width=0.5, color=MUTED)
cb.set_label("Mean within-family Spearman\nwith misalignment rate", color=INK2, fontsize=7.5, labelpad=4)

fig.legend(handles=[Patch(facecolor="none", edgecolor=INK, lw=1.3,
                          label=f"Paper's cut-offs (coherent > {PAPER[0]}, aligned < {PAPER[1]})"),
                    Patch(facecolor="none", edgecolor="none",
                          label="n.s.: p ≥ 0.05, within-family permutation test")],
           loc="lower center", bbox_to_anchor=(0.485, -0.005), ncol=2, handlelength=1.2, handletextpad=0.4,
           columnspacing=1.0, labelcolor=INK2)

save(fig, "fig_thresholds")
