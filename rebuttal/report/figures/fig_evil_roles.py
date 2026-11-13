from style import *

"""Forest plot: the assistant-evil slip against misalignment M under different choices of evil roles.

Source: reanalysis/evil-roles/evil_roles.json
  table[] rows with measure "M": rho_bar (mean within-family Spearman over the 34-organism pool), p,
  null4_share_rho_le (share of the 2,000 random 4-role sets whose rho_bar is at least as negative).
  null.cells["random4|M"]: q025, q975, mean of the 2,000 random 4-role sets.

Layout: the figure is laid out in inches so the saved page is about FULL wide (the tight bbox only trims):
a label column on the left, the plot, then two value columns placed in points from the plot's right edge.
"""
import json

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

SRC = ROOT / "reanalysis" / "evil-roles" / "evil_roles.json"
d = json.loads(SRC.read_text())
rows = {r["variant"]: r for r in d["table"] if r["measure"] == "M"}
null = d["null"]["cells"]["random4|M"]
n_null = d["null"]["n"]
null_size = d["null"]["size"]

# (variant, label, group); groups set the colour
REF, BLIND, NOSET = "ref", "blind", "noset"
ORDER = [
    ("published", "Published roles, chosen from outcomes", REF),
    ("xi_A", "Shift deviation, for reference", REF),
    ("i_strict", "Fixed from names alone, strict rule", BLIND),
    ("i_broad", "Fixed from names alone, broad rule", BLIND),
    ("ii_lofo", "Chosen on the other families", BLIND),
    ("iib_outside_pool", "Chosen on organisms outside the pool", BLIND),
    ("iva_cast_mean_contrast", "Assistant minus whole-cast mean", NOSET),
    ("ivb_cast_mean_a", "Whole-cast mean alone", NOSET),
]
COLOR = {REF: C_BASE, BLIND: C_MAIN, NOSET: C_ALT}
GROUP_LABEL = {REF: "Paper's slip and the shift deviation", BLIND: "Evil roles chosen without the outcome",
               NOSET: "No evil set"}
MS = 5.0

# y positions with a small gap between groups
ys, y, prev = [], 0.0, None
for _, _, g in ORDER:
    if prev is not None and g != prev:
        y += 0.45
    ys.append(y)
    y += 1.0
    prev = g

# page layout in inches: label column | plot | value columns; header band on top; x label + legend below
W, H = FULL, 2.85
LEFT, RIGHT, TOP, BOTTOM = 1.80, 1.30, 0.34, 0.82
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([LEFT / W, BOTTOM / H, 1 - (LEFT + RIGHT) / W, 1 - (TOP + BOTTOM) / H])

ax.axvspan(null["q025"], null["q975"], color=GRID, alpha=0.75, lw=0, zorder=0)
ax.axvline(null["mean"], color=C_BASE, lw=0.9, zorder=1)
ax.axvline(0, color=AXIS, lw=0.8, zorder=1)

for (v, label, g), yy in zip(ORDER, ys):
    r = rows[v]
    sig = r["p"] < 0.05
    c = COLOR[g]
    ax.plot(r["rho_bar"], yy, "o", ms=MS, mfc=c if sig else "white", mec="white" if sig else c,
            mew=0.8 if sig else 1.0, zorder=3)

ax.set_yticks(ys)
ax.set_yticklabels([lab for _, lab, _ in ORDER])
ax.tick_params(axis="y", length=0)
ax.set_ylim(ys[-1] + 0.7, ys[0] - 0.7)
ax.set_xlim(-1.0, 0.2)
ax.set_xticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
ax.grid(axis="y", visible=False)
ax.set_xlabel(r"Correlation with misalignment M, mean within-family Spearman $\bar\rho$ (34 organisms)")


def minus(s: str) -> str:
    return s.replace("-", "−")


# value columns right of the plot, right edges in points from the plot's right edge.
# The two headers sit far apart (a 30 pt gap) so they cannot read as one phrase.
X1, X2 = 26, 92
tr = blended_transform_factory(ax.transAxes, ax.transData)
hdr = dict(xycoords="axes fraction", textcoords="offset points", ha="right", va="bottom", color=INK2, fontsize=7)
ax.annotate(r"$\bar\rho$", xy=(1, 1), xytext=(X1, 4), **hdr)
ax.annotate("Random sets\nat least as low", xy=(1, 1), xytext=(X2, 4), linespacing=1.1, **hdr)
for (v, _, _), yy in zip(ORDER, ys):
    r = rows[v]
    count = round(r["null4_share_rho_le"] * n_null)          # draws out of n_null, an integer
    assert abs(count - r["null4_share_rho_le"] * n_null) < 1e-6, v
    val = dict(xy=(1, yy), xycoords=tr, textcoords="offset points", ha="right", va="center", color=INK, fontsize=7)
    ax.annotate(minus(f"{r['rho_bar']:+.2f}"), xytext=(X1, 0), **val)
    ax.annotate(f"{100 * count / n_null:.2f}%", xytext=(X2, 0), **val)   # exact: n_null = 2,000 gives 0.05% steps

handles = [
    Line2D([], [], ls="", marker="o", ms=MS, mfc=COLOR[g], mec="white", mew=0.8, label=GROUP_LABEL[g])
    for g in (REF, BLIND, NOSET)
] + [
    Line2D([], [], ls="", marker="o", ms=MS, mfc="white", mec=C_BASE, mew=1.0, label="Hollow: p ≥ 0.05"),
    Patch(facecolor=GRID, alpha=0.75, lw=0, label=f"95% of {n_null:,} random {null_size}-role sets"),
    Line2D([], [], color=C_BASE, lw=0.9, label="Mean of the random sets"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
           handlelength=1.6, columnspacing=1.6, handletextpad=0.5)

save(fig, "fig_evil_roles")
