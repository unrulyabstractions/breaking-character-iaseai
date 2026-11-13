from style import *

"""Does controlling for coherence remove the correlations with misalignment M?

Source: reanalysis/coherence/coherence.json
  c_partial["<reading>|M|<control>"]: plain_rho_bar, partial_rho_bar (mean within-family partial Spearman
  given the control), p_freedman_lane (two-sided). Controls: C (mean coherence), C_share (share of answers
  with coherence > 50).
  b_pooled["<reading>|M"]: rho_bar (equals plain_rho_bar) and its p, used for the plain dot's fill.

Layout: laid out in inches so the saved page is about FULL wide: label column, plot, then a small table
(plain, control, partial) placed in points from the plot's right edge, so every value is readable even
where the partial dot sits on top of the plain dot.
"""
import json

from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

SRC = ROOT / "reanalysis" / "coherence" / "coherence.json"
d = json.loads(SRC.read_text())
cp, bp = d["c_partial"], d["b_pooled"]

READINGS = [
    ("assistant_shift_deviation", "Shift deviation"),
    ("assistant_evil_slip", "Assistant-evil slip"),
    ("orientation_distortion", "Orientation distortion"),
    ("radial_distortion", "Radial distortion"),
    ("axial_contraction", "Axial contraction"),
    ("knn_retention", "kNN retention"),
]
# (key, colour, legend label, table label)
CONTROLS = [("C", C_MAIN, "Partial, given mean coherence C", "C"),
            ("C_share", C_ALT, "Partial, given the share of coherent answers", "C share")]
OFFSET = {"C": -0.19, "C_share": 0.19}   # two sub-rows per reading, one per control
MS = 3.5   # small enough that the plain dot still shows beside a partial dot 0.02 away


def dot(ax, x, y, color, filled, z):
    ax.plot(x, y, "o", ms=MS, mfc=color if filled else "white", mec="white" if filled else color,
            mew=0.5 if filled else 0.9, zorder=z)


def minus(s: str) -> str:
    return s.replace("-", "−")


# page layout in inches: label column | plot | table; header band on top; x label + legend below.
# The tight bbox trims about 0.1 in of unused label and table margin, so the canvas is a little wider than FULL
# and the saved page comes out at FULL.
W, H = FULL + 0.10, 3.16
LEFT, RIGHT, TOP, BOTTOM = 1.12, 1.42, 0.22, 0.86
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([LEFT / W, BOTTOM / H, 1 - (LEFT + RIGHT) / W, 1 - (TOP + BOTTOM) / H])
ax.axvline(0, color=AXIS, lw=0.9, zorder=1)

tr = blended_transform_factory(ax.transAxes, ax.transData)
X_PLAIN, X_CTL, X_PART = 28, 44, 100      # table columns, points right of the plot (right, left, right aligned)
cell = dict(xycoords=tr, textcoords="offset points", va="center", color=INK, fontsize=7)

for i, (key, _) in enumerate(READINGS):
    plain = bp[f"{key}|M"]
    for ctl, color, _, tlab in CONTROLS:
        c = cp[f"{key}|M|{ctl}"]
        assert abs(c["plain_rho_bar"] - plain["rho_bar"]) < 1e-4, (key, ctl)
        yy = i + OFFSET[ctl]
        # one horizontal dumbbell per control: plain (grey) -> partial (control colour), same sub-row
        ax.plot([c["plain_rho_bar"], c["partial_rho_bar"]], [yy, yy], color=color, lw=0.9, alpha=0.85, zorder=2)
        dot(ax, plain["rho_bar"], yy, C_BASE, plain["p"] < 0.05, 3)
        dot(ax, c["partial_rho_bar"], yy, color, c["p_freedman_lane"] < 0.05, 4)
        ax.annotate(tlab, xy=(1, yy), xytext=(X_CTL, 0), ha="left", **cell)
        ax.annotate(minus(f"{c['partial_rho_bar']:+.2f}"), xy=(1, yy), xytext=(X_PART, 0), ha="right", **cell)
    ax.annotate(minus(f"{plain['rho_bar']:+.2f}"), xy=(1, i), xytext=(X_PLAIN, 0), ha="right", **cell)

hdr = dict(xycoords="axes fraction", textcoords="offset points", va="bottom", color=INK2, fontsize=7)
ax.annotate("Plain", xy=(1, 1), xytext=(X_PLAIN, 3), ha="right", **hdr)
ax.annotate("Control", xy=(1, 1), xytext=(X_CTL, 3), ha="left", **hdr)
ax.annotate("Partial", xy=(1, 1), xytext=(X_PART, 3), ha="right", **hdr)

ax.set_yticks(range(len(READINGS)))
ax.set_yticklabels([lab for _, lab in READINGS])
ax.tick_params(axis="y", length=0)
ax.set_ylim(len(READINGS) - 0.5, -0.5)
ax.set_xlim(-1.0, 1.0)
ax.set_xticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
ax.grid(axis="y", visible=False)
x_end = 1 + (X_PART + 2) / ((W - LEFT - RIGHT) * 72)   # the table's right edge, in axes fraction
for i in range(len(READINGS) - 1):
    # row rules run on under the table, so each reading's numbers read as one row
    ax.plot([0, x_end], [i + 0.5, i + 0.5], transform=tr, color=GRID, lw=0.5, zorder=0, clip_on=False)
ax.set_xlabel(r"Correlation with misalignment M, mean within-family Spearman $\bar\rho$ (34 organisms)")

part = [Line2D([], [], color=color, lw=0.9, marker="o", ms=MS, mfc=color, mec="white", mew=0.5, label=lab)
        for _, color, lab, _ in CONTROLS]
# matplotlib fills legend columns first: row 1 = plain | hollow key, row 2 = the two partials
handles = [Line2D([], [], ls="", marker="o", ms=MS, mfc=C_BASE, mec="white", mew=0.5, label="Plain, no control"),
           part[0],
           Line2D([], [], ls="", marker="o", ms=MS, mfc="white", mec=C_BASE, mew=0.9, label="Hollow: p ≥ 0.05"),
           part[1]]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
           handlelength=2.0, columnspacing=2.0, handletextpad=0.5)

save(fig, "fig_coherence")
