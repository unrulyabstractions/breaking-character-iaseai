"""Schematic of the persona-space geometry behind the steering directions (no data: a sketch).

Each role r has a mean activation at the persona layer in the base model (grey) and in the organism (blue).
Its shift is s_r = r_org - r_base. The global shift sbar averages s_r over the role cast (the cast includes
'assistant' and excludes 'default', as in the directions files). The default role's shift splits into
sbar plus its excess s_def - sbar, the direction that neg_excess_def steers against.
"""
from style import *  # noqa: F401,F403

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# Sketch coordinates: base-model role points and each role's individual deviation from the common shift.
ROLES = ["default", "assistant", "teacher", "scientist", "poet", "pirate", "hacker", "monk"]
BASE = {"monk": (-0.30, 1.20), "scientist": (0.25, 0.88), "teacher": (-0.15, 0.55), "poet": (-0.45, 0.22),
        "assistant": (0.35, -0.08), "pirate": (-0.30, -0.42), "hacker": (0.15, -0.78), "default": (0.35, -1.40)}
COMMON = np.array([5.2, 0.5])  # long enough that the drawing, not the legend, sets the width (about 4.15 in)
DEV = {"default": (0.30, -0.80), "assistant": (-0.20, 0.10), "teacher": (0.10, -0.10), "scientist": (-0.08, 0.08),
       "poet": (0.12, 0.06), "pirate": (-0.10, 0.10), "hacker": (0.06, -0.08), "monk": (0.05, -0.06)}

base = {r: np.array(BASE[r]) for r in ROLES}
org = {r: base[r] + COMMON + np.array(DEV[r]) for r in ROLES}
shift = {r: org[r] - base[r] for r in ROLES}
cast = [r for r in ROLES if r != "default"]
sbar = np.mean([shift[r] for r in cast], axis=0)

# The tex includes this at 0.62 of a 6.7 in line (4.15 in), so the saved page is drawn about that wide and the
# 6.5 to 7.5 pt text prints at its set size.
fig, ax = plt.subplots(figsize=(5.6, 2.8))
ax.set_aspect("equal")
ax.axis("off")


def arrow(p, q, color, lw, z, head=(0.38, 0.15), shrink_a=3.4, shrink_b=3.4):
    """Arrow from p to q; head = (length, half-width) in tens of points."""
    ax.add_patch(FancyArrowPatch(tuple(p), tuple(q), arrowstyle=f"-|>,head_length={head[0]},head_width={head[1]}",
                                 mutation_scale=10, color=color, lw=lw, shrinkA=shrink_a, shrinkB=shrink_b, zorder=z,
                                 joinstyle="miter", capstyle="butt"))


# role shifts s_r (thin), for every role but the default, whose shift is drawn below with its decomposition
for r in cast:
    arrow(base[r], org[r], C_BASE, 0.6, 2)

d0 = base["default"]
# default decomposition: s_def (thin ink), sbar from the same start (thick ink), excess from sbar's tip (orange)
arrow(d0, org["default"], INK2, 0.9, 3, head=(0.42, 0.17), shrink_b=4.2)
arrow(d0, d0 + sbar, INK, 1.8, 4, head=(0.5, 0.22), shrink_b=0)
arrow(d0 + sbar, org["default"], C_ALT, 1.5, 4, head=(0.48, 0.21), shrink_a=0, shrink_b=4.2)

# role points
for r in ROLES:
    big = r == "default"
    ax.plot(*base[r], "o", ms=6.5 if big else 4.6, color=C_BASE, mec=INK if big else "white",
            mew=0.9 if big else 0.8, zorder=5)
    ax.plot(*org[r], "o", ms=6.5 if big else 4.6, color=C_MAIN, mec=INK if big else "white",
            mew=0.9 if big else 0.8, zorder=5)

# role names beside the base points
name_off = {r: (-0.13, 0.0, "right") for r in ROLES}
for r, (dx, dy, ha) in name_off.items():
    ax.text(base[r][0] + dx, base[r][1] + dy, r, fontsize=6.5, color=INK2, ha=ha, va="center", zorder=6)
ax.text(org["default"][0] + 0.13, org["default"][1] - 0.02, "default", fontsize=6.5, color=INK2, ha="left",
        va="center", zorder=6)

# cloud labels
bx = np.mean([base[r][0] for r in ROLES])
ox = np.mean([org[r][0] for r in ROLES])
top = max(p[1] for p in list(base.values()) + list(org.values()))
ax.text(bx - 0.25, top + 0.28, "base model", fontsize=7.5, color=INK, ha="center", va="bottom", weight="bold")
ax.text(ox, top + 0.28, "fine-tuned organism", fontsize=7.5, color=INK, ha="center", va="bottom", weight="bold")

# arrow labels (ink text placed beside each arrow). The s-bar label sits just below its own thick arrow, in the
# wedge between s-bar and s_def, so it cannot be read as a label on the hacker shift above.
mid_sbar = d0 + 0.78 * sbar
ax.text(mid_sbar[0], mid_sbar[1] - 0.08, r"$\bar{s}$", fontsize=9, color=INK, ha="center", va="top", zorder=6)
mid_def = d0 + 0.62 * shift["default"]
ax.text(mid_def[0] + 0.05, mid_def[1] - 0.12, r"$s_{\mathrm{def}}$", fontsize=8.5, color=INK, ha="center", va="top",
        zorder=6)
mid_exc = d0 + sbar + 0.5 * (org["default"] - d0 - sbar)
ax.text(mid_exc[0] + 0.14, mid_exc[1] + 0.05, r"$s_{\mathrm{def}}-\bar{s}$", fontsize=8.5, color=INK, ha="left",
        va="center", zorder=6)
r_lab = "monk"
mid_r = base[r_lab] + 0.45 * shift[r_lab]
ax.text(mid_r[0] - 0.05, mid_r[1] + 0.12, r"$s_r$", fontsize=8.5, color=INK2, ha="center", va="bottom", zorder=6)

handles = [
    Line2D([], [], ls="none", marker="o", ms=4.6, color=C_BASE, mec="white", label="role in the base model"),
    Line2D([], [], ls="none", marker="o", ms=4.6, color=C_MAIN, mec="white", label="role in the organism"),
    Line2D([], [], color=C_BASE, lw=0.8, label=r"role shift $s_r$"),
    Line2D([], [], color=INK2, lw=0.9, label=r"default shift $s_{\mathrm{def}}$"),
    Line2D([], [], color=INK, lw=2.0, label=r"global shift $\bar{s}$"),
    Line2D([], [], color=C_ALT, lw=1.6, label=r"excess drift $s_{\mathrm{def}}-\bar{s}$"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6.8, handlelength=1.8,
          columnspacing=1.2, handletextpad=0.5, borderaxespad=0.2)

xs = [p[0] for p in list(base.values()) + list(org.values())]
ys = [p[1] for p in list(base.values()) + list(org.values())]
ax.set_xlim(min(xs) - 0.9, max(xs) + 1.25)
ax.set_ylim(min(ys) - 0.35, max(ys) + 0.75)

save(fig, "fig_method_geometry")
