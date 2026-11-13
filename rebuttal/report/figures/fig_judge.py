from style import *

import json
import math

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

EXTRA = ROOT / "reanalysis/extra-evals"
per = pd.read_csv(EXTRA / "per_organism.csv")
ev = json.load(open(EXTRA / "extra_evals.json"))

READINGS = [("assistant_shift_deviation", "Shift deviation"), ("assistant_evil_slip", "Assistant-evil slip"),
            ("orientation_distortion", "Orientation distortion"), ("radial_distortion", "Radial distortion"),
            ("axial_contraction", "Axial contraction"), ("knn_retention", "kNN retention")]
JUDGES = [("M_paper", "Paper judge (gpt-4o)", C_MAIN), ("M_second", "Second judge (DeepSeek-V4-Pro)", C_ALT)]

# layout in inches, so that marker sizes in points can be placed without overlap
W, H = FULL, 4.12
TOP = 3.87                                   # top edge of both panels
A_L, A_S = 0.50, 2.58                        # panel a: left, side (square)
B_L, B_W, B_H = 4.27, 2.35, 2.90             # panel b: left, width, height
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([A_L / W, (TOP - A_S) / H, A_S / W, A_S / H])
bx = fig.add_axes([B_L / W, (TOP - B_H) / H, B_W / W, B_H / H])

# ---------------------------------------------------------------- (a) misalignment rate per organism, two judges
d = per.dropna(subset=["M_paper", "M_second"])
pool, out = d[d.set == "pool"], d[d.set == "outside"]
top = math.ceil(max(d.M_paper.max(), d.M_second.max()) / 0.1) * 0.1
S_A, RING_A = 18, 0.6                         # marker area (pt^2) and ring width (pt)
pt_per_unit = A_S * 72 / top
PAD = 2.2 * (math.sqrt(S_A) / 2 + RING_A) / pt_per_unit  # room below 0, so organisms at exactly 0 are drawn whole


def draw_points(a, pl, ou, s):
    a.plot([-1, 2], [-1, 2], ls=(0, (4, 3)), lw=0.8, color=C_BASE, zorder=1)
    a.scatter(pl.M_paper, pl.M_second, s=s, color=C_MAIN, edgecolor="white", linewidth=RING_A, zorder=3,
              clip_on=False)
    a.scatter(ou.M_paper, ou.M_second, s=s, facecolor="none", edgecolor=C_ALT, linewidth=1.0, zorder=4,
              clip_on=False)


draw_points(ax, pool, out, S_A)
ax.set_xlim(-PAD, top)
ax.set_ylim(-PAD, top)
ax.set_aspect("equal")
ticks = np.round(np.arange(0, top + 1e-9, 0.1), 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xlabel("Misalignment rate, gpt-4o paper judge (0-1)")
ax.set_ylabel("Misalignment rate, DeepSeek-V4-Pro (0-1)")
ax.set_title("a   Misalignment rate per organism", loc="left")

# label the organism with the highest paper-judge rate, below it, with a leader
x = d.loc[d.M_paper.idxmax()]
ax.annotate(f"{x.family}\n{x.organism}", (x.M_paper, x.M_second), xytext=(x.M_paper + 0.005, x.M_second - 0.16),
            ha="right", va="top", fontsize=6.5, color=INK2, linespacing=1.1,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=3))
ax.text(top * 0.84, top * 0.90, "y = x", rotation=45, rotation_mode="anchor", ha="center", va="bottom",
        fontsize=6.5, color=INK2)
rho = ev["measure_vs_M_paper"]["M_second"]["rho_bar"]
ax.text(top * 0.985, 0.0, f"Agreement in the pool:\nmean within-family\nSpearman {rho:.2f}", ha="right",
        va="bottom", fontsize=6.5, color=INK2, linespacing=1.1)

# zoom on the crowded corner near (0, 0): every organism with a paper-judge rate up to ZX, drawn in the empty space
# above y = x. The zoom is wider than tall because these organisms all sit low on the second judge's axis; the top
# keeps 0.02 of headroom for the note on the (0, 0) stack.
ZX = 0.1
zin = d[d.M_paper <= ZX]
ZY = math.ceil((zin.M_second.max() + 0.02) / 0.01) * 0.01
zscale = 2.6                                            # magnification
zpad = 2.2 * (math.sqrt(S_A + 4) / 2 + RING_A) / (pt_per_unit * zscale)
zw, zh = (ZX + zpad) * zscale, (ZY + zpad) * zscale     # inset size in panel a data units
zx0 = 0.06 * top
zy0 = zx0 + zw + 0.065                                  # bottom-right corner and its tick labels stay above y = x
assert zy0 + zh + 0.03 < top, "the inset and its title do not fit under the top of panel a"
axz = ax.inset_axes([zx0, zy0, zw, zh], transform=ax.transData, zorder=6)
axz.set_facecolor("white")
draw_points(axz, zin[zin.set == "pool"], zin[zin.set == "outside"], S_A + 4)
axz.set_xlim(-zpad, ZX)
axz.set_ylim(-zpad, ZY)
axz.set_aspect("equal")
axz.set_xticks([0, ZX / 2, ZX])
axz.set_yticks(np.arange(0, ZY + 1e-9, 0.025))
axz.tick_params(length=2, pad=1.5, labelsize=6)
axz.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
axz.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
for s in axz.spines.values():
    s.set_visible(True)
    s.set_color(AXIS)
    s.set_linewidth(0.6)
# the zoom window on panel a
ax.add_patch(plt.Rectangle((-PAD * 0.7, -PAD * 0.7), ZX + PAD * 0.7, ZY + PAD * 0.7, fill=False, ec=MUTED, lw=0.6,
                           zorder=2))
axz.set_title(f"Zoom on the grey box ({len(zin)} organisms)", fontsize=6, color=INK2, pad=2, loc="left")

# organisms at exactly (0, 0) under both judges stack into one mark: say how many
zero = d[(d.M_paper == 0) & (d.M_second == 0)]
nzp, nzo = int((zero.set == "pool").sum()), int((zero.set == "outside").sum())
# (every route for a leader into that corner crosses a neighbouring mark, so the note names the point instead)
axz.text(0.06 * ZX, 0.97 * ZY, f"{len(zero)} organisms at (0, 0):\n{nzp} pool, {nzo} outside",
         ha="left", va="top", fontsize=6, color=INK2, linespacing=1.05, zorder=7,
         bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

ax.legend(handles=[
    Line2D([], [], ls="", marker="o", ms=4.5, color=C_MAIN, mec="white", mew=RING_A,
           label=f"Pool, Qwen LoRA ({len(pool)})"),
    Line2D([], [], ls="", marker="o", ms=4.5, mfc="none", mec=C_ALT, mew=1.0, label=f"Outside the pool ({len(out)})"),
], loc="upper left", bbox_to_anchor=(-0.02, -0.15), ncol=2, handletextpad=0.3, columnspacing=1.0,
    labelcolor=INK2, borderaxespad=0)

# ---------------------------------------------------------------- (b) each reading against M, under each judge
S_BIG, S_SMALL, RING = 24, 8, 0.5             # marker areas (pt^2) and the white ring width (pt)
R_BIG, R_SMALL = math.sqrt(S_BIG) / 2, math.sqrt(S_SMALL) / 2
GAP_PT = 0.35
XLIM = (-1, 1)
pt_per_x = B_W * 72 / (XLIM[1] - XLIM[0])
pt_per_row = B_H * 72 / len(READINGS)
off = {"M_paper": -0.2, "M_second": 0.2}      # judge bands within a row (rows run top to bottom)
side = {"M_paper": -1, "M_second": 1}          # each judge keeps to its own half of the row


def tint(c, f=0.5):
    r, g, b = to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def place(xp, r, placed, yc, sd):
    """Vertical position (pt from the row centre) nearest the band centre yc at which a dot of radius r at
    x = xp touches no placed dot and stays in its own half of the row."""
    def free(c):
        return all(math.hypot(xp - ox, c - oy) >= r + orr + GAP_PT + RING - 1e-6 for ox, oy, orr in placed)

    cands = [yc]
    for ox, oy, orr in placed:
        D = r + orr + GAP_PT + RING
        dx = xp - ox
        if abs(dx) < D:
            h = math.sqrt(D * D - dx * dx)
            cands += [oy + h, oy - h]
    ok = [c for c in cands if free(c) and c * sd >= r + RING]
    return min(ok, key=lambda c: (round(abs(c - yc), 6), 0 if (c - yc) * sd >= 0 else 1))


max_extent, n_small = 0.0, 0
for i, (rd, _) in enumerate(READINGS):
    if i % 2 == 0:
        bx.axhspan(i - 0.5, i + 0.5, color="#f6f5f2", lw=0, zorder=0)
    row = {m: ev["readings_vs_measure"][f"{rd}|{m}"] for m, _, _ in JUDGES}
    # the two means are fixed on their band; family dots are stacked around them so that no dot hides another
    placed = [(row[m]["rho_bar"] * pt_per_x, off[m] * pt_per_row, R_BIG) for m, _, _ in JUDGES]
    for m, _, col in JUDGES:
        t = row[m]
        sig = t["p"] < 0.05
        bx.scatter([t["rho_bar"]], [i + off[m]], s=S_BIG, zorder=4, linewidth=1.1 if not sig else RING,
                   facecolor=col if sig else "white", edgecolor=col if not sig else "white")
        fx, fy = [], []
        for v in sorted(t["per_family_rho"].values(), key=lambda v: abs(v - t["rho_bar"])):
            xp = v * pt_per_x
            y = place(xp, R_SMALL, placed, off[m] * pt_per_row, side[m])
            placed.append((xp, y, R_SMALL))
            fx.append(v)
            fy.append(i + y / pt_per_row)
            max_extent = max(max_extent, abs(y) + R_SMALL + RING)
        n_small += len(fx)
        bx.scatter(fx, fy, s=S_SMALL, color=tint(col), edgecolor="white", linewidth=RING, zorder=5)
    # no two dots in the row overlap, rings included
    for a_ in range(len(placed)):
        for b_ in range(a_ + 1, len(placed)):
            (x1, y1, r1), (x2, y2, r2) = placed[a_], placed[b_]
            assert math.hypot(x1 - x2, y1 - y2) >= r1 + r2 + RING - 1e-6, (rd, placed[a_], placed[b_])
print(f"{n_small} family dots, none overlapping; they reach {max_extent:.1f} pt from a row centre "
      f"(half a row is {pt_per_row / 2:.1f} pt)")
assert max_extent <= pt_per_row / 2, "family dots spill into the next row: make the rows taller"

bx.axvline(0, color=AXIS, lw=0.8, zorder=1)
bx.set_xlim(*XLIM)
bx.set_ylim(len(READINGS) - 0.5, -0.5)
bx.set_yticks(range(len(READINGS)))
bx.set_yticklabels([lab for _, lab in READINGS])
bx.tick_params(axis="y", length=0)
bx.grid(axis="y", visible=False)
bx.set_xticks([-1, -0.5, 0, 0.5, 1])
bx.set_xticklabels(["−1", "−0.5", "0", "0.5", "1"])
bx.set_xlabel("Spearman of reading with misalignment rate (−1 to 1)")
LEFT = -0.40  # panel b's title and legend start under its row labels
bx.set_title("b   Readings against misalignment rate, pool", loc="left", x=LEFT)
nfam = len(ev["readings_vs_measure"][f"{READINGS[0][0]}|M_paper"]["per_family_rho"])
bx.legend(handles=[
    Line2D([], [], ls="", marker="o", ms=5, color=C_MAIN, mec="white", mew=RING, label="Paper judge (gpt-4o)"),
    Line2D([], [], ls="", marker="o", ms=5, color=C_ALT, mec="white", mew=RING, label="Second judge (DeepSeek-V4-Pro)"),
    Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=C_BASE, mew=1.1, label="Hollow: p ≥ 0.05"),
    Line2D([], [], ls="", marker="o", ms=5, color=C_BASE, mec="white", mew=RING,
           label=f"Large dot: mean of {nfam} families"),
    Line2D([], [], ls="", marker="o", ms=2.8, color=tint(C_BASE), mec="white", mew=RING,
           label="Small dot: one family"),
], loc="upper left", bbox_to_anchor=(LEFT, -0.13), ncol=2, handletextpad=0.3, columnspacing=1.0,
    labelcolor=INK2, borderaxespad=0)

save(fig, "fig_judge")
