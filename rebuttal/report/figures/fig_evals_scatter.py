"""Three independent evaluations against the paper's M, in the 34-organism pool, both axes centred within family.

Centring subtracts each family's mean from x and from y, so the plot shows the within-family variation that the
pooled within-family Spearman rho_bar uses. Sources: reanalysis/extra-evals/per_organism.csv (values) and
reanalysis/extra-evals/extra_evals.json (measure_vs_M_paper rho_bar in the titles).

Every organism stays visible:
- A single organism is a hollow marker. Each marker's page width and height (outline plus edge line) are measured
  from its path.
- A stack is two organisms of one family (same marker) whose centres lie within STACK_PT points of each other on the
  page in both x and y, so that one would hide the other. Both members are drawn at their own positions as filled
  markers, and the legend cell says that a filled marker holds two organisms.
- Markers that would overlap are moved sideways, along paper M only, by the smallest least-squares moves that keep
  GAP_PT points of white between any two marker boxes (with SHAPE_P = inf). Two organisms less than ORDER_PT apart
  vertically keep their x order. The members of a stack move together. Vertical positions are never moved.
The script prints every stack and every moved point with its true and plotted x, asserts that no x order flipped
among vertically close organisms, and prints the largest move, which the figure states in its legend cell.
"""
from style import *

import json
import math
import os

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from scipy.optimize import minimize

EV = ROOT / "reanalysis/extra-evals"
DF = pd.read_csv(EV / "per_organism.csv")
J = json.load(open(EV / "extra_evals.json"))
POOL = DF[DF["set"] == "pool"].reset_index(drop=True)
assert len(POOL) == J["n_pool"], (len(POOL), J["n_pool"])
assert set(POOL["family"]) == set(POOL_FAMILIES), set(POOL["family"])
N = len(POOL)
FAM = POOL["family"].to_numpy()

PANELS = [("betley48_M", "Betley 48 M", "Betley 48 M (rate 0-1)"),
          ("mgs4_strict", "UK AISI MGS, 4 evals", "MGS 4-eval rate (rate 0-1)"),
          ("tqa_mc2_drop", "TruthfulQA MC2 drop", "TruthfulQA MC2 drop (score 0-1)")]
# PAPER=1 draws the paper appendix copy: MGS and TruthfulQA only, one row with the legend cell at its end
# (author, 2026-11-13: "remove the pre-registered ... no reviewer brings it up"), saved as fig_evals_scatter_paper
PAPER = os.environ.get("PAPER") == "1"
if PAPER:
    PANELS = PANELS[1:]
EDGE = 0.75  # marker edge line, points
GAP_PT = 0.4  # white kept between two side-by-side neighbours, points
STACK_PT = 1.5  # same-family centres closer than this in x and in y form a stack, points
ORDER_PT = 10.0  # organisms closer than this vertically keep their x order, points
SHAPE_P = np.inf  # spacing rule (1) below: inf = full box test, 2 = ellipse
# marker area in points^2; the diamond is drawn smaller because matplotlib's "D" box is 1.41 times a square's
AREA = {f: (10.5 if FAMILY_MARKER[f] == "D" else 18.0) for f in POOL_FAMILIES}
FAM_RANK = {f: i for i, f in enumerate(POOL_FAMILIES)}
KEY = [(FAM_RANK[f], o) for f, o in zip(POOL["family"], POOL["organism"])]  # tie order for identical points


def box_pt(fam):
    """Width and height in points of one marker as drawn: its path extent times sqrt(area), plus the edge line."""
    m = MarkerStyle(FAMILY_MARKER[fam])
    ext = m.get_path().transformed(m.get_transform()).get_extents()
    side = math.sqrt(AREA[fam])
    return ext.width * side + EDGE, ext.height * side + EDGE


BOX = np.array([box_pt(f) for f in FAM])  # (n, 2) width, height in points


def centred(col):
    return (POOL[col] - POOL.groupby("family")[col].transform("mean")).to_numpy()


def limits(v, pad=0.07):
    lo, hi = float(np.min(v)), float(np.max(v))
    return lo - pad * (hi - lo), hi + pad * (hi - lo)


def find_stacks(px, py):
    """Groups (lists of row indices, size >= 2) of same-family organisms within STACK_PT of each other."""
    root = list(range(N))

    def top(i):
        while root[i] != i:
            i = root[i]
        return i

    for i in range(N):
        for j in range(i + 1, N):
            if FAM[i] == FAM[j] and abs(px[i] - px[j]) < STACK_PT and abs(py[i] - py[j]) < STACK_PT:
                root[top(j)] = top(i)
    groups = {}
    for i in range(N):
        groups.setdefault(top(i), []).append(i)
    return [sorted(g, key=lambda i: KEY[i]) for g in groups.values() if len(g) > 1]


def push_apart_x(px, py, stacks):
    """Smallest sideways moves (page points, least squares) such that
    (1) two organisms outside a common stack, i left of j, with w = mean width + GAP_PT and h = mean height + GAP_PT:
        if |dy| < h then dx >= w (SHAPE_P = inf, a full box test: GAP_PT of white between the two marker boxes), or
        (dx / w)^P + (|dy| / h)^P >= 1 for a finite SHAPE_P = P;
    (2) two organisms less than ORDER_PT apart vertically never swap their order along x (organisms at exactly the
        same x may come in either order);
    (3) the members of a stack move together.
    y never changes."""
    same = {(i, j) for g in stacks for i in g for j in g if i != j}
    order = sorted(range(N), key=lambda i: (px[i], KEY[i]))
    rank = {i: r for r, i in enumerate(order)}
    pairs, need = [], []
    for i in range(N):  # (2)
        for j in range(N):
            if px[i] < px[j] - 1e-9 and abs(py[i] - py[j]) < ORDER_PT:
                pairs.append((i, j))
                need.append(0.0)
    for i in range(N):  # (1)
        for j in range(N):
            if rank[i] >= rank[j] or (i, j) in same:
                continue
            w, h = (BOX[i, 0] + BOX[j, 0]) / 2 + GAP_PT, (BOX[i, 1] + BOX[j, 1]) / 2 + GAP_PT
            dy = py[j] - py[i]
            if abs(dy) < h:  # j sits at least this far right of i
                pairs.append((i, j))
                need.append(float(w if np.isinf(SHAPE_P) else w * (1 - abs(dy / h) ** SHAPE_P) ** (1 / SHAPE_P)))
    A = np.zeros((len(pairs), N))
    for k, (i, j) in enumerate(pairs):
        A[k, i], A[k, j] = -1.0, 1.0
    need = np.array(need)
    eq = [(g[0], j) for g in stacks for j in g[1:]]  # (3)
    E = np.zeros((max(len(eq), 1), N))
    for k, (i, j) in enumerate(eq):
        E[k, i], E[k, j] = -1.0, 1.0
    e0 = E @ px
    cons = [{"type": "ineq", "fun": lambda q: A @ q - need, "jac": lambda q: A},
            {"type": "eq", "fun": lambda q: E @ q - e0, "jac": lambda q: E}]
    res = minimize(lambda q: float(np.sum((q - px) ** 2)), px.astype(float), jac=lambda q: 2 * (q - px),
                   constraints=cons, method="SLSQP", options={"maxiter": 2000, "ftol": 1e-12})
    assert res.success, res.message
    assert np.all(A @ res.x - need >= -1e-6), float(np.min(A @ res.x - need))
    assert np.allclose(E @ res.x, e0, atol=1e-6)
    return res.x


x = centred("M_paper")
OUT = int(np.argmax(np.abs(x)))  # the organism furthest from its family mean on the paper's M: labelled directly
OUT_LABEL = f"{POOL.at[OUT, 'family']}\n{POOL.at[OUT, 'organism']}"
XLIM = limits(x)

# two by two: three panels and a legend cell, so each panel is wide enough to keep every organism apart
if PAPER:
    fig, grid = plt.subplots(1, 3, figsize=(FULL, 2.6))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.2, wspace=0.42)
    axes, LEG = [grid[0], grid[1]], grid[2]
else:
    fig, grid = plt.subplots(2, 2, figsize=(FULL, 4.7))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.1, wspace=0.3, hspace=0.66)
    axes, LEG = [grid[0, 0], grid[0, 1], grid[1, 0]], grid[1, 1]

largest, stack_sizes = 0.0, []
for ax, (col, title, ylab) in zip(axes, PANELS):
    y = centred(col)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*limits(y))
    # page scale of this panel: data units per point
    pos = ax.get_position()
    sx = (XLIM[1] - XLIM[0]) / (pos.width * fig.get_figwidth() * 72)
    sy = (ax.get_ylim()[1] - ax.get_ylim()[0]) / (pos.height * fig.get_figheight() * 72)
    stacks = find_stacks(x / sx, y / sy)
    stack_sizes += [len(g) for g in stacks]
    xp = push_apart_x(x / sx, y / sy, stacks) * sx
    assert XLIM[0] < xp.min() and xp.max() < XLIM[1], (xp.min(), xp.max(), XLIM)
    # x order swaps (beyond solver tolerance): none allowed within ORDER_PT vertically; any others are printed
    flips = [(i, j) for i in range(N) for j in range(N) if x[i] < x[j] - 1e-9 and xp[i] > xp[j] + 1e-6]
    near = [(i, j) for i, j in flips if abs(y[i] - y[j]) / sy < ORDER_PT]
    moved = np.flatnonzero(np.abs(xp - x) > 5e-4)
    largest = max(largest, float(np.max(np.abs(xp - x))))
    print(f"[{col}] {len(stacks)} stacks, {len(moved)} points moved sideways, largest move "
          f"{np.max(np.abs(xp - x)):.4f}, x-order swaps {len(flips)} ({len(near)} within {ORDER_PT:g} pt "
          f"vertically = {ORDER_PT * sy:.3f} on y):")
    for g in stacks:
        print(f"    STACK x{len(g)}: " + "; ".join(
            f"{POOL.at[i, 'family']} {POOL.at[i, 'organism']} ({x[i]:+.4f}, {y[i]:+.4f})" for i in g))
    for i in moved[np.argsort(x[moved])]:
        print(f"    {POOL.at[i, 'family']} {POOL.at[i, 'organism']}: true ({x[i]:+.4f}, {y[i]:+.4f}), "
              f"plotted x {xp[i]:+.4f} (moved {xp[i] - x[i]:+.4f})")
    for i, j in flips:
        print(f"    SWAP {POOL.at[i, 'family']} {POOL.at[i, 'organism']} ({x[i]:+.4f}, {y[i]:+.4f}) -> x {xp[i]:+.4f} "
              f"and {POOL.at[j, 'family']} {POOL.at[j, 'organism']} ({x[j]:+.4f}, {y[j]:+.4f}) -> x {xp[j]:+.4f}")
    assert not near, f"{col}: sideways moves swapped the x order of {len(near)} vertically close pairs"
    ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
    ax.axvline(0, color=AXIS, lw=0.6, zorder=1)
    filled = np.zeros(N, dtype=bool)
    for g in stacks:
        filled[g] = True
    for fam in POOL_FAMILIES:
        for fill in (False, True):
            k = (FAM == fam) & (filled == fill)
            if k.any():
                ax.scatter(xp[k], y[k], marker=FAMILY_MARKER[fam], s=AREA[fam], facecolor=C_MAIN if fill else "none",
                           edgecolor=C_MAIN, linewidth=EDGE, zorder=3)
    ax.annotate(OUT_LABEL, (xp[OUT], y[OUT]), xytext=(0.99, 0.03), textcoords="axes fraction", ha="right",
                va="bottom", fontsize=6.3, color=INK2, linespacing=1.15,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=3.5, relpos=(0.93, 1.0)))
    rho = J["measure_vs_M_paper"][col]["rho_bar"]
    ax.set_title(f"{title}\n" + r"$\bar\rho$" + f" = {rho:.2f} with paper M", fontsize=7.5, linespacing=1.3)
    ax.set_ylabel(f"{ylab},\nminus family mean", fontsize=7)
    ax.set_xlabel("Paper M (rate 0-1),\nminus family mean", fontsize=7)

assert set(stack_sizes) <= {2}, stack_sizes  # the note below says "two organisms"
handles = [Line2D([], [], ls="none", marker=FAMILY_MARKER[f], markersize=AREA[f] ** 0.5, markerfacecolor="none",
                  markeredgecolor=C_MAIN, markeredgewidth=EDGE, label=f) for f in POOL_FAMILIES]
LEG.axis("off")
leg = LEG.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=1, handletextpad=0.5,
                 labelspacing=0.55, frameon=False, borderaxespad=0,
                 title=f"Model family\n({N} organisms in all)", title_fontsize=7)
leg.get_title().set_color(INK)
leg.get_title().set_multialignment("center")
bound = math.ceil(largest * 1000) / 1000  # the largest sideways move, rounded up to 0.001
LEG.text(0.5, 0.0, "A filled marker holds two organisms of one family\nat the same point. "
         f"Markers that would overlap are\nmoved sideways, at most {bound:.3f} on paper M.\n"
         "Vertical positions are exact.", transform=LEG.transAxes, ha="center", va="top", fontsize=6.5,
         color=INK2, linespacing=1.35)

save(fig, "fig_evals_scatter" + ("_paper" if PAPER else ""))
