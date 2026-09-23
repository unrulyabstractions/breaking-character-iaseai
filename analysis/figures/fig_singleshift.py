"""Figure 8: one shared shift, at every exported depth, in the full residual space.

In:  the sweep cache (see geometry/README.md).
Out: <paper>/figures/shift/single_shift.{pdf,png}
Feeds: Section 4.3.
"""
# One shared shift, at every exported depth, in the full residual space (redesign, 2026-09-11: the
# earlier figure read one depth inside a 3-D joint basis and spent a page on 34 bars).
# LEFT : the share of each organism's drift energy that the single mean shift carries,
#        |g|^2 / mean_i |d_i|^2 with d_i = org_i - base_i and g = mean_i d_i, against depth.
# RIGHT: per-role agreement with that shift, cos(d_i, g): the median over roles (line) and the
#        smallest role (marker) per organism and depth. Data: the raw per-role vectors behind
#        zoo_cast.json (scratchpad/excess_sweep_cache/*.npz), every organism the cache holds,
#        controls included.
#
# COLOUR (redesign, 2026-09-14). The figure drew 44 lines in the eight family colours. Eight
# categorical colours is past the limit a reader can hold (Wilke, "Color pitfalls": three to five
# work, eight is the hard ceiling), and no single line was traceable through the bundle anyway, so
# the hue was decoration. The two groups the text actually names are drawn instead: every organism
# in grey, the aligned controls in green, and the median organism as one dark line. The per-family
# lines live in the appendix shift fields.
import glob, os, sys, pathlib as _pl
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
from _paperstyle import CANON, INK, use
from _organism_meta import is_control
use(8)
CACHE = _paths.SWEEP_CACHE
DEPTHS = ['10%', '25%', '50%', '70%', '90%']; X = [10, 25, 50, 70, 90]

GREYL = "#bdbdbd"     # one organism
CTRL  = "#2f7d5d"     # an aligned control
MED   = "#1a1a1a"     # the median organism
WORST = "#b23a48"     # the least agreeing role of an organism

rows = []
ALLC = [[] for _ in DEPTHS]          # every role of every organism, per depth: the distribution the panel draws
for fam in CANON:
    bp = CACHE / f"{fam}__base.npz"
    if not bp.exists(): continue
    zb = np.load(bp, allow_pickle=True); X0 = zb["X"].astype(np.float64)
    for f in sorted(glob.glob(str(CACHE / f"{fam}__*.npz"))):
        og = os.path.basename(f)[len(fam) + 2:-4]
        if og == "base": continue
        # the topic-locked control carries no reading (Table 4), so it is not drawn here either (review 2026-09-18)
        if og.startswith("mismatch"): continue
        zo = np.load(f, allow_pickle=True); XO = zo["X"].astype(np.float64)
        share, med, mn = [], [], []
        for k in range(len(DEPTHS)):
            D = XO[k] - X0[k]; g = D.mean(0)
            share.append(float(g @ g / (np.linalg.norm(D, axis=1) ** 2).mean()))
            c = (D @ g) / (np.linalg.norm(D, axis=1) * np.linalg.norm(g) + 1e-12)
            med.append(float(np.median(c))); mn.append(float(c.min())); ALLC[k].extend(c.tolist())
        rows.append(dict(fam=fam, og=og, ctrl=is_control(og, fam), share=share, med=med, mn=mn, n=XO.shape[1]))
print(f"{len(rows)} organisms from the cache; families: {sorted({r['fam'] for r in rows})}")
n_ctrl = sum(r["ctrl"] for r in rows)
chance = max(1.0 / r["n"] for r in rows)   # independent directions: E|g|^2/mean|d|^2 = 1/n

# The canvas is the printed one: single_shift.pdf is included at width=\columnwidth and
# iaseai26.sty sets textwidth=5.5in, so LaTeX scales it by 1.0 and a point here is a point on the
# page. Dropping the eight-family legend strip gave the panels 0.9 in of height back.
# WAY SMALLER (author, 2026-09-16): 0.655 of the column, the type kept legible by shortening what it says
fig, (aL, aR) = plt.subplots(1, 2, figsize=(3.3125, 2.3))   # one column of the AAAI kit (2026-09-16)
fig.subplots_adjust(left=0.19, right=0.985, top=0.88, bottom=0.31, wspace=0.52)

# Draw order is the ink hierarchy: the bundle first and faint, the named groups over it.
for r in rows:
    if r["ctrl"]: continue
    aL.plot(X, [100 * v for v in r["share"]], color=GREYL, lw=0.6, zorder=2)
    aR.plot(X, r["med"], color=GREYL, lw=0.6, zorder=2)
# The controls carry a dash as well as a hue. In greyscale the green and the dark median line come
# out at almost the same grey, and only the line weight separated them (checked 2026-09-14).
CTRL_LS = (0, (2.6, 1.6))
for r in rows:
    if not r["ctrl"]: continue
    aL.plot(X, [100 * v for v in r["share"]], color=CTRL, lw=0.9, ls=CTRL_LS, zorder=4)
    aR.plot(X, r["med"], color=CTRL, lw=0.9, ls=CTRL_LS, zorder=4)
# the median and the mean run over the ORGANISMS; the controls are drawn and counted separately (review 2026-09-18)
ORGS = [r for r in rows if not r["ctrl"]]   # the organisms; the controls are drawn and counted separately (review 2026-09-18)
medshare = [100 * float(np.median([r["share"][k] for r in ORGS])) for k in range(len(X))]
medmed = [float(np.median([r["med"][k] for r in ORGS])) for k in range(len(X))]
aL.plot(X, medshare, color=MED, lw=1.6, zorder=6, solid_capstyle="round")
aR.plot(X, medmed, color=MED, lw=1.6, zorder=6, solid_capstyle="round")
# the mean beside the median, dashed (author, 2026-09-16: "also show avg")
# The mean ran in the median's colour and weight on top of the median, so a column scan of the
# printed page found ONE black line where the key promised two (audit, 2026-09-17). The mean now
# carries a white casing: where it lies on the median, the gaps between its dashes cut the median
# open, so the reader sees a dashed line over a solid one.
MEAN_LS = (0, (2.6, 1.9))
MEAN_PE = [pe.withStroke(linewidth=2.7, foreground="white")]
meanshare = [100 * float(np.mean([r["share"][k] for r in ORGS])) for k in range(len(X))]
meanmed = [float(np.mean([r["med"][k] for r in ORGS])) for k in range(len(X))]
aL.plot(X, meanshare, color=MED, lw=1.1, ls=MEAN_LS, zorder=7, path_effects=MEAN_PE)
aR.plot(X, meanmed, color=MED, lw=1.1, ls=MEAN_LS, zorder=7, path_effects=MEAN_PE)

# ---- left: how much of the drift one direction carries
aL.axhline(100 * chance, color="#777", lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
aL.text(91, 100 * chance + 2.0, f"chance, {100*chance:.1f}%",
        fontsize=5.2, color="#777", ha="right", va="bottom")
aL.set_ylim(0, 100); aL.set_yticks([0, 25, 50, 75, 100])
aL.set_ylabel("drift energy in the\nshared shift (%)", fontsize=6.4)
# the floor of the bundle, named where it happens: the reader should not have to trust "most"
lo_k = int(np.argmin([min(100 * v for v in r["share"]) for r in rows]))
lo_r = rows[lo_k]; lo_i = int(np.argmin(lo_r["share"]))
aL.annotate(f"lowest, {100*lo_r['share'][lo_i]:.0f}%", (X[lo_i], 100 * lo_r["share"][lo_i]),
            textcoords="data", xytext=(X[lo_i] + 12, 47), fontsize=5.2, color="#555", ha="left",
            va="center", zorder=7,
            bbox=dict(fc="white", ec="none", pad=1.0),
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#999", shrinkA=0, shrinkB=2))

# One key for both panels, in a strip under the figure: inside the left panel it sat over the data
# (author, 2026-09-16: "polish").
from matplotlib.lines import Line2D as _L2
from matplotlib.patches import Patch as _Pt
_key = [_L2([0], [0], color=MED, lw=1.6, label="median over the organisms"),   # no count: a count here read against the Methods' count of organisms and controls (review, 2026-09-18)
        _L2([0], [0], color=MED, lw=1.1, ls=MEAN_LS, path_effects=MEAN_PE,
            label="mean over the organisms"),
        _L2([0], [0], color=CTRL, lw=0.9, ls=CTRL_LS, label="aligned controls"),
        _L2([0], [0], color=GREYL, lw=0.9, label="one organism"),
        _Pt(fc=WORST, alpha=0.45, ec=WORST, lw=0.6, label="roles, 25th to 75th"),
        _Pt(fc=WORST, alpha=0.14, ec="none", label="roles, 5th to 95th")]
_lg = fig.legend(handles=_key, loc="lower left", bbox_to_anchor=(0.02, 0.005), ncol=3, fontsize=5.0,
                 frameon=False, handlelength=1.5, handletextpad=0.45, columnspacing=1.15, borderpad=0.0)
for _t in _lg.get_texts():
    _t.set_color("#444")

# ---- right: whether any role moves against the shift
# The 44 minima were drawn as 44 open triangles per depth, a red wall that outweighed the medians
# it sits under and swallowed its own label. They are now the band they always were: the spread of
# the least agreeing role across organisms, with its median and the single lowest reading named.
# THE DISTRIBUTION OF EVERY ROLE'S AGREEMENT, not the least agreeing role. A band from the 5th to
# the 95th percentile over all roles of all organisms, a darker one from the 25th to the 75th, and
# under the zero line the count of role readings that move against the shift (author, 2026-09-16:
# "is there a better way to show this?"; the minimum of five thousand readings was one number).
Q = {q: [float(np.percentile(ALLC[k], q)) for k in range(len(X))] for q in (5, 25, 75, 95)}
NNEG = [int(sum(1 for v in ALLC[k] if v < 0)) for k in range(len(X))]
NALL = [len(ALLC[k]) for k in range(len(X))]
# two bands a reader can tell apart: the inner one darker with its own edge (author, 2026-09-16)
aR.fill_between(X, Q[5], Q[95], color=WORST, alpha=0.14, lw=0, zorder=2)
aR.fill_between(X, Q[25], Q[75], color=WORST, alpha=0.45, lw=0, zorder=2)
for q in (25, 75):
    aR.plot(X, Q[q], color=WORST, lw=0.6, alpha=0.9, zorder=3)
# the count row under the zero line came off (author, 2026-09-16: "what is this text below");
# the caption states the two readings below zero
# THE RANGE FOLLOWS THE DATA. On 0 to 1 every drawn reading sat in the top fifth of the panel and
# the middle-half band printed 10 px tall, so the bands, the median and the mean read as one strip
# (audit, 2026-09-17). Nothing drawn here falls below the 5th percentile of the role readings, so
# the floor is that percentile rounded down; the zero line and its "orthogonal" tag came off with
# the empty space, and the caption carries the two readings below zero.
_drawn = ([v for r in rows for v in r["med"]] + medmed + meanmed
          + [Q[q][k] for q in (5, 95) for k in range(len(X))])
_ylo = np.floor((min(_drawn) - 0.02) * 20) / 20
aR.set_ylim(_ylo, 1.005)
aR.set_yticks([t / 10 for t in range(int(np.ceil(_ylo * 10)), 11)])
aR.set_ylabel("per-role agreement\n" + r"$\cos(s_i,\bar s)$", fontsize=6.4)
print("roles below zero per depth:", dict(zip(DEPTHS, NNEG)), "of", NALL)

def _fit_title(ax, options, fs=5.7):
    """Set the first title that fits inside the panel's own box, and refuse when none does.

    The left title ran 0.29 in past its panel and ended over the right panel's top tick label
    (audit, 2026-09-17). A title is now measured against the box it names."""
    r = fig.canvas.get_renderer()
    w = ax.get_window_extent(r).width
    for text in options:
        t = ax.set_title(text, fontsize=fs, loc="left", color="#333", pad=3, fontweight="bold")
        if t.get_window_extent(r).width <= w:
            print(f"  title: {text!r} ({t.get_window_extent(r).width:.0f} of {w:.0f} px)")
            return
    raise SystemExit(f"fig_singleshift: no title in {options} fits the {w:.0f} px panel at {fs} pt.")


for ax, t in ((aL, ["One direction carries the drift", "One direction carries it",
                    "One shared direction"]),
              (aR, ["Every role moves with it", "Every role agrees"])):
    # The depths sit 15 units apart at the low end, so "10%" and "25%" touched and printed as one
    # token (audit, 2026-09-17). The per cent sign moves to the axis title and the labels are bare
    # numbers, which leaves the gap between any two of them wider than the gaps inside one.
    ax.set_xticks(X); ax.set_xticklabels([f"{x}" for x in X], fontsize=5.8)
    ax.set_xlabel("depth (%)", fontsize=6.4, labelpad=2); ax.tick_params(axis="y", labelsize=5.8)
    ax.set_xlim(6, 94)
    # horizontal rules only: depth is the categorical axis and the comparison runs up the panel
    ax.grid(axis="y", lw=0.4, color="#dddddd"); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    _fit_title(ax, t)
# the right panel spans a quarter of a unit, where a minor rule every 0.025 is noise
aR.grid(False, which="minor", axis="y")

# ---- the guards: every one of these caught a defect on the printed page
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_PX = fig.dpi * fig.get_figwidth()
if _lg.get_window_extent(_r).x1 > _PX - 0.02 * fig.dpi:
    raise SystemExit("fig_singleshift: the key runs past the canvas; shorten a label or drop a column.")
for _ax, _nm in ((aL, "left"), (aR, "right")):
    _bbs = sorted((t.get_window_extent(_r) for t in _ax.get_xticklabels()), key=lambda b: b.x0)
    _gap = min(b.x0 - a.x1 for a, b in zip(_bbs, _bbs[1:]))
    if _gap < 0.02 * fig.dpi:
        raise SystemExit(f"fig_singleshift: the {_nm} panel's x tick labels are {_gap:.1f} px apart.")
    _tb = _ax.title.get_window_extent(_r); _ab = _ax.get_window_extent(_r)
    if _tb.x1 > _ab.x1 + 1:
        raise SystemExit(f"fig_singleshift: the {_nm} title runs {(_tb.x1 - _ab.x1) / fig.dpi:.3f} in past its panel.")
PDF = f"{_paths.FIGURES}/shift/single_shift.pdf"
PNG = f"{_paths.FIGURES}/shift/single_shift.png"
os.makedirs(os.path.dirname(PDF), exist_ok=True)
fig.savefig(PDF); fig.savefig(PNG, dpi=300); plt.close(fig)
lo = min(min(r["share"]) for r in rows); mn_all = min(min(r["mn"]) for r in rows)
print(f"wrote single_shift.pdf/png: {len(rows)} organisms ({n_ctrl} controls), share min {100*lo:.0f}%, chance {100*chance:.1f}%, smallest role cos {mn_all:+.2f}")
neg = [(r["fam"], r["og"], DEPTHS[k], r["mn"][k]) for r in rows for k in range(len(X)) if r["mn"][k] < 0]
print("  organism-depth cells whose least agreeing role falls below zero:", neg)
