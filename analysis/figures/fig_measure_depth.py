#!/usr/bin/env python
"""A small single-column figure for the results: how each reading's pooled within-family coefficient
against G and against U moves with depth (author, 2026-09-16: "small figure ... how the two metrics
U, G change per reading as depth % changes").

Five panels in two rows (three over two), one per reading; x is the depth of the layer stack, y the
pooled within-family coefficient over the six low-rank families: Spearman rho-bar solid, Pearson
r-bar dashed, G in one colour and U in another. Everything is read from zoo_cast.json,
graph_features.json and behaviour_bundle.json through fig_metric_slices, so it cannot drift from
Figure 5. A double-column version sat at the top of the references page and the kit's float
barrier then broke the column under it (2026-09-17), so the figure is one column wide.

In:  data/raw/{zoo_cast,graph_features,behaviour_bundle}.json, through fig_metric_slices.
Out: <paper>/figures/metrics/measure_depth.{pdf,png}
Feeds: Figure 7 and Section 4.2.
"""
import sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _paths
from _paperstyle import use, tex_printed_width, assert_canvas
import fig_metric_slices as MS
use(8)

PAPER = _paths.PAPER
TEX_REL = "sections/results_geometry.tex"
READS = [("REF_assistant_excess", "assistant shift\ndeviation"), ("assistant_minus_evil_excess", "assistant\u2013evil\nslip"),
         ("along_cos_sd", "orientation\ndistortion"), ("radial_distortion", "radial\ndistortion"),
         ("axial_contraction", "axial\ncontraction"), ("knn25pct_keep_mean", "kNN\nretention")]   # the paper's six readings in its three groups (2026-09-17)
# The group bands of Figure 6, in one neutral grey at three alphas, no hue the data uses. Two
# alphas five levels apart named three groups and put the topology reading in the detachment
# group's tint, and nothing on the page said what a band meant (audit, 2026-09-17). The three
# tints are now far enough apart to read and the key under the panels names them.
GROUP = [0, 0, 1, 1, 1, 2]
GNAME = ["assistant detachment", "cast deformation", "cast topology"]
TINT = [(0.549, 0.561, 0.596, a) for a in (0.04, 0.12, 0.22)]
MEAS = [("RB", "$G$", "#1f4e79"), ("RC+(1-C)", "$U$", "#b23a48")]
DEPTHS = MS.DEPTHS; XD = [int(d.rstrip("%")) for d in DEPTHS]


def pooled(xk, m, dep):
    rs, rr = [], []
    for f, ks in MS.BODY.items():
        x = np.array([MS.val(k, dep, xk) for k in ks], float); y = np.array([MS.measures(MS.BEH[k])[m] for k in ks], float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 3 and np.std(y[ok]) > 0 and np.std(x[ok]) > 0:
            rs.append(spearmanr(x[ok], y[ok])[0]); rr.append(pearsonr(x[ok], y[ok])[0])
    return float(np.mean(rs)), float(np.mean(rr))


W = tex_printed_width(TEX_REL, "metrics/measure_depth")
NCOL = 3
# the band every panel shows, read off the data: the smallest and largest coefficient in absolute
# value, with a marker's clearance at each end
_ALL = [abs(v) for xk, _ in READS for m, _, _ in MEAS for d in DEPTHS for v in pooled(xk, m, d)]
MLO = float(np.floor((min(_ALL) - 0.06) * 20) / 20)
MHI = float(min(1.03, np.ceil((max(_ALL) + 0.06) * 20) / 20))
TICKS = [t / 100 for t in range(0, 101, 25) if MLO <= t / 100 <= MHI]
TICKLAB = [abs(t * 4 - round(t * 4)) < 1e-9 and (round(t * 4) % 2 == 0) for t in TICKS]
# BOT holds the depth label and the two key rows; GAP holds each panel's own y tick labels, which
# it must, because the panels no longer share one range. RIGHT balances the y label on the left, so
# the drawn block sits centred in the column.
LEFT, RIGHT, TOP, MID, BOT, GAP = 0.427, 0.038, 0.42, 0.44, 0.66, 0.28
PW = (W - LEFT - RIGHT - (NCOL - 1) * GAP) / NCOL; PH = 0.66
H = TOP + 2 * PH + MID + BOT
fig = plt.figure(figsize=(W, H))
for i, (xk, name) in enumerate(READS):
    r, c = divmod(i, NCOL)
    y0 = H - TOP - PH - r * (PH + MID)
    ax = fig.add_axes([(LEFT + c * (PW + GAP)) / W, y0 / H, PW / W, PH / H])
    ax.set_facecolor(TINT[GROUP[i]])   # the reading's group (author, 2026-09-17)
    got = []
    for m, mlab, col in MEAS:
        rho = [pooled(xk, m, d)[0] for d in DEPTHS]; rr = [pooled(xk, m, d)[1] for d in DEPTHS]
        ax.plot(XD, rho, "-", color=col, lw=1.0, marker="o", ms=2.2, zorder=3)
        ax.plot(XD, rr, "--", color=col, lw=0.8, marker="o", ms=2.6, mfc="white", mew=0.55, zorder=3)
        got += rho + rr
    ax.axvline(50, color="#777", lw=0.6, ls=(0, (2.5, 1.8)), zorder=1)   # the persona layer
    # EACH PANEL SHOWS ITS OWN BAND. On -1 to 1 every reading sat in a fifth of the panel, the
    # markers at the ends were cut in half by the frame, and the middle half of every panel was
    # empty (audit, 2026-09-17). The window is the band the readings occupy, the same width in
    # every panel so the panels still compare, and it carries the sign of the panel it belongs to.
    sign = 1.0 if np.mean(got) > 0 else -1.0
    ax.set_ylim(*sorted((sign * MLO, sign * MHI)))
    ax.set_yticks([sign * t for t in TICKS])
    # zero carries no sign: an inverted axis printed "-0" beside the "0" of its neighbours (review, 2026-09-22)
    ax.set_yticklabels([("\u2212" if sign < 0 and t != 0 else "") + f"{t:g}" if lab else ""
                        for t, lab in zip(TICKS, TICKLAB)])
    # the 10 and 25 labels touched and read as "1025" at 5.4 pt (review, 2026-09-22): 25 keeps its tick, not its label
    ax.set_xticks(XD); ax.set_xticklabels([f"{d}" if d != 25 else "" for d in XD], fontsize=5.4)
    ax.set_xlim(4, 96)
    ax.tick_params(axis="y", labelsize=5.2, length=1.8, pad=1.4); ax.tick_params(axis="x", length=1.8, pad=1.2)
    ax.grid(axis="y", color="#e8e8e8", lw=0.45); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(name, fontsize=6.2, pad=2.5, linespacing=1.1)
fig.text(0.077 / W, (BOT + PH + MID / 2) / H, "pooled coefficient", rotation=90, va="center", ha="center", fontsize=6.4)
CX = (LEFT + (W - RIGHT)) / 2 / W
fig.text(CX, 0.40 / H, "depth of the layer stack (%)", fontsize=6.4, ha="center", va="bottom")
# every key handle carries the mark its series carries, so the key describes what is drawn
key = [Line2D([0], [0], color=MEAS[0][2], lw=1.0, marker="o", ms=2.2, label="$G$"),
       Line2D([0], [0], color=MEAS[1][2], lw=1.0, marker="o", ms=2.2, label="$U$"),
       Line2D([0], [0], color="#444", lw=1.0, marker="o", ms=2.2, label=r"$\bar\rho$"),
       Line2D([0], [0], color="#444", lw=0.8, ls="--", marker="o", ms=2.6, mfc="white", mew=0.55,
              label=r"$\bar r$")]
# the key runs under the six panels (a sixth reading took the empty cell, 2026-09-17)
fig.legend(handles=key, loc="lower center", bbox_to_anchor=(CX, 0.22 / H), ncol=4,
           fontsize=6.4, frameon=False, handlelength=1.6, columnspacing=1.2, handletextpad=0.5, borderpad=0)
# the panel tints, named: a code on the page that the reader cannot decode is worse than no code
gkey = [Patch(fc=TINT[g], ec="#c8c8c8", lw=0.4, label=GNAME[g]) for g in range(3)]
_gl = fig.legend(handles=gkey, loc="lower center", bbox_to_anchor=(CX, 0.02 / H), ncol=3,
                 fontsize=5.6, frameon=False, handlelength=1.0, handleheight=0.9,
                 columnspacing=0.7, handletextpad=0.35, borderpad=0)   # TrueType metrics run a little wider than Type 3 (2026-09-17)
for _t in _gl.get_texts():
    _t.set_color("#555")
# the guards: the key must fit the column and no tick label may run into the panel beside it
fig.canvas.draw()
_r = fig.canvas.get_renderer()
for _lg, _nm in ((_gl, "group key"), (fig.legends[0], "series key")):
    _b = _lg.get_window_extent(_r)
    if _b.x0 < 0 or _b.x1 > W * fig.dpi:
        raise SystemExit(f"fig_measure_depth: the {_nm} is wider than the column.")
for _i, _ax in enumerate(fig.axes):
    _room = (GAP if _i % NCOL else LEFT - 0.13) * fig.dpi   # column 0 also carries the y label
    _ab = _ax.get_window_extent(_r)
    for _t in _ax.get_yticklabels():
        if _t.get_text() and _ab.x0 - _t.get_window_extent(_r).x0 > _room:
            raise SystemExit(f"fig_measure_depth: the y tick label {_t.get_text()!r} of panel "
                             f"{_i} needs {(_ab.x0 - _t.get_window_extent(_r).x0) / fig.dpi:.3f} in "
                             f"of the {_room / fig.dpi:.3f} in beside it; widen GAP.")
assert_canvas(fig, TEX_REL, "metrics/measure_depth")
out = _paths.FIGURES / "metrics"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "measure_depth.pdf"); fig.savefig(out / "measure_depth.png", dpi=300); plt.close(fig)
for xk, name in READS:
    print(name.replace("\n", " "), " ".join(f"{d}: G {pooled(xk, 'RB', d)[0]:+.2f}/{pooled(xk, 'RB', d)[1]:+.2f} U {pooled(xk, 'RC+(1-C)', d)[0]:+.2f}/{pooled(xk, 'RC+(1-C)', d)[1]:+.2f}" for d in DEPTHS))
print(f"wrote figures/metrics/measure_depth.pdf, canvas {W:.2f} x {H:.2f} in")
