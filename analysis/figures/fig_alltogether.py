#!/usr/bin/env python
"""Every organism together: the six readings with every family in one panel each (author, 2026-09-23:
"have version of this in paper in column but comparing all samples all together").

One column wide, two rows of three panels in the order and tints of Figure 6. Every rated fine-tune of
the study is one point, coloured by family, a triangle for a LoRA and a circle for a full fine-tune, an
open green ring on a control, a square for the baselines (which all sit at the reading's null and a
rate of zero), and a dashed least-squares line through all of them (author, 2026-09-23: "draw line
dashes, make more readable, polished, smaller"). No intervals: fifty-six whiskers in a panel an inch
wide read as noise, and Appendix B draws them per family. The detachment readings are read against U
and the others against G, as in Figure 4. Each panel prints Spearman rho and Pearson r over all
organisms with a two-sided Monte Carlo permutation p (20,000 draws), bold below .05, so the
across-family coefficient can be set beside the within-family ones of Figure 4.

Reads what Figure 4 reads, through fig_excess_by_tuning, so it cannot drift from it.

In:  the same data as fig_excess_by_tuning.
Out: <paper>/figures/excess/alltogether.{pdf,png}
Feeds: Figure 10 of the Results.
"""
import sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import rankdata
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _paths
from _paperstyle import use, tex_printed_width, assert_canvas, FAMC, CANON   # noqa: E402
import fig_excess_by_tuning as F                                               # noqa: E402
use(8)

PAPER = _paths.PAPER
TEX_REL = "sections/results_geometry.tex"
NAME = "excess/alltogether"
# the paper's six readings in its three groups, each with the measure Figure 4 reads it against
READS = [("REF_assistant_excess", "assistant shift\ndeviation", "U"),
         ("assistant_minus_evil_excess", "assistant–evil\nslip", "U"),
         ("along_cos_sd", "orientation\ndistortion", "G"),
         ("radial_distortion", "radial\ndistortion", "G"),
         ("axial_contraction", "axial\ncontraction", "G"),
         ("knn25pct_keep_mean", "kNN\nretention", "G")]
GROUP = [0, 0, 1, 1, 1, 2]
GNAME = ["assistant detachment", "cast deformation", "cast topology"]
TINT = [(0.549, 0.561, 0.596, a) for a in (0.04, 0.12, 0.22)]   # Figure 6's three tints
MEAS = {"U": F._Concern(), "G": F._MB()}
MLAB = {"U": "$U$", "G": "$G$"}
N_PERM = 20000

# every fine-tune with a rate; the topic-locked control has none
ROWS = [r for r in F.EXC if r["em"] is not None and "mismatch" not in r["o"]]
# the paper's family order: the Qwen pool as in Table 4, then Mistral, Llama, Gemma
ORDER = ["Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-7B", "Qwen3.5-9B", "Qwen3-32B", "Mistral-24B", "Llama-3.1-8B", "Gemma-3-12B"]
FAMS = [f for f in ORDER + [f for f in CANON if f not in ORDER] if any(r["f"] == f for r in ROWS)]


def x_of(r, xk):
    return float(r["x"]) if xk == "REF_assistant_excess" else float(F.reading_x(r["f"], r["o"], xk))


def xci_of(r, xk):
    ci = F.excess_ci(r["f"], r["o"]) if xk == "REF_assistant_excess" else F.reading_ci_of(xk)(r["f"], r["o"])
    return (float(ci[0]), float(ci[1])) if ci is not None else (np.nan, np.nan)


def coef(x, y, stat):
    a, b = (rankdata(x), rankdata(y)) if stat == "rho" else (x, y)
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float(a @ b / d) if d > 0 else np.nan


def perm_p(x, y, stat, seed=0):
    """Two-sided Monte Carlo permutation p over N_PERM shuffles of y, with the observed arrangement counted
    once (the n! orderings of the grids' exact p are out of reach at n above ten)."""
    rng = np.random.default_rng(seed)
    obs = coef(x, y, stat)
    null = np.array([coef(x, rng.permutation(y), stat) for _ in range(N_PERM)])
    return obs, float(((np.abs(null) >= abs(obs) - 1e-12).sum() + 1) / (N_PERM + 1))


def _num(v):
    return f"{v:+.2f}".replace("0.", ".").replace("-", "−")


def _ps(q):
    return "<.001" if q < 0.001 else f"{q:.3f}".lstrip("0")


def _mark(r):
    t = "lora" if r["o"] in F.RANK1 else F.tuning_of(r["f"], r["o"])
    return F.TUNE_MARK["lora" if t in ("lora", "rank1", "general") else "full"]


W = tex_printed_width(TEX_REL, NAME) or 3.3125
NCOL = 3
# LEFT holds each row's y label and tick labels; TOP and MID hold a panel's four header lines (the reading's
# two-line name and its two coefficient lines); BOT holds the three key rows
LEFT, RIGHT, TOP, MID, BOT, GAP = 0.36, 0.05, 0.50, 0.58, 0.86, 0.30   # BOT: the bottom row's tick labels end 0.12 in under its panels; the family key (two rows) starts under that
PW = (W - LEFT - RIGHT - (NCOL - 1) * GAP) / NCOL; PH = 0.62
H = TOP + 2 * PH + MID + BOT
fig = plt.figure(figsize=(W, H))
report = []
for i, (xk, name, m) in enumerate(READS):
    YM = MEAS[m]
    rr, cc = divmod(i, NCOL)
    y0 = H - TOP - PH - rr * (PH + MID)
    ax = fig.add_axes([(LEFT + cc * (PW + GAP)) / W, y0 / H, PW / W, PH / H])
    ax.set_facecolor(TINT[GROUP[i]])
    xs, ys = [], []
    for r in ROWS:
        x = x_of(r, xk); y = YM.value(r, r["f"])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xs.append(x); ys.append(y)
        col = FAMC[r["f"]]
        mk = _mark(r)
        ax.plot(x, y, mk, color=col, ms=2.5 if mk == "o" else 3.1, zorder=3, markeredgecolor="white", markeredgewidth=0.3, alpha=0.92)
        if F.is_ctl(r):
            ax.plot(x, y, "o", mfc="none", mec="#2f9e6b", mew=0.6, ms=3.9, zorder=4)
    xnull = F.XNULL.get(xk, 0.0)
    ax.axvline(xnull, color="#999", lw=0.5, zorder=1)
    ax.plot(xnull, 0.0, marker="s", mfc="white", mec="#444", mew=0.7, ms=2.6, ls="none", zorder=5)   # every baseline
    x = np.array(xs); y = np.array(ys)
    rho, p_rho = perm_p(x, y, "rho"); r_, p_r = perm_p(x, y, "r")
    # the least-squares line through every organism, dashed, over the data's own range
    _m, _b = np.polyfit(x, y, 1); _xx = np.array([x.min(), x.max()])
    ax.plot(_xx, _m * _xx + _b, ls=(0, (3.2, 2.0)), color="#333", lw=0.75, zorder=2.5)
    report.append((xk, m, len(x), rho, p_rho, r_, p_r))
    lo, hi = min(x.min(), xnull), max(x.max(), xnull); pad = 0.10 * max(hi - lo, 0.05)
    ax.set_xlim(lo - pad, hi + pad)
    ytop = max(y.max(), 0.0); ax.set_ylim(-0.06 * ytop - 0.01, ytop * 1.10 + 0.01)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5], min_n_ticks=2))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5], min_n_ticks=2))
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v:.2f}".replace("0.", ".").replace("-", "−") if abs(v) < 10 else f"{v:g}")
    ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis="both", labelsize=5.0, length=1.6, pad=1.3)
    ax.grid(True, color="#888", alpha=0.18, lw=0.45); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel(MLAB[m], fontsize=6.2, labelpad=1.5, rotation=0, ha="right", va="center")
    # the header: the reading's name, then its two coefficient lines, each bold when its own p is below .05
    ax.set_title(name, fontsize=6.0, pad=2.2 + 2 * 5.3 * 1.22, linespacing=1.08)
    import matplotlib.transforms as _mt
    for k, (sym, bsym, v, q) in enumerate(reversed([("$\\rho$", "$\\boldsymbol{\\rho}$", rho, p_rho), ("$r$", "$\\boldsymbol{r}$", r_, p_r)])):
        bold = q < 0.05
        ln = f"{bsym if bold else sym} {_num(v)} ({_ps(q)})"
        tr = _mt.offset_copy(ax.transAxes, fig=fig, x=0, y=k * 5.3 * 1.22, units="points")
        ax.text(0.0, 1.02, ln, transform=tr, ha="left", va="bottom", fontsize=5.3,
                color="#222" if bold else "#555", fontweight="bold" if bold else "normal")

CX = (LEFT + (W - RIGHT)) / 2 / W
# the keys: families, then the marks, then the group tints
fam_key = [Line2D([0], [0], marker="o", ls="none", color=FAMC[f], ms=3.0, label=f) for f in FAMS]
lg1 = fig.legend(handles=fam_key, loc="lower center", bbox_to_anchor=(CX, 0.44 / H), ncol=4, fontsize=5.2,
                 frameon=False, handlelength=1.0, columnspacing=0.9, handletextpad=0.35, borderpad=0)
mark_key = [Line2D([0], [0], marker="o", ls="none", color="#555", ms=3.0, label="full fine-tune"),
            Line2D([0], [0], marker="^", ls="none", color="#555", ms=3.6, label="LoRA"),
            Line2D([0], [0], marker="o", ls="none", mfc="white", mec="#2f9e6b", mew=0.8, ms=4.2, label="control"),
            Line2D([0], [0], marker="s", ls="none", mfc="white", mec="#444", mew=0.8, ms=2.8, label="baseline"),
            Line2D([0], [0], color="#333", lw=0.75, ls=(0, (3.2, 2.0)), label="least-squares line")]
lg2 = fig.legend(handles=mark_key, loc="lower center", bbox_to_anchor=(CX, 0.24 / H), ncol=5, fontsize=5.2,
                 frameon=False, handlelength=1.0, columnspacing=0.9, handletextpad=0.35, borderpad=0)
gkey = [Patch(fc=TINT[g], ec="#c8c8c8", lw=0.4, label=GNAME[g]) for g in range(3)]
lg3 = fig.legend(handles=gkey, loc="lower center", bbox_to_anchor=(CX, 0.05 / H), ncol=3, fontsize=5.4,
                 frameon=False, handlelength=1.0, handleheight=0.9, columnspacing=0.7, handletextpad=0.35, borderpad=0)
for lg in (lg1, lg2, lg3):
    for t in lg.get_texts():
        t.set_color("#555")
n_all = len(ROWS)   # the caption carries n and the family count

# guards: every key inside the column and clear of the panels above it, no tick label into the panel beside it,
# the canvas as printed
fig.canvas.draw(); _r = fig.canvas.get_renderer()
_low = min(t.get_window_extent(_r).y0 for ax in fig.axes[NCOL:] for t in ax.get_xticklabels() if t.get_text())
for lg, nm in ((lg1, "family key"), (lg2, "mark key"), (lg3, "group key")):
    b = lg.get_window_extent(_r)
    if b.x0 < 0 or b.x1 > W * fig.dpi:
        raise SystemExit(f"fig_alltogether: the {nm} is wider than the column.")
    if b.y1 > _low - 2:
        raise SystemExit(f"fig_alltogether: the {nm} runs into the bottom row's tick labels; raise BOT.")
for i, ax in enumerate(fig.axes):
    room = (GAP if i % NCOL else LEFT - 0.12) * fig.dpi
    ab = ax.get_window_extent(_r)
    for t in ax.get_yticklabels():
        if t.get_text() and ab.x0 - t.get_window_extent(_r).x0 > room:
            raise SystemExit(f"fig_alltogether: the y tick label {t.get_text()!r} of panel {i} needs "
                             f"{(ab.x0 - t.get_window_extent(_r).x0) / fig.dpi:.3f} in of the {room / fig.dpi:.3f} in beside it.")
assert_canvas(fig, TEX_REL, NAME)
out = _paths.FIGURES / "excess"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "alltogether.pdf"); fig.savefig(out / "alltogether.png", dpi=300); plt.close(fig)
for xk, m, n, rho, p_rho, r_, p_r in report:
    print(f"  {xk:28s} vs {m}  n {n}  rho {rho:+.2f} (p {p_rho:.4f})  r {r_:+.2f} (p {p_r:.4f})")
print(f"wrote figures/excess/alltogether.pdf, canvas {W:.2f} x {H:.2f} in, {n_all} organisms in {len(FAMS)} families")
