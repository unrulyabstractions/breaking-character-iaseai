#!/usr/bin/env python
"""Every organism together against two cloud-similarity metrics (author, 2026-10-03: "Other metrics TODO,
include ALL TOGETHER plot for M vs linear CKA and M vs Procrustes disparity").

One column wide, one row of two panels in the style of Figure 10: every rated fine-tune of the study is one
point, coloured by family, a triangle for a LoRA and a circle for a full fine-tune, an open green ring on a
control, a square for the baselines (a baseline's cloud is identical to itself: CKA 1, disparity 0, rate 0),
and a dashed least-squares line through all of them. The x axis is the linear CKA, or the Procrustes
disparity, between the baseline's cloud and the organism's at the persona layer, as Aneja et al. read it
(blog/data/ig_reconcile.json, the same file Appendix C read). The y axis is the misalignment rate M. Each
panel prints Spearman rho and Pearson r over all organisms with a two-sided Monte Carlo permutation p
(20,000 draws), bold below .05.

In:  data/raw/ig_reconcile.json and the same data as fig_excess_by_tuning.
Out: <paper>/figures/excess/alltogether_ig.{pdf,png}
Feeds: Figure 11 of the Discussion.
"""
import sys, json, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms as _mt
from matplotlib.lines import Line2D
from scipy.stats import rankdata
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _paths
from _paperstyle import use, tex_printed_width, assert_canvas, FAMC, CANON   # noqa: E402
import fig_excess_by_tuning as F                                               # noqa: E402
use(8)

PAPER = _paths.PAPER
TEX_REL = "sections/discussion.tex"
NAME = "excess/alltogether_ig"
READS = [("ours|raw|cka", "linear CKA", 1.0), ("ours|raw|procrustes", "Procrustes disparity", 0.0)]   # key, name, a baseline's value
N_PERM = 20000

IG = {(r["f"], r["o"]): r for r in json.load(open(_paths.RAW / "ig_reconcile.json"))["rows"]}
ROWS = [r for r in F.EXC if r["em"] is not None and "mismatch" not in r["o"]]   # every fine-tune with a rate
missing = [(r["f"], r["o"]) for r in ROWS if (r["f"], r["o"]) not in IG]
if missing:
    raise SystemExit(f"fig_alltogether_ig: no cloud-similarity row for {missing}")
ORDER = ["Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-7B", "Qwen3.5-9B", "Qwen3-32B", "Mistral-24B", "Llama-3.1-8B", "Gemma-3-12B"]
FAMS = [f for f in ORDER + [f for f in CANON if f not in ORDER] if any(r["f"] == f for r in ROWS)]
YM = F._Rate()   # the misalignment rate M, in percent


def coef(x, y, stat):
    a, b = (rankdata(x), rankdata(y)) if stat == "rho" else (x, y)
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float(a @ b / d) if d > 0 else np.nan


def perm_p(x, y, stat, seed=0):
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
NCOL = 2
LEFT, RIGHT, TOP, BOT, GAP = 0.40, 0.05, 0.50, 0.74, 0.34   # BOT: the x labels end 0.3 in under the panels; the two key rows sit under that
PW = (W - LEFT - RIGHT - (NCOL - 1) * GAP) / NCOL; PH = 0.95
H = TOP + PH + BOT
fig = plt.figure(figsize=(W, H))
report = []
for i, (xk, name, xbase) in enumerate(READS):
    ax = fig.add_axes([(LEFT + i * (PW + GAP)) / W, BOT / H, PW / W, PH / H])
    xs, ys = [], []
    for r in ROWS:
        x = IG[(r["f"], r["o"])].get(xk); y = YM.value(r, r["f"])
        if x is None or not (np.isfinite(x) and np.isfinite(y)):
            continue
        xs.append(float(x)); ys.append(float(y))
        mk = _mark(r)
        ax.plot(x, y, mk, color=FAMC[r["f"]], ms=2.5 if mk == "o" else 3.1, zorder=3, markeredgecolor="white", markeredgewidth=0.3, alpha=0.92)
        if F.is_ctl(r):
            ax.plot(x, y, "o", mfc="none", mec="#2f9e6b", mew=0.6, ms=3.9, zorder=4)
    ax.axvline(xbase, color="#999", lw=0.5, zorder=1)
    ax.plot(xbase, 0.0, marker="s", mfc="white", mec="#444", mew=0.7, ms=2.6, ls="none", zorder=5)   # every baseline
    x = np.array(xs); y = np.array(ys)
    rho, p_rho = perm_p(x, y, "rho"); r_, p_r = perm_p(x, y, "r")
    _m, _b = np.polyfit(x, y, 1); _xx = np.array([x.min(), x.max()])
    ax.plot(_xx, _m * _xx + _b, ls=(0, (3.2, 2.0)), color="#333", lw=0.75, zorder=2.5)
    report.append((xk, len(x), rho, p_rho, r_, p_r))
    lo, hi = min(x.min(), xbase), max(x.max(), xbase); pad = 0.08 * max(hi - lo, 0.05)
    ax.set_xlim(lo - pad, hi + pad)
    ytop = max(y.max(), 0.0); ax.set_ylim(-0.06 * ytop - 0.5, ytop * 1.10 + 0.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5], min_n_ticks=2))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5], min_n_ticks=2))
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v:.2f}".replace("0.", ".").replace("-", "−") if abs(v) < 2 else f"{v:g}")
    ax.xaxis.set_major_formatter(fmt)
    ax.tick_params(axis="both", labelsize=5.2, length=1.6, pad=1.3)
    ax.grid(True, color="#888", alpha=0.18, lw=0.45); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(name, fontsize=6.0, labelpad=2)
    if i == 0:
        ax.set_ylabel("misalignment rate $M$ (%)", fontsize=6.0, labelpad=2)
    for k, (sym, bsym, v, q) in enumerate(reversed([("$\\rho$", "$\\boldsymbol{\\rho}$", rho, p_rho), ("$r$", "$\\boldsymbol{r}$", r_, p_r)])):
        bold = q < 0.05
        tr = _mt.offset_copy(ax.transAxes, fig=fig, x=0, y=k * 5.6 * 1.22, units="points")
        ax.text(0.0, 1.02, f"{bsym if bold else sym} {_num(v)} ({_ps(q)})", transform=tr, ha="left", va="bottom",
                fontsize=5.6, color="#222" if bold else "#555", fontweight="bold" if bold else "normal")

CX = (LEFT + (W - RIGHT)) / 2 / W
fam_key = [Line2D([0], [0], marker="o", ls="none", color=FAMC[f], ms=3.0, label=f) for f in FAMS]
lg1 = fig.legend(handles=fam_key, loc="lower center", bbox_to_anchor=(CX, 0.22 / H), ncol=4, fontsize=5.2,
                 frameon=False, handlelength=1.0, columnspacing=0.9, handletextpad=0.35, borderpad=0)
mark_key = [Line2D([0], [0], marker="o", ls="none", color="#555", ms=3.0, label="full fine-tune"),
            Line2D([0], [0], marker="^", ls="none", color="#555", ms=3.6, label="LoRA"),
            Line2D([0], [0], marker="o", ls="none", mfc="white", mec="#2f9e6b", mew=0.8, ms=4.2, label="control"),
            Line2D([0], [0], marker="s", ls="none", mfc="white", mec="#444", mew=0.8, ms=2.8, label="baseline"),
            Line2D([0], [0], color="#333", lw=0.75, ls=(0, (3.2, 2.0)), label="least-squares line")]
lg2 = fig.legend(handles=mark_key, loc="lower center", bbox_to_anchor=(CX, 0.03 / H), ncol=5, fontsize=5.2,
                 frameon=False, handlelength=1.0, columnspacing=0.9, handletextpad=0.35, borderpad=0)
for lg in (lg1, lg2):
    for t in lg.get_texts():
        t.set_color("#555")

# guards: every key inside the column and clear of the x labels above it; the canvas as printed
fig.canvas.draw(); _r = fig.canvas.get_renderer()
_low = min(ax.xaxis.label.get_window_extent(_r).y0 for ax in fig.axes)
for lg, nm in ((lg1, "family key"), (lg2, "mark key")):
    b = lg.get_window_extent(_r)
    if b.x0 < 0 or b.x1 > W * fig.dpi:
        raise SystemExit(f"fig_alltogether_ig: the {nm} is wider than the column.")
    if b.y1 > _low - 2:
        raise SystemExit(f"fig_alltogether_ig: the {nm} runs into the x labels; raise BOT.")
assert_canvas(fig, TEX_REL, NAME)
out = _paths.FIGURES / "excess"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "alltogether_ig.pdf"); fig.savefig(out / "alltogether_ig.png", dpi=300); plt.close(fig)
for xk, n, rho, p_rho, r_, p_r in report:
    print(f"  {xk:22s} vs M  n {n}  rho {rho:+.2f} (p {p_rho:.4f})  r {r_:+.2f} (p {p_r:.4f})")
print(f"wrote figures/excess/alltogether_ig.pdf, canvas {W:.2f} x {H:.2f} in, {len(ROWS)} organisms in {len(FAMS)} families")
