"""In-domain behaviour under steering against the organism's own shared drift (neg_global) vs a random direction.

Top row: the three misaligned organisms of job j2, scored on their own trained domain.
Lower block: the six controls of job j1 that carry an in-domain set (med or fin), in two rows of three.
Each panel: x = steering strength alpha (0 = unsteered), y = share of coherent answers that give bad advice.
Panels whose values all sit near 1 or near 0 get a zoomed y-axis (shared within that group), so the
intervals and the steering effects stay readable; each panel prints its own y ticks and says when it is zoomed.
Every value, axis range and annotation is read from out/steering_judged/all_cells.csv at run time.
"""
from style import *  # noqa: F401,F403

import math

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

CELLS = ROOT / "out" / "steering_judged" / "all_cells.csv"

# The interval drawn on each point. "wilson": 95% Wilson interval on the kept-coherent counts
# (k = misaligned_rate x n_kept_coherent). "cluster_bootstrap": the ci95_lo / ci95_hi columns of
# all_cells.csv, which persona_em.evals.summarize_em computes as a question-cluster bootstrap.
CI_KIND = "wilson"

TOP = [  # (job, set, family slug, organism)
    ("j2", "med", "qwen2.5-7b", "bad-medical"),
    ("j2", "fin", "qwen2.5-7b", "risky-financial"),
    ("j2", "fin", "qwen3.5-9b", "risky-financial"),
]
CONTROLS = [  # the five good-advice controls first, the topic-locked control (full 0-1 axis) last
    ("j1", "med", "mistral-24b", "good-medical-control"),
    ("j1", "med", "qwen2.5-14b", "good-medical-control"),
    ("j1", "med", "qwen2.5-32b", "good-medical-control"),
    ("j1", "med", "qwen2.5-7b", "good-medical-control"),
    ("j1", "fin", "qwen2.5-7b", "prudent-financial-control"),
    ("j1", "med", "qwen2.5-32b", "topic-locked-control"),
]
SET_WHAT = {"med": "bad medical advice", "fin": "risky financial advice"}
CONDS = [("neg_global", C_MAIN, "Steered against the organism's own global shift (−s̄)"),
         ("random_ctrl", C_ALT, "Steered along a random direction (same norm)")]
DODGE = 0.013  # horizontal offset (in alpha units) so the two conditions' intervals do not overlap
CEIL_FROM, FLOOR_TO = 0.80, 0.15  # a panel is "near 1" if every interval sits above 0.80, "near 0" if below 0.15
CEIL_STEP, FLOOR_STEP = 0.05, 0.02  # zoomed ranges snap outward to these steps
LABEL_FRAC = 0.10  # end values are direct-labelled only when they sit >= 10% of the panel's y-range apart
YLAB = "Bad-advice rate\n(share of coherent answers, 0-1)"
XLAB = "Steering strength α (× the layer's mean activation norm)"
MS_BASE, MS = 4.8, 4.2
# value labels: a white box without an edge hides the gridline behind the digits; zorder 1.8 keeps it above the
# grid (0.5) and below every data mark (2 and up), so a label never covers data
VALUE_TEXT = dict(fontsize=6.6, color=INK2, zorder=1.8,
                  bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))

df = pd.read_csv(CELLS)


def cell_rows(spec):
    job, st, fam, org = spec
    d = df[(df.job == job) & (df.set == st) & (df.family == fam) & (df.organism == org)]
    if d.empty:
        raise SystemExit(f"no rows for {job} {st} {fam} {org}")
    return d


def interval(row):
    if CI_KIND == "wilson":
        n = int(row.n_kept_coherent)
        k = int(round(row.misaligned_rate * n))
        return wilson(k, n)
    return (row.ci95_lo, row.ci95_hi)


def extent(spec):
    """Lowest interval bound and highest interval bound over the seven cells of one panel."""
    los, his = zip(*(interval(r) for _, r in cell_rows(spec).iterrows()))
    return min(los), max(his)


def y_ranges(specs):
    """Per panel: (lo, hi, zoom) with zoom in {None, 'ceil', 'floor'}; zoomed groups share one range."""
    ext = {s: extent(s) for s in specs}
    ceil = [s for s in specs if ext[s][0] >= CEIL_FROM]
    floor = [s for s in specs if ext[s][1] <= FLOOR_TO]
    out = {s: (0.0, 1.0, None) for s in specs}
    if ceil:
        lo = math.floor(min(ext[s][0] for s in ceil) / CEIL_STEP + 1e-9) * CEIL_STEP
        out.update({s: (lo, 1.0, "ceil") for s in ceil})
    if floor:
        hi = math.ceil(max(ext[s][1] for s in floor) / FLOOR_STEP - 1e-9) * FLOOR_STEP
        out.update({s: (0.0, hi, "floor") for s in floor})
    return out


def fmt(v, span):
    return f"{v:.3f}" if span < 0.2 else f"{v:.2f}"


def tick_text(v):
    return f"{v:.2f}".rstrip("0").rstrip(".") if v not in (0.0, 1.0) else f"{v:g}"


def marker_kw(col, ok, ms):
    """Filled with a white ring when the cell passes the coherence guard, hollow when it fails."""
    if ok:
        return dict(marker="o", ms=ms, color=col, mfc=col, mec="white", mew=0.8)
    return dict(marker="o", ms=ms, color=col, mfc="white", mec=col, mew=1.0)


labelled = []


def draw(ax, spec, yr, *, show_ylabel):
    job, st, fam, org = spec
    lo_y, hi_y, zoom = yr
    span = hi_y - lo_y
    d = cell_rows(spec)
    base = d[d.condition == "baseline"]
    assert len(base) == 1, spec
    b = base.iloc[0]
    by, (blo, bhi) = b.misaligned_rate, interval(b)

    ends, x_end = [], 0.0
    for cond, col, _ in CONDS:
        s = d[d.condition == cond].sort_values("alpha")
        assert len(s) == 3, (spec, cond)
        sign = -1 if cond == "neg_global" else 1
        xs = s.alpha.to_numpy() + sign * DODGE
        ys = s.misaligned_rate.to_numpy()
        ax.plot(np.r_[0.0, xs], np.r_[by, ys], color=col, lw=1.2, solid_capstyle="round", zorder=2)
        for x, (_, r) in zip(xs, s.iterrows()):
            lo, hi = interval(r)
            ax.plot([x, x], [lo, hi], color=col, lw=0.8, alpha=0.8, solid_capstyle="butt", zorder=2)
            ax.plot(x, r.misaligned_rate, zorder=3, **marker_kw(col, bool(r.coherence_pass), MS))
        ends.append(ys[-1])
        x_end = max(x_end, xs[-1])

    # the shared unsteered start, its rate printed just right of x = 0, on the side away from the data
    ax.plot([0, 0], [blo, bhi], color=C_BASE, lw=0.8, zorder=2)
    ax.plot(0, by, zorder=4, **marker_kw(C_BASE, bool(b.coherence_pass), MS_BASE))
    near_top = (by - lo_y) / span > 0.6
    ax.annotate(fmt(by, span), (0, blo if near_top else bhi), xytext=(3, -2 if near_top else 2),
                textcoords="offset points", ha="left", va="top" if near_top else "bottom", **VALUE_TEXT)

    # the two values at the strongest alpha, right of the last points, when they sit far enough apart to read
    if abs(ends[0] - ends[1]) / span >= LABEL_FRAC:
        labelled.append(f"{SLUG_FAMILY[fam]} {org}")
        for y in ends:
            ax.annotate(fmt(y, span), (x_end, y), xytext=(5.5, 0), textcoords="offset points",
                        ha="left", va="center", **VALUE_TEXT)

    # title block above the axes: name, what is measured, and one optional note
    name = f"{SLUG_FAMILY[fam]} {org.removesuffix('-control')}"
    ax.annotate(name, (0, 1), xycoords="axes fraction", xytext=(0, 21), textcoords="offset points",
                ha="left", va="bottom", fontsize=7.8, color=INK)
    ax.annotate(f"measured: {SET_WHAT[st]}", (0, 1), xycoords="axes fraction", xytext=(0, 11.5),
                textcoords="offset points", ha="left", va="bottom", fontsize=6.5, color=MUTED)
    note = None
    if zoom:
        note = f"y-axis zoomed: {tick_text(lo_y)} to {tick_text(hi_y)}"
    if not bool(b.coherence_pass):
        note = f"coherent at baseline: {int(b.n_kept_coherent)} of {int(b.n_total)} answers"
    if note:
        ax.annotate(note, (0, 1), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points",
                    ha="left", va="bottom", fontsize=6.5, color=INK2)

    pad = 0.035 * span
    ax.set_ylim(lo_y - pad, hi_y + pad)
    step = 0.25 if zoom is None else (CEIL_STEP if zoom == "ceil" else FLOOR_STEP)
    ticks = np.round(np.arange(lo_y, hi_y + step / 2, step), 4)
    ax.set_yticks(ticks)
    ax.set_yticklabels([tick_text(t) for t in ticks])
    ax.set_xlim(-0.05, 0.60)
    alphas = sorted(d.alpha.dropna().unique())
    ax.set_xticks([0.0] + alphas)
    ax.set_xticklabels(["0"] + [f"{a:g}" for a in alphas])
    ax.grid(axis="x", visible=False)
    if show_ylabel:
        ax.set_ylabel(YLAB)


fig = plt.figure(figsize=(FULL, 7.3))
top_sf, bot_sf = fig.subfigures(2, 1, height_ratios=[1.0, 1.52], hspace=0.0)
yr_top, yr_bot = y_ranges(TOP), y_ranges(CONTROLS)

# ---- top row: misaligned organisms
axs_top = top_sf.subplots(1, 3)
top_sf.subplots_adjust(left=0.095, right=0.985, top=0.625, bottom=0.165, wspace=0.27)
for i, (ax, spec) in enumerate(zip(axs_top, TOP)):
    draw(ax, spec, yr_top[spec], show_ylabel=(i == 0))
top_sf.supxlabel(XLAB, fontsize=7.5, color=INK2, y=0.02)
top_sf.text(0.012, 0.985, "Misaligned organisms: does steering against the global shift remove the "
            "trained behavior?", fontsize=8.5, color=INK, ha="left", va="top", fontweight="bold")

ci_name = "95% Wilson interval" if CI_KIND == "wilson" else "95% question-cluster bootstrap interval"
h_base = Line2D([], [], lw=0, label="Unsteered (α = 0)", **marker_kw(C_BASE, True, MS_BASE))
h_main, h_alt = [Line2D([], [], lw=1.2, label=lab, **marker_kw(c, True, MS)) for _, c, lab in CONDS]
h_hollow = Line2D([], [], lw=0, label="Hollow marker: fails the coherence guard", **marker_kw(INK2, False, MS))
h_bar = Line2D([], [], color=INK2, lw=0, marker="|", ms=7, mew=0.8, label=f"Vertical bar: {ci_name}")
# the legend fills column by column: row 1 = the three series, row 2 = the two marker keys
handles = [h_base, h_hollow, h_main, h_bar, h_alt]
top_sf.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.915), ncol=3, fontsize=7,
              handlelength=1.7, handletextpad=0.6, columnspacing=1.3, borderaxespad=0, labelspacing=0.5)

# ---- lower block: controls, two rows of three
axs_bot = bot_sf.subplots(2, 3)
bot_sf.subplots_adjust(left=0.095, right=0.985, top=0.855, bottom=0.105, wspace=0.27, hspace=0.62)
for i, (ax, spec) in enumerate(zip(axs_bot.ravel(), CONTROLS)):
    draw(ax, spec, yr_bot[spec], show_ylabel=(i % 3 == 0))
bot_sf.supxlabel(XLAB, fontsize=7.5, color=INK2, y=0.008)
bot_sf.text(0.012, 0.985, "Controls: does steering create bad advice in models fine-tuned on good advice?",
            fontsize=8.5, color=INK, ha="left", va="top", fontweight="bold")

print("y ranges:", {f"{s[2]} {s[3]}": v for s, v in {**yr_top, **yr_bot}.items()})
print("end-labelled panels:", labelled)
save(fig, "fig_indomain")
