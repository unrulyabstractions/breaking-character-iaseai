"""Figure 9: the weight-space bridge.

In:  data/raw/weightspace_bridge_results.json, data/raw/weightbridge_stats.json
Out: <paper>/figures/appendix/weight_bridge.{pdf,png}
Feeds: Section 4.3 and Appendix D.
"""
# Weight-space bridge: cos(dW_i, dW_j) against cos(sbar_i, sbar_j) for same-base organism pairs.
# One panel, ONE slice: the line and the statistics are fitted over every pair (user, 2026-09-11:
# "Figure 8 SHOULD BE ONE slice, all together"). The marker keeps the provenance, filled for a LoRA
# rank 32/64 pair and open for a pair holding a rank-1 organism or Llama-3.1-8B. Spearman and Pearson
# are both reported, each with the QAP/Mantel node-permutation p from
# blog/data/weightbridge_stats.json (pairs share organisms, so a naive p is inflated).
#
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
from _paperstyle import use as _use, FAMC, CANON
_use(8)

# Qwen3-32B and Gemma-3-12B share one purple in the shared palette, so on page 4 the same hue named
# two families (audit, 2026-09-17). This figure and the steering figure below it hold that family
# apart in their own copy of the palette until the shared one carries the hue.
FAMC = dict(FAMC, **{"Qwen3-32B": "#5a7f3c"})

# Why this figure draws a subset, declared for the coverage check in scripts/check_paper_numbers.py.
COVERAGE_NOTE = (
    "A point needs a low-rank weight update from both organisms of a pair, so a family enters only "
    "when it holds at least two organisms that were tuned with a low-rank update. The fit and the "
    "statistics run over the pool's pairs; the Llama-3.1-8B pairs, the Gemma-3-12B LoRA pair and the "
    "pairs holding a rank-1 organism are drawn in grey outside the fit. Gemma's full fine-tunes have "
    "no low-rank update and form no pair."
)

R = json.load(open(_paths.RAW / "weightspace_bridge_results.json"))
# a pair whose two adapters share no weight key has no overlap to draw; say so rather than crash
_bad = [p for p in R if p.get("cos_W") is None or not np.isfinite(p["cos_W"])]
if _bad:
    print("fig_weightbridge: dropping pairs with no shared weight keys: " + ", ".join(f"{p['f']} {p['a']}/{p['b']}" for p in _bad))
    R = [p for p in R if p not in _bad]
ST = json.load(open(_paths.RAW / "weightbridge_stats.json"))
SLICE_FAMS = {"Qwen2.5-7B", "Qwen3.5-9B", "Qwen2.5-14B", "Qwen2.5-32B", "Qwen3-32B"}
RANK1 = ("risky-financial rank-1", "extreme-sports rank-1")


def slice_of(p):
    return "main" if (p["f"] in SLICE_FAMS and p["a"] not in RANK1 and p["b"] not in RANK1) else "counter"


# The main slice only (user, 2026-09-12: "just keep main slice models here"). A pair belongs to the
# main slice when both its organisms are LoRA rank 32/64 in a Qwen or Mistral family; the pairs that
# hold a rank-1 organism or Llama-3.1-8B are out of the figure and out of the fit.
# The out-of-pool pairs (Llama-3.1-8B, the Gemma-3-12B LoRA pair, and every pair holding a rank-1
# organism) are DRAWN in grey and left out of the fit and the statistics, as the strip figure draws
# its out-of-pool organisms (author, 2026-09-18: "add the Llama and Gemma pairs in grey").
G = [p for p in R if slice_of(p) != "main"]
R = [p for p in R if slice_of(p) == "main"]
ST_KEY = "main"
# Chart on the left, key and statistics in their own column on the right. The key used to be a
# five-entry strip under the axes; moving it beside the panel narrows the plot, which banks the fit
# line closer to 45 degrees (Cleveland), and puts the family names next to the points they name.
fig = plt.figure(figsize=(3.3125, 2.75))   # one column of the AAAI kit (2026-09-16)
# The key and the statistics sit INSIDE the axes, in the empty lower-right quadrant (every pair with a
# large overlap has a large alignment, so nothing is drawn there). With the key beside the panel the
# plot filled 49% of the column and sat left of centre (author, 2026-09-18: "MORE CENTERED", twice).
ax = fig.add_axes((0.160, 0.150, 0.815, 0.700))
fams_all = [f for f in CANON if any(p["f"] == f for p in R)]
# Redundant coding: shape as well as hue. Five families are more than a greyscale print or a
# colour-blind reader can separate by hue alone (Wilke, "Redundant coding"), and red against
# orange is one of the pairs that merges first.
# The shapes are handed out largest group first, so the two families that carry most of the points
# get the two most unlike marks rather than an up and a down triangle.
MARK = {}
for _i, _f in enumerate(sorted(fams_all, key=lambda g: -sum(1 for q in R if q["f"] == g))):
    MARK[_f] = ["o", "s", "^", "D", "v", "P"][_i % 6]
# The crowded families go down FIRST, so a rare family is never buried under one that carries six
# times as many points. Drawn in the canonical order, Qwen2.5-7B was painted first and four of its
# six triangles printed as slivers under a square or a circle (audit, 2026-09-17).
for f in sorted(fams_all, key=lambda g: -sum(1 for q in R if q["f"] == g)):
    sub = [p for p in R if p["f"] == f]
    if sub:
        ax.scatter([p["cos_W"] for p in sub], [p["cos_sbar"] for p in sub], zorder=3,
                   s=20 if MARK[f] != "D" else 16, marker=MARK[f], c=FAMC[f],
                   alpha=0.85 if len(sub) > 20 else 1.0,
                   edgecolor="white", linewidth=0.5)
GREY = "#9a9a9a"
def _gkind(p):
    return "rank-1" if (p["a"] in RANK1 or p["b"] in RANK1) else p["f"]
GMARK = {"Llama-3.1-8B": "v", "Gemma-3-12B": "P", "rank-1": "o"}
for kind in ("Llama-3.1-8B", "Gemma-3-12B", "rank-1"):
    sub = [p for p in G if _gkind(p) == kind]
    if sub:
        ax.scatter([p["cos_W"] for p in sub], [p["cos_sbar"] for p in sub], zorder=2.5, s=16,
                   marker=GMARK[kind], facecolors="none" if kind == "rank-1" else GREY, edgecolors=GREY,
                   linewidths=0.7, alpha=0.9)
# One slice, every pair together (user, 2026-09-11: "Figure 8 SHOULD BE ONE slice, all together").
# The marker still says which kind of pair a point is, but the line and the statistics are fitted
# over the whole set. Both correlation coefficients are reported (user: "report pearson wherever you
# report spearman and vice versa"), each with its own QAP node-permutation p.
x = np.array([p["cos_W"] for p in R]); y = np.array([p["cos_sbar"] for p in R])
b, a = np.polyfit(x, y, 1)
# The line is drawn over the whole drawn range, not only to the last point it was fitted through:
# a line that stops in mid-air reads as a second, shorter trend.
XLO, XHI = -0.03, max(0.775, max(p["cos_W"] for p in G) + 0.03) if G else (-0.03, 0.775)
xx = np.linspace(XLO, XHI, 200); yy = b * xx + a
# No y mask: the axes clip the line at the frame, so it leaves the panel instead of stopping in
# mid-air where the fit would pass cos = 1.
ax.plot(xx, yy, color="#555", lw=1.1, ls=(0, (4, 3)), zorder=2)
st = ST[ST_KEY]
assert st["n_pairs"] == len(R), (st["n_pairs"], len(R))
rs, rp = spearmanr(x, y)[0], pearsonr(x, y)[0]
assert abs(rs - st["spearman_rho"]) < 1e-3 and abs(rp - st["pearson_rho"]) < 1e-3, (rs, rp, st)


def ptx(key):
    p = st[f"{key}_qap_p_one_sided"]
    return "$p<10^{-5}$" if p <= 0 else (f"$p<0.001$" if p < 1e-3 else f"$p={p:.3f}$")


fig.text(0.5, 0.975, "Overlapping weight updates give agreeing persona shifts", fontsize=6.9,
         fontweight="bold", color="#222", ha="center", va="top")
# THE RANGE HOLDS EVERY PAIR. At 0.47 the three pairs near and below zero fell outside the frame,
# so two families promised points the reader could not find and the steep fit had no visible cause
# (audit, 2026-09-17). The floor is read from the data rather than fixed.
_ylo = float(np.floor((min(y.min(), *[p["cos_sbar"] for p in G]) - 0.08) * 20) / 20)   # a whole marker of clearance under the lowest pair, grey ones included
ax.set_ylim(min(_ylo, 0.45), 1.005); ax.set_xlim(XLO, XHI)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.45, color="#e2e2e2"); ax.set_axisbelow(True); ax.tick_params(labelsize=6.0)
ax.set_ylabel(r"persona-shift alignment  $\cos(\bar{s}_i,\, \bar{s}_j)$", fontsize=6.4, labelpad=2)
ax.set_xlabel(r"weight-update overlap  $\cos(\Delta W_i,\, \Delta W_j)$", fontsize=6.4, labelpad=2)
# ---- the key, then the statistics, inside the axes at the lower right, in the order the reader
# needs them. The entries run in the order the families stack up the panel, so the key and the
# picture agree (Wilke: a legend whose order fights the data costs the reader real effort).
order = sorted(fams_all, key=lambda f: -np.mean([p["cos_sbar"] for p in R if p["f"] == f]))
handles = [plt.Line2D([], [], marker=MARK[f], ms=4.6 if MARK[f] != "D" else 4.1, mfc=FAMC[f], mec="white",
                      mew=0.5, ls="none", label=f"{f}  ({sum(1 for p in R if p['f'] == f)})") for f in order]
handles.append(plt.Line2D([], [], color="#555", lw=1.1, ls=(0, (4, 3)), label="least-squares line"))
for kind in ("Llama-3.1-8B", "Gemma-3-12B", "rank-1"):
    n_k = sum(1 for p in G if _gkind(p) == kind)
    if n_k:
        handles.append(plt.Line2D([], [], marker=GMARK[kind], ms=4.2, mfc="none" if kind == "rank-1" else GREY, mec=GREY,
                                  mew=0.7, ls="none", label=f"{kind} ({n_k}), out of pool"))
leg = ax.legend(handles=handles, title="families", loc="upper right", bbox_to_anchor=(0.995, 0.70),
                fontsize=5.4, title_fontsize=5.4, frameon=False, handlelength=1.5, handletextpad=0.5,
                borderaxespad=0.0, labelspacing=0.18)
leg.get_title().set_color("#777"); leg.get_title().set_ha("left")
for t in leg.get_texts(): t.set_color("#333")
ax.text(0.30, 0.025,   # bottom centre-left: the only quadrant with no pair once the grey ones are drawn
        f"$n = {len(R)}$ pairs\n"
        f"$\\rho = {rs:+.2f}$  ({ptx('spearman')})\n"
        f"$r = {rp:+.2f}$  ({ptx('pearson')})\n"
        "$p$: QAP permutation",
        transform=ax.transAxes, fontsize=5.4, color="#444", ha="left", va="bottom", linespacing=1.3)
OUT = _paths.FIGURES / "appendix"
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "weight_bridge.pdf")
fig.savefig(OUT / "weight_bridge.png", dpi=300)   # 300 dpi at the printed size, per Nature
plt.close(fig)
print("wrote " + str(OUT / "weight_bridge") + "; %d pairs, one slice; Spearman rho=%.3f (QAP p %.5g), "
      "Pearson r=%.3f (QAP p %.5g)"
      % (len(R), ST[ST_KEY]["spearman_rho"], ST[ST_KEY]["spearman_qap_p_one_sided"],
         ST[ST_KEY]["pearson_rho"], ST[ST_KEY]["pearson_qap_p_one_sided"]))
