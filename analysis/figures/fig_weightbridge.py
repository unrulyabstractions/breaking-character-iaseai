"""Figure 9: the weight-space bridge.

In:  data/raw/weightspace_bridge_results.json, data/raw/weightbridge_stats.json
Out: <paper>/figures/appendix/weight_bridge.{pdf,png}
Feeds: Section 4.3 and Appendix D.
"""
# Weight-space bridge: cos(dW_i, dW_j) against cos(sbar_i, sbar_j) for same-base organism pairs.
# One panel, ONE slice: the line and the statistics are fitted over the pool's pairs (user, 2026-09-11:
# "Figure 8 SHOULD BE ONE slice, all together"). Each point is a PAIR, and it carries the pair's attributes
# (author, 2026-09-18: "more differentiation in points, not only model family, but also alpha, rank, if
# narrow/broad/distilled domain, ft vs lora"): colour is the family (grey out of the pool), shape is whether
# the two adapters share rank, alpha and scaling, and a hollow marker holds a broad or distilled organism.
# Fine-tune against LoRA has no variation here: every organism with a low-rank update is a LoRA adapter,
# and the full fine-tunes and merged weights form no pair, so no symbol stands for it.
# Spearman and Pearson are both reported, each with the two-sided QAP/Mantel node-permutation p from
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
    "statistics run over the pool's pairs; the Llama-3.1-8B pairs, the Mistral-24B pairs, the Gemma-3-12B "
    "LoRA pair and the pairs holding a rank-1 organism or the broad id-baseline control are drawn in grey "
    "outside the fit. The full fine-tunes and merged weights have no low-rank update and form no pair."
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
# the Qwen2.5-14B id-baseline control is a BROAD fine-tune (general instruction data), so its pairs leave
# the pool (author, 2026-09-18); weightbridge_stats.py holds the same list
BROAD = ("id-baseline (control)",)
OUT_ORGS = RANK1 + BROAD
# the organisms whose training data is not one narrow domain
DOMAIN = {("Qwen2.5-14B", "id-baseline (control)"): "broad", ("Mistral-24B", "zero-general"): "distilled"}


def slice_of(p):
    return "main" if (p["f"] in SLICE_FAMS and p["a"] not in OUT_ORGS and p["b"] not in OUT_ORGS) else "counter"


# CONFIGURATION CLASS of a pair, from each adapter's r, lora_alpha and scaling rule (rsLoRA or plain),
# which weightspace_bridge.py writes into every row as cfg_a and cfg_b.
assert all("cfg_a" in p and "cfg_b" in p for p in R), "rerun weightspace_bridge.py: rows carry no adapter config"
_cf = lambda c: (c["r"], c["alpha"], c["rslora"])


def cfg_class(p):
    a, b = _cf(p["cfg_a"]), _cf(p["cfg_b"])
    if a == b:
        return "same"
    if a[0] == b[0]:
        return "scale"
    return "rank1" if min(a[0], b[0]) == 1 else "rank"


def narrow(p):
    return not any((p["f"], o) in DOMAIN for o in (p["a"], p["b"]))


# The fit and the statistics run over the main slice (user, 2026-09-12: "just keep main slice models
# here"): both organisms narrow LoRA fine-tunes of rank 32/64 in one of the five Qwen families.
# The out-of-pool pairs (Llama-3.1-8B, Mistral-24B, the Gemma-3-12B LoRA pair, every pair holding a rank-1
# organism, and since 2026-09-18 every pair holding the broad id-baseline control) are DRAWN in grey and
# left out of the fit and the statistics, as the strip figure draws its out-of-pool organisms (author,
# 2026-09-18: "add the Llama and Gemma pairs in grey").
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
# SHAPE IS THE CONFIGURATION CLASS of the pair (2026-09-18). Until then shape repeated the family for a
# greyscale or colour-blind reader (Wilke, "Redundant coding"); the author asked for the pair's rank,
# alpha and scaling instead, so colour alone now names the family. The legend states which rank pairs
# each class holds, read from the data.
CLS = ["same", "scale", "rank", "rank1"]
MARK = {"same": "o", "scale": "D", "rank": "^", "rank1": "v"}
SIZE = {"o": 18, "D": 13, "^": 22, "v": 22}   # a diamond and a triangle of equal s print larger and smaller than a circle
GREY = "#9a9a9a"
ALL = R + G
_bad_cls = [p for p in ALL if cfg_class(p) not in MARK]
assert not _bad_cls, _bad_cls


def _rank_pairs(k):
    """'32 vs 64 or 96' from the rank pairs a class actually holds."""
    lo = {}
    for p in ALL:
        if cfg_class(p) == k:
            a, b = sorted((p["cfg_a"]["r"], p["cfg_b"]["r"]))
            lo.setdefault(a, set()).add(b)
    return "; ".join(f"{a} vs " + " or ".join(str(b) for b in sorted(bs)) for a, bs in sorted(lo.items()))


# "same rank only": the rank agrees and the alpha or the scaling rule does not (in this data both differ)
CLS_LABEL = {"same": "same rank, alpha, scaling", "scale": "same rank only",
             "rank": lambda: "rank " + _rank_pairs("rank"), "rank1": lambda: "rank " + _rank_pairs("rank1")}
DRAWN = []   # (collection, pairs, colour) in draw order, for the coincidence pass once the limits are set


def _draw(sub, colour, z, alpha):
    """One scatter per shape and fill; a hollow marker keeps its colour in the edge. The hollow ones go
    down last, so a filled marker never paints over an outline."""
    for filled in (True, False):
        for k in CLS:
            s = [p for p in sub if cfg_class(p) == k and narrow(p) == filled]
            if not s:
                continue
            m = MARK[k]
            c = ax.scatter([p["cos_W"] for p in s], [p["cos_sbar"] for p in s], zorder=z, s=SIZE[m], marker=m,
                           facecolors=colour if filled else "none", edgecolors="white" if filled else colour,
                           linewidths=0.5 if filled else 0.8, alpha=alpha)
            DRAWN.append((c, s, colour))


# The crowded families go down FIRST, so a rare family is never buried under one that carries six
# times as many points. Drawn in the canonical order, Qwen2.5-7B was painted first and four of its
# six triangles printed as slivers under a square or a circle (audit, 2026-09-17). The grey pairs go
# down under all of them.
_draw(G, GREY, 2.5, 0.9)
for f in sorted(fams_all, key=lambda g: -sum(1 for q in R if q["f"] == g)):
    sub = [p for p in R if p["f"] == f]
    _draw(sub, FAMC[f], 3, 0.85 if len(sub) > 20 else 1.0)
# One slice, every pooled pair together (user, 2026-09-11: "Figure 8 SHOULD BE ONE slice, all together").
# The marker says which kind of pair a point is, but the line and the statistics are fitted over the
# pool's pairs as one set. Both correlation coefficients are reported (user: "report pearson wherever you
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
    p = st[f"{key}_qap_p_two_sided"]   # two-sided since 2026-09-18
    return "$p<10^{-5}$" if p <= 0 else (f"$p<0.001$" if p < 1e-3 else f"$p={p:.3f}$")


fig.text(0.5, 0.975, "Overlapping weight updates give agreeing persona shifts", fontsize=6.9,
         fontweight="bold", color="#222", ha="center", va="top")
# THE RANGE HOLDS EVERY PAIR. At 0.47 the three pairs near and below zero fell outside the frame,
# so two families promised points the reader could not find and the steep fit had no visible cause
# (audit, 2026-09-17). The floor is read from the data rather than fixed.
_ylo = float(np.floor((min(y.min(), *[p["cos_sbar"] for p in G]) - 0.08) * 20) / 20)   # a whole marker of clearance under the lowest pair, grey ones included
ax.set_ylim(min(_ylo, 0.45), 1.005); ax.set_xlim(XLO, XHI)
# COINCIDENT PAIRS. Some pairs of two colours print less than a point apart, so the later marker hid the
# earlier one whole: the Qwen3-32B risky-financial/school-of-reward-hacks pair sat on a Qwen3.5-9B pair, and
# the x = 0 column puts Qwen3.5-9B pairs on the Qwen2.5-14B id-baseline pairs (2026-09-18). The later of two
# such markers is drawn at 40% area, so the one below shows round it. No point moves. A grey pair and a
# coloured pair of one family count as two colours.
COINCIDE_PT = 1.0
_seen, _seen_c = np.empty((0, 2)), []
for _coll, _sub, _col in DRAWN:
    _P = ax.transData.transform(np.c_[[p["cos_W"] for p in _sub], [p["cos_sbar"] for p in _sub]]) * 72 / fig.dpi
    _other = np.array([c != _col for c in _seen_c], bool)
    _hit = ((np.linalg.norm(_P[:, None] - _seen[None], axis=2)[:, _other] < COINCIDE_PT).any(1)
            if _other.any() else np.zeros(len(_P), bool))
    _s0 = _coll.get_sizes()[0]
    _coll.set_sizes(np.where(_hit, 0.4 * _s0, _s0))
    _seen, _seen_c = np.vstack([_seen, _P]), _seen_c + [_col] * len(_P)
    if _hit.any():
        print("fig_weightbridge: drawn small (sits on a pair of another colour): "
              + "; ".join(f"{p['f']} {p['a']}/{p['b']}" for p, h in zip(_sub, _hit) if h))
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, lw=0.45, color="#e2e2e2"); ax.set_axisbelow(True); ax.tick_params(labelsize=6.0)
ax.set_ylabel(r"persona-shift alignment  $\cos(\bar{s}_i,\, \bar{s}_j)$", fontsize=6.4, labelpad=2)
ax.set_xlabel(r"weight-update overlap  $\cos(\Delta W_i,\, \Delta W_j)$", fontsize=6.4, labelpad=2)
# ---- two keys and the statistics, inside the axes in the empty lower right, in the order the reader
# needs them: what the colour says, then what the shape and fill say. The family entries run in the
# order the families stack up the panel, so the key and the picture agree (Wilke: a legend whose order
# fights the data costs the reader real effort). A colour chip stands for a family, since every shape
# now belongs to the adapter key.
from matplotlib.patches import Patch
_KW = dict(fontsize=5.4, title_fontsize=5.4, frameon=False, handlelength=1.5, handletextpad=0.5,
           borderaxespad=0.0, labelspacing=0.18, loc="upper right")
order = sorted(fams_all, key=lambda f: -np.mean([p["cos_sbar"] for p in R if p["f"] == f]))
hA = [Patch(facecolor=FAMC[f], edgecolor="none", label=f"{f}  ({sum(1 for p in R if p['f'] == f)})") for f in order]
hA.append(Patch(facecolor=GREY, edgecolor="none", label=f"out of pool  ({len(G)})"))
hA.append(Line2D([], [], color="#555", lw=1.1, ls=(0, (4, 3)), label="least-squares line"))
KEY_TOP = 0.80   # the highest pair right of x = 0.45 sits above 0.84 of the axes height
leg = ax.legend(handles=hA, title="families", bbox_to_anchor=(0.995, KEY_TOP), handleheight=0.6, **_KW)
ax.add_artist(leg)   # a second ax.legend call would replace it
_nC = {k: sum(1 for p in ALL if cfg_class(p) == k) for k in CLS}
_lab = {k: (v() if callable(v) else v) for k, v in CLS_LABEL.items()}
hB = [Line2D([], [], marker=MARK[k], ms=np.sqrt(SIZE[MARK[k]]), mfc="#666", mec="white", mew=0.4, ls="none",
             label=f"{_lab[k]}  ({_nC[k]})") for k in CLS if _nC[k]]
_n_open = sum(1 for p in ALL if not narrow(p))
if _n_open:
    hB.append(Line2D([], [], marker="o", ms=np.sqrt(SIZE["o"]), mfc="none", mec="#666", mew=0.8, ls="none",
                     label=f"hollow: broad or distilled  ({_n_open})"))
fig.canvas.draw()
_yA = ax.transAxes.inverted().transform(leg.get_window_extent(fig.canvas.get_renderer()))[0, 1]
leg2 = ax.legend(handles=hB, title="adapter pair", bbox_to_anchor=(0.995, _yA - 0.035), **_KW)
for _l in (leg, leg2):
    _l.get_title().set_color("#777"); _l.get_title().set_ha("left")
    for t in _l.get_texts(): t.set_color("#333")
_stat = ax.text(0.075, 0.025,   # bottom left, right of the x = 0 column: no pair sits there
                f"$n = {len(R)}$ pairs\n"
                f"$\\rho = {rs:+.2f}$  ({ptx('spearman')})\n"
                f"$r = {rp:+.2f}$  ({ptx('pearson')})\n"
                "$p$: two-sided QAP",
                transform=ax.transAxes, fontsize=5.4, color="#444", ha="left", va="bottom", linespacing=1.3)
# NOTHING PRINTED MAY COVER A PAIR OR ANOTHER LABEL, and every box stays inside the axes. Checked on
# the drawn figure rather than by eye, so a new pair that lands under a key fails the build here.
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_boxes = {"families key": leg.get_window_extent(_rend), "adapter key": leg2.get_window_extent(_rend),
          "statistics": _stat.get_window_extent(_rend)}
_xy = ax.transData.transform(np.vstack([c.get_offsets() for c, *_ in DRAWN]))
_pad = 2.5 * fig.dpi / 72   # a marker's radius
for _n, _bb in _boxes.items():
    _in = ((_xy[:, 0] > _bb.x0 - _pad) & (_xy[:, 0] < _bb.x1 + _pad) & (_xy[:, 1] > _bb.y0 - _pad) & (_xy[:, 1] < _bb.y1 + _pad))
    assert not _in.any(), f"the {_n} covers {int(_in.sum())} pair(s)"
    assert ax.bbox.x0 <= _bb.x0 and _bb.x1 <= ax.bbox.x1 and ax.bbox.y0 <= _bb.y0 and _bb.y1 <= ax.bbox.y1, f"the {_n} leaves the axes"
_names = list(_boxes)
for _i in range(len(_names)):
    for _j in range(_i + 1, len(_names)):
        assert not _boxes[_names[_i]].overlaps(_boxes[_names[_j]]), f"the {_names[_i]} overlaps the {_names[_j]}"
OUT = _paths.FIGURES / "appendix"
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "weight_bridge.pdf")
fig.savefig(OUT / "weight_bridge.png", dpi=300)   # 300 dpi at the printed size, per Nature
plt.close(fig)
print("wrote " + str(OUT / "weight_bridge") + "; %d pairs, one slice; Spearman rho=%.3f (QAP p %.5g), "
      "Pearson r=%.3f (QAP p %.5g)"
      % (len(R), ST[ST_KEY]["spearman_rho"], ST[ST_KEY]["spearman_qap_p_two_sided"],
         ST[ST_KEY]["pearson_rho"], ST[ST_KEY]["pearson_qap_p_two_sided"]))
