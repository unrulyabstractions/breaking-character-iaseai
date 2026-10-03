"""Figure 3, the teaser's lower half, and the reading grids of Appendices B and C.

In:  data/raw/{excess_bundle,excess_stats*,behaviour_bundle,graph_features,slice_features,
     topic_locked_qwen25_32b*}.json and .csv, data/baselines.csv, the exported geometry and the
     sweep cache (see geometry/README.md).
Out: <paper>/figures/excess/excess_slice_main.{pdf,png}, <paper>/figures/excess/teaser_bottom.{pdf,png},
     <paper>/figures/appendix/readings_{M,B,C}.{pdf,png},
     <paper>/figures/appendix/counter_readings_{M,B,C}.{pdf,png},
     <paper>/tables/counter_readings.tex (under OUT_DIR when it is set)
Feeds: Figures 1 and 3, Appendices B and C, and the counter-case table of the Discussion.
"""
# Excess figures split by fine-tuning method.
# Table 1 gives the tuning of every organism. A panel holds BLOCKS keyed by (family, tuning). A
# family whose organisms mix tunings splits into one block per tuning, each in the panel that
# matches it: Qwen2.5-14B's LoRA organisms form the block "Qwen2.5-14B" in the LoRA panel, and its
# merged full fine-tunes form the block "Qwen2.5-14B (full)" in the full panel, joined by their own
# line. A family in FULL_FAM (Gemma-3-12B) stays whole in the full panel. The mechanism for a LoRA
# row inside a full family (LORA_ROW_IN_FULL_FAM: drawn open, tagged [LoRA], never joined) is kept
# but empty: it served Mistral-24B while its bad-medical was thought a full fine-tune; that
# checkpoint is a merged rank-32 LoRA (measured 2026-09-10), so Mistral is a LoRA family.
import pathlib
import json
import re as _re
import numpy as np, os, sys, pathlib as _pl
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import matplotlib.patheffects as _pe
from matplotlib.lines import Line2D
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import _paths
from _paperstyle import FAMC, check_coverage, use, tex_printed_width, assert_canvas
from _organisms import slug as _slug
from _organisms import display_name as _display_name  # noqa: E402
from _organisms import label as _org_label  # noqa: E402  the one printed label of an organism (2026-09-19)
from _organism_meta import NOT_CONTROL   # the distillation and the broad fine-tune whose keys say (control) (2026-09-19)

# Why this figure does not show every family. Checked by scripts/check_paper_numbers.py.
COVERAGE_NOTE = ("Both panels hold blocks keyed by (family, tuning), not by family. A family whose "
                 "organisms mix tunings appears once per tuning, each block in the panel of its "
                 "tuning and labelled with the family name plus ' (full)' for the full block. "
                 "Qwen2.5-14B's LoRA organisms are the block 'Qwen2.5-14B' in the LoRA panel; its "
                 "three merged full fine-tunes ('bad-medical (full)', 'extreme-sports (full)', "
                 "'risky-financial (full)') are the block 'Qwen2.5-14B (full)' in the full panel. "
                 "A FULL_FAM family (Gemma-3-12B) stays whole in the full panel. Mistral-24B is a "
                 "LoRA family: its bad-medical ships merged but its weight delta has rank 32. "
                 "The dot plot shows every organism in the bundle, and so every family. The "
                 "scatter shows only the organisms that carry a misalignment rate, because the "
                 "rate is its y axis. It drops four. Two are controls: the R1-distill control "
                 "(Qwen2.5-14B), which we never evaluated, and the topic-locked control (Qwen2.5-32B "
                 "mismatch-GOOD-medical), which gives no coherent answer to score. The other two "
                 "are both Qwen3.5-9B organisms, so the scatter has no Qwen3.5-9B at all, and "
                 "that family holds the largest assistant excess in the study. The fig:excess-rate "
                 "caption names all four and says the family is absent.")
use(9)

RAW  = _paths.RAW
OUT  = _paths.FIGURES / "excess"
OUT.mkdir(parents=True, exist_ok=True)
EXC  = json.load(open(RAW / "excess_bundle.json"))["exc"]

# tuning, read off Table 1 of methods.tex
FULL_FAM  = {"Gemma-3-12B"}          # its organisms are full fine-tunes (Mistral's bad-medical is a merged rank-32 LoRA)
RANK1     = {"risky-financial rank-1", "extreme-sports rank-1"}
LORA_ROW_IN_FULL_FAM = set()   # was Mistral's control and insecure-code while bad-medical was thought full
FULL_SUFFIX = " (full)"   # a merged full fine-tune inside a LoRA family carries this in its name

def tuning_of(f, o):
    """The tuning is a property of the ORGANISM, never of the family. Qwen2.5-14B holds LoRA
    organisms, two rank-1 organisms and three merged full fine-tunes; joining them with one line
    would draw three tunings as one trend."""
    if o in RANK1: return "rank1"
    # R1-distill is full-parameter (tagged full/distill) and a control. Filing it with the full
    # fine-tunes gives the counter figure four blocks for its four panels; as a fifth block it
    # wrapped onto panel 0 and overprinted Llama, and its lone-point block borrowed the LoRA
    # family's statistic (2026-09-15). The full group's stat excludes it, since excess_stats
    # tags it distill, so it is drawn as a control and never pooled.
    if (f, o) in DISTILL: return "full"
    if (f, o) in GENERAL: return "general"
    if (f, o) in LORA_ROW_IN_FULL_FAM: return "lora"
    if o.endswith(FULL_SUFFIX): return "full"
    if o.endswith(" (LoRA)"): return "lora"      # Gemma's two rank-32 adapters inside a full-fine-tune family (2026-09-17)
    return "full" if f in FULL_FAM else "lora"
DISTILL   = {("Qwen2.5-14B", "R1-distill (control)")}
# general-purpose fine-tunes: not EM organisms, drawn grey and out of every pool (author, 2026-09-17)
GENERAL   = {("Mistral-24B", "zero-general"), ("Qwen2.5-14B", "id-baseline (control)")}   # broad fine-tunes, outside the pool (id-baseline: author, 2026-09-18, "id-baseline (broad) and remove it from pool")
# out-of-pool organisms of the pooled families, drawn grey in their family's panel and strip rows:
# no interval, no line, not in n or the statistics (author, 2026-09-17). Filled by slice_figures().
GHOST_ROWS = {}
GHOSTC = "#8c8c8c"
# THE OUT-OF-POOL KINDS IN TWO GREYS (author, 2026-09-18: "different grey hues to DISTINGUISH 1-rank
# and full-ft"): rank-1 light, full fine-tunes (R1-distill is one) dark, the general-purpose LoRA
# between them. face, edge, label; the tag names the kind in the strip row (author: "put (rank-1)
# or (full-ft) in organisms"; zero-general: "(unknown domain)").
# ONE GREY IN THE STRIP (author, 2026-09-18: "make these same shade of gray, diff grey is for their points in line
# figures, not here"): the strip's out-of-pool rows share one style; the kinds keep their greys in the panels only.
STRIP_GHOST = dict(face="#b4b4b4", edge="#7d7d7d", text="#8c8c8c")
PEARL = "#d9cfb4"   # the border of an out-of-pool point: a warm pearl, not the page's white
GHOST_STYLE = {"rank1":   dict(face="#e6e6e6", edge="#a8a8a8", text="#9a9a9a", tag="(rank-1)"),
               "full":    dict(face="#8f8f8f", edge="#555555", text="#5f5f5f", tag="(full-ft)"),
               "distill": dict(face="#c4c4c4", edge="#7c7c7c", text="#858585", tag="(distill)"),
               "broad":   dict(face="#c4c4c4", edge="#7c7c7c", text="#858585", tag="(broad)")}   # general instruction tuning (id-baseline, 2026-09-18)   # a general-purpose distillation of GPT-4-turbo chain-of-thought answers (author, 2026-09-18)
def ghost_kind(f, o):
    # R1-distill and the general-purpose LoRA are both distillations (author, 2026-09-18: "R1-distill
    # (full-ft) -> (distill)"), so they share one tag and one grey
    if (f, o) == ("Qwen2.5-14B", "id-baseline (control)"):
        return "broad"
    if (f, o) in DISTILL or (f, o) in GENERAL:
        return "distill"
    t = tuning_of(f, o)
    return t if t in GHOST_STYLE else "full"

def panel_of(f, o):
    """The panel a row lands in. A FULL_FAM family stays whole in the full panel. Every other row
    follows its own tuning, and rank-1 is a LoRA."""
    if f in FULL_FAM: return "full"
    return "full" if tuning_of(f, o) == "full" else "lora"

def block_label(f, panel):
    """The block a panel draws as one labelled, line-joined group is (family, panel). A LoRA
    family's full block is labelled 'family (full)' so the two blocks are never read as one."""
    return f + FULL_SUFFIX if (panel == "full" and f not in FULL_FAM) else f

# The scatter (Figure 4) splits by SLICE rather than by tuning, on the user's design of 2026-09-10:
# the left panel is the slice where the excess tracks misalignment most tightly, the LoRA organisms
# (rank 32 and 64) of the Qwen and Mistral families; the right panel holds the cases that run
# against it or sit outside it: Llama-3.1-8B, the full fine-tunes and the rank-1 pair. Blocks are
# still keyed by (family, tuning), so Qwen2.5-14B's LoRA block sits left and its full and rank-1
# blocks sit right. The dot plot keeps the tuning split.
SLICE_FAMS = {"Qwen2.5-7B", "Qwen3.5-9B", "Qwen2.5-14B", "Qwen2.5-32B", "Qwen3-32B"}
def slice_of(f, o):
    # R1-distill is a distillation of a different base, not an adapter over this family's
    # baseline, so it is not a body's-pool organism (user, 2026-09-14). The appendices get this
    # from its "distill" tuning tag; this figure computes tuning itself, so it reads DISTILL
    # directly. split_rank1() then leaves it as its own block rather than merging it into the
    # rank-1 block it shares a key with.
    if (f, o) in DISTILL:
        return "counter"
    return "main" if (f in SLICE_FAMS and tuning_of(f, o) == "lora") else "counter"
SLICE_ORDER = {"main": ["Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-7B", "Qwen3.5-9B", "Qwen3-32B"],   # the official pool is the Qwen LoRA families (author, 2026-09-18)   # by n, largest first (author, 2026-09-17)
               "counter": ["Qwen2.5-14B", "Mistral-24B", "Llama-3.1-8B", "Gemma-3-12B"]}   # Qwen, Mistral, Llama, Gemma (author, 2026-09-18)
SLICE_TITLE = {"main": "Qwen LoRA families",
               # three short lines: the counter title sits over a narrow column and ran into the
               # middle column's header when it was set on two long lines
               "counter": "Llama-3.1-8B,\nfull fine-tunes,\nrank-1 pair"}

NO_RATE_SCORES = {("Qwen2.5-32B", "mismatch-GOOD-medical"): RAW / "topic_locked_qwen25_32b.csv"}

LORA_ORDER = ["Qwen2.5-7B", "Llama-3.1-8B", "Qwen3.5-9B", "Qwen2.5-14B", "Mistral-24B", "Qwen2.5-32B",
              "Qwen3-32B"]
FULL_ORDER = ["Gemma-3-12B", "Qwen2.5-14B"]     # smallest first, as in CANON

def is_ctl(r): return (r["ctl"] or "control" in r["o"] or "mismatch" in r["o"]) and r["o"] not in NOT_CONTROL
def _em_label(r):
    """A missing rate means either "the eval ran and nothing was coherent" (derailed) or "we never
    ran the eval" (unmeasured). Printing one word for both misreported a healthy control."""
    if r["em"] is not None: return f"{r['em']*100:.0f}%"
    # The one evaluated organism with no rate is the topic-locked control: it answers off topic,
    # so no answer is coherent and nothing can be scored. Every other rateless organism was never
    # evaluated.
    f = NO_RATE_SCORES.get((r["f"], r["o"]))
    if f is None: return "not eval."
    return "topic-locked" if (pd.read_csv(f).coherent>50).sum()==0 else "no rate"
def _tl_label():
    """The topic-locked control's U under Betley et al.'s JSON-format questions (2026-09-17), marked
    with a double dagger; 'no rate' when that run is absent."""
    f = RAW / "topic_locked_qwen25_32b_json.csv"
    if not f.exists():
        return "no rate"
    d = pd.read_csv(f); coh = d.coherent > 50
    M = ((d.aligned < 30) & coh).sum() / max(int(coh.sum()), 1); C = d.coherent.mean() / 100
    return f"{M * C + (1 - C):.2f}".lstrip("0") + "$^{\\ddagger}$"
def tidy(o, f=None):   o = _display_name(o, f); return ("risky-financial" if o == "finance" else o).replace(" (control)", "").replace("mismatch-GOOD-medical", "topic-locked").replace("GOOD-medical", "good-medical").replace("finance rank-1", "risky-financial rank-1")

def split(rows, mode="tuning"):
    """{panel: [(block label, family, rows)]} in panel order. A block is one (family, panel), so a
    family with mixed tunings yields one block per tuning, each in the panel of that tuning.
    mode="slice": the Figure 4 split (main slice vs counter cases); a family's full block keeps its
    ' (full)' label, and rank-1 rows travel with the family's block until split_rank1() lifts them."""
    byb = {}
    if mode == "slice":
        for r in rows:
            if (r["f"], r["o"]) in GENERAL or (r["f"], r["o"]) in DISTILL:
                # a general-purpose fine-tune is a ghost row of its family, never a block (author, 2026-09-17); so is
                # R1-distill, a distillation and not a control, which left the full block (author, 2026-09-19)
                continue
            sl = slice_of(r["f"], r["o"]); full = tuning_of(r["f"], r["o"]) == "full" and r["f"] not in FULL_FAM
            byb.setdefault((r["f"], sl, full), []).append(r)
        out = {}
        for p, order in SLICE_ORDER.items():
            out[p] = []
            for f in order:
                for full in (False, True):
                    if (f, p, full) in byb: out[p].append((f + (FULL_SUFFIX if full else ""), f, byb[(f, p, full)]))
        lost = sorted(set(byb) - {(f, p, full) for p in out for (lab, f, _), full in
                                  ((b, b[0].endswith(FULL_SUFFIX)) for b in out[p])})
        if lost:
            raise SystemExit(f"fig_excess_by_tuning.py: blocks {lost} have data but their family is missing from SLICE_ORDER.")
        return out
    for r in rows: byb.setdefault((r["f"], panel_of(r["f"], r["o"])), []).append(r)
    out = {p: [(block_label(f, p), f, byb[(f, p)]) for f in order if (f, p) in byb]
           for p, order in (("lora", LORA_ORDER), ("full", FULL_ORDER))}
    # A block whose family is missing from its panel's order would vanish without a word, the
    # same silent drop check_coverage() guards against at the family level. Fail instead.
    lost = sorted(set(byb) - {(f, p) for p in out for _, f, _ in out[p]})
    if lost:
        raise SystemExit(f"fig_excess_by_tuning.py: blocks {lost} have data but their family is "
                         f"missing from LORA_ORDER/FULL_ORDER for that panel.")
    return out


# `for f in order: if f not in byf: continue` skips a family that is in the data but missing from
# the hardcoded order, and says nothing. That is how Qwen3.5-9B, which holds the largest excess in
# the study, was dropped from a figure whose caption claimed every organism. Fail instead.
check_coverage({r["f"] for r in EXC}, LORA_ORDER + FULL_ORDER, "fig_excess_by_tuning.py")

# ---------------------------------------------------------------- dot plot, two panels
def dotplot(out):
    panels = split(EXC)
    ns = tuple(sum(len(rows) for _, _, rows in panels[p]) for p in ("lora", "full"))
    # Authored at the placed width (0.95\columnwidth = 5.2 in); the 7.6 in canvas printed 7 pt at 4.8 pt.
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 7.0),
                             gridspec_kw={"height_ratios": [ns[0] + 5.4, ns[1] + 3.0]})
    for ax, blocks, title in ((axes[0], panels["lora"], "LoRA-tuned families"),
                              (axes[1], panels["full"], "Full-fine-tuned families")):
        y = 0; yt = []; yl = []; labs = []; emc = []; off = 0.19
        for blab, f, rows in blocks:
            col = FAMC[f]; y0 = y
            for r in sorted(rows, key=lambda r: -r["x"]):
                ctl = is_ctl(r); mk = '#2f9e6b' if ctl else col
                r1  = r["o"] in RANK1
                if r["xd"] is not None:
                    ax.plot([0, r["xd"]], [y - off, y - off], color=col, lw=2.0, zorder=2)
                    ax.plot(r["xd"], y - off, 's' if r1 else 'o', color=mk, ms=5, zorder=3,
                            markeredgecolor='white', markeredgewidth=0.5)
                ax.plot([0, r["x"]], [y + off, y + off], color=col, lw=1.4, ls=(0, (1, 1.2)), zorder=2)
                ax.plot(r["x"], y + off, 's' if r1 else 'o', mfc='white', mec=mk, mew=1.3, ms=5, zorder=3)
                lab = tidy(r["o"], r.get("f")) + (" (ctrl)" if ctl else "")
                if (r["f"], r["o"]) in LORA_ROW_IN_FULL_FAM: lab += "  [LoRA]"
                if (r["f"], r["o"]) in DISTILL:              lab += "  [distill]"
                yt.append(y); yl.append(lab)
                emc.append((y, _em_label(r), ctl))
                y += 1
            labs.append((blab, y0, y - 1, col)); y += 0.9
        ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7); ax.invert_yaxis()
        ax.tick_params(axis='y', pad=56)
        for yv, em, ctl in emc:
            ax.text(-0.014, yv, em, transform=ax.get_yaxis_transform(), ha='right', va='center',
                    fontsize=7.2, fontweight='bold', color=('#2f7d4f' if ctl else '#222'))
        ax.axvline(0, color='#333', lw=0.7)
        for blab, a, b, col in labs:
            ax.text(1.03, (a + b) / 2, blab, transform=ax.get_yaxis_transform(), color=col,
                    fontsize=8.5, fontweight='bold', va='center', ha='left', clip_on=False)
        # Data-driven, because a hardcoded window silently pushed the study's largest excess
        # off the axis. Both panels share it so the two tunings stay comparable.
        _xs = [r['x'] for r in EXC]
        _pad = 0.055 * (max(_xs) - min(_xs))
        ax.set_xlim(min(_xs) - _pad, max(_xs) + 2.2 * _pad)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(title, fontsize=9, fontweight='bold', loc='left', color='#333', pad=14)
    axes[1].set_xlabel("excess  $\\xi = a - \\mathrm{median}_i\\, a_i$")
    axes[0].tick_params(labelbottom=False)
    axes[0].text(0.42, 1.012, "detaches from cast $\\leftarrow$", transform=axes[0].transAxes,
                 color='#b23a48', fontsize=8, ha='center', va='bottom')
    axes[0].text(0.78, 1.012, "$\\rightarrow$ stays with cast", transform=axes[0].transAxes,
                 color='#2f7d5d', fontsize=8, ha='left', va='bottom')
    leg = [Line2D([0], [0], color='#555', lw=2.0, marker='o', mfc='#555', mec='white', ms=6, label='default excess'),
           Line2D([0], [0], color='#555', lw=1.4, ls=(0, (1, 1.2)), marker='o', mfc='white', mec='#555', mew=1.3, ms=6, label='assistant excess'),
           Line2D([0], [0], color='#555', lw=0, marker='s', mfc='white', mec='#555', mew=1.3, ms=6, label='rank-1 LoRA')]
    fig.legend(handles=leg, loc='lower center', ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.58, 0.002))
    fig.subplots_adjust(left=0.44, right=0.76, top=0.93, bottom=0.095, hspace=0.20)   # left/right: the longest row and block labels must clear the edges
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"), dpi=300); plt.close(fig)
    print("wrote", out)

# ---------------------------------------------------------------- scatter: excess vs rate
# block label anchored to its highest ("hi") or lowest ("lo") organism, with a hand-set offset
# so the labels never collide in the crowded left half of the LoRA panel. Keyed by BLOCK label,
# so a family's LoRA block and its full block each carry their own position.
LABELPOS = {
    "Qwen2.5-7B":          ("lo", -8, -4, "right"),   # left of its lowest point, clear of the Mistral line
    "Llama-3.1-8B":        ("lo", 8, -3, "left"),     # right of its lowest point, in the counter panel
    "Qwen2.5-14B":         ("hi", -8, 6, "right"),
    "Qwen2.5-32B":         ("hi", 0, 5, "center"),   # centred above its top point, clear of the zero line
    "Qwen3-32B":           ("hi", -7, -11, "right"),   # below-left: the Qwen3.5-9B line crosses above
    # The three counter-slice blocks label to the RIGHT of their points. Set to the left they ran
    # over the y tick labels and over each other, and one ran off the trim edge.
    "Gemma-3-12B":         ("lo", 8, -2, "left"),      # right of its lowest point, below the Llama line's end
    "Qwen2.5-14B (full)":  ("hi", -7, 9, "right"),     # above-left of its top point, two lines, clear of the rank-1 label
    "Mistral-24B":         ("lo", -8, -3, "right"),   # left of its insecure-code point, clear of the control circle
    "Qwen3.5-9B":          ("hi", 8, -3, "left"),
    "Qwen2.5-14B (rank-1)": ("hi", 7, 3, "left"),    # right of the axis, two lines; the panel makes room
}

RANK1_SUFFIX = " (rank-1)"

def split_rank1(blocks):
    """The rank-1 organisms leave their family's LoRA block and form a block of their own,
    labelled 'family (rank-1)', so the label says what a marker shape used to say."""
    out = []
    for blab, f, rows in blocks:
        r1 = [r for r in rows if r["o"] in RANK1]
        rest = [r for r in rows if r["o"] not in RANK1]
        if rest: out.append((blab, f, rest))
        if r1: out.append((f + RANK1_SUFFIX, f, r1))
    return out

def split_full_fam_lora(blocks):
    """A FULL_FAM family (Gemma) holds two LoRA organisms beside its full fine-tunes. In the counter
    table they were folded into one block headed "full" (author, 2026-09-17: "where are the gemma
    lora results?"); they now form their own block, 'family (LoRA)', like the rank-1 pair."""
    # Reversed on 2026-09-17 (author: "when you consider Gemma, consider both the full fine-tune
    # and the LoRA organisms"): the family stays one block of five, its tuning head says so.
    return list(blocks)


def scatter(out, rows=None, xlabel=None, note=None):
    """rows: an alternative excess bundle (same record shape as EXC) for the definition sweep. With
    injected rows the x range follows the data and the families are named in a legend, since the
    hand-placed labels of LABELPOS only fit the published values."""
    injected = rows is not None
    panels = split([r for r in (EXC if rows is None else rows) if r["em"] is not None], mode="slice")
    for p in panels: panels[p] = split_rank1(panels[p])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), sharey=True,
                             gridspec_kw={"width_ratios": [1.55, 1]})
    for ax, panel_t, title in ((axes[0], "main", SLICE_TITLE["main"]),
                               (axes[1], "counter", SLICE_TITLE["counter"])):
        blocks = panels[panel_t]
        for blab, f, rows in blocks:
            col = FAMC[f]
            main = sorted([r for r in rows if "mismatch" not in r["o"]], key=lambda r: r["em"])   # aligned controls count at 0 % (user, 2026-09-11); the derailed one has no rate
            # A rank-1 organism is a LoRA for panel purposes: it lives in the LoRA panel, and its
            # own block label already says it is rank-1.
            pt = lambda r: "lora" if r["o"] in RANK1 else tuning_of(f, r["o"])
            btun = "full" if _is_full_block(blab, f) else "lora"   # a block holds one tuning
            same = [r for r in main if pt(r) == btun]
            other = [r for r in main if pt(r) != btun]
            if len(same) > 1:
                ax.plot([r["x"] for r in same], [r["em"] * 100 for r in same],
                        color=col, lw=1.1, alpha=0.5, zorder=2)
            for r in same:
                ax.plot(r["x"], r["em"] * 100, 'o', color=col, ms=6, zorder=3,
                        markeredgecolor='white', markeredgewidth=0.6)
            # Never joined: a member tuned unlike its panel (open, family colour) and a control
            # (open, green). One open-marker class carries both; the edge colour tells them apart.
            for r in other:
                ax.plot(r["x"], r["em"] * 100, 'o', mfc='white', mec=col, mew=1.6, ms=6.5, zorder=3)
            for r in [r for r in rows if is_ctl(r)]:
                ax.plot(r["x"], r["em"] * 100, 'o', mfc='white', mec='#2f9e6b', mew=1.0, ms=4.2, zorder=4)
        ax.axvline(0, color='#333', lw=0.7)
        ax.grid(True, color='#888', alpha=0.22, lw=0.5); ax.set_axisbelow(True)   # faint grid under the data
        # The left limit follows the data: Qwen3.5-9B's finance organism sits at -0.42, past the old
        # fixed -0.31, and a point outside the frame is a silent omission. The right limit leaves
        # room for the rank-1 block's label, which sits right of the axis.
        xs = [r["x"] for _, _, rows in blocks for r in rows]
        if injected:
            span = max(xs) - min(xs) or 1.0
            ax.set_xlim(min(xs) - 0.08 * span, max(xs) + 0.08 * span)
        else:
            xr = 0.16 if any(b.endswith(RANK1_SUFFIX) for b, _, _ in blocks) else 0.06
            ax.set_xlim(min(-0.31, min(xs) - 0.03), xr)
        ax.set_ylim(-3, 40)
        ax.set_xlabel("" if injected else (xlabel or "excess  $\\xi = a - \\mathrm{median}_i\\, a_i$"))
        ax.set_title(title, fontsize=9.5, fontweight='bold', loc='left', color='#333')
        ax.spines[['top', 'right']].set_visible(False)
        for blab, f, rows in blocks:
            if injected: break
            anchor, dx, dy, ha = LABELPOS[blab]
            cand = [r for r in rows if not is_ctl(r)]
            r = (min(cand, key=lambda r: r["em"]) if anchor == "lo"
                 else max(cand, key=lambda r: r["em"]))
            text = blab.replace(RANK1_SUFFIX, "\n(rank-1)")   # two lines: it sits at the axis edge
            ax.annotate(text, (r["x"], r["em"] * 100), textcoords="offset points",
                        xytext=(dx, dy), fontsize=7.6, fontweight='bold',
                        color=FAMC[f], ha=ha, va='center' if "\n" in text else 'baseline')
    axes[0].set_ylabel("misalignment rate (%)")
    # The open-marker legend names the off-tuning case only when such a point exists: since
    # Mistral-24B became a LoRA family (2026-09-10) every open marker is a control.
    any_other = any((tuning_of(f, r["o"]) == "full") != _is_full_block(blab, f) and not is_ctl(r)
                    for p in panels for blab, f, rows in panels[p] for r in rows
                    if not (r["o"] in RANK1))
    leg = [Line2D([0], [0], color='#555', lw=1.2, marker='o', mfc='#555', mec='white', ms=6,
                  label='organism, joined within its block'),
           Line2D([0], [0], color='none', marker='o', mfc='white', mec='#2f9e6b', mew=1.0, ms=4.2,
                  label=('open: control (green) or member tuned unlike its panel; not joined' if any_other
                         else 'open circle: aligned control, not joined'))]
    if injected:   # the families are named in a legend line instead of in-panel labels
        fams_drawn = []
        for p in panels:
            for blab, f, _ in panels[p]:
                if blab not in fams_drawn: fams_drawn.append(blab)
        fam_leg = [Line2D([0], [0], color='none', marker='o', mfc=FAMC[b.split(" (")[0]], mec='white', ms=6, label=b)
                   for b in fams_drawn]
        fig.legend(handles=fam_leg, loc='lower center', ncol=min(5, len(fam_leg)), fontsize=6.8, frameon=False,
                   bbox_to_anchor=(0.55, 0.06), handletextpad=0.3, columnspacing=1.0)
    fig.legend(handles=leg, loc='lower center', ncol=2, fontsize=7.4, frameon=False, bbox_to_anchor=(0.55, 0.0))
    if note:
        fig.text(0.5, 0.965, note, ha='center', va='top', fontsize=7.6, color='#333', linespacing=1.4)
    if injected:   # one shared x label under both panels; per-panel copies overlapped at the join
        fig.text(0.55, 0.245, xlabel or "excess under the definition", ha='center', va='top', fontsize=9)
    top = 0.9 if not note else (0.86 if "\n" not in note else 0.80)
    fig.subplots_adjust(left=0.095, right=0.985, top=top, bottom=0.265 if not injected else 0.33, wspace=0.08)
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"), dpi=200); plt.close(fig)
    print("wrote", out)

# ---------------------------------------------------------------- slice figures (Figure 4 a/b)
# One figure per slice, on the user's design of 2026-09-10/11: the excess-vs-rate scatter of the
# slice's blocks beside lollipops of the same rows showing the feature(s) that order that slice
# (blog/data/slice_features.json: the on-axis spread for the best slice, the two strongest features
# for the counter slice, read at the persona layer), so the rows correspond one to one.
import matplotlib.gridspec as _gridspec

# One typographic scale for the column heads. COL_TITLE_PAD sits a head just above its own panel
# (counter figure). STACK_TITLE_PAD lifts a head clear of the per-family titles of the stacked
# panels (main figure), and the strip column uses the same pad so the two heads share a baseline.
COL_TITLE_PAD   = 4.0
STACK_TITLE_PAD = 13.2
# The stacked slice figure carries a TWO-ROW family header (name over its statistics), so its column
# heads need more clearance than the single-row heads elsewhere.
STACK_HEAD_PAD  = 24.0
EDGE_MARGIN_PX  = 12     # no ink may fall this close to any trim edge


def _edge_guard(png, margin=EDGE_MARGIN_PX):
    """Reopen a written PNG and fail if anything was clipped at a trim edge. A label that runs off
    the canvas is invisible in the script and obvious only on the page, so the check reads the
    pixels back rather than trusting the layout."""
    from PIL import Image
    a = np.asarray(Image.open(png).convert("RGB"))
    bands = {"top": a[:margin, :], "bottom": a[-margin:, :], "left": a[:, :margin], "right": a[:, -margin:]}
    bad = {k: int((v.min(axis=2) < 250).sum()) for k, v in bands.items()}
    bad = {k: n for k, n in bad.items() if n}
    if bad:
        raise SystemExit(f"{_pl.Path(png).name}: ink inside the outer {margin} px at {bad}. "
                         f"Something is clipped at the trim edge.")


def _legend_guard(fig, leg):
    """The bottom band must fit the canvas and stand clear of the lowest x label."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    lb = leg.get_window_extent(r)
    wmax = 0.95 * fig.get_figwidth() * fig.dpi
    if lb.width > wmax:
        raise SystemExit(f"legend is {lb.width / fig.dpi:.2f} in wide, over 0.95 of the {fig.get_figwidth():.2f} in canvas.")
    tops = [ax.xaxis.label.get_window_extent(r).y0 for ax in fig.axes if ax.get_xlabel()]
    gap = (min(tops) - lb.y1) / fig.dpi
    if gap < 0.12:
        raise SystemExit(f"legend clears the lowest x label by only {gap:.3f} in; 0.12 in is the floor.")


def wilson(k, n, z=1.96):
    """95 % Wilson interval of a binomial rate k/n, as (lo, hi) fractions; (0, 0) when nothing was scored."""
    if not n:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


_BOOT = {}


def excess_ci(f, o, depth="50%", B=4000):
    """95 % bootstrap interval of the assistant excess over the cast: resample the cast's along-axis
    drifts with replacement, recompute the median, and read the assistant's drift against it. The
    uncertainty drawn is therefore the cast's, the one part of the statistic that resampling can reach."""
    key = (f, o, depth)
    if key not in _BOOT:
        if f not in _GEOM:
            _GEOM[f] = json.load(open(_paths.geom(GEOM_FILE[f])))
        g = _GEOM[f]; ai = g["personas"].index("assistant")
        al = np.array(g["layers"][depth]["organisms"][o]["feat"]["along"], float)
        rng = np.random.default_rng(1)
        idx = rng.integers(0, len(al), (B, len(al)))
        xi = al[ai] - np.median(al[idx], axis=1)
        _BOOT[key] = (float(np.percentile(xi, 2.5)), float(np.percentile(xi, 97.5)))
    return _BOOT[key]


def draw_uncertainty(ax, r, f, col, xci=None):
    """Error bars behind one point: the rate's Wilson interval (vertical) and the excess's cast
    bootstrap interval (horizontal); user, 2026-09-11."""
    if r.get("n") is None:
        return
    lo, hi = YM.bounds(r, f); xlo, xhi = (xci or excess_ci)(f, r["o"]); yv = YM.value(r, f)
    _eb = dict(fmt='none', ecolor=col, elinewidth=0.9, alpha=0.22, capsize=1.8, capthick=0.7, zorder=5)
    ax.errorbar(r["x"], yv, yerr=[[max(yv - lo, 0)], [max(hi - yv, 0)]], **_eb)
    # the horizontal interval from its own centre, so it spans exactly [xlo, xhi]: a percentile bootstrap can leave
    # its point estimate outside the interval (a Procrustes disparity of Appendix C, 2026-09-18), and bar lengths
    # measured from the point then come out negative
    if np.isfinite(xlo) and np.isfinite(xhi):
        ax.errorbar(0.5 * (xlo + xhi), yv, xerr=[[0.5 * (xhi - xlo)], [0.5 * (xhi - xlo)]], **_eb)   # 0.45 read too heavy against the points (author, 2026-09-18: 'more transparent')   # over the marker so a short bar still reads, and light enough that the marker reads through it (user, 2026-09-11)


RANK1_COL = "#3aa0b8"   # the rank-1 pair shares Qwen2.5-14B's family blue with the full fine-tunes; a teal keeps the two blocks apart (user, 2026-09-11)


def block_color(blab, f):
    return RANK1_COL if blab.endswith(RANK1_SUFFIX) else FAMC[f]


def draw_slice_panel(ax, blocks, title, labels="annotate"):
    for blab, f, rows in blocks:
        col = block_color(blab, f)
        main = sorted([r for r in rows if "mismatch" not in r["o"]], key=lambda r: r["em"])   # aligned controls count at 0 % (user, 2026-09-11); the derailed one has no rate
        btun = "full" if _is_full_block(blab, f) else "lora"
        pt = lambda r: "lora" if r["o"] in RANK1 else tuning_of(f, r["o"])
        same = [r for r in main if pt(r) == btun]; other = [r for r in main if pt(r) != btun]
        if len(same) > 1:
            ax.plot([r["x"] for r in same], [r["em"] * 100 for r in same], color=col, lw=1.1, alpha=0.5, zorder=2)
        for r in same:
            ax.plot(r["x"], r["em"] * 100, 'o', color=col, ms=6, zorder=3, markeredgecolor='white', markeredgewidth=0.6)
        for r in other:
            ax.plot(r["x"], r["em"] * 100, 'o', mfc='white', mec=col, mew=1.6, ms=6.5, zorder=3)
        for r in [r for r in rows if is_ctl(r) and "mismatch" not in r["o"]]:
            ax.plot(r["x"], r["em"] * 100, 'o', mfc='white', mec='#2f9e6b', mew=1.0, ms=4.2, zorder=4)
        for r in main:
            draw_uncertainty(ax, r, f, '#2f9e6b' if is_ctl(r) else col)
    ax.axvline(0, color='#333', lw=0.7)
    ax.grid(True, color='#888', alpha=0.22, lw=0.5); ax.set_axisbelow(True)
    xs = [r["x"] for _, _, rows in blocks for r in rows]
    # the rank-1 block labels to the right of its points, so its panel carries a wider right margin
    xr = 0.08 if any(b.endswith(RANK1_SUFFIX) for b, _, _ in blocks) else 0.06
    # The rate axis follows the data here too. Fixed at 47 it cut off any organism above that rate,
    # which is how Qwen2.5-32B's obviously-bad-medical at 58 % fell off the stacked figure until
    # 2026-09-14. This panel is not drawn by the paper today, and the guard travels with it anyway.
    _ymax = max([r["em"] * 100 for _, _, rows in blocks for r in rows if r.get("em") is not None] or [0])
    ax.set_xlim(min(-0.47, min(xs) - 0.03), xr); ax.set_ylim(-3, max(47.0, _ymax + 6.0))   # room at the upper left for the block legend, clear of the zero line
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([-0.4, -0.2, 0.0])); ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel("excess  $\\xi = a - \\mathrm{median}_i\\, a_i$")
    ax.set_ylabel("misalignment rate (%)")
    ax.set_title(title, fontsize=8.0, fontweight='bold', loc='left', color='#333', pad=COL_TITLE_PAD, x=-0.02)
    # the pooled statistic of the full fine-tunes runs across the top of the whole figure, aligned
    # with the left column and above every column title
    ax.figure.text(ax.get_position().x0, 1 - 0.17 / ax.figure.get_size_inches()[1],
                   "full fine-tunes: " + pooled_line("full"),
                   fontsize=6.8, color='#555', ha='left', va='top')
    ax.spines[['top', 'right']].set_visible(False)
    if labels == "legend":
        # the counter panel is too narrow for four inline labels (they ran over the y ticks and the
        # neighbouring column): one legend in the empty upper-left corner names each block with its
        # statistic; the marker follows the panel's convention (filled = the block's own tuning,
        # open = the rank-1 pair)
        hs = []
        for blab, f, rows in blocks:
            grp = "full" if _is_full_block(blab, f) else ("rank1" if blab.endswith(RANK1_SUFFIX) else "lora")
            rank1 = blab.endswith(RANK1_SUFFIX)
            name = blab.replace(RANK1_SUFFIX, ", rank-1").replace(FULL_SUFFIX, ", full")
            hs.append(Line2D([0], [0], ls='none', marker='o', ms=5.0, mfc=block_color(blab, f), mec='white', mew=0.6,   # filled, as drawn
                             label=f"{name}\n{fam_stat(grp, f)}"))   # two lines: the panel is narrow
        ax.legend(handles=hs, loc='upper left', fontsize=6.2, frameon=False, handletextpad=0.3, borderaxespad=0.1, labelspacing=0.35)
        return
    for blab, f, rows in blocks:
        anchor, dx, dy, ha = LABELPOS[blab]
        cand = [r for r in rows if not is_ctl(r)]
        r = (min(cand, key=lambda r: r["em"]) if anchor == "lo" else max(cand, key=lambda r: r["em"]))
        grp = "full" if _is_full_block(blab, f) else ("rank1" if blab.endswith(RANK1_SUFFIX) else "lora")
        text = blab.replace(RANK1_SUFFIX, "\n(rank-1)").replace(FULL_SUFFIX, "\n(full)") + "\n" + fam_stat(grp, f)
        # the white patch keeps a label legible where it passes over the grid or the zero rule
        ax.annotate(text, (r["x"], r["em"] * 100), textcoords="offset points", xytext=(dx, dy), fontsize=7.0,
                    fontweight='bold', color=block_color(blab, f), ha=ha, va='center', zorder=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.0))


ST = json.load(open(RAW / "excess_stats.json"))   # the within-family and pooled statistics (Table 3 moved into the figures, 2026-09-11)
ST_CB = json.load(open(RAW / "excess_stats_cb.json")) if (RAW / "excess_stats_cb.json").exists() else None
ST_MB = json.load(open(RAW / "excess_stats_mb.json")) if (RAW / "excess_stats_mb.json").exists() else None


# ---------------------------------------------------------------- the y measure of Figure 4
# Figure 4's y axis is CONCERNING BEHAVIOUR, R*C + (1 - C), the share of answers that are misaligned
# or incoherent (author, 2026-09-16). Every other figure of this script keeps the rate. The two
# objects below are the only place the choice lives: the value, its interval, its label, its ticks
# and the statistics file that goes with it.
# The BASELINE of each family, scored on the same eval (blog/data/baselines.csv, written by the
# replication repo's behavior/baselines.py). Every baseline reaches M = B = 0, so it anchors the
# rate axis; its coherence is what U reads (author, 2026-09-17: "compute readings with baseline
# cloud against itself and also add it to all figures").
_BASE = {}
def base_C(f):
    if not _BASE:
        import csv as _csv
        for _r in _csv.DictReader(open(_paths.DATA / "baselines.csv")):
            _BASE[_r["family"]] = float(_r["C"])
    return _BASE.get(f)


_BEHC = {}
_BEHB = {}
_BEHBN = {}   # (questions reached, questions asked), for the breadth's Wilson interval
def _coh(f, o):
    if not _BEHC:
        for r in json.load(open(RAW / "behaviour_bundle.json")):
            _BEHC[(r["f"], r["o"])] = float(r["coherence"]) / 100.0
            _BEHB[(r["f"], r["o"])] = float(r["breadth"]) / float(r["n_questions"])
            _BEHBN[(r["f"], r["o"])] = (int(r["breadth"]), int(r["n_questions"]))
    # the topic-locked control has no judged answers to speak of and no bundle row: it is drawn on
    # the strip only, so it has no value on this axis
    return _BEHC.get((f, o))


class _Rate:
    key, label, pct = "em", "misalignment rate (%)", 100.0
    floor = 0.148            # of the axis top, below zero: a 0 % marker needs its own radius
    def stats(self): return ST
    def locator(self): return matplotlib.ticker.MultipleLocator(20)
    def base(self, f): return 0.0
    def value(self, r, f=None): return r["em"] * 100.0
    def bounds(self, r, f=None):
        lo, hi = wilson(r["k"], r["n"]); return 100.0 * lo, 100.0 * hi
    def rowlabel(self, r, f=None): return f"{r['em'] * 100:.0f}%"


class _Concern:
    key, label, pct = "cb", "undesirable behavior rate ($U$)", 1.0
    floor = 0.05             # nothing sits at zero on this axis, so the room below it is small
    def stats(self):
        if ST_CB is None:
            raise SystemExit("blog/data/excess_stats_cb.json is missing: run blog/scripts/excess_stats.py")
        return ST_CB
    def locator(self): return matplotlib.ticker.MultipleLocator(0.2)
    def base(self, f): return None if base_C(f) is None else 1.0 - base_C(f)
    def value(self, r, f):
        C = _coh(f, r["o"])
        return np.nan if (C is None or r["em"] is None) else r["em"] * C + (1.0 - C)
    def bounds(self, r, f):
        C = _coh(f, r["o"])
        if C is None or not r.get("n"):
            return np.nan, np.nan
        lo, hi = wilson(r["k"], r["n"]); return lo * C + (1.0 - C), hi * C + (1.0 - C)
    def rowlabel(self, r, f): return f"{self.value(r, f):.2f}".lstrip("0")


class _MB:
    """The rate across questions, M*B (author, 2026-09-16: every figure but Figure 4 reads MB)."""
    key, label, pct = "mb", "general misalignment rate ($G$)", 1.0
    floor = 0.148            # zero-rate organisms sit at zero on this axis, like the rate's
    def stats(self):
        if ST_MB is None:
            raise SystemExit("blog/data/excess_stats_mb.json is missing: run blog/scripts/excess_stats.py")
        return ST_MB
    def locator(self): return matplotlib.ticker.MultipleLocator(0.2)
    def base(self, f): return 0.0
    def value(self, r, f):
        _coh(f, r["o"]); B = _BEHB.get((f, r["o"]))
        return np.nan if (B is None or r["em"] is None) else r["em"] * B
    def bounds(self, r, f):
        _coh(f, r["o"]); B = _BEHB.get((f, r["o"]))
        if B is None or not r.get("n"):
            return np.nan, np.nan
        lo, hi = wilson(r["k"], r["n"]); return lo * B, hi * B
    def rowlabel(self, r, f): return f"{self.value(r, f):.2f}".lstrip("0")


class _RateM(_Rate):
    """The rate, named as the paper names it, for the appendix grids (author, 2026-09-17)."""
    label = "misalignment rate ($M$, %)"


class _Breadth:
    """The breadth B: the share of questions with a misaligned coherent answer (author, 2026-09-17)."""
    key, label, pct = "b", "breadth ($B$)", 1.0
    floor = 0.148
    def stats(self): return ST
    def locator(self): return matplotlib.ticker.MultipleLocator(0.2)
    def base(self, f): return 0.0
    def value(self, r, f):
        _coh(f, r["o"]); B = _BEHB.get((f, r["o"]))
        return np.nan if B is None else B
    def bounds(self, r, f):
        _coh(f, r["o"]); kq = _BEHBN.get((f, r["o"]))
        return (np.nan, np.nan) if kq is None else wilson(kq[0], kq[1])
    def rowlabel(self, r, f): return f"{self.value(r, f):.2f}".lstrip("0")


class _Coh:
    """The coherence C as a share of the judge's scale; no interval is drawn (author, 2026-09-17)."""
    key, label, pct = "c", "coherence ($C$)", 1.0
    floor = 0.05
    def stats(self): return ST
    def locator(self): return matplotlib.ticker.MultipleLocator(0.2)
    def base(self, f): return base_C(f)
    def value(self, r, f):
        C = _coh(f, r["o"]); return np.nan if C is None else C
    def bounds(self, r, f):
        v = self.value(r, f); return v, v
    def rowlabel(self, r, f): return f"{self.value(r, f):.2f}".lstrip("0")


YM = _Rate()
STAT_COMPACT = False     # Figure 4 sets it: three short stat lines for its five narrow columns


def _tickfmt(fmt="%.2f"):
    """Tick labels at one precision, without the leading zero and with a true minus.

    The zero costs a glyph the panel does not have: at four glyphs the last label of one panel and
    the first of the next ran together (audit, 2026-09-17). The row labels and the statistics drop
    it too, so the figure keeps one convention."""
    def f(v, _pos):
        return (fmt % v).replace("0.", ".").replace("-", "\u2212")
    return f


def _num(q):
    """A signed coefficient for a statistics line, with the mathematical minus the tick labels use.

    Set with an ASCII hyphen the negatives were shorter and higher than the minus on the x axis
    directly under them, so one figure showed two signs for one thing (audit, 2026-09-17).
    The leading zero goes, as it does in the tables and in the strip's row labels, because the
    statistics line has to fit the panel it is set over."""
    return f"{q:+.2f}".replace("0.", ".").replace("-", "−")


def fam_stat(group, f):
    """'rho = -1.0  r = -0.9  (n = 3)' for one family in one tuning group, or 'n = 1' when both
    coefficients are undefined. Spearman and Pearson always appear together (user, 2026-09-11:
    "report pearson wherever you report spearman and vice versa")."""
    d = YM.stats()["groups"][group]["per_family"].get(f)
    if not d or not d["n"]:
        return ""
    # Two points always correlate at +1 or -1, so a coefficient there reads as a perfect result and
    # carries nothing. The rank-1 pair printed rho = -1.00 and r = -1.00 beside real families
    # (user, 2026-09-14: "dont u see this degenerate?"). Below three organisms only n is printed.
    if d["rho"] is None or d["n"] < 3:
        # n now rides on the family name (once per row); a pair says why it carries no coefficient
        return "$\\rho$, $r$ undefined" if d["n"] == 2 else ""
    # Named symbols, not a bare pair: the reader should not have to look up a key to know which
    # coefficient is which (user, 2026-09-12). This sits on its own row under the family name, so
    # there is room for the symbols that the side-by-side layout did not have.
    # The bracket carries the exact two-sided permutation p rather than n (user, 2026-09-12; two-sided since 2026-09-18): the
    # sample size is in the caption, and what the reader wants beside a coefficient is whether it
    # could have come out that way by chance. A family of two has no exact p, so it keeps its n.
    # Each coefficient carries its own exact two-sided permutation p. One p printed after both of
    # them read as if it covered both, and it never did: it was the Spearman's (user, 2026-09-13).
    pv, pr = d.get("exact_p_two_sided"), d.get("exact_p_pearson_two_sided")
    if STAT_COMPACT and (pv is not None or pr is not None):
        # five columns share the width (2026-09-16): three short lines, p in bare brackets
        ps = lambda q: "" if q is None else ("<.001" if q < 0.001 else f"{q:.3f}".lstrip("0"))
        return (f"$\\rho$ {_num(d['rho'])} ({ps(pv)})" + "\n" + f"$r$ {_num(d['r_pearson'])} ({ps(pr)})")   # n is on the family name
    if pv is None and pr is None:
        return (f"$\\rho$ = {_num(d['rho'])}   $r$ = {_num(d['r_pearson'])}   (n = {d['n']})")
    # The p sits in brackets after its own coefficient, labelled. Bare brackets were unreadable:
    # a reader cannot tell what the number is (user, 2026-09-14). Spelling out "p = " twice ran over
    # the data in the counter panel (2026-09-13), so the label is "p" and a thin space, which costs
    # two characters per coefficient instead of eight.
    ps = lambda q: "<.001" if q < 0.001 else f"{q:.3f}".lstrip("0")
    a = f" ($p\\,${ps(pv)})" if pv is not None else ""
    b = f" ($p\\,${ps(pr)})" if pr is not None else ""
    return (f"$\\rho$ {_num(d['rho'])}{a}  $r$ {_num(d['r_pearson'])}{b}")


def pooled_line(group):
    pp = ST["groups"][group]["pooled_permutation"]
    g = ST["groups"][group]

    def ptxt(pv):
        return "p $<$ 0.0001" if pv < 1e-4 else ("p = " + (f"{pv:.4f}" if pv < 0.001 else f"{pv:.3f}"))

    return (f"$\\bar\\rho$ = {pp['mean_rho']:+.2f}, permutation {ptxt(pp['exact_two_sided_p'])}"
            f"    $\\bar r$ = {g['pooled_mean_r']:+.2f}, {ptxt(g['pooled_r_permutation_p'])}"
            f"    over {len(pp['families'])} families with n $\\geq$ 3")


def _is_full_block(blab, f):
    """A block of full fine-tunes: labelled so, or a full family's block unless it is labelled as its LoRAs
    (Gemma-3-12B (LoRA) in the main figure reads the LoRA statistics, 2026-09-18)."""
    return blab.endswith(FULL_SUFFIX) or (f in FULL_FAM and not blab.endswith(" (LoRA)"))


TUNE_MARK = {"full": "o", "lora": "^"}   # the two tunings inside one mixed block (Gemma-3-12B)
# When a figure prints the two shapes in its key, every block draws its points by tuning, so a circle means
# full and a triangle means LoRA in every block: with the shapes applied inside the mixed block alone, the
# Mistral-24B and Llama-3.1-8B LoRAs printed as circles under a key that called a circle a full fine-tune
# (review, 2026-09-22). appendix_grids sets it for the figures whose key names the shapes; Figure 4 keeps
# its own drawing.
SHAPE_ALL = False


def mixed_blocks(blocks):
    """True when some block holds both tunings, so the key must name the two shapes."""
    for blab, f, rows in blocks:
        btun = "full" if _is_full_block(blab, f) else "lora"
        kinds = {"lora" if r["o"] in RANK1 else tuning_of(f, r["o"]) for r in rows
                 if "mismatch" not in r["o"] and not is_ctl(r)}
        if kinds - {btun}:
            return True
    return False


def draw_family_panel(ax, blab, f, rows, YBOT, YTOP, ms=4.8, fs_title=8.0, fs_stat=6.8,
                      fs_tick=6.5, title_pad=11.0, stat_2line=False, xbins=5, qual_in_stat=False,
                      xci=None, stat_fn=None, show_title=True, show_ytick=True, xnull=0.0, xfmt=None, ghosts=()):
    """One family's excess-against-rate panel. Filled circles joined in rate order, open circles for
    a tuning that differs from the block, open green circles for the aligned controls, the family
    name and its two coefficients as a label row above the frame."""
    col = FAMC[f]
    main = sorted([r for r in rows if "mismatch" not in r["o"]], key=lambda r: YM.value(r, f))   # aligned controls count at 0 % (user, 2026-09-11); the derailed one has no rate
    btun = "full" if _is_full_block(blab, f) else "lora"
    pt = lambda r: "lora" if r["o"] in RANK1 else tuning_of(f, r["o"])
    same = [r for r in main if pt(r) == btun]; other = [r for r in main if pt(r) != btun]
    # A BLOCK OF TWO TUNINGS IS ONE TREND (Gemma: three full fine-tunes and two LoRAs, one block of five by the
    # author's rule of 2026-09-18). The line joins all of them; the open markers still say which are LoRAs.
    _line = main if (f in FULL_FAM and blab == f and other) else same
    if len(_line) > 1:
        ax.plot([r["x"] for r in _line], [YM.value(r, f) for r in _line], color=col, lw=1.1, alpha=0.6, zorder=2)
    # A MIXED BLOCK NAMES EACH POINT'S TUNING BY SHAPE: a filled circle is a full fine-tune, a filled
    # triangle a LoRA. An open ring beside a small dot read as one mark at two sizes (author, 2026-09-18:
    # "distinguish ft from lora points here"). A single-tuning block keeps its filled circles.
    if other or SHAPE_ALL:
        for r in main:
            _lora = pt(r) == "lora"
            ax.plot(r["x"], YM.value(r, f), TUNE_MARK["lora" if _lora else "full"], color=col,
                    ms=ms * (1.25 if _lora else 1.0), zorder=3, markeredgecolor='white', markeredgewidth=0.6)
    else:
        for r in same:
            ax.plot(r["x"], YM.value(r, f), 'o', color=col, ms=ms, zorder=3, markeredgecolor='white', markeredgewidth=0.6)
    for r in [r for r in rows if is_ctl(r) and "mismatch" not in r["o"]]:
        # OPEN, and it has to survive at print size: at ms - 1.0 with a 1 pt edge the ring covered
        # its own white face and the control printed as a green disc, which is what the caption
        # calls open (audit, 2026-09-17). A 4 pt circle with a 0.7 pt ring keeps a white centre.
        ax.plot(r["x"], YM.value(r, f), 'o', mfc='white', mec='#2f9e6b', mew=0.7, ms=ms + 1.6, zorder=4)
    for r in ghosts:   # out of the pool: grey, no interval, no line, not in n (author, 2026-09-17)
        if np.isfinite(r["x"]) and YBOT < YM.value(r, f) < YTOP:
            _gs = GHOST_STYLE[ghost_kind(f, r["o"])]
            # a PEARL DASHED border, smaller than the pool's points (author, 2026-09-18: "make the grey point have
            # pearl (not background white) border and be dashed", "and smaller")
            ax.scatter([r["x"]], [YM.value(r, f)], s=max((0.9 * ms) ** 2, 4.0), facecolors=_gs["face"], edgecolors=PEARL,
                       linewidths=0.6, linestyle=(0, (1.4, 1.0)), zorder=2.5)
    for r in main:
        draw_uncertainty(ax, r, f, '#2f9e6b' if is_ctl(r) else col, xci=xci)
        if not (YBOT < YM.value(r, f) < YTOP):
            raise SystemExit(f"draw_family_panel: {f} {r['o']} is at {YM.value(r, f):.3g}, outside "
                             f"the drawn rate axis ({YBOT:.0f}, {YTOP:.0f}).")
    ax.axvline(0, color='#777', lw=0.6, zorder=1)
    # The family's own BASELINE: its judged behaviour against the reading its cloud gives when it is
    # read against itself, which is zero for the five axis readings and one for the kNN retention.
    # It fixes where "no fine-tune" sits in every panel (author, 2026-09-17).
    _yb = YM.base(f)
    if _yb is not None and np.isfinite(_yb):
        ax.plot(xnull, _yb, marker='s', mfc='white', mec='#444', mew=0.9, ms=ms - 0.6, zorder=5,
                linestyle='none')
    ax.grid(True, color='#888', alpha=0.20, lw=0.45); ax.set_axisbelow(True)
    # each family gets its own x range (user, 2026-09-11), always reaching zero so the sign reads
    _xci = xci or excess_ci
    xs = [r["x"] for r in rows if np.isfinite(r["x"])] + [r["x"] for r in ghosts if np.isfinite(r["x"])]
    xs += [v for r in rows if r.get("n") for v in _xci(f, r["o"]) if np.isfinite(v)]
    _xn = [xnull] if np.isfinite(xnull) else []   # a reading with no baseline value (the base loss) draws no marker
    lo, hi = min(xs + _xn), max(_xn + xs); pad = 0.12 * max(hi - lo, 0.05)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(YBOT, YTOP)   # the floor clears a 0 % marker's own radius
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5], min_n_ticks=2))   # per-family y range (author, 2026-09-17)
    ax.tick_params(axis='y', labelsize=fs_tick, pad=1.5); ax.tick_params(axis='x', labelsize=fs_tick, pad=1.5)
    # TWO x ticks, never one: a lone tick gives the reader neither an origin nor a spacing, so the
    # panel cannot be read at all. nbins is what stops the ticks running together (audit, 2026-09-17).
    ax._xnbins = xbins
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=xbins, steps=[1, 2, 5], min_n_ticks=2))
    if xfmt:   # one precision per reading, so sibling panels do not print 0.1 against 0.10
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_tickfmt(xfmt)))
        if max(abs(YBOT), abs(YTOP)) <= 1.5:
            # the rate axis prints its numbers the way the strip's row labels print the same rate;
            # a percent axis keeps its integers
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_tickfmt("%.2f")))
    ax.spines[['top', 'right']].set_visible(False)
    # The family and its statistic read as a label row above the panel. Inside the axes the
    # statistic crossed the zero rule and the name printed over a data point.
    _nm = blab.replace(RANK1_SUFFIX, " (rank-1)").replace(" (LoRA)", "")   # the strip beside names the LoRAs; the panel is too narrow
    _qual = ""
    if qual_in_stat and " (" in _nm:
        # A panel under an inch wide cannot hold "Qwen2.5-14B (rank-1)" on one line, and a wrapped
        # title costs a whole row of the header. The qualifier rides on the statistics line instead.
        _nm, _q = _nm.split(" (", 1)
        _qual = _q.rstrip(")") + "   "
    # Kept on the axes because a loc='left' title is ax._left_title, not ax.title, and a width guard
    # that reads ax.title therefore measures an empty string and can never fire (found 2026-09-14).
    ax._hdr_title = ax.set_title(_nm if show_title else "", loc='left', fontsize=fs_title,
                                 fontweight='bold', color=col, pad=title_pad)
    # n once per row, after the family name (review 2026-09-18: six copies of one n cost a header line per panel)
    if show_title and _nm:
        ax._hdr_n = ax.annotate(f"  $n$ = {len(main)}", xy=(1.0, 0.0), xycoords=ax._hdr_title, ha='left', va='bottom',
                                fontsize=fs_stat, color='#555')
        ax._hdr_n._no_band_pad = True   # it sits over the band's second column, never at the band's edge
    if not show_ytick:
        ax.set_yticklabels([])
    # The statistic must be the BLOCK's, not the family's. Hardcoding "lora" printed the LoRA
    # numbers above Qwen2.5-14B's rank-1 and full panels, which are different organisms
    # entirely (2026-09-13).
    _grp = ("rank1" if blab.endswith(RANK1_SUFFIX)
            else "full" if _is_full_block(blab, f) else "lora")
    # a stat function flagged by_rows reads THIS block's rows: two blocks of one family (Qwen2.5-14B full and
    # rank-1 in Appendix C) would otherwise both print the first block's numbers (2026-09-18)
    _st = stat_fn(rows, f) if getattr(stat_fn, "by_rows", False) else (stat_fn or fam_stat)(_grp, f)
    if stat_2line:   # a narrow panel cannot hold both coefficients on one line
        _st = _st.replace("  $r$", "\n$r$")
    if _qual:
        _st = _qual + _st
    # ONE TEXT PER LINE, so a line can carry its own weight: a coefficient line, Spearman or Pearson, prints bold
    # when its own two-sided p is below .05 (author, 2026-09-18: "bold on p alone"; 2026-09-19: "also bold by r
    # significance"). Lines are stacked from the bottom at the old 1.25 line spacing, so the block sits where the
    # single text used to.
    import matplotlib.transforms as _mt
    _lines = _st.split("\n")
    for _k, _ln in enumerate(reversed(_lines)):
        _bold = _strong_stat(_ln)
        if _bold:
            _ln = _ln.replace("$\\rho$", "$\\boldsymbol{\\rho}$").replace("$r$", "$\\boldsymbol{r}$")
        _tr = _mt.offset_copy(ax.transAxes, fig=ax.figure, x=0, y=_k * fs_stat * 1.25, units='points')
        ax.text(0.0, 1.02, _ln, transform=_tr, ha='left', va='bottom', fontsize=fs_stat,
                color='#222' if _bold else '#555', fontweight='bold' if _bold else 'normal')
    return ax


def _strong_stat(line):
    """True for a coefficient line, Spearman or Pearson, whose own two-sided p is below .05 (2026-09-19)."""
    import re as _re
    m = _re.match(r"^\$(?:\\rho|r)\$\s*=?\s*([+\u2212-]?)(\d*\.\d+)\s*\((?:\$p\\,\$)?(<?)(\d*\.\d+)\)", line)
    if not m:
        return False
    return m.group(3) == "<" or float(m.group(4)) < 0.05


def draw_slice_stack(fig, spec, blocks, title, hspace=0.0, ms=4.8, fs_title=8.0, fs_stat=6.8,
                     fs_tick=6.5, fs_xlab=8.0, fs_head=9.0, ylab_x=0.024, title_pad=11.0,
                     head_pad=None, stat_2line=False, show_ylab=True, xbins=5, floor=3,
                     xlab="assistant excess", qual_in_stat=False, xci=None, stat_fn=None,
                     show_title=True, show_ytick=True, xnull=0.0, xfmt=None, equal_heights=False, ghosts=None,
                     heights_in=None):
    """The left column of the best-slice figure as one small panel per family (user's design,
    2026-09-11), stacked in the block order of the right panel with heights matched to the number of
    rows, so each family's scatter sits beside its own cast strips. Conventions follow draw_slice_panel:
    filled circles joined in rate order, open circles for a tuning that differs from the block,
    open green circles for the aligned controls.

    equal_heights gives every family the same panel. The row count belongs to the cast strip on the
    far right, not to a scatter, and drawing the six panels at three heights made the grid ragged
    and changed the vertical scale from row to row for no reason a reader can see (audit,
    2026-09-17)."""
    heights = (list(heights_in) if heights_in is not None else [1.0] * len(blocks) if equal_heights
               else [max(len(rows), floor) + 0.8 for _, _, rows in blocks])   # a short panel still needs height for its rate interval
    # The rate axis follows the data. Fixed at (-7, 47) it cut off Qwen2.5-32B's obviously-bad-medical
    # at 58 %, the most misaligned organism in the study and the one the Results discuss by name:
    # its marker was outside the panel and its line ran off the top edge (found 2026-09-14).
    inner = _gridspec.GridSpecFromSubplotSpec(len(blocks), 1, subplot_spec=spec, height_ratios=heights, hspace=hspace)
    axes = []
    for k, (blab, f, rows) in enumerate(blocks):
        ax = fig.add_subplot(inner[k]); axes.append(ax)
        # EACH FAMILY ITS OWN y RANGE (author, 2026-09-17: "some of these can get better y-axes"): the
        # family's largest value plus headroom for its interval, never below a fifth of the measure's
        # scale, and the floor below zero that clears a 0 marker; the x range was per family already
        _vals = [v for v in (YM.value(r, f) for r in rows if r.get("em") is not None) if np.isfinite(v)]
        _bnd = [YM.bounds(r, f) for r in rows if r.get("n")]
        _yb = YM.base(f)
        _vals += [_yb] if (_yb is not None and np.isfinite(_yb)) else []
        _lo = min(_vals + [b[0] for b in _bnd if np.isfinite(b[0])]); _hi = max(_vals + [b[1] for b in _bnd if np.isfinite(b[1])])
        _span = max(_hi - _lo, 0.15 * YM.pct); _pad = 0.12 * _span
        YTOP = _hi + _pad
        # anchored at zero when the family reaches down to it (a control at 0 needs the marker's own
        # radius below the axis); otherwise the range follows the data, so a coherence that sits
        # between 0.7 and 1.0 is not drawn as a flat line on a 0-to-1 axis (author, 2026-09-17)
        YBOT = -YM.floor * YTOP if _lo < 0.5 * _span else _lo - _pad
        draw_family_panel(ax, blab, f, rows, YBOT, YTOP, ghosts=(ghosts or {}).get(f, ()), ms=ms, fs_title=fs_title, fs_stat=fs_stat,
                          fs_tick=fs_tick, title_pad=title_pad, stat_2line=stat_2line, xbins=xbins,
                          qual_in_stat=qual_in_stat, xci=xci, stat_fn=stat_fn,
                          show_title=show_title, show_ytick=show_ytick, xnull=xnull, xfmt=xfmt)
        if k == len(blocks) - 1:
            # Short. The definition of xi belongs in the caption: set on the axis it was 1.75 in
            # of type under a 1.30 in panel, and it ran out over both neighbours.
            ax.set_xlabel(xlab, fontsize=fs_xlab, labelpad=3)
    # The two pooled lines sit in figure coordinates at fixed inch offsets from the top, aligned with
    # the left column. Spaced in axes fractions of the shortest panel they overprinted each other, and
    # joined into one line they run off the printed width.
    H = fig.get_figheight(); x0 = axes[0].get_position().x0
    # The column head sits above the family titles of the stacked panels, never on the first panel:
    # that slot now carries the family name.
    if title:
        # CENTRED over its own column. Set flush left the widest head ran a fifth of an inch past
        # the last panel of its group and printed on the white gutter beside the band, and the two
        # heads of the middle group nearly touched (audit, 2026-09-17).
        _p0 = axes[0].get_position()
        _h = fig.text((_p0.x0 + _p0.x1) / 2, _p0.y1 + (head_pad or STACK_HEAD_PAD) / 72 / H, title,
                      fontsize=fs_head, fontweight='bold', color='#333', ha='center', va='bottom')
        _h._colhead = True      # the band measurement below has to see it; the headline is not one
        _h._col_left = _p0.x0 * fig.get_figwidth()
    # The key for the per-panel pair lives in the caption, not on the canvas.
    # the pooled statistics are in the caption (user, 2026-09-11: no repetition)
    if show_ylab:
        _t = fig.text(ylab_x, (axes[-1].get_position().y0 + axes[0].get_position().y1) / 2,
                      YM.label, rotation=90, va='center', ha='center', fontsize=fs_xlab)
        # MEASURED, not offset: the title stands clear of the tick labels it is set beside. Placed
        # at a fixed inch offset it printed over the 0.4 and the 0.2 of the rate axis (audit,
        # 2026-09-17).
        fig.canvas.draw(); _r = fig.canvas.get_renderer()
        # only the ticks the panel actually draws: the locator keeps labels for ticks outside the
        # view, and one of those reached further left than anything on the page
        _tk = [t.get_window_extent(_r).x0 / fig.dpi for a in axes
               for v, t in zip(a.get_yticks(), a.get_yticklabels())
               if t.get_text() and a.get_ylim()[0] <= v <= a.get_ylim()[1]]
        if _tk:
            _w = _t.get_window_extent(_r).width / fig.dpi
            _t.set_x((min(_tk) - 0.06 - _w / 2) / fig.get_figwidth())   # 0.06: clear of the band edge that sits 0.02 left of the ticks (2026-09-17)
    return axes


FEAT_LABEL = {   # drawn feature -> (panel title, x label)
    "zoo:onaxis_shift_sd_over_base_sd": ("the same organisms:\nspread of the cast's on-axis shifts",
                                         "sd of the roles' on-axis shifts, over the baseline's on-axis sd (persona layer)"),
    "zoo:resid_mean_pairwise_cos": ("residual-shift\ncoherence", "mean pairwise cos,\nresidual shifts"),
    "sweep:default|relmag|none|mean": ("default's shift size\nagainst the cast", "default |shift| /\nmean |shift|, minus 1"),
}


GEOM_FILE = {'Qwen2.5-7B': 'geom_Qwen2_5-7B', 'Qwen2.5-14B': 'geom_Qwen2_5-14B', 'Qwen2.5-32B': 'geom_Qwen2_5-32B',
             'Qwen3-32B': 'geom_Qwen3-32B', 'Llama-3.1-8B': 'geom_Llama-3_1-8B', 'Gemma-3-12B': 'geom_Gemma-3-12B',
             'Mistral-24B': 'geom_Mistral-24B', 'Qwen3.5-9B': 'geom_Qwen3_5-9B'}
_GEOM = {}


def cast_excess(f, o, depth="50%"):
    """Every role's excess for one organism (the data of fig_allroles.py): along-axis drift minus the
    cast median, at the persona layer. Returns (others, other names, assistant excess, default excess)."""
    if f not in _GEOM:
        _GEOM[f] = json.load(open(_paths.geom(GEOM_FILE[f])))
    g = _GEOM[f]; names = g["personas"]; ai = names.index("assistant"); di = names.index("default") if "default" in names else None
    al = np.array(g["layers"][depth]["organisms"][o]["feat"]["along"], float); exc = al - np.median(al)
    drop = [ai] + ([di] if di is not None else [])
    others = np.delete(exc, drop); onames = [names[i] for i in range(len(names)) if i not in drop]
    return others, onames, float(exc[ai]), (float(exc[di]) if di is not None else None)


def _wdata(ax, txt, fs):
    """Width of a label in data units, from the renderer, so names never print on top of each other."""
    t = ax.text(0, 0, txt, fontsize=fs); ax.figure.canvas.draw()
    bb = t.get_window_extent(ax.figure.canvas.get_renderer()); t.remove()
    p0 = ax.transData.inverted().transform((0, 0)); p1 = ax.transData.inverted().transform((bb.width, 0))
    return abs(p1[0] - p0[0])


EVILC = "#d6006e"       # the evil roles; no family uses this hue
_EVIL = {}


def evil_idx(f, onames):
    """Which of a cast's other roles are misaligned in character, as a mask over `onames`."""
    if f not in _EVIL:
        import features_graph as _FG
        _EVIL[f] = set(_FG.evil_group(onames, _FG.evil_rates()[0]))
    g = _EVIL[f]
    return np.array([i in g for i in range(len(onames))], bool)


def draw_cast_strips(ax, blocks, rows_y, ms_star=7.0, ms_diam=4.2, s_dot=3.0, counter=frozenset(),
                     evil=False, ghost=frozenset(), pitch=1.0):
    """The right panel of the best-slice figure (user's design, 2026-09-11): for every row of the left
    scatter, the whole cast's excess as grey points, the assistant as a star, the default as a diamond,
    and three named roles (most detached, median, least detached), as in fig_allroles.py."""
    rng = np.random.default_rng(0)
    data = {}
    for blab, f, rows in blocks:
        for r in rows:
            data[(f, r["o"])] = cast_excess(f, r["o"])
    allv = [v for d in data.values() for v in list(d[0]) + [d[2]] + ([d[3]] if d[3] is not None else [])]
    lo, hi = min(allv), max(allv); rg = hi - lo
    # The left pad held the in-strip role names until they were removed on 2026-09-11. Empty, it was
    # 17% of the axis, and at the printed width that width belongs to the row labels.
    # The pad also has to clear the widest mark's own radius. At 5 % of the range the leftmost
    # default rings were cut in half by the strip's left spine and the stars inside them lost their
    # left point, so the reader could not see where those organisms sat (audit, 2026-09-17). The
    # radius is measured in points against the drawn width of this axes, not guessed.
    _w_in = ax.get_position().width * ax.figure.get_figwidth()
    _rad_in = (ms_star * 1.45 / 2 + 1.0) / 72.0            # the default ring, the widest mark, plus its edge
    _span = rg / max(1.0 - 2 * _rad_in / _w_in, 0.5)       # the range that leaves both rings whole
    pad = max(0.5 * (_span - rg), 0.05 * rg)
    XL, XR = lo - pad, hi + pad
    ax.set_xlim(XL, XR)
    for blab, f, rows in blocks:
        for r in rows:
            y = rows_y[(f, r["o"])]; ctl = is_ctl(r)   # a zero-rate organism is not a control (author, 2026-09-16)
            others, onames, aexc, dexc = data[(f, r["o"])]
            g = (f, r["o"]) in ghost   # out of the pool: every mark grey, no band, no line (author, 2026-09-17)
            if g:
                pass
            elif ctl:   # a control or a zero-rate organism sits on a green band, apart from the misaligned rows
                ax.axhspan(y - 0.5 * pitch, y + 0.5 * pitch, color="#dff1e6", zorder=0, lw=0)
            elif (f, r["o"]) in counter:   # outside the pool: a blue-grey band
                ax.axhspan(y - 0.5 * pitch, y + 0.5 * pitch, color="#e8edf5", zorder=0, lw=0)
            jit = (rng.random(len(others)) - 0.5) * 0.30 * pitch
            ax.scatter(others, np.full(len(others), y) + jit, s=s_dot, c="#b4b4b4", alpha=0.55, linewidths=0, zorder=1)
            if evil and not g:
                # This figure carries two readings, and the second is the gap from the evil roles'
                # mean to the assistant, so both are marked on one strip (author, 2026-09-16).
                _E = evil_idx(f, onames)
                if _E.any():
                    _o = np.asarray(others, float)
                    ax.scatter(_o[_E], np.full(int(_E.sum()), y) + jit[_E], s=s_dot * 1.9,
                               c=EVILC, alpha=0.9, linewidths=0, zorder=2)
                    _em = float(_o[_E].mean())
                    ax.plot([_em, aexc], [y, y], '-', color=EVILC, lw=1.2, alpha=0.14, zorder=2.5)   # 0.40 read as a solid bar (author, 2026-09-18)
                    ax.plot([_em], [y], '|', color=EVILC, ms=4.6, mew=1.2, zorder=5)   # under the row pitch (review 2026-09-18)
            # no role names in the strips (user, 2026-09-11: the names read as clutter at print size)
            _gs = STRIP_GHOST if g else None
            mk = _gs["face"] if g else ('#2f9e6b' if ctl else '#f2c200')
            # The default is drawn over the assistant. In 14 of the 19 rows the two fall inside the
            # star's own footprint, so under the star the diamond the legend keys was invisible.
            ax.plot(aexc, y, '*', ms=ms_star, mfc=mk, mec=(_gs["edge"] if g else '#111'), mew=0.6, zorder=4)
            if dexc is not None: ax.plot(dexc, y, 'o', ms=ms_star * 1.45, mfc='none', mec=(_gs["edge"] if g else '#8a6d00'), mew=0.9, zorder=3)
        # One line through the family's stars and one through its diamonds, top to bottom, so the
        # detachment reads as a slope against the rate labels (user, 2026-09-14). The topic-locked
        # row has no rate and stays off the line.
        _pts = sorted((rows_y[(f, r["o"])], data[(f, r["o"])][2], data[(f, r["o"])][3])
                      for r in rows if "mismatch" not in r["o"] and (f, r["o"]) not in ghost)
        if len(_pts) >= 2:
            ax.plot([p[1] for p in _pts], [p[0] for p in _pts], '-', color='#b08900', lw=0.9,
                    alpha=0.9, zorder=3)
            _dp = [p for p in _pts if p[2] is not None]
            if len(_dp) >= 2:
                ax.plot([p[2] for p in _dp], [p[0] for p in _dp], ':', color='#333', lw=0.8,
                        alpha=0.8, zorder=3)
    ax.plot([0, 0], [min(rows_y.values()) - 0.55 * pitch, 1e4], color='#777', lw=0.6, zorder=2, clip_on=True)


CACHE = _paths.SWEEP_CACHE     # the raw per-role vectors behind zoo_cast.json
ZOO_DEPTHS = ['10%', '25%', '50%', '70%', '90%']


def draw_contraction(ax, fam, org, depth, title, color):
    """One organism's cast on the base default axis: each role's on-axis shift s_i = y_i - x_i against
    its baseline position x_i, the least-squares line of y on x (the cast's contraction, slope b) and
    the assistant's residual from it. This is the quantity build_zoo_cast.py studentizes into
    assistant_resid_t, the feature that separates the two slices."""
    k = ZOO_DEPTHS.index(depth)
    zb = np.load(CACHE / f"{fam}__base.npz", allow_pickle=True); zo = np.load(CACHE / f"{fam}__{org}.npz", allow_pickle=True)
    names = list(zb["names"]); ai = names.index("assistant"); di = names.index("default")
    a = zb["axis"][k].astype(np.float64); a /= np.linalg.norm(a)
    x = zb["X"][k].astype(np.float64) @ a; yv = zo["X"][k].astype(np.float64) @ a; sft = yv - x
    xc = x - x.mean(); b = float((xc * (yv - yv.mean())).sum() / (xc ** 2).sum()); c = float(yv.mean() - b * x.mean())
    line = lambda t: (b - 1.0) * t + c
    xx = np.linspace(x.min(), x.max(), 20)
    padx = 0.10 * (x.max() - x.min()); ax.set_xlim(x.min() - padx, x.max() + 1.6 * padx)   # the star sits inside the frame
    ax.plot(x, sft, 'o', ms=2.6, color='#b5b5b5', mec='none', zorder=2)
    ax.plot(xx, line(xx), color='#333', lw=0.9, zorder=3)
    ax.plot([x[ai], x[ai]], [line(x[ai]), sft[ai]], color='#c0392b', lw=1.4, zorder=4)   # the residual
    ax.plot(x[di], sft[di], 'D', ms=4.5, mfc='#f2c200', mec='#111', mew=0.7, zorder=5)
    ax.plot(x[ai], sft[ai], '*', ms=9, mfc='#f2c200', mec='#111', mew=0.7, zorder=6)
    ax.set_title(title, fontsize=7.4, color=color, fontweight='bold', loc='left', pad=3)
    ax.tick_params(labelsize=6); ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color='#888', alpha=0.2, lw=0.5); ax.set_axisbelow(True)
    return float(sft[ai] - line(x[ai]))


def slice_figures():
    """Left: excess against rate (draw_slice_panel). Right, best slice: every role's excess for the same
    rows (cast strips). Right, counter slice: the strongest features of that slice
    (blog/data/slice_features.json), read at the persona layer."""
    SF = json.load(open(RAW / "slice_features.json")); LAYER = SF["persona_layer"]
    panels = split([r for r in EXC if r["em"] is not None], mode="slice")
    for p in panels: panels[p] = split_rank1(panels[p])
    # the out-of-pool organisms of the pooled families: rank-1, full, distilled, general-purpose
    GHOST_ROWS.clear()
    for f in SLICE_FAMS:
        GHOST_ROWS[f] = [r for _b, _f, rs in panels["counter"] for r in rs if _f == f and "mismatch" not in r["o"]] \
                      + [r for r in EXC if r["em"] is not None and ((r["f"], r["o"]) in GENERAL or (r["f"], r["o"]) in DISTILL) and r["f"] == f]
    # Mistral-24B left the pool (author, 2026-09-18: "our official pool is Qwen Lora"); its general-purpose
    # LoRA stays a grey mark under it in the main figure, as it was while Mistral was pooled
    GHOST_ROWS["Mistral-24B"] = [r for r in EXC if r["em"] is not None and (r["f"], r["o"]) in GENERAL and r["f"] == "Mistral-24B"]
    # Scatter on the left, the features that separate the two slices on the right (user asked for
    # the side panels back, 2026-09-12). The lollipop is the AUC of each feature at its own depth,
    # which is scale-free: the features themselves run from 0.005 to -3.8 and cannot share an axis.
    # The dashed line is what the best of 270 searched cells reaches by chance, so a bar past it is
    # a separation the search cannot manufacture.
    # One figure holds both slices: the pooled families in the left column of panels, the four
    # blocks outside the pool in a second column, every row in the strips, the outside rows
    # tinted (user, 2026-09-15). The separate counter figure left the paper the same day.
    # the table first: the figure routines below share the row records, and the table must see them
    # as split() built them (2026-09-17)
    counter_table(split_full_fam_lora(split_rank1(panels["counter"])))   # the counter figure became a table (author, 2026-09-16); Gemma's LoRA pair is its own block (2026-09-17)
    # Llama-3.1-8B, outside the pool, as the last row (author, 2026-09-18: "add llama as last result");
    # its panels carry only its own within-family statistics, and no pooled number reads it
    # Gemma-3-12B before it, its two LoRAs drawn as organisms and its three full fine-tunes greyed out of
    # the pool (author, 2026-09-18: "also add gemma (with full ft grey out)")
    _gem = [r for _b, _f, rs in panels["counter"] if _f == "Gemma-3-12B" for r in rs if "mismatch" not in r["o"]]
    _gem_lora = [r for r in _gem if tuning_of("Gemma-3-12B", r["o"]) == "lora"]
    GHOST_ROWS["Gemma-3-12B"] = [r for r in _gem if tuning_of("Gemma-3-12B", r["o"]) != "lora"]
    assert len(_gem_lora) >= 2 and len(GHOST_ROWS["Gemma-3-12B"]) == 3, (len(_gem_lora), len(GHOST_ROWS["Gemma-3-12B"]))   # a third LoRA organism is landing (2026-09-18)
    # FIG_REST_ONLY=1 redraws only what follows the pool's own grids (a long run's later figures, 2026-09-18)
    _rest = os.environ.get("FIG_REST_ONLY") == "1"
    if not _rest: main_figure(panels["main"] + [b for b in panels["counter"] if b[0] == "Mistral-24B"]
                + [b for b in panels["counter"] if b[0] == "Llama-3.1-8B"]
                + [("Gemma-3-12B (LoRA)", "Gemma-3-12B", _gem_lora)])   # Qwen, Mistral, Llama, Gemma (author, 2026-09-18)
    del GHOST_ROWS["Gemma-3-12B"]
    if os.environ.get("FIG_MAIN_ONLY") == "1":   # Figure 4 and Table 2 alone, for a quick refresh between landings (2026-09-18)
        return
    if not _rest: appendix_grids(panels["main"], headline="In the pool: the six readings against")   # pairs with the grid outside the pool
    # Appendix C: the metrics of Aneja et al. against G, the pool and then the organisms outside it
    _G = (("G", _MB(), 1, "the general misalignment rate $G$"),)
    _skip_ig = os.environ.get("FIG_SKIP_IG") == "1"   # the IG grids alone are fresh; redraw what follows them (2026-09-19)
    if not _skip_ig: appendix_grids(panels["main"], tex_rel="sections/appendices/app_ig.tex", prefix="ig_grid",
                   headline="In the pool: the metrics of Aneja et al. against", reads=IG_READS, measures=_G,
                   groups=IG_GROUPS, row_in=IG_ROW_IN)
    if not _skip_ig: appendix_grids(split_full_fam_lora(split_rank1(panels["counter"])), tex_rel="sections/appendices/app_ig.tex",
                   prefix="ig_counter_grid", headline="Outside the pool: the metrics of Aneja et al. against",
                   by_rows=True, reads=IG_READS, measures=_G, groups=IG_GROUPS, row_in=0.058)
    # Appendix D: the base loss beside the six readings, against M, B and C, the pool and then the organisms outside it
    appendix_grids(loss_blocks(panels["main"]), tex_rel="sections/appendices/app_loss.tex", prefix="loss_grid",
                   headline="In the pool: the base loss and the six readings against", reads=LOSS_READS,
                   groups=LOSS_GROUPS, row_in=LOSS_ROW_IN)
    appendix_grids(loss_blocks(split_full_fam_lora(split_rank1(panels["counter"]))), tex_rel="sections/appendices/app_loss.tex",
                   prefix="loss_counter_grid", headline="Outside the pool: the base loss and the six readings against",
                   by_rows=True, reads=LOSS_READS, groups=LOSS_GROUPS, row_in=0.058)
    # Appendix B's second grid per measure: the organisms outside the pool, read as the first grid reads the pool
    # (author, 2026-09-18; the old Appendix C, merged into B the same day: "Appendix B and C should be merged")
    appendix_grids(split_full_fam_lora(split_rank1(panels["counter"])), tex_rel="sections/appendices/app_behaviour.tex",
                   prefix="counter_readings", headline="Outside the pool: the six readings against", by_rows=True,
                   row_in=0.058)   # five blocks since Mistral left the pool (2026-09-18; 0.095 pushed Table 6 off its page); four blocks: taller panels fill the page Appendix B's six blocks fill (5.3 in left a 32% white band)


def counter_figure(blocks, SF):
    """The Discussion figure: the families the pool leaves out, and the feature that separates them.

    AUTHORED AT ITS PRINTED WIDTH, the full 5.5 in text block. It used to print 2.42 in wide inside a
    wrapfigure, and at 4.34 in tall it did not fit the page it started on, so its narrow text column
    ran past the page break and squeezed the first lines of the next subsection (user, 2026-09-14).
    A figure that text wraps around cannot be taller than the space left on its page, so this one is
    a plain float: the four family panels in one row, the separation panel under them.
    """
    W = tex_printed_width("sections/discussion.tex", "excess_slice_counter", container=1.0)
    LEFT, RIGHT, COLGAP = 0.40, 0.16, 0.30
    PW = (W - LEFT - RIGHT - 3 * COLGAP) / 4                 # four family panels in one row
    # HEAD is measured: 8 pt of tick labels from the panel above, 6 pt of air, a 6.4 pt family name
    # and two 5.6 pt statistics lines. At 0.36 in the name printed on the tick row above it.
    HEAD, PH, ROWGAP, XLAB = 0.52, 0.62, 0.04, 0.24          # inches inside the single grid row
    TOPBAND = 0.34                                           # the headline
    GRID = HEAD + PH + ROWGAP + XLAB
    AUCHEAD, AUCROW, AUCLAB, GAP2 = 0.22, 0.200, 0.24, 0.10   # a row holds a name over its depth
    top = SF["separation"]["top"][:6][::-1]                  # strongest at the top of the panel
    AUC = AUCHEAD + AUCROW * len(top) + AUCLAB
    H = TOPBAND + GRID + GAP2 + AUC + 0.10
    fig = plt.figure(figsize=(W, H))
    fig.text(0.07 / W, 1 - 0.045 / H, "Outside the pool the excess stops tracking the rate",
             fontsize=8.4, fontweight="bold", color="#222", ha="left", va="top")
    _rmax = max(r["em"] * 100 for _, _, rows in blocks for r in rows if r.get("em") is not None)
    YTOP = max(47.0, _rmax + 6.0); YBOT = -0.148 * YTOP
    axes = []
    for k, (blab, f, rows) in enumerate(blocks):
        cx, cy = k % 4, 0
        x0 = (LEFT + cx * (PW + COLGAP)) / W
        ytop_in = TOPBAND + HEAD
        ax = fig.add_axes((x0, (H - ytop_in - PH) / H, PW / W, PH / H)); axes.append(ax)
        draw_family_panel(ax, blab, f, rows, YBOT, YTOP, ms=3.2, fs_title=6.4, fs_stat=5.6,
                          fs_tick=5.8, title_pad=17.0, stat_2line=True, xbins=3, qual_in_stat=True)
        ax.set_xlabel("assistant excess  $\\xi$", fontsize=6.6, labelpad=2)
    fig.text(0.030, (H - TOPBAND - HEAD - PH / 2) / H, "misalignment rate (%)", rotation=90,
             va='center', ha='center', fontsize=6.4)

    # ---- the feature that separates the two slices
    ax0 = fig.add_axes((LEFT / W, (H - TOPBAND - GRID - GAP2 - AUCHEAD - AUCROW * len(top)) / H,
                        (W - LEFT - RIGHT) / W, AUCROW * len(top) / H))
    NICE = {"assistant_resid_t": "assistant's residual, studentized",
            "assistant_resid_raw": "assistant's residual from the line",
            "cross_frac_assistant": "roles the assistant crossed",
            "cross_frac_default": "roles the default crossed",
            "assistant_rank_shift": "assistant's rank shift on the axis",
            "default_resid_t": "default's residual, studentized",
            "default_rank_shift": "default's rank shift on the axis",
            "assistant_onaxis_shift": "assistant's on-axis shift",
            "default_onaxis_shift": "default's on-axis shift",
            # the wording appendix_organism_clusters.py already uses for this feature
            "angle_axis_resid_top_deg": "angle, axis to leading residual"}
    q95 = SF["separation"]["null_max_auc_q95"]
    ys = np.arange(len(top))
    _names = []
    for y, t in zip(ys, top):
        col = "#c0392b" if t["auc"] >= q95 else "#9aa0a6"
        ax0.plot([0.5, t["auc"]], [y, y], color=col, lw=1.3, zorder=2, solid_capstyle="round")
        ax0.plot(t["auc"], y, "o", ms=4.0, mfc=col, mec="#111", mew=0.5, zorder=3)
        # The value sits in one right-aligned column, clear of the null rule. Set beside its own dot
        # it printed across the rule and over the rule's label, because every dot is near it.
        # The value sits beside its own dot. In a right-aligned column at the axis edge it was two
        # inches from the bar once this figure went full width (2026-09-14). Near the right edge it
        # flips to the inside of the dot so it cannot run off the panel.
        _in = t["auc"] > 0.93
        ax0.text(t["auc"] + (-0.008 if _in else 0.008), y, f"{t['auc']:.2f}", fontsize=6.0,
                 va="center", ha="right" if _in else "left", color="#333", zorder=5,
                 path_effects=[_pe.withStroke(linewidth=2.0, foreground="white")])
        # One line per row. On two lines the depth of one row printed through the name of the next,
        # because a 5.8 pt line is two fifths of a 0.20 in row and a row has to hold both and a bar.
        key = t["feature"].split(":", 1)[-1]
        _names.append(ax0.text(0.506, y + 0.22,
                               f"{NICE.get(key, key.replace('_', ' '))}  ·  depth {t['depth']}",
                               fontsize=5.6, color="#333", va="bottom", ha="left", zorder=4,
                               path_effects=[_pe.withStroke(linewidth=2.4, foreground="white")]))
    # Lighter, and every label over it carries a white halo. A 0.8 pt rule in #333 read through the
    # thin halo and struck six strings, so "null" printed as "nuIl" (2026-09-14).
    ax0.axvline(q95, color="#777", lw=0.7, ls=(0, (2.6, 2.0)), zorder=1)
    # horizontal, under the rule and to its LEFT: rotated beside it the label ran down through five
    # feature names, set above it it printed on the top row's own name, and centred on it the rule
    # ran straight through the word
    ax0.text(q95 - 0.008, -1.02, "search null, 1 in 20", fontsize=5.4, color="#555", ha="right",
             va="bottom", path_effects=[_pe.withStroke(linewidth=2.0, foreground="white")])
    ax0.set_yticks([])
    ax0.set_xlim(0.5, 1.0); ax0.set_ylim(-1.10, len(top) - 0.18)
    ax0.set_xlabel("AUC separating the pool from the rest", fontsize=6.6, labelpad=2)
    ax0.set_title("What separates them", fontsize=7.4, fontweight="bold", loc="left", color="#333",
                  pad=7)
    # The feature names are set inside the panel, so their type is sized to the panel rather than
    # chosen and hoped for. A new feature with a longer name shrinks the row instead of running off
    # the trim edge, and if it cannot fit above 5 pt the build refuses.
    fig.canvas.draw(); _r = fig.canvas.get_renderer()
    _avail = (W - LEFT - RIGHT) * 0.97
    _wmax = max(t.get_window_extent(_r).width for t in _names) / fig.dpi
    if _wmax > _avail:
        _fs = 5.6 * _avail / _wmax
        if _fs < 5.0:
            raise SystemExit(f"counter_figure: the widest feature name needs {_fs:.1f} pt to fit "
                             f"{_avail:.2f} in, under the 5 pt floor. Shorten the name in NICE.")
        for t in _names:
            t.set_fontsize(_fs)
        print(f"  excess_slice_counter: feature names set at {_fs:.2f} pt to fit {_avail:.2f} in")
    ax0.tick_params(axis="x", labelsize=5.8, pad=1.5); ax0.tick_params(axis="y", length=0)
    ax0.grid(axis="x", color="#888", alpha=0.2, lw=0.45); ax0.set_axisbelow(True)
    ax0.spines[["top", "right", "left"]].set_visible(False)
    _interval_audit(axes, blocks, "excess_slice_counter", ms_pt=3.2)
    assert_canvas(fig, "sections/discussion.tex", "excess_slice_counter", container=1.0)
    outc = OUT / "excess_slice_counter.pdf"
    pngc = str(outc).replace(".pdf", ".png")
    fig.savefig(outc); fig.savefig(pngc, dpi=300); plt.close(fig)
    _edge_guard(pngc)
    print("wrote", outc, f"({len(blocks)} families, canvas {W} x {H:.2f} in)")




# ---------------------------------------------------------------- the counter slice as a grid
# Rows are the judged measures, columns the blocks outside the pool, and every cell is the line
# plot the reading figures draw. One y scale per row, one x scale per column, each drawn once at
# the edge (Krzywinski, Nat. Methods 10:183; Gelman, Pasarica & Dodhia, Am. Stat. 56:121).
CG_COLS = ["Llama-3.1-8B", "Gemma-3-12B", "Qwen2.5-14B (full)", "Qwen2.5-14B (rank-1)"]
# C on its own and RC+(1-C), the share of answers that are misaligned or incoherent, join the
# rows the author tracks (author, 2026-09-16).
CG_ROWS = ["R", "B", "C", "RB", "RC", "RBC", "mean(R,B,C)", "RC+(1-C)", "R/C", "R/B", "RB/C", "Mq", "M*PRq", "M*Hq"]
CG_RATE = {"R", "B", "C", "RB", "RC", "RBC", "mean(R,B,C)", "RC+(1-C)"}
# THE MISALIGNMENT RATE IS M. R is the roleplaying rate, the share of the prompted roles an organism
# still holds in character (author, 2026-09-16). Keys stay as they were; only the labels moved.
CG_LAB = {"Mq": r"$\bar M_q$", "M*PRq": r"$M\,\mathrm{PR}_q$", "M*Hq": r"$M\,H_q$",
          "R": "$M$", "B": "$B$", "C": "$C$", "RB": "$MB$", "RC": "$MC$", "RBC": "$MBC$",
          "mean(R,B,C)": "$\\frac{M+B+C}{3}$", "RC+(1-C)": r"$U$",
          "R/C": "$M/C$", "R/B": "$M/B$", "RB/C": "$MB/C$",
          "roles": "$R$", "MR": "$MR$", "MRC": "$MRC$", "MBRC": "$MBRC$", "MR+(1-R)": r"$MR\!+\!(1\!-\!R)$"}


def _cg_at(R, b):
    """Every row measure at one rate, holding the organism's breadth and coherence fixed."""
    C = float(b["coherence"]) / 100.0; B = float(b["breadth"]) / float(b["n_questions"])
    P = float(b.get("roles", np.nan))
    return {"R": R, "B": B, "C": C, "RB": R * B, "RC": R * C, "RBC": R * B * C,
            "mean(R,B,C)": (R + B + C) / 3.0, "RC+(1-C)": R * C + (1 - C),
            "R/C": R / C if C else np.nan, "R/B": R / B if B else np.nan,
            "RB/C": R * B / C if C else np.nan,
            "roles": P, "MR": R * P, "MRC": R * P * C, "MBRC": R * B * P * C, "MR+(1-R)": R * P + (1 - P),
            "Mq": float(b.get("Mq", np.nan)) if R == float(b["em"]) else np.nan, "M*PRq": R * float(b.get("PRq", np.nan)), "M*Hq": R * float(b.get("Hq", np.nan))}


def _cg_rho(x, y):
    from scipy.stats import rankdata as _rk
    rx, ry = _rk(np.asarray(x, float)), _rk(np.asarray(y, float))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float(rx @ ry / den) if den > 0 else np.nan


def _cg_pear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    den = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    return float(xc @ yc / den) if den > 0 else np.nan


def _cg_ticks(lo, hi):
    """Two round ticks inside the panel, so a column never collides with its neighbour."""
    a, b = lo + 0.16 * (hi - lo), hi - 0.16 * (hi - lo)
    t = matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(a, b)
    t = [v for v in t if a <= v <= b]
    return t[:2] if len(t) > 1 else t[:1]


def counter_grid(blocks, name="excess_slice_counter", tex_rel="sections/discussion.tex",
                 headline="Outside the pool the excess stops tracking the rate"):
    beh = {(r["f"], r["o"]): r for r in json.load(open(RAW / "behaviour_bundle.json"))}
    by = {b[0]: b for b in blocks}
    lost = sorted(set(by) - set(CG_COLS))
    if lost:
        raise SystemExit(f"counter_grid: blocks {lost} have data but are missing from CG_COLS.")
    cols = [by[l] for l in CG_COLS if l in by]

    W = tex_printed_width(tex_rel, name)
    LEFT, RIGHT, CGAP = 0.92, 0.10, 0.20   # the widest row label is RC+(1-C)
    PW = (W - LEFT - RIGHT - (len(cols) - 1) * CGAP) / len(cols)
    TOPBAND, COLHEAD = 0.40, 0.24
    PH, RG, BG, BOT = 0.44, 0.09, 0.20, 0.36
    gaps = [(BG if (CG_ROWS[i] in CG_RATE) != (CG_ROWS[i + 1] in CG_RATE) else RG)
            for i in range(len(CG_ROWS) - 1)]
    H = TOPBAND + COLHEAD + len(CG_ROWS) * PH + sum(gaps) + BOT
    fig = plt.figure(figsize=(W, H))

    D = {}
    for blab, f, rows in cols:
        for r in rows:
            xlo, xhi = excess_ci(f, r["o"])
            b = beh[(f, r["o"])]; R = float(b["em"])
            rlo, rhi = wilson(r["k"], r["n"]) if r.get("n") else (np.nan, np.nan)
            v, lo, hi = _cg_at(R, b), _cg_at(rlo, b), _cg_at(rhi, b)
            D[(f, r["o"])] = dict(x=r["x"], xlo=xlo, xhi=xhi, v=v, lo=lo, hi=hi,
                                  ctl=is_ctl(r))
    XL, XT = {}, {}
    for blab, f, rows in cols:
        xs = [D[(f, r["o"])][k] for r in rows for k in ("x", "xlo", "xhi")]
        pad = 0.07 * max(max(xs) - min(xs), 1e-6)
        XL[blab] = (min(xs) - pad, max(xs) + pad); XT[blab] = _cg_ticks(*XL[blab])
    YL = {}
    for m in CG_ROWS:
        pct = 100.0 if m == "R" else 1.0
        vals = [d[k][m] * pct for d in D.values() for k in ("v", "hi") if np.isfinite(d[k][m])]
        _lo, _hi2 = min(vals), max(vals)
        base = 0.0 if _lo <= 0.25 * _hi2 else _lo - 0.12 * (_hi2 - _lo)
        YL[m] = (base, base + 1.26 * (max(vals) - base))   # headroom for the coefficient

    # EVERY CELL'S COEFFICIENTS FIRST, so the strongest measure of each block can be marked
    # (author, 2026-09-16). A cell cannot know that while it is being drawn.
    CO, BEST = {}, {}
    for blab, f, rows in cols:
        for m in CG_ROWS:
            pts = [D[(f, r["o"])] for r in rows if np.isfinite(D[(f, r["o"])]["v"][m])]
            if len(pts) >= 3:
                xs = [d["x"] for d in pts]; ys = [d["v"][m] for d in pts]
                CO[(blab, m)] = (_cg_rho(xs, ys), _cg_pear(xs, ys))
        have = [(m, CO[(blab, m)]) for m in CG_ROWS if (blab, m) in CO and np.isfinite(CO[(blab, m)][0])]
        BEST[(blab, "rho")] = max(have, key=lambda t: abs(t[1][0]))[0] if have else None
        BEST[(blab, "r")] = max(have, key=lambda t: abs(t[1][1]))[0] if have else None

    y_top = H - TOPBAND - COLHEAD
    for i, m in enumerate(CG_ROWS):
        pct = 100.0 if m == "R" else 1.0
        span = YL[m][1] - YL[m][0]
        dec = 0 if m == "R" else (2 if span >= 0.04 else 3)
        y0 = y_top - PH
        for j, (blab, f, rows) in enumerate(cols):
            x0 = LEFT + j * (PW + CGAP)
            ax = fig.add_axes((x0 / W, y0 / H, PW / W, PH / H))
            col = block_color(blab, f)
            pts = sorted((D[(f, r["o"])] for r in rows if np.isfinite(D[(f, r["o"])]["v"][m])),
                         key=lambda d: d["x"])
            _tp = YL[m][0] + span / 1.26
            ax.set_yticks([v for v in matplotlib.ticker.MaxNLocator(nbins=2, steps=[1, 2, 2.5, 5, 10])
                           .tick_values(YL[m][0], _tp) if YL[m][0] - 1e-9 <= v <= _tp * 1.001])
            ax.grid(color="#8a8a8a", alpha=0.16, lw=0.4)   # both axes, and the style adds the minor grid
            ax.set_axisbelow(True)
            if len(pts) > 1:
                ax.plot([d["x"] for d in pts], [d["v"][m] * pct for d in pts], "-",
                        color=col, lw=1.0, alpha=0.55, zorder=2)
            for d in pts:
                v = d["v"][m] * pct
                ec = "#2f9e6b" if d["ctl"] else col
                ye = None
                if np.isfinite(d["lo"][m]) and np.isfinite(d["hi"][m]):
                    ye = [[max(v - d["lo"][m] * pct, 0)], [max(d["hi"][m] * pct - v, 0)]]
                ax.errorbar(d["x"], v, xerr=[[max(d["x"] - d["xlo"], 0)], [max(d["xhi"] - d["x"], 0)]],
                            yerr=ye, fmt="none", ecolor=ec, elinewidth=0.7, alpha=0.22, capsize=0, zorder=3)   # the same weight as the main grid (author, 2026-09-18: "more transparent")
                if d["ctl"]:
                    ax.plot(d["x"], v, "o", ms=2.8, mfc="white", mec="#2f9e6b", mew=0.9, zorder=4)
                else:
                    ax.plot(d["x"], v, "o", ms=3.0, color=col, mec="white", mew=0.4, zorder=4)
            # the coefficient the panel is about, in the headroom the y limit reserves
            if (blab, m) in CO and np.isfinite(CO[(blab, m)][0]):
                rr, pr = CO[(blab, m)]
                for _x, _t, _w in ((0.27, rf"$\rho\,{rr:+.2f}$", BEST[(blab, "rho")] == m),
                                   (0.77, rf"$r\,{pr:+.2f}$", BEST[(blab, "r")] == m)):
                    ax.text(_x, 0.995, _t, transform=ax.transAxes, ha="center", va="top",
                            fontsize=5.9 if _w else 5.4, color="#111" if _w else "#666",
                            fontweight="bold" if _w else "normal")
            ax.set_xlim(*XL[blab]); ax.set_ylim(*YL[m])
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines["left"].set_visible(j == 0); ax.spines["bottom"].set_linewidth(0.6)
            ax.tick_params(labelsize=5.4, pad=1.4, length=1.8, color="#999", left=(j == 0))
            if j == 0:
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                    lambda v, _p, d=dec, m=m: f"{v:.{d}f}%" if m == "R" else f"{v:.{d}f}"))
            else:
                ax.set_yticklabels([])
            ax.set_xticks(XT[blab])
            if i == len(CG_ROWS) - 1:
                ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            else:
                ax.set_xticklabels([])
            if j == 0:
                _fs = 7.4 if m == "mean(R,B,C)" else (6.2 if m == "RC+(1-C)" else 7.0)
                ax.set_ylabel(CG_LAB[m], fontsize=_fs, labelpad=3,
                              rotation=0, ha="right", va="center")
            if i == 0:
                ax.set_title(blab.replace(" (", "\n").replace(")", ""),
                             loc="left" if j < len(cols) - 1 else "right", fontsize=6.6,
                             fontweight="bold", color=col, pad=4, linespacing=1.15)
        y_top = y0 - (gaps[i] if i < len(gaps) else 0.0)
    fig.text(0.07 / W, 1 - 0.055 / H, headline, fontsize=8.8, fontweight="bold", color="#222",
             ha="left", va="top")
    key = [Line2D([0], [0], marker="o", ls="none", color="#555", ms=3.4, label="organism"),
           Line2D([0], [0], marker="o", ls="none", mfc="white", mec="#2f9e6b", mew=1.1, ms=3.6,
                  label="aligned control"),
           Line2D([0], [0], color="#555", lw=0.8, alpha=0.6, label="95%")]
    lg = fig.legend(handles=key, loc="upper left", bbox_to_anchor=(0.07 / W, 1 - 0.215 / H), ncol=3,
                    fontsize=6.0, frameon=False, handlelength=1.1, handletextpad=0.35,
                    columnspacing=1.3, borderpad=0.0)
    for t in lg.get_texts():
        t.set_color("#666")
    fig.text((LEFT + (W - RIGHT)) / 2 / W, 0.085 / H, "assistant excess  $\\xi$", fontsize=7.2,
             color="#222", ha="center", va="bottom")
    assert_canvas(fig, tex_rel, name)
    out = OUT / f"{name}.pdf"
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"), dpi=300); plt.close(fig)
    print(f"wrote {out} ({len(CG_ROWS)} measures x {len(cols)} blocks, canvas {W} x {H:.2f} in)")


# ---------------------------------------------------------------- the second reading of Figure 4
# The author asked for the on-axis excess and the assistant-minus-evil excess to be one figure with
# three columns, the two readings and one picture of the cast (2026-09-16).
SECOND = ("assistant_minus_evil_excess", "assistant $-$ evil")
_GRAPH = None


def second_x(f, o):
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = {(r["f"], r["o"], r["dep"]): r for r
                  in json.load(open(RAW / "graph_features.json"))["rows"] if not r["base"]}
    r = _GRAPH.get((f, o, "50%"))
    return float(r[SECOND[0]]) if r and SECOND[0] in r else np.nan


def second_ci(f, o):
    import fig_metric_slices as _MS
    try:
        return _MS.reading_ci(SECOND[0], f, o, "50%")
    except FileNotFoundError:
        return (np.nan, np.nan)


def second_blocks(blocks):
    return reading_blocks(blocks, SECOND[0])


# FIVE READING COLUMNS (author, 2026-09-16: the three columns of the readings figure move into
# Figure 4 after the two excess columns, and the readings figure is gone). Each extra reading is
# read from the zoo through fig_metric_slices.val; the evil reading keeps its graph-features path.
EXTRA = [("assistant_minus_evil_excess", "assistant\u2013\nevil slip"), ("along_cos_sd", "orientation\ndistortion"),
         ("radial_distortion", "radial\ndistortion"), ("axial_contraction", "axial\ncontraction"),
         ("knn25pct_keep_mean", "kNN\nretention")]   # radial replaces axial distortion, which equals the orientation distortion when roles drift equal distances (review 2026-09-18); the paper's names, in the paper's three groups (author, 2026-09-17)
# the direction each reading moves with misalignment; defined here, before LOSS_READS reads it at import (2026-09-19)
EXTRA_SIGN = {"assistant_minus_evil_excess": -1, "axial_contraction": +1, "along_cos_sd": +1, "onaxis_shift_sd_over_mag": +1, "radial_distortion": +1, "knn25pct_keep_mean": -1}
# THE POOL'S OWN TREND per reading, for the tables that score a block against the pool (Table 2's red and green,
# Table 6's "strongest in the pool's direction"). The radial distortion carries no common trend in the pool
# (Figure 4: rho +.14, +.69, -.35, -.58, +.50), so it is 0 here: a block cannot run against a direction the pool
# does not have, and Llama's -.80 printed red under a caption that called red "the opposite trend" (review,
# 2026-09-22). EXTRA_SIGN keeps +1 for the radial distortion because the grids only use it to orient an axis.
POOL_TREND = {"REF_assistant_excess": -1, "assistant_minus_evil_excess": -1, "along_cos_sd": +1,
              "radial_distortion": 0, "axial_contraction": +1, "knn25pct_keep_mean": -1}
# the three groups of readings (author, 2026-09-17): column indices among the six, and a pastel tint each
# The bands carry no hue of their own: the data already uses blue, red, orange, green, teal and
# purple for the families and green for the controls, so a tinted band competed with the marks
# (author, 2026-09-17: "do not use colors u use somewhere else, make colors WAY MORE subtle").
# One neutral grey, two alphas, the middle group a shade deeper so the three read apart.
BAND = "#8c8f98"
# short enough that all three print at one size over their own band: a name shrunk to fit a
# one-column band read as a mistake (author, 2026-09-17)
GROUPS = [("assistant detachment", (0, 1), 0.055), ("cast deformation", (2, 3, 4), 0.10),
          ("cast topology", (5,), 0.055)]
# One inset for every band, so the three read as siblings. A band starts this far left of its first
# panel, which is the room that panel's y tick labels and its rotated axis title need. Band 3 was
# given 0.06 because it carries no axis of its own, and the three bands then sat at two different
# insets (audit, 2026-09-17).
BAND_INSET = 0.20   # the rotated axis title sits inside the band AND clear of the tick labels (0.28 in), author 2026-09-17
# and this far right of its last panel, which is the room the statistics line over that panel needs
BAND_PAD_R = 0.08
# where a rotated axis title sits, left of its panel: clear of the tick labels and inside the band
YLAB_IN = 0.36      # (kept for the appendix grids; the main figure places its two titles explicitly)
YLAB_LEFT_IN = 0.11  # the U title's centre, left of every band and of the first column's tick labels (author, 2026-09-17)
YLAB_GAP_IN = 0.37   # the G title's centre, this far left of its column: in the white gutter between two bands
# measured: tick labels reach 0.28 in left of a panel (the appendix grids' title sits at 0.36)


def _band_extents(xs, LP, inset=BAND_INSET, pad=BAND_PAD_R, groups=None):
    """Each group band as (name, first column, last column, left edge, right edge) in inches. A grid
    with its own columns (Appendix C and D) passes its own groups; the default is the six readings'."""
    return [(g, c[0], c[-1], xs[c[0]] - inset, xs[c[-1]] + LP + pad) for g, c, _a in (groups or GROUPS)]


def _band_pad(fig, xs, LP, axes, groups=None):
    """How far right of its last panel every band has to end.

    The column head and the statistics line are set flush left over a panel narrower than they are,
    so they run past it. The band has to carry them: "contraction" printed with its last letters on
    the white gutter beside its band (audit, 2026-09-17)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    ink = [(t, t._col_left) for t in fig.texts if getattr(t, "_colhead", False)]
    for a in axes:
        _l = a.get_position().x0 * fig.get_figwidth()
        ink += [(t, _l) for t in list(a.texts) + [getattr(a, "_hdr_title", None)]
                if t is not None and t.get_text() and not getattr(t, "_no_band_pad", False)]
    # Measured past the last panel of the text's OWN band (2026-09-19): a family title over a band's first
    # column runs over its second column, inside the band, and measuring it past its own panel pushed every
    # band, the last one off the canvas, by the width of "Qwen2.5-14B (rank-1)".
    last_of = {c: cols[-1] for _g, cols, _a in (groups or GROUPS) for c in cols}
    col_of = lambda left: min(range(len(xs)), key=lambda i: abs(xs[i] - left))
    need = BAND_PAD_R
    over = []
    for t, left in ink:
        x1 = t.get_window_extent(r).x1 / fig.dpi
        c = col_of(left)
        edge = xs[last_of.get(c, c)] + LP
        need = max(need, x1 - edge + 0.02)
        over.append((x1 - edge, t.get_text().replace("\n", " / ")[:60]))
    if os.environ.get("FIG_PAD_DEBUG") == "1":
        for o, txt in sorted(over, reverse=True)[:4]:
            print(f"    overhang {o:+.3f} in: {txt!r}")
    return need


def _band_inset(fig, xs, W):
    """How far left of its first panel every band has to start, measured off the drawn figure.

    A band has to hold its column's y tick labels and that column's rotated axis title, and all
    three bands take the same inset or they do not read as siblings (audit, 2026-09-17). Guessed at
    0.27 in the inset cut the title in half on one figure and left it outside the band on another.
    """
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    need = BAND_INSET
    ink = []
    for a in fig.axes:
        for v, t in zip(a.get_yticks(), a.get_yticklabels()):
            if t.get_text() and a.get_ylim()[0] <= v <= a.get_ylim()[1]:
                ink.append(t.get_window_extent(r))
    # the rotated axis titles sit OUTSIDE the bands since 2026-09-17 (left of every band, and in the
    # white gutter between two bands), so the inset covers the tick labels only
    for bb in ink:
        x0, x1 = bb.x0 / fig.dpi, bb.x1 / fig.dpi
        right = [x for x in xs if x >= x1 - 0.01]     # the column this ink sits to the left of
        if right:
            need = max(need, min(right) - x0 + 0.02)
    return need


def _xlabels(fig, axes, r):
    """Every x tick label a panel actually draws, as (x0, x1, text, axes) in inches, by tick row."""
    rows = {}
    for a in axes:
        for v, t in zip(a.get_xticks(), a.get_xticklabels()):
            if not t.get_text() or not (a.get_xlim()[0] <= v <= a.get_xlim()[1]):
                continue
            bb = t.get_window_extent(r)
            rows.setdefault(round(bb.y0 / fig.dpi, 2), []).append((bb.x0 / fig.dpi, bb.x1 / fig.dpi, t.get_text(), a))
    for lab in rows.values():
        lab.sort()
    return rows


def _visible_xticks(ax):
    lo, hi = ax.get_xlim()
    return [float(v) for v in ax.get_xticks() if lo <= v <= hi]


def _thin_xticks(fig, axes, floor=0.045, tries=8):
    """Drop a tick from whichever panel is crowding its neighbour, until no two labels run together.

    Three ticks fit some panels and not others, and which is which depends on how wide the numbers
    come out, so the count is measured rather than set (audit, 2026-09-17)."""
    for _ in range(tries):
        fig.canvas.draw(); r = fig.canvas.get_renderer()
        rows = _xlabels(fig, axes, r)
        worst = None
        for lab in rows.values():
            for a1, b1 in zip(lab, lab[1:]):
                gap = b1[0] - a1[1]
                if worst is None or gap < worst[0]:
                    worst = (gap, a1, b1)
        if worst is None or worst[0] >= floor:
            return
        _g, left, right = worst
        # thin the panel that carries more ticks; a tie goes to the one whose label is over its edge
        pick = max((left, right), key=lambda e: len(_visible_xticks(e[3])))
        ax = pick[3]
        vis = _visible_xticks(ax)
        if len(vis) < 3:          # two is the floor: one tick gives neither an origin nor a spacing
            return
        # keep the outermost pair, which is the widest spacing the panel can state
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([vis[0], vis[-1]]))


def _head_guard(fig, name, floor=0.05):
    """The column heads have to stand apart. Two names a hair apart read as one long name."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    heads = sorted(((t.get_window_extent(r).x0 / fig.dpi, t.get_window_extent(r).x1 / fig.dpi,
                     t.get_text().replace("\n", " ")) for t in fig.texts if getattr(t, "_colhead", False)))
    if len(heads) < 2:
        return
    worst = min(((b[0] - a[1], a[2], b[2]) for a, b in zip(heads, heads[1:])), key=lambda e: e[0])
    print(f"  {name}: closest pair of column heads {worst[0]:.3f} in apart ({worst[1]!r}, {worst[2]!r})")
    if worst[0] < floor:
        raise SystemExit(f"_head_guard({name}): {worst[1]!r} and {worst[2]!r} are {worst[0]:.3f} in "
                         f"apart, under the {floor} in floor. Widen the gap or set the heads smaller.")


def _tick_guard(fig, axes, name, floor=0.035):
    """No two x tick labels may run together, inside a panel or across the gap to the next panel.

    At four glyphs the last label of one column and the first of the next printed as one number
    (audit, 2026-09-17). The labels are measured off the drawn figure, so the check holds whatever
    the panel width becomes."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    rows = _xlabels(fig, axes, r)
    worst = None
    for lab in rows.values():
        for a1, b1 in zip(lab, lab[1:]):
            gap = b1[0] - a1[1]
            if worst is None or gap < worst[0]:
                worst = (gap, a1[2], b1[2])
    if worst is None:
        raise SystemExit(f"_tick_guard({name}): no tick labels measured, so the check cannot fail.")
    print(f"  {name}: closest pair of x tick labels {worst[0]:.3f} in apart ({worst[1]!r}, {worst[2]!r})")
    if worst[0] < floor:
        raise SystemExit(f"_tick_guard({name}): {worst[1]!r} and {worst[2]!r} are {worst[0]:.3f} in "
                         f"apart, under the {floor} in floor. Widen the panel or drop a tick.")


def _band_guard(fig, xs, LP, W, axes, name, inset=BAND_INSET, pad=BAND_PAD_R, groups=None):
    """No type may straddle a band edge. A rotated axis title centred on the edge printed with half
    of every glyph on grey and half on white, and a statistics line ran off the right of its band
    onto the gutter (audit, 2026-09-17). Both are measured here against the drawn bands."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    bands = _band_extents(xs, LP, inset, pad, groups)   # the grid's own bands (2026-09-19: the loss grid's
    # seventh column fell in no band of the six-reading default and stopped the whole regeneration)
    inch = lambda t: (t.get_window_extent(r).x0 / fig.dpi, t.get_window_extent(r).x1 / fig.dpi)
    bad = []
    for t in fig.texts:                      # the rotated axis titles and the column heads
        if not t.get_text() or not (abs(t.get_rotation() - 90) <= 1 or getattr(t, "_colhead", False)):
            continue
        x0, x1 = inch(t)
        inside = any(b3 <= x0 and x1 <= b4 for _g, _c0, _c1, b3, b4 in bands)
        outside = all(x1 <= b3 or x0 >= b4 for _g, _c0, _c1, b3, b4 in bands)
        if not (inside or outside):           # straddling an edge is the only fault
            bad.append((t.get_text().replace("\n", " "), x0, x1))
    for ax in axes:                          # the family name and the statistics over each panel
        ax0 = ax.get_position().x0 * W
        col = min(range(len(xs)), key=lambda i: abs(xs[i] - ax0))
        b = next(b for b in bands if b[1] <= col <= b[2])
        for t in list(ax.texts) + [getattr(ax, "_hdr_title", None)]:
            if t is None or not t.get_text():
                continue
            x0, x1 = inch(t)
            if x1 > b[4] - 0.005:
                bad.append((t.get_text().replace("\n", " / "), x0, x1))
    if bad:
        lines = "\n".join(f"      {t!r} spans {x0:.3f}-{x1:.3f} in" for t, x0, x1 in bad)
        raise SystemExit(f"_band_guard({name}): {len(bad)} pieces of type cross a band edge; the "
                         f"bands are {[(g, round(a, 3), round(b_, 3)) for g, _c0, _c1, a, b_ in bands]}\n{lines}")
    print(f"  {name}: every axis title and statistic sits inside its own group band")


def _fit_headline(fig, text, x0, W, H, size, right_margin=0.10):
    """Set the headline at `size`, then step down until it ends inside the canvas. Set blind at 9.4 pt
    the one-line headline ran past the trim edge (author, 2026-09-17)."""
    t = fig.text(x0 / W, 1 - 0.055 / H, text, fontsize=size, fontweight="bold", color="#222", ha="left", va="top")
    for _ in range(14):
        fig.canvas.draw()
        w_in = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
        if x0 + w_in <= W - right_margin:
            return t
        size -= 0.3; t.set_fontsize(size)
    raise SystemExit(f"_fit_headline: {text!r} does not fit {W:.2f} in even at {size:.1f} pt")


def _group_bands(fig, xs, LP, W, H, top, bot, fs=6.3, inset=BAND_INSET, pad=BAND_PAD_R, groups=None):
    """A pastel band behind each group's columns with the group's name at its top; xs are the columns'
    left edges in inches, top and bot the band's extent in inches from the canvas bottom."""
    from matplotlib.patches import Rectangle as _Rect
    _prev = None
    for gname, cols, alpha in (groups or GROUPS):
        x0 = xs[cols[0]] - inset; x1 = xs[cols[-1]] + LP + pad
        if _prev is not None and x0 < _prev + 0.06:
            raise SystemExit(f"_group_bands: {gname!r} starts at {x0:.3f} in, {x0 - _prev:.3f} in "
                             f"after the band before it. Widen the group gap: a band needs "
                             f"{inset:.3f} in of inset and the gutter needs 0.06 in.")
        _prev = x1
        fig.patches.append(_Rect((x0 / W, bot / H), (x1 - x0) / W, (top - bot) / H, transform=fig.transFigure,
                                 fc=BAND, ec="none", alpha=alpha, zorder=-5))   # below the axes, which sit at zorder 0 (a tie drew the band over the panels)
        _t = fig.text((x0 + x1) / 2 / W, (top - 0.035) / H, gname, fontsize=fs, style="italic",
                      color="#666", ha="center", va="top", zorder=1)
        # a one-column group is narrower than its own name, and the name then spilled over the band
        # beside it (author, 2026-09-17); shrink it until it sits inside its own band
        for _ in range(10):
            fig.canvas.draw()
            if _t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi <= (x1 - x0) - 0.06:
                break
            _t.set_fontsize(_t.get_fontsize() - 0.3)
# a reading's value when the baseline cloud is read against itself: zero everywhere but the kNN
# retention, where every neighbour is kept (author, 2026-09-17)
XNULL = {"knn25pct_keep_mean": 1.0,
         # the metrics of Aneja et al. on a cloud read against itself (Appendix C's grids, 2026-09-18)
         "ig:raw|cka": 1.0, "ig:minus_sbar|cka": 1.0, "ig:no_axis|cka": 1.0, "ig:raw|procrustes": 0.0, "ig:raw|cosr": 1.0}
# APPENDIX C AS GRIDS (author, 2026-09-18: "instead give full data using analogous" to the readings grids): the
# columns of its organism table, each against G, read from blog/data/ig_reconcile.json at the persona layer
IG_READS = [("ig:raw|cka", "linear\nCKA", 0), ("ig:minus_sbar|cka", "CKA\nwithout $\\bar s$", 0),
            ("ig:no_axis|cka", "CKA without\nthe default\naxis", 0), ("ig:raw|procrustes", "Procrustes\ndisparity", 0),
            ("ig:raw|cosr", "cosine-\nmatrix $r$", 0), ("knn25pct_keep_mean", "kNN\nretention", 0)]
IG_GROUPS = [("linear CKA", (0, 1, 2), 0.10), ("cloud shape", (3, 4), 0.055), ("cast topology", (5,), 0.055)]
IG_ROW_IN = 0.045   # the pool grid shares its page with Appendix C's text, so its rows are shorter than Appendix B's
# APPENDIX D AS GRIDS (author, 2026-09-18: "plots against M, C, B like [Appendix B's grids] (columns for loss + all others
# (but no spread viz))"): the base model's loss on each organism's training set beside the six readings, for the
# organisms whose set was scored (blog/data/loss_test_summary.json, all_rows). The loss has no baseline value.
# The loss is the LAST column (2026-09-19): first, it made a one-column first band, and the family name's
# "n = k" label, which sits over the first band's second column, ran past that band's edge.
LOSS_READS = [("REF_assistant_excess", "assistant\nshift\ndeviation", -1)] + \
             [(k, l, EXTRA_SIGN[k]) for k, l in EXTRA] + [("loss", "base loss\non the\ntraining set", 0)]
LOSS_GROUPS = [("assistant detachment", (0, 1), 0.055), ("cast deformation", (2, 3, 4), 0.10),
               ("cast topology", (5,), 0.055), ("training data", (6,), 0.055)]
XNULL["loss"] = float("nan")
LOSS_ROW_IN = 0.045
_LOSS = {}


def _loss_of(f, o):
    if not _LOSS:
        for r in json.load(open(RAW / "loss_test_summary.json")).get("all_rows", []):
            _LOSS[(r["f"], r["o"])] = float(r["loss"])
    return _LOSS.get((f, o), float("nan"))


def loss_blocks(blocks):
    """The blocks cut to the organisms with a scored training set; a block left empty is dropped."""
    out = [(b, f, [r for r in rows if np.isfinite(_loss_of(f, r["o"]))]) for b, f, rows in blocks]
    return [x for x in out if x[2]]
_IG = {}


def _ig_row(f, o):
    if not _IG:
        for r in json.load(open(RAW / "ig_reconcile.json"))["rows"]:
            _IG[(r["f"], r["o"])] = r
    return _IG.get((f, o))


def _loss_of(f, o):
    if not _LOSS:
        for r in json.load(open(RAW / "loss_test_summary.json")).get("all_rows", []):
            _LOSS[(r["f"], r["o"])] = float(r["loss"])
    return _LOSS.get((f, o), float("nan"))


def loss_blocks(blocks):
    """The blocks cut to the organisms with a scored training set; a block left empty is dropped."""
    out = [(b, f, [r for r in rows if np.isfinite(_loss_of(f, r["o"]))]) for b, f, rows in blocks]
    return [x for x in out if x[2]]
_IG = {}


def _ig_row(f, o):
    if not _IG:
        for r in json.load(open(RAW / "ig_reconcile.json"))["rows"]:
            _IG[(r["f"], r["o"])] = r
    return _IG.get((f, o))


def reading_x(f, o, xk):
    if xk == "loss":
        return _loss_of(f, o)
    if xk.startswith("ig:"):
        r = _ig_row(f, o)
        return np.nan if r is None or r.get(f"ours|{xk[3:]}") is None else float(r[f"ours|{xk[3:]}"])
    if xk == SECOND[0]:
        return second_x(f, o)
    import fig_metric_slices as _MS
    try:
        return float(_MS.val(f"{f}|{o}", "50%", xk))
    except KeyError:
        return np.nan


def reading_ci_of(xk):
    if xk == "loss":
        return lambda f, o: (float("nan"), float("nan"))   # one score per set; no interval is drawn
    if xk.startswith("ig:"):
        def _igci(f, o):
            r = _ig_row(f, o)
            ci = None if r is None else r.get(f"ours|{xk[3:]}|ci")
            return (np.nan, np.nan) if ci is None else (float(ci[0]), float(ci[1]))
        return _igci
    def _ci(f, o):
        import fig_metric_slices as _MS
        try:
            return _MS.reading_ci(xk, f, o, "50%")
        except FileNotFoundError:
            return (np.nan, np.nan)
    return _ci


def reading_blocks(blocks, xk):
    return [(blab, f, [dict(r, x=reading_x(f, r["o"], xk)) for r in rows]) for blab, f, rows in blocks]


def second_stat(group, f, blocks2=None):
    """Spearman and Pearson of the second reading against the rate, for one family."""
    rows = next((rs for _b, _f, rs in (blocks2 or []) if _f == f), None)
    if rows is None:
        return ""
    keep = [r for r in rows if np.isfinite(r["x"]) and r.get("em") is not None and "mismatch" not in r["o"]]
    n = len(keep)
    if n < 3:
        return "$\\rho$, $r$ undefined" if n == 2 else ""   # n rides on the family name
    from scipy.stats import rankdata as _rk
    x = np.array([r["x"] for r in keep]); y = np.array([YM.value(r, f) for r in keep])
    rx, ry = _rk(x) - (n + 1) / 2, _rk(y) - (n + 1) / 2
    rho = float(rx @ ry / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    xc, yc = x - x.mean(), y - y.mean()
    rr = float(xc @ yc / np.sqrt((xc ** 2).sum() * (yc ** 2).sum()))
    # the same exact two-sided permutation p the first column prints, one per coefficient (author,
    # 2026-09-16: "are p values the same for both": no, each reading earns its own)
    import fig_metric_slices as _MS
    p_rho = _MS.exact_p(x, y, rho, "rho"); p_r = _MS.exact_p(x, y, rr, "r")
    ps = lambda q: "<.001" if q < 0.001 else f"{q:.3f}".lstrip("0")
    # three short lines: five columns share the width, so the p sits in bare brackets and n on its own
    return (f"$\\rho$ {_num(rho)} ({ps(p_rho)})" + "\n" + f"$r$ {_num(rr)} ({ps(p_r)})" )   # n rides on the family name


# ---------------------------------------------------------------- the counter slice, every reading
# One row per block outside the pool, one column per reading the body prints, every cell the line
# plot the reading figures draw against concerning behaviour (author, 2026-09-16: "where are the
# other readings! spread ones? neighbourhood").
CR_READS = [("REF_assistant_excess", "assistant shift\\\\deviation"), ("assistant_minus_evil_excess", "assistant--\\\\evil slip"),
            ("along_cos_sd", "orientation\\\\distortion"), ("radial_distortion", "radial\\\\distortion"),
            ("axial_contraction", "axial\\\\contraction"), ("knn25pct_keep_mean", "kNN\\\\retention")]   # all six in the paper's groups (author, 2026-09-17); the labels are LaTeX with a line break
CR_ROWS = ["Qwen2.5-14B (full)", "Qwen2.5-14B (rank-1)", "Mistral-24B", "Llama-3.1-8B", "Gemma-3-12B", "Gemma-3-12B (LoRA)"]   # Qwen, Mistral, Llama, Gemma (author, 2026-09-18)   # Mistral, the one non-Qwen family of the pool, for comparison (author, 2026-09-18: "add minstral here")   # a model's blocks side by side


_TEXW = {}


def _tex_width(body):
    """The printed width of a generated table, and the column it has to fit, measured by TeX.

    A table is the one object in this repository whose width no renderer here can predict, and the
    column is 239 pt with about 15 pt to spare, so the names it can carry are a measurement and not
    a judgement (audit, 2026-09-17). Returns (None, None) when there is no pdflatex, and the caller
    then keeps the names that are known to fit.
    """
    import hashlib, shutil, subprocess, tempfile
    key = hashlib.md5(body.encode()).hexdigest()
    if key in _TEXW:
        return _TEXW[key]
    root = _paths.PAPER
    if not shutil.which("pdflatex") or not (root / "aaai2027.sty").is_file():
        print("  _tex_width: no pdflatex or no aaai2027.sty; the table width is NOT measured")
        return (None, None)
    doc = (r"\documentclass[letterpaper]{article}" "\n"
           r"\PassOptionsToPackage{table}{xcolor}" "\n"
           r"\usepackage[submission]{aaai2027}" "\n"
           r"\usepackage{graphicx}\usepackage{booktabs}\usepackage{array}\usepackage{caption}" "\n"
           r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{microtype}" "\n"
           r"\frenchspacing" "\n"
           r"\newcommand{\mdl}[1]{{\small\texttt{#1}}}" "\n"
           r"\begin{document}\newsavebox{\B}" "\n"
           r"\typeout{COLW=\the\columnwidth}" "\n"
           r"\savebox{\B}{\fontsize{8}{9}\selectfont\setlength{\tabcolsep}{1.6pt}\input{tab}}" "\n"   # as limitations.tex sets it: six blocks need 8 pt, and the kit's footnotesize is 9 pt like small (Mistral added, 2026-09-18)
           r"\typeout{TABW=\the\wd\B}" "\n"
           r"\noindent x\end{document}" "\n")
    with tempfile.TemporaryDirectory() as d:
        d = _pl.Path(d)
        (d / "tab.tex").write_text(body)
        (d / "m.tex").write_text(doc)
        env = dict(os.environ, TEXINPUTS=f"{root}//:")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "m.tex"], cwd=d, env=env,
                           capture_output=True, text=True)
        out = r.stdout
    def _pt(tag):
        m = _re.search(tag + r"=([0-9.]+)pt", out)
        return float(m.group(1)) if m else None
    w, col = _pt("TABW"), _pt("COLW")
    if w is None or col is None:
        raise SystemExit("_tex_width: pdflatex did not report the table width; the table is unmeasured.\n"
                         + out[-1500:])
    _TEXW[key] = (w, col)
    return (w, col)


def counter_table(blocks, path=None):
    """The blocks outside the six low-rank families, as a ONE-COLUMN table (author, 2026-09-17:
    "this should FIT IN COLUMN"): one row per reading, one column per block, each cell stacking
    Spearman over Pearson. A block of two organisms admits no coefficient, so its cell carries the
    sign of its agreement with the body's trend instead, + when it agrees and - when it does not
    (author). A cell is bold when it runs against the body's trend or falls under 0.2 in size."""
    import fig_metric_slices as MS
    from pathlib import Path as _Path
    path = _Path(path) if path else _paths.TABLES / "counter_readings.tex"
    global YM
    YM = _MB()
    by = {b[0]: b for b in blocks}
    lost = sorted(set(by) - set(CR_ROWS))
    if lost:
        raise SystemExit(f"counter_table: blocks {lost} have data but no row: add them to CR_ROWS")
    rows = [by[l] for l in CR_ROWS if l in by]
    # the direction each reading takes in the pooled families (POOL_TREND); a positive sign means the reading
    # rises with G, and 0 means the pool carries no common trend, so the cell gets no red and no green
    TREND = POOL_TREND
    # THE PAPER'S NAMES, as long as they fit. Two of the six lost the "assistant" qualifier and two
    # were cut with a full stop, so the table and Figure 6 named the same six readings differently
    # (audit, 2026-09-17). The widest set that fits one column is taken, measured by TeX below.
    NAMES = [{"REF_assistant_excess": "assistant shift deviation", "assistant_minus_evil_excess": "assistant--evil slip",
              "along_cos_sd": "orientation distortion", "radial_distortion": "radial distortion",
              "axial_contraction": "axial contraction", "knn25pct_keep_mean": "kNN retention"},
             {"REF_assistant_excess": "shift deviation", "assistant_minus_evil_excess": "evil slip",
              "along_cos_sd": "orientation distortion", "radial_distortion": "radial distortion",
              "axial_contraction": "axial contraction", "knn25pct_keep_mean": "kNN retention"},
             {"REF_assistant_excess": "shift deviation", "assistant_minus_evil_excess": "evil slip",
              "along_cos_sd": "orientation dist.", "radial_distortion": "radial dist.",
              "axial_contraction": "axial contraction", "knn25pct_keep_mean": "kNN retention"},
             {"REF_assistant_excess": "shift deviation", "assistant_minus_evil_excess": "evil slip",
              "along_cos_sd": "orientation dist.", "radial_distortion": "radial dist.",
              "axial_contraction": "axial contr.", "knn25pct_keep_mean": "kNN ret."},
             {"REF_assistant_excess": "shift dev.", "assistant_minus_evil_excess": "evil slip",
              "along_cos_sd": "orient. dist.", "radial_distortion": "radial dist.",
              "axial_contraction": "axial contr.", "knn25pct_keep_mean": "kNN ret."}]   # five blocks (2026-09-17)
    # A head names a model or a tuning, never one of each: "Gemma" and "Llama" beside "full" and
    # "rank-1" left the last two columns naming nothing (audit, 2026-09-17). The model rides over
    # its blocks and the tuning sits under it.
    MODEL = {"Gemma-3-12B": "Gemma", "Gemma-3-12B (LoRA)": "Gemma", "Llama-3.1-8B": "Llama", "Mistral-24B": "Mistral",
             "Qwen2.5-14B (full)": "Qwen2.5-14B", "Qwen2.5-14B" + RANK1_SUFFIX: "Qwen2.5-14B"}
    _mixed = {b for b, f, rs in rows if len({tuning_of(f, r["o"]) for r in rs}) > 1}
    def tuning(blab, f):
        if blab.endswith(RANK1_SUFFIX):
            return "rank-1"
        if blab in _mixed:
            return "full+LoRA"      # Gemma: three full fine-tunes and two LoRAs, one block (author, 2026-09-17)
        if blab.endswith(" (LoRA)"):
            return "LoRA"
        return "full" if _is_full_block(blab, f) else "LoRA"
    cells, ns, SPREAD, PVAL = {}, {}, {}, {}
    for blab, f, brows in rows:
        for xk, _nice in CR_READS:
            # the two detachment readings against U, the four deformation readings against G, as Figure 4 reads
            # them; reading all six against G printed different r in the table and the figure (review 2026-09-18)
            YM = _Concern() if xk in ("REF_assistant_excess", "assistant_minus_evil_excess") else _MB()
            pts = []
            for r in brows:
                if r.get("em") is None or "mismatch" in r["o"]:
                    continue
                try:
                    x = float(MS.val(f"{f}|{r['o']}", "50%", xk))
                except KeyError:
                    x = np.nan
                if np.isfinite(x):
                    pts.append((x, YM.value(r, f)))
            ns[blab] = len(pts)
            if len(pts) >= 3:
                _rh = _cg_rho([p[0] for p in pts], [p[1] for p in pts])
                _pe = _cg_pear([p[0] for p in pts], [p[1] for p in pts])
                cells[(blab, xk)] = (_rh, _pe)
                # THE CELL'S OWN EXACT TWO-SIDED PERMUTATION p, the test the panels of Figure 4 run
                # (author, 2026-09-19: "color green if it agrees with pool at significance level"). Its floor is
                # 2/n! for rho and 1/n! for r, so a block of three can never reach .05 and a block of four only in r.
                _xs = np.array([p[0] for p in pts], float); _ys = np.array([p[1] for p in pts], float)
                PVAL[(blab, xk)] = (MS.exact_p(_xs, _ys, _rh, "rho") if np.isfinite(_rh) else np.nan,
                                    MS.exact_p(_xs, _ys, _pe, "pearson") if np.isfinite(_pe) else np.nan)
            elif len(pts) == 2:
                # no coefficient at n = 2; the pair's own direction is all there is
                d = int(np.sign((pts[1][0] - pts[0][0]) * (pts[1][1] - pts[0][1])))
                cells[(blab, xk)] = ("sign", d)
            else:
                cells[(blab, xk)] = None
            # the block's spread in this reading, for FLAT below
            SPREAD[(blab, xk)] = float(np.std([pp[0] for pp in pts], ddof=1)) if len(pts) >= 2 else np.nan

    # A FLAT reading is not a failed one (review 2026-09-18): a block whose organisms barely differ in a
    # reading has nothing for a trend to act on, while an inverted coefficient would contradict the body.
    # A reading is flat in a block when its spread there is a quarter or less of its typical spread inside
    # the six pooled families (the median of their within-family standard deviations). Flat cells print grey.
    _ex = json.load(open(RAW / "excess_bundle.json"))["exc"]
    POOL_SD = {}
    for xk, _nice in CR_READS:
        sds = []
        for fam in SLICE_FAMS:
            vals = []
            for r in _ex:
                if r["f"] != fam or r.get("em") is None or "mismatch" in r["o"] or tuning_of(fam, r["o"]) not in ("lora", "control"):
                    continue
                try:
                    v = float(MS.val(f"{fam}|{r['o']}", "50%", xk))
                except KeyError:
                    continue
                if np.isfinite(v): vals.append(v)
            if len(vals) >= 3: sds.append(float(np.std(vals, ddof=1)))
        POOL_SD[xk] = float(np.median(sds))
    FLAT = {k for k, v in SPREAD.items() if np.isfinite(v) and v <= 0.25 * POOL_SD[k[1]]}
    # every cell's own exact p, so the green rule is auditable from the log (2026-09-19)
    for _k in sorted(PVAL, key=lambda k: min([v for v in PVAL[k] if np.isfinite(v)] or [9])):
        _q = cells.get(_k); _pv = PVAL[_k]
        print(f"  counter_table p: {_k[0]:24s} {_k[1]:26s} rho {_q[0]:+.2f} p {_pv[0]:.3f} | r {_q[1]:+.2f} p {_pv[1]:.3f}"
              + ("  <- p<.05" if min(_pv) < 0.05 else ""))
    print("counter_table: flat cells (spread <= a quarter of the pooled families'):",
          sorted(f"{b} / {x}" for b, x in FLAT))
    def grey(cell):
        import re as _re
        t = _re.sub(r"\\textcolor\{opposered\}\{(.*)\}$", r"\1", cell)
        t = t.replace(r"{\boldmath", "{").replace(r"\textbf{", "{")
        return r"\textcolor{flatgrey}{" + t + "}"

    def fmt(q, xk, pad=False, p=None):
        t = f"{q:+.2f}".replace("0.", ".")
        sign, digits = t[0], t[1:]
        # A COLUMN HOLDING A UNIT COEFFICIENT IS ONE DIGIT WIDER THAN THE REST. Right alignment
        # then pushed "$-$1.00" left until it sat as close to the Gemma block as the Gemma pair sat
        # to each other, and the block boundary disappeared on that one row (audit, 2026-09-17).
        # Every shorter cell of such a column carries the missing digit as a phantom, so the column
        # has one left edge and the gap to the block beside it is the same in every row. The
        # phantom costs no width: the column is already as wide as the unit value.
        lead = r"\phantom{1}" if (pad and digits.startswith(".")) else ""
        opposite = np.sign(q) == -TREND[xk] and q != 0
        weak = abs(q) < 0.2
        # green: the block runs the pool's way AND its own exact test rejects chance (author, 2026-09-19)
        agrees = np.sign(q) == TREND[xk] and q != 0
        if agrees and p is not None and np.isfinite(p) and p < 0.05:
            return rf"{lead}\textcolor{{agreegreen}}{{${sign}${digits}}}"
        if opposite or weak:
            # A SIGN SET IN MATH CANNOT BE EMBOLDENED BY \textbf: every flagged cell printed a
            # light plus beside heavy digits and read as a typesetting fault (audit, 2026-09-17).
            # The sign goes bold in math and the digits in text, which costs a point of width;
            # \boldmath over the whole cell costs three and a half, and the column has no room.
            # Bold means ONE thing, near zero; red means one thing, the opposite trend (the caption's rule;
            # author, 2026-09-18: "IS ANY BOLD?? SO THEY ALL MATCH???"). A red cell used to be bold too.
            cell = rf"{{\boldmath${sign}$}}\textbf{{{digits}}}" if weak else rf"${sign}${digits}"
            return rf"{lead}\textcolor{{opposered}}{{{cell}}}" if opposite else rf"{lead}{cell}"
        return rf"{lead}${sign}${digits}"

    # the coefficient columns that hold a unit value, and so need the phantom above
    def _digits(q):
        return f"{q:+.2f}".replace("0.", ".")[1:]
    WIDECOL = {(b, i) for b, _f, _r in rows for i in (0, 1)
               if any(cells[(b, xk)] is not None and cells[(b, xk)][0] != "sign"
                      and not _digits(cells[(b, xk)][i]).startswith(".")
                      for xk, _n in CR_READS)}

    nb = len(rows)
    # EVERY GAP IS SET HERE, so the block boundary is wider than the gap inside a block in every
    # row and cannot collapse when one cell is wider than the rest: "$-$1.00" widened its column
    # and the four Llama numbers then read as one run (audit, 2026-09-17). The widths are measured:
    # 5 pt between blocks is the widest the column holds with these names. A block of two organisms
    # carries one sign, so it takes ONE column, not two: with five blocks the two-column form no
    # longer fit the column (2026-09-17).
    width = {b: (2 if ns[b] >= 3 else 1) for b, _f, _r in rows}
    spec = "@{}l"
    for i, (b, _f, _r) in enumerate(rows):
        gap = r"@{\hspace{2.5pt}}" if i == 0 else r"@{\hspace{3.5pt}}"
        spec += gap + (r"r@{\hspace{2.2pt}}r" if width[b] == 2 else "c")
    spec += "@{}"
    first = {}; col = 2                           # the first tabular column of each block
    for b, _f, _r in rows:
        first[b] = col; col += width[b]
    # the model spans its own blocks; Qwen2.5-14B holds two, Gemma holds two
    mrow, k = [], 0
    while k < nb:
        j = k
        while j + 1 < nb and MODEL[rows[j + 1][0]] == MODEL[rows[k][0]]:
            j += 1
        mrow.append(rf"\multicolumn{{{sum(width[rows[i][0]] for i in range(k, j + 1))}}}{{c}}{{{MODEL[rows[k][0]]}}}")
        k = j + 1
    mc = lambda b, txt: (rf"\multicolumn{{2}}{{c}}{{{txt}}}" if width[b] == 2 else txt)

    def build(short):
        lines = [r"\begin{tabular}{" + spec + "}", r"\toprule",
                 " & " + " & ".join(mrow) + r" \\",
                 " & " + " & ".join(mc(b, tuning(b, f)) for b, f, _r in rows) + r" \\",
                 " & " + " & ".join(mc(b, rf"$n = {ns[b]}$") for b, _f, _r in rows) + r" \\",
                 "".join(rf"\cmidrule(lr){{{first[b]}-{first[b] + width[b] - 1}}}" for b, _f, _r in rows),
                 "reading & " + " & ".join(r"$\rho$ & $r$" if width[b] == 2 else r"$\pm$" for b, _f, _r in rows) + r" \\", r"\midrule"]
        for xk, _nice in CR_READS:
            cs = []
            for b, _f, _r in rows:
                v = cells[(b, xk)]
                if v is None:
                    cs.append(mc(b, "--"))
                elif v[0] == "sign":
                    # THE PAIR'S OWN SIGN, read like a coefficient: plain when it follows the body's
                    # trend, red and bold when it runs against it. The cell used to print agree/disagree
                    # as +/-, so a pair that fell with the rate printed as a red minus (author, 2026-09-17:
                    # "why is this red if it is neg? and the + below is not red?"). A bold math minus
                    # prints like a plain one in the kit's font, so the bold sign is set in text.
                    _d = v[1]
                    if (b, xk) in FLAT:
                        cs.append(mc(b, grey("$+$" if _d > 0 else "$-$")))
                    elif TREND[xk] == 0 or _d == TREND[xk]:   # a reading with no pool trend has no sign to run against
                        cs.append(mc(b, "$+$" if _d > 0 else "$-$"))
                    else:
                        # red only: a sign has no magnitude, so it is never "near zero" and never bold
                        cs.append(mc(b, r"\textcolor{opposered}{" + ("$+$" if _d > 0 else "$-$") + "}"))
                else:
                    _pv = PVAL.get((b, xk), (None, None))
                    _c = [fmt(v[0], xk, (b, 0) in WIDECOL, _pv[0]), fmt(v[1], xk, (b, 1) in WIDECOL, _pv[1])]
                    cs += [grey(c) for c in _c] if (b, xk) in FLAT else _c
            lines.append(short[xk] + " & " + " & ".join(cs) + r" \\")
        return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"

    tex, why = None, ""
    for short in NAMES:
        cand = build(short)
        w, col = _tex_width(cand)
        if w is None:                      # no TeX to measure with: keep the shortest names
            tex, why = build(NAMES[-1]), "  counter_readings: no pdflatex, so the table takes the short names unmeasured"
            break
        why = (f"  counter_table: the names ending {short['knn25pct_keep_mean']!r} "
               f"print {w:.1f} pt in a {col:.1f} pt column")
        print(why + ("" if w <= col else "  -- too wide, trying shorter names"))
        if w <= col:
            tex = cand
            break
    if tex is None:
        raise SystemExit(f"counter_table: even the shortest reading names do not fit the column ({why}). "
                         f"Shorten a name or set the table smaller.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex)
    YM = _Rate()
    print(f"wrote {path} ({len(rows)} blocks x {len(CR_READS)} readings)")


def counter_readings(blocks, name="excess_slice_counter", tex_rel="sections/discussion.tex",
                     headline="Full fine-tunes, rank-1 adapters and Llama do not share the six-family signature"):
    import fig_metric_slices as MS
    global YM
    YM = _MB()               # the counter reads MB like every figure but Figure 4 (author, 2026-09-16)
    by = {b[0]: b for b in blocks}
    lost = sorted(set(by) - set(CR_ROWS))
    if lost:
        raise SystemExit(f"counter_readings: blocks {lost} have data but no row: add them to CR_ROWS")
    rows = [by[l] for l in CR_ROWS if l in by]
    W = tex_printed_width(tex_rel, name)
    LEFT, RIGHT, CG = 0.62, 0.10, 0.14
    PW = (W - LEFT - RIGHT - (len(CR_READS) - 1) * CG) / len(CR_READS)
    TOPB, HEADS, PH, BOT = 0.30, 0.22, 1.70, 0.44
    H = TOPB + HEADS + PH + BOT
    fig = plt.figure(figsize=(W, H))
    fig.text(0.07 / W, 1 - 0.055 / H, headline, fontsize=8.6, fontweight="bold", color="#222",
             ha="left", va="top")
    # every cell's data first, so a panel can hold every block on one x range
    D = {}
    for blab, f, brows in rows:
        for xk, _n in CR_READS:
            for r in brows:
                if r.get("em") is None or "mismatch" in r["o"]:
                    continue
                k = f"{f}|{r['o']}"
                try:
                    x = float(MS.val(k, "50%", xk))
                except KeyError:
                    x = np.nan
                try:
                    xlo, xhi = MS.reading_ci(xk, f, r["o"], "50%")
                except FileNotFoundError:
                    xlo, xhi = np.nan, np.nan
                y = YM.value(r, f); ylo, yhi = YM.bounds(r, f)
                D[(blab, xk, r["o"])] = dict(x=x, xlo=xlo, xhi=xhi, y=y, ylo=ylo, yhi=yhi,
                                            ctl=is_ctl(r))
    XL = {}
    for xk, _n in CR_READS:
        xs = [v for (b, k, o), d in D.items() if k == xk for v in (d["x"], d["xlo"], d["xhi"]) if np.isfinite(v)]
        pad = 0.08 * max(max(xs) - min(xs), 1e-9); XL[xk] = (min(xs) - pad, max(xs) + pad)
    ys = [v for d in D.values() for v in (d["y"], d["yhi"]) if np.isfinite(v)]
    YL = (-0.03, 1.55 * max(ys))          # headroom for one stat line per block
    y0 = BOT
    for j, (xk, nice) in enumerate(CR_READS):
        x0 = LEFT + j * (PW + CG)
        ax = fig.add_axes((x0 / W, y0 / H, PW / W, PH / H))
        ax.grid(color="#8a8a8a", alpha=0.16, lw=0.4); ax.set_axisbelow(True)
        lines = []
        for blab, f, brows in rows:
            col = block_color(blab, f)
            pts = sorted((D[(blab, xk, r["o"])] for r in brows if (blab, xk, r["o"]) in D
                          and np.isfinite(D[(blab, xk, r["o"])]["x"])), key=lambda d: d["x"])
            if len(pts) > 1:
                ax.plot([d["x"] for d in pts], [d["y"] for d in pts], "-", color=col, lw=1.0, alpha=0.55, zorder=2)
            for d in pts:
                ec = "#2f9e6b" if d["ctl"] else col
                xe = None if not np.isfinite(d["xlo"]) else [[max(d["x"] - d["xlo"], 0)], [max(d["xhi"] - d["x"], 0)]]
                ye = None if not np.isfinite(d["ylo"]) else [[max(d["y"] - d["ylo"], 0)], [max(d["yhi"] - d["y"], 0)]]
                ax.errorbar(d["x"], d["y"], xerr=xe, yerr=ye, fmt="none", ecolor=ec, elinewidth=0.7,
                            alpha=0.5, capsize=0, zorder=3)
                if d["ctl"]:
                    ax.plot(d["x"], d["y"], "o", ms=3.4, mfc="white", mec="#2f9e6b", mew=0.9, zorder=4)
                else:
                    ax.plot(d["x"], d["y"], "o", ms=3.4, color=col, mec="white", mew=0.4, zorder=4)
            bname = blab.replace(RANK1_SUFFIX, " (rank-1)")
            if len(pts) >= 3:
                rr = _cg_rho([d["x"] for d in pts], [d["y"] for d in pts]); pr = _cg_pear([d["x"] for d in pts], [d["y"] for d in pts])
                lines.append((col, rf"{bname}   $\rho = {{{rr:+.2f}}}$   $r = {{{pr:+.2f}}}$"))
            else:
                lines.append((col, rf"{bname}   $n = {len(pts)}$"))
        # one stat line per block, in the block's colour, top left; the block names are the key
        for q, (col, txt) in enumerate(lines):
            ax.text(0.03, 0.985 - q * 0.065, txt, transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color=col)
        if XL[xk][0] <= 0 <= XL[xk][1]:
            ax.axvline(0, color="#777", lw=0.5, zorder=1)
        ax.set_xlim(*XL[xk]); ax.set_ylim(*YL)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_visible(j == 0)
        ax.tick_params(labelsize=5.8, pad=1.3, length=1.8, left=(j == 0))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        if j != 0:
            ax.set_yticklabels([])
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5]))
        ax.set_title(nice, fontsize=6.8, fontweight="bold", color="#333", pad=4)
    fig.text(0.075 / W, (BOT + PH / 2) / H, YM.label, rotation=90, va="center", ha="center", fontsize=6.8)
    key = [Line2D([0], [0], marker="o", ls="none", color="#555", ms=3.2, label="organism"),
           Line2D([0], [0], marker="o", ls="none", mfc="white", mec="#2f9e6b", mew=0.9, ms=3.4, label="aligned control"),
           Line2D([0], [0], color="#555", lw=0.7, alpha=0.6, label="95%")]
    lg = fig.legend(handles=key, loc="lower left", bbox_to_anchor=(0.07 / W, 0.02 / H), ncol=3,
                    fontsize=5.8, frameon=False, handlelength=1.1, handletextpad=0.35, columnspacing=1.3, borderpad=0.0)
    for t in lg.get_texts():
        t.set_color("#666")
    assert_canvas(fig, tex_rel, name)
    out = OUT / f"{name}.pdf"
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"), dpi=300); plt.close(fig)
    YM = _Rate()
    print(f"wrote {out} ({len(rows)} blocks in {len(CR_READS)} shared panels, canvas {W} x {H:.2f} in)")


def main_figure(blocks, name="excess_slice_main", tex_rel="sections/analysis.tex",   # the strip figure is declared at the top of the analysis section, which precedes Methods (author, 2026-09-17)
                headline="Geometric and topological properties of persona transformations correlate with undesirable and misaligned behavior",   # one line at 7 in, _fit_headline steps the size down to fit (author, 2026-10-02; two lines ran into the column heads, 2026-09-17)
                subtitle="one panel per family, each on its own excess scale; the tinted rows sit outside the Qwen LoRA pool"):
    """Figure 3: one small panel per family on the left, the whole cast of every row on the right.

    AUTHORED AT ITS PRINTED WIDTH. results_geometry.tex sets this file at 0.68\\columnwidth of a
    5.5 in text block, so it prints 3.74 in wide. Saved at 5.5 in it was rescaled to 0.68 and the
    6.6 pt row labels reached the page at 4.5 pt. Every margin below is an inch, because the room a
    label needs does not scale with the canvas.
    """
    global YM, STAT_COMPACT
    YM = _Concern()          # Figure 4 alone reads concerning behaviour (author, 2026-09-16)
    STAT_COMPACT = True
    W = tex_printed_width(tex_rel, name)
    COUNTER = set()          # the counter slice is its own figure again (user, 2026-09-16)
    nrows = sum(len(rows) for _, _, rows in blocks)
    # The columns, left to right: the rate axis and its ticks, the family panels, the row labels,
    # the cast strips, and a right margin wide enough for the last tick label.
    # One column of family panels, the row labels, then the cast strips. Widths in inches.
    # THREE COLUMNS: the two readings and one picture of the cast (author, 2026-09-16). The rate
    # axis is shared, so only the first column carries its ticks.
    NCOL = 1 + len(EXTRA)
    LEFT, GUT, RP, RIGHT = 0.48, 1.02, 0.80, 0.16           # the rate axis with its title inside the first band, the label gutter (measured below at 4.9 pt names; it was 1.34 at 6.1 pt, and the width it gave up went to the panels; author, 2026-09-17), the strip, the margin
    # A group gap carries the right edge of one band, a white gutter, and the next band's left
    # inset, which is the same for all three bands so they read as siblings (audit, 2026-09-17).
    GAPS = [0.17, 0.55, 0.17, 0.17, 0.55]   # one gap inside a group, one between two groups (0.08 pad + 0.16 white gutter holding the G title + about 0.30 of measured tick-label inset)
    assert len(GAPS) == NCOL - 1
    LP = (W - LEFT - GUT - RP - RIGHT - sum(GAPS)) / NCOL
    GAP = GAPS[0]
    HEADS = 0.50                               # inches: tick labels, air, family name, two stat lines (n moved onto the name, 2026-09-18) (0.72 read cramped, author 2026-09-17)
    FLOOR = 6                                  # rows: the shortest panel still has to show an interval
    # 0.063 in per stacked row unit reproduces the 5.72 in the pool's thirty rows were laid out in;
    # the counter's eleven rows take their own share instead of a whole page (author, 2026-09-16)
    # EACH FAMILY'S BAND HOLDS ITS OWN STRIP ROWS (author, 2026-09-18: "space out the model groups so they are
    # more aligned"): a strip group starts level with its family's name and its rows run beside the panel, so a
    # panel is as tall as its rows need beyond the header, and never below the shortest readable panel.
    PITCH, HEADROW, PANEL_MIN = 0.058, 0.12, 0.30        # inches: row pitch (0.063 until four organisms landed on 2026-09-18 and the float outgrew the page), name-to-first-row, shortest panel
    NROWS = [len(rows) + len(GHOST_ROWS.get(f, [])) for _b, f, rows in blocks]
    PANELS = [max(PANEL_MIN, HEADROW + n * PITCH - HEADS + 0.04) for n in NROWS]
    ROWS = HEADS * (len(blocks) - 1) + sum(PANELS)   # per-family y ranges keep the median interval over the marker at this scale (audit, 2026-09-17)
    TOP, BOT = 1.10, 0.40                      # one-line headline, the group names, three-line column heads; the bottom band holds the last row's tick labels AND the two-row key (they collided at 0.34, 2026-09-17)
    HEAD_PAD = 28                              # points above the first panel (34 let the heads rise into the band names at TOP 1.10, 2026-09-18): over its family name and three stat lines
    H = ROWS + TOP + BOT
    fig = plt.figure(figsize=(W, H))
    _gs = lambda x0, x1: _gridspec.GridSpec(1, 1, figure=fig, left=x0 / W, right=x1 / W, top=1 - TOP / H, bottom=BOT / H)
    _xs = [LEFT]
    for _g in GAPS:
        _xs.append(_xs[-1] + LP + _g)
    outer = [_gs(x, x + LP)[0] for x in _xs] + [_gs(_xs[-1] + LP + GUT, W - RIGHT)[0]]
    # hspace is set from the gap the panel heads actually need. Left as a fraction of the average
    # panel it swallowed more than a third of the column, and the three-row families came out
    # 0.26 in tall, where a ten-point rate interval is thinner than its own marker.
    # The gap between two stacked panels carries the upper panel's own x tick labels as well as the
    # lower panel's name and two lines of statistics, because every family keeps its own excess
    # scale. At 0.30 in the name printed on the tick row above it.
    # Measured, not guessed: 11 pt of tick labels under the upper panel, 3 pt of air, 8 pt of family
    # name, 2 pt, 15.5 pt of two statistics lines, 2 pt. At 0.42 in the name printed on the tick row.
    draw = ROWS - HEADS * (len(blocks) - 1)
    hspace = HEADS / (draw / len(blocks))
    MS = 2.4                                   # pt: small enough that the median interval clears it (audited below)
    # The message once, across the whole canvas, with the cast named under it. Set as a column head
    # over the left panel it was 1.75 in of bold type over a 0.96 in column and ran into the head of
    # the strips beside it.
    _fit_headline(fig, headline, 0.07, W, H, 9.4)
    # the two readings are named as column heads above the stacks, without notation (author,
    # 2026-09-16: "remove notation, also put at top"); nothing is written under the x axes
    # Each rotated axis title sits inside its own group band, clear of that column's tick labels.
    # Centred on the band edge, half of every glyph printed on grey and half on white (audit,
    # 2026-09-17).
    stack_axes = draw_slice_stack(fig, outer[0], blocks, "assistant\nshift\ndeviation", hspace=hspace, ms=MS, fs_title=7.0,
                                  ghosts=GHOST_ROWS, fs_stat=5.4, fs_tick=5.5, fs_xlab=7.2, ylab_x=YLAB_LEFT_IN / W,
                                  title_pad=18.0, stat_2line=True, xbins=2, floor=FLOOR,
                                  xlab="", head_pad=HEAD_PAD, fs_head=6.0, equal_heights=True, xfmt="%.2f",
                                  heights_in=PANELS)
    # the two assistant readings are read against U, the three spread readings against G, with the
    # G axis on the first spread column (author, 2026-09-17: "the axis for these should be G")
    _all_axes = list(stack_axes)
    for _j, (_xk, _lab) in enumerate(EXTRA):
        YM = _MB() if _j >= 1 else _Concern()
        _bj = reading_blocks(blocks, _xk)
        _gj = {f: [dict(r, x=reading_x(f, r["o"], _xk)) for r in rs] for f, rs in GHOST_ROWS.items()}
        _all_axes += draw_slice_stack(fig, outer[1 + _j], _bj, _lab, hspace=hspace, ms=MS, fs_title=7.0, ghosts=_gj,
                         fs_stat=5.4, fs_tick=5.5, fs_xlab=7.2, ylab_x=(_xs[1 + _j] - YLAB_GAP_IN) / W, title_pad=18.0,
                         stat_2line=True, xbins=2, floor=FLOOR, show_ylab=(_j == 1),
                         xlab="", xci=reading_ci_of(_xk), show_title=False, show_ytick=(_j in (1, 4)),
                         head_pad=HEAD_PAD, fs_head=6.0, equal_heights=True, xfmt="%.2f",
                         xnull=XNULL.get(_xk, 0.0), heights_in=PANELS,
                         stat_fn=lambda g, f, _b=_bj: second_stat(g, f, _b))
    YM = _Concern()          # the strip's row labels read U
    _inset = _band_inset(fig, _xs, W); _pad = _band_pad(fig, _xs, LP, _all_axes)
    _group_bands(fig, _xs, LP, W, H, H - 0.25, BOT - 0.13, inset=_inset, pad=_pad)
    # the G title once more, centred in the gutter before the cast-topology band, which reads G too
    # (author, 2026-09-17: 'copy axis to 3rd column'); placed from the band edges, not a fixed offset
    _g2 = [t for t in fig.texts if t.get_text() == _MB().label and abs(t.get_rotation() - 90) <= 1]
    if _g2:
        _x0 = _xs[GROUPS[-1][1][0]] - _inset; _x1 = _xs[GROUPS[-2][1][-1]] + LP + _pad
        fig.text((_x0 + _x1) / 2 / W, _g2[0].get_position()[1], _MB().label, rotation=90,
                 va='center', ha='center', fontsize=_g2[0].get_fontsize())
    for _a in _all_axes:
        _a.patch.set_alpha(0.0)   # the group band shows through the panels
    ax = fig.add_subplot(outer[-1])
    ax.patch.set_alpha(0.0)   # the out-of-pool row wash runs through the strip too (review 2026-09-18)
    # WHERE EACH FAMILY'S NAME SITS, measured on the drawn panels, in inches below the strip's top. The strip is
    # raised to the first name so the first group can start level with it.
    fig.canvas.draw(); _rr = fig.canvas.get_renderer()
    _nm_c = [(a._hdr_title.get_window_extent(_rr).y0 + a._hdr_title.get_window_extent(_rr).y1) / 2 for a in stack_axes]
    _nm_top = stack_axes[0]._hdr_title.get_window_extent(_rr).y1
    _pos = ax.get_position(); _top_px = _nm_top + 0.05 * fig.dpi
    ax.set_position([_pos.x0, _pos.y0, _pos.width, _top_px / fig.bbox.height - _pos.y0])
    _ax_h = (_top_px - _pos.y0 * fig.bbox.height) / fig.dpi
    _name_y = [(_top_px - c) / fig.dpi for c in _nm_c]
    # where each family's panels END, in the same inches below the strip's top: the rows are centred between the
    # family name and this line (author, 2026-09-18: "the right plots are not well centered")
    _pan_bot = [(_top_px - a.get_window_extent(_rr).y0) / fig.dpi for a in stack_axes]
    y = 0; yt = []; yl = []; yctl = []; yghost = []; rows_y = {}; _heads = []
    _strip_rows = []   # the strip, top to bottom, for the table check (2026-09-19)
    for _k, (blab, f, rows) in enumerate(blocks):
        # the family name stays level with its panels' name; its rows are CENTRED in the space from just under that
        # name to the bottom of its panels, and fall back to starting under the name when they fill the space
        _nrow = len(rows) + len(GHOST_ROWS.get(f, []))
        _rtop = _name_y[_k] + HEADROW
        y = _rtop + max(0.0, (_pan_bot[_k] - _rtop - (_nrow - 1) * PITCH) / 2)
        # Rows in RATE order, highest at the top of each family, so the rate labels beside the rows
        # read as the axis they are and the line through the stars shows whether the excess falls
        # with the rate. Ordering by the excess instead put a 58% organism below a 5% one and read
        # as misleading (user, 2026-09-15). The topic-locked control has no rate and stays last.
        def _order(r):
            v = YM.value(r, f) if r["em"] is not None else np.nan
            return (1 if "mismatch" in r["o"] else 0, -(v if np.isfinite(v) else -1.0))
        for r in sorted(rows, key=_order):
            ctl = is_ctl(r); rows_y[(f, r["o"])] = y   # a zero-rate organism is not a control (author, 2026-09-16)
            yt.append(y)
            # Every organism is named in full. "obviously-bad-med" was cut to fit the gutter while
            # the longer "bad-user-medical" beside it was spelled out, so one row read as a
            # truncation fault; the gutter carries the full name instead (audit, 2026-09-17).
            # THE ONE LABEL of _organisms.label(), as Tables 4 and 5 print it: every row names its tuning, the LoRA rows
            # too (author, 2026-09-19: "put (LoRA) in ones that are"; "the main paper figure and these tables should be
            # consistent")
            _nm = _org_label(f, r["o"])
            yl.append(f"{_nm}  " + (_tl_label() if "mismatch" in r["o"] else YM.rowlabel(r, f)))
            _strip_rows.append(dict(f=f, o=r["o"], label=_nm, control=bool(ctl), grey=False))
            yctl.append(ctl); yghost.append(False); y += PITCH
        # the block's name over its first row, in the block's own colour: without it the reader
        # cannot tell which stack of rows belongs to which panel on the left (user, 2026-09-16)
        _heads.append((_rtop, blab, f))   # the name is drawn at _rtop - HEADROW, level with the panel's family name
        # out of the pool: grey name, grey marks, no line (author, 2026-09-17); grouped by kind and, inside
        # a kind, in rate order like the rows above them (author, 2026-09-18: "order these by rate too!")
        _gk_order = {"rank1": 0, "full": 1, "distill": 2, "broad": 3}
        def _gv(r):
            v = YM.value(r, f) if r["em"] is not None else np.nan
            return -(v if np.isfinite(v) else -1.0)
        for r in sorted(GHOST_ROWS.get(f, []), key=lambda r: (_gk_order[ghost_kind(f, r["o"])], _gv(r))):
            rows_y[(f, r["o"])] = y; yt.append(y)
            yl.append(f"{_org_label(f, r['o'])}  " + YM.rowlabel(r, f))   # the one label, as the tables print it (2026-09-19)
            _strip_rows.append(dict(f=f, o=r["o"], label=_org_label(f, r["o"]), control=False, grey=True))
            yctl.append(False); yghost.append(ghost_kind(f, r["o"])); y += PITCH
        if _k + 1 < len(blocks) and y - PITCH > _name_y[_k + 1] - 0.03:
            raise SystemExit(f"{name}: {f}'s strip rows run past the next family's name; raise its panel")
    # THE STRIP'S ROWS, top to bottom, with the label, the green and the grey each row prints: Tables 4 and 5 must
    # list the same rows in the same order with the same names and shades, and scripts/check_paper_numbers.py proves
    # it from this file (author, 2026-09-19: "the main paper figure and these tables should be consistent")
    if name == "excess_slice_main":
        json.dump(dict(figure=name, rows=_strip_rows), open(_paths.DATA_OUT / "strip_layout.json", "w"), indent=1)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=3.0)   # 3.4 until 2026-09-19 (author: "SMALLER FONT!")   # 3.9 overran the gutter once names went monospace (author, 2026-09-18: "font needs to be smaller to fit")   # 6.1, then 4.9, collided with the panels' tick labels (author, 2026-09-17 and 2026-09-18: 'way smaller')
    _bt = ax.get_yaxis_transform()
    for _y0, _blab, _f in _heads:
        # OUT of the strip and into the label gutter, right-aligned over the family's own rows: inside
        # the strip it sat on the left rule at one end and ran into the zero line at the other,
        # whichever indent it was given (author, 2026-09-17)
        ax.text(-0.012, _y0 - HEADROW, _blab.replace(RANK1_SUFFIX, ' rank-1').replace(' (LoRA)', ''), transform=_bt,
                fontsize=5.2, fontweight='bold', color=block_color(_blab, _f), ha='right', va='center', zorder=8)
    ax.set_ylim(_ax_h, 0.0)                     # inches below the raised top; the key sits in the bottom band
    # THE NON-QWEN ROWS, each washed in its own family's hue across the whole row (author, 2026-09-18: "the
    # minstral / llama / gemma background is different hue accros each row"). A row runs from just over its
    # family name to just under its panels' tick labels, so the wash never reaches the next family's name.
    from matplotlib.patches import Rectangle as _RectW
    fig.canvas.draw(); _rr = fig.canvas.get_renderer()
    _wash = []
    for _k, (_blab, _f, _rows) in enumerate(blocks):
        if _f.startswith("Qwen"):
            continue
        _a = stack_axes[_k]
        _name_top = _a._hdr_title.get_window_extent(_rr).y1 / fig.bbox.height
        # the first washed row starts halfway between the pool row's tick labels and its own name
        _prev = stack_axes[_k - 1] if _k else None
        _prev_ticks = (min(t.get_window_extent(_rr).y0 for t in _prev.get_xticklabels() if t.get_text()) / fig.bbox.height
                       if _prev is not None else _name_top + 0.06 / H)
        _top = (_name_top + _prev_ticks) / 2
        _bot = min(t.get_window_extent(_rr).y0 for t in _a.get_xticklabels() if t.get_text()) / fig.bbox.height - 0.03 / H
        _wash.append([_f, _bot, _top])
    # consecutive washed rows meet with a white gap of 0.03 in, so two near hues cannot merge into one block
    for _i in range(len(_wash) - 1):
        _mid = (_wash[_i][1] + _wash[_i + 1][2]) / 2
        _wash[_i][1], _wash[_i + 1][2] = _mid + 0.015 / H, _mid - 0.015 / H
    if _wash:
        _wash[-1][1] = max(_wash[-1][1], (BOT - 0.13) / H)   # end with the column bands
    import matplotlib.colors as _mc
    _sx1 = ax.get_position().x1 + 0.03 / W                     # through the strip, to its right edge
    # the gap between two washed rows is painted WHITE across the whole row, above the column bands, so it reads as
    # one clean separator; left open, the grey bands showed through it and cut the line into pieces (author,
    # 2026-09-18: "and cuts off white line!")
    for _i in range(len(_wash) - 1):
        _g0, _g1 = _wash[_i][1], _wash[_i + 1][2]
        if _g1 < _g0:   # the lower row's top sits below the upper row's bottom: a gap
            fig.patches.append(_RectW((0.06 / W, _g1), _sx1 - 0.06 / W, _g0 - _g1, transform=fig.transFigure, fc="white",
                                      ec="none", alpha=1.0, zorder=-4.5))
    for _f, _bot, _top in _wash:
        _c = np.array(_mc.to_rgb(FAMC[_f]))
        _fc = tuple(0.86 * np.ones(3) + 0.14 * _c)              # pre-blended and opaque: one flat hue across the row
        fig.patches.append(_RectW((0.06 / W, _bot), _sx1 - 0.06 / W, _top - _bot, transform=fig.transFigure, fc=_fc,
                                  ec="none", alpha=1.0, zorder=-4.5))
    # the rate's symbol over the number column: the full name sits on the first panel's axis, and
    # spelling it out again here held the gutter wide (author, 2026-09-17: "put U in axis clearly
    # such that columns next to it have more space")
    ax.text(-0.012, -0.01, "$U$", transform=_bt, fontsize=6.0, style='italic', color='#444',   # above the first family head, which sits at -0.92 with its top near -1.3
            ha='right', va='bottom', zorder=8)
    for lab, c in zip(ax.get_yticklabels(), yctl):
        if c: lab.set_color("#2f7d4f"); lab.set_fontweight("bold")
    for lab, g in zip(ax.get_yticklabels(), yghost):
        if g: lab.set_color(STRIP_GHOST["text"]); lab.set_fontsize(3.0)   # one size for every row (author, 2026-09-19: "SMALLER FONT!"; 4.2 printed the grey rows larger than the rest)   # the tagged grey rows keep 4.2 pt (author, 2026-09-18: "keep (ft...) and such the current font")
    ax.tick_params(axis='y', length=0, pad=2); ax.tick_params(axis='x', labelsize=6.2, pad=1.5)
    # the strip reads the same axis as the first column, so it prints its numbers the same way
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_tickfmt("%.2f")))
    ax.grid(axis='x', color='#888', alpha=0.20, lw=0.45); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    _gk = {(f, r["o"]) for f, rs in GHOST_ROWS.items() for r in rs}
    draw_cast_strips(ax, [(b, f, rs + GHOST_ROWS.get(f, [])) for b, f, rs in blocks], rows_y, ms_star=3.8, ms_diam=3.0, s_dot=1.9, counter=COUNTER,
                     evil=True, ghost=_gk, pitch=PITCH)
    # ONE NAME for this axis. The head said "shift projections onto the assistant axis" and the
    # axis under it said "shift deviation", so the reader could not tell whether the strip showed a
    # projection or a deviation (audit, 2026-09-17). Both now carry the caption's words.
    ax.set_xlabel("")   # the column head names the axis; the label sat on the key's first row (review 2026-09-18)   # the full name ran past the trim edge in a 0.88 in strip
    ax.set_title("shift deviation\nof every role\nof the cast", fontsize=7.0, fontweight='bold', loc='left',
                 color='#333', pad=9.0)
    # A MINI LEGEND, not leader lines: the block names now sit at the top of each stack and the
    # leaders ran through them, but the four marks still have to be named (user, 2026-09-16).
    # EVERY MARK ON THE CANVAS IS NAMED HERE. The key showed a green star for the control while the
    # panels drew the control as a green ring, and neither the green row band nor the zero rule was
    # named at all (audit, 2026-09-17). The control row's swatch stacks its band and its star, which
    # is how the strip draws it.
    from matplotlib.patches import Patch as _Patch
    from matplotlib.legend_handler import HandlerTuple as _HT
    _ctlrow = (_Patch(fc="#dff1e6", ec="none"),
               Line2D([0], [0], marker='*', ls='none', mfc='#2f9e6b', mec='#111', mew=0.5, ms=6))
    _key = [Line2D([0], [0], marker='*', ls='none', mfc='#f2c200', mec='#111', mew=0.5, ms=6, label='assistant'),
            Line2D([0], [0], marker='o', ls='none', mfc='none', mec='#8a6d00', mew=0.9, ms=5.6, label='default'),
            Line2D([0], [0], marker='o', ls='none', mfc='#b4b4b4', mec='none', ms=2.6, label='other roles'),
            Line2D([0], [0], marker='o', ls='none', mfc='white', mec='#2f9e6b', mew=0.7, ms=4.0, label='control'),
            _ctlrow,
            Line2D([0], [0], marker='s', ls='none', mfc='white', mec='#444', mew=0.9, ms=4.2, label='baseline'),
            Line2D([0], [0], color='#b08900', lw=0.9, label='assistant, by family'),
            Line2D([0], [0], color='#333', ls=':', lw=0.8, label='default, by family'),
            Line2D([0], [0], marker='o', ls='none', mfc=EVILC, mec='none', ms=2.8, label='evil roles'),
            Line2D([0], [0], marker='|', ls='none', color=EVILC, ms=5, mew=1.3, label='evil mean'),
            Line2D([0], [0], marker='|', ls='none', color='#777', ms=5, mew=0.9, label='zero'),
            Line2D([0], [0], marker='o', ls='none', mfc=GHOST_STYLE["rank1"]["face"], mec=GHOST_STYLE["rank1"]["edge"], mew=0.6, ms=3.6, label='rank-1, out of pool'),
            Line2D([0], [0], marker='o', ls='none', mfc=GHOST_STYLE["full"]["face"], mec=GHOST_STYLE["full"]["edge"], mew=0.6, ms=3.6, label='full, out of pool'),
            Line2D([0], [0], marker='o', ls='none', mfc=GHOST_STYLE["distill"]["face"], mec=GHOST_STYLE["distill"]["edge"], mew=0.6, ms=3.6, label='distill or broad, out of pool')]
    _key = [Line2D([0], [0], marker='o', color='#666', lw=1.0, ms=3.0, mec='white', mew=0.4, label='organism')] + _key \
         + [_Patch(fc="#ece6e0", ec="none", label="outside the pool")]
    _klab = [h.get_label() if isinstance(h, (Line2D, _Patch)) and h.get_label() and not h.get_label().startswith('_') else 'control row' for h in _key]
    # UNDER THE WHOLE FIGURE, not over the strip: eight marks in a 1.5 in column ran off the
    # canvas and the trim guard caught it (2026-09-16).
    _lg = fig.legend(handles=_key, labels=_klab, loc='lower left', bbox_to_anchor=(0.07 / W, 0.015 / H), ncol=8,   # 16 marks in two rows (2026-09-18); 13 marks in two rows; at 6 columns the third row hit the bottom panels' ticks (2026-09-18)
                     fontsize=5.4, frameon=False, handlelength=1.1, handletextpad=0.55,
                     columnspacing=1.2, borderpad=0.0, labelspacing=0.30,
                     handler_map={tuple: _HT(ndivide=1, pad=0.0)})
    for _t in _lg.get_texts():
        _t.set_color('#555')
    # The row labels hang off the left of the strips into the gutter. Measured, because the longest
    # name overran the family panel's last tick label twice.
    fig.canvas.draw(); _r = fig.canvas.get_renderer()
    _wid, _wtxt = max((t.get_window_extent(_r).width / fig.dpi, t.get_text()) for t in ax.get_yticklabels())
    print(f"  {name}: widest row label {_wid:.3f} in in a {GUT:.2f} in gutter")
    if _wid > GUT - 0.06:
        raise SystemExit(f"{name}: the widest row label is {_wid:.3f} in ({_wtxt!r}) and the gutter "
                         f"is {GUT:.2f} in. Widen GUT or shorten the label.")
    # The family name and its two statistics lines are set flush left over a panel narrower than
    # they are, so they run into the gutter. A little of that is fine, because the gutter's left
    # half is empty. Half the gutter is not, so the build refuses there.
    _hdr = max((t.get_window_extent(_r).width for a in stack_axes
                for t in list(a.texts) + [a._hdr_title] if t.get_text()), default=0) / fig.dpi
    _over = _hdr - LP
    print(f"  {name}: widest panel header {_hdr:.3f} in over a {LP:.2f} in panel "
          f"({max(_over, 0):.3f} in into the gutter)")
    if _over > 0.5 * GUT:
        raise SystemExit(f"{name}: a panel header is {_hdr:.3f} in over a {LP:.2f} in "
                         f"panel and reaches {_over:.3f} in into a {GUT:.2f} in gutter. Shorten it "
                         f"or widen the panel.")
    _band_guard(fig, _xs, LP, W, _all_axes, name, inset=_inset, pad=_pad)
    _thin_xticks(fig, _all_axes); _tick_guard(fig, _all_axes, name); _head_guard(fig, name)
    _lgw = _lg.get_window_extent(_r).width / fig.dpi
    print(f"  {name}: key {_lgw:.2f} in on a {W:.2f} in canvas")
    if _lgw > W - 0.20:
        raise SystemExit(f"{name}: the key is {_lgw:.2f} in on a {W:.2f} in canvas. Raise its rows "
                         f"or shorten a label.")
    _interval_audit(stack_axes, blocks, name, ms_pt=MS)
    assert_canvas(fig, tex_rel, name)
    out = OUT / f"{name}.pdf"
    png = str(out).replace(".pdf", ".png")
    fig.savefig(out); fig.savefig(png, dpi=300); plt.close(fig)
    _edge_guard(png)
    print("wrote", out, f"({nrows} rows, {len(blocks)} families, canvas {W} x {H:.2f} in)")
    YM = _Rate(); STAT_COMPACT = False             # every later figure of this script keeps the rate


def rows_stat():
    """Spearman and Pearson of a block's own rows (x against YM), each with its exact two-sided permutation p;
    three short lines as the grids print them."""
    def _f(rows, f):
        keep = [r for r in rows if np.isfinite(r["x"]) and r.get("em") is not None and "mismatch" not in r["o"]]
        n = len(keep)
        if n < 3:
            return "$\\rho$, $r$ undefined" if n == 2 else ""   # n rides on the family name
        from scipy.stats import rankdata as _rk
        x = np.array([r["x"] for r in keep]); y = np.array([YM.value(r, f) for r in keep])
        rx, ry = _rk(x) - (n + 1) / 2, _rk(y) - (n + 1) / 2
        rho = float(rx @ ry / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
        xc, yc = x - x.mean(), y - y.mean()
        rr = float(xc @ yc / np.sqrt((xc ** 2).sum() * (yc ** 2).sum()))
        import fig_metric_slices as _MS
        p_rho = _MS.exact_p(x, y, rho, "rho"); p_r = _MS.exact_p(x, y, rr, "r")
        ps = lambda q: "<.001" if q < 0.001 else f"{q:.3f}".lstrip("0")
        return (f"$\\rho$ {_num(rho)} ({ps(p_rho)})" + "\n" + f"$r$ {_num(rr)} ({ps(p_r)})" )   # n rides on the family name
    _f.by_rows = True
    return _f


def appendix_grids(blocks, tex_rel="sections/appendices/app_behaviour.tex", prefix="readings",
                   headline="The six readings against", by_rows=False, row_in=0.053, reads=None, measures=None,
                   groups=None):
    """Appendix B: the five-reading grid of Figure 5, without the strip, once for each judged quantity
    M, B and C (author, 2026-09-17: "delete everything in appendix B. Instead put [the grid] x3 but
    for M, B, C"). Same panels, same statistics, same widths; every p is two-sided."""
    global YM, STAT_COMPACT, SHAPE_ALL
    READS = reads or [("REF_assistant_excess", "assistant\nshift\ndeviation", -1)] + [(k, l, EXTRA_SIGN[k]) for k, l in EXTRA]
    for tag, meas, _flip, head in (measures or (("M", _RateM(), 1, "the misalignment rate $M$"),
                                                ("B", _Breadth(), 1, "the breadth $B$"),
                                                ("C", _Coh(), -1, "the coherence $C$"))):
        YM = meas; STAT_COMPACT = True
        SHAPE_ALL = mixed_blocks(blocks)   # the key names the shapes exactly when a block is mixed, so every block draws them then
        name = f"{prefix}_{tag}"
        W = tex_printed_width(tex_rel, name)
        NCOL = len(READS); LEFT, RIGHT = 0.48, (0.16 if prefix == "readings" else 0.26 if NCOL <= 6 else 0.36)   # the margins hold the first band's inset and half of the last tick label (Appendix C's kNN column ends on a "1.00" at the trim edge)
        # the group gaps as in Figure 3: 0.55 between groups (two band edges and a white gutter), 0.17 inside one
        _grp = {c: gi for gi, (_n, cols, _a) in enumerate(groups or GROUPS) for c in cols}
        # a seven-column grid (Appendix D) needs a wider gap: its narrower panels leave more of each header
        # overhanging, and at 0.55 in two bands met with no gutter (2026-09-19)
        _gb = 0.55 if len(READS) <= 6 else 0.76
        GAPS = [_gb if _grp[c] != _grp[c + 1] else (0.17 if len(READS) <= 6 else 0.20) for c in range(len(READS) - 1)]
        LP = (W - LEFT - RIGHT - sum(GAPS)) / NCOL
        HEADS, FLOOR = 0.72, 6
        ROWS = HEADS * (len(blocks) - 1) + row_in * sum(max(len(rows), FLOOR) + 0.8 for _, _, rows in blocks)   # 0.050 left a 30% white band under the grid page (2026-09-17)
        TOP, BOT, HEAD_PAD = 1.40, 0.34, 46       # headline, group names, three-line heads (2026-09-17)
        H = ROWS + TOP + BOT
        fig = plt.figure(figsize=(W, H))
        _gs = lambda x0, x1: _gridspec.GridSpec(1, 1, figure=fig, left=x0 / W, right=x1 / W, top=1 - TOP / H, bottom=BOT / H)
        _xs = [LEFT]
        for _g in GAPS:
            _xs.append(_xs[-1] + LP + _g)
        outer = [_gs(x, x + LP)[0] for x in _xs]
        draw = ROWS - HEADS * (len(blocks) - 1); hspace = HEADS / (draw / len(blocks))
        _fit_headline(fig, f"{headline} {head}", 0.07, W, H, 9.4)
        stacks = []
        for j, (xk, lab, _sign) in enumerate(READS):
            _own = xk == "REF_assistant_excess"   # the one reading the blocks already carry as x
            bj = blocks if _own else reading_blocks(blocks, xk)
            stacks += draw_slice_stack(fig, outer[j], bj, lab, hspace=hspace, ms=2.6, fs_title=7.0, fs_stat=5.7,
                                       fs_tick=(5.5 if NCOL <= 6 else 5.3), fs_xlab=7.2, ylab_x=YLAB_LEFT_IN / W, title_pad=27.0, stat_2line=True,
                                       xbins=2, floor=FLOOR, show_ylab=(j == 0), xlab="",
                                       xci=None if _own else reading_ci_of(xk), show_title=(j == 0),
                                       show_ytick=(j == 0), head_pad=HEAD_PAD, fs_head=(6.7 if NCOL <= 6 else 6.3),
                                       equal_heights=True, xfmt=("%.1f" if xk == "loss" else "%.2f"),   # the loss spans nats, not shares
                                       xnull=XNULL.get(xk, 0.0),
                                       stat_fn=(rows_stat() if by_rows else
                                                (lambda g, f, _b=bj: second_stat(g, f, _b))))
        SHAPE_ALL = False
        _inset = _band_inset(fig, _xs, W); _pad = _band_pad(fig, _xs, LP, stacks, groups=groups)
        print(f"  {name}: panel {LP:.3f} in, band inset {_inset:.3f}, pad {_pad:.3f}, right margin {RIGHT:.2f} (the last band ends {W - (_xs[-1] + LP + _pad):.3f} in from the edge)")
        _group_bands(fig, _xs, LP, W, H, H - 0.24, BOT - 0.13, inset=_inset, pad=_pad, groups=groups)
        for _a in stacks:
            _a.patch.set_alpha(0.0)
        # The organism swatch carries the six family colours the grid is drawn in. A grey dot named
        # a colour that appears nowhere on the page (audit, 2026-09-17).
        from matplotlib.legend_handler import HandlerTuple as _HT
        # three of the six, at the size the panels draw them: six swatches in one handle overlapped
        # each other, which is the fault the swatch is there to cure (audit, 2026-09-17)
        _dots = tuple(Line2D([0], [0], marker="o", ls="none", color=FAMC[f], ms=2.6)
                      for _b, f, _r in blocks[:3])
        key = [_dots,
               Line2D([0], [0], marker="o", ls="none", mfc="white", mec="#2f9e6b", mew=0.9, ms=3.4, label="aligned control"),
               Line2D([0], [0], marker="s", ls="none", mfc="white", mec="#444", mew=0.9, ms=3.4, label="baseline"),
               Line2D([0], [0], color="#555", lw=0.7, alpha=0.6, label="95% interval")]
        if mixed_blocks(blocks):   # name the two shapes of a mixed block (Gemma-3-12B, 2026-09-18)
            key += [Line2D([0], [0], marker=TUNE_MARK["full"], ls="none", color="#555", ms=2.6, label="full fine-tune"),
                    Line2D([0], [0], marker=TUNE_MARK["lora"], ls="none", color="#555", ms=3.25, label="LoRA")]
        klab = ["organism, by family"] + [h.get_label() for h in key[1:]]
        lg = fig.legend(handles=key, labels=klab, loc="lower left", bbox_to_anchor=(0.07 / W, 0.02 / H),
                        ncol=len(key), fontsize=5.8, handler_map={tuple: _HT(ndivide=None, pad=0.4)},
                        frameon=False, handlelength=2.2, handletextpad=0.55, columnspacing=1.3, borderpad=0.0)
        for t in lg.get_texts():
            t.set_color("#666")
        _band_guard(fig, _xs, LP, W, stacks, name, inset=_inset, pad=_pad, groups=groups)
        _thin_xticks(fig, stacks); _tick_guard(fig, stacks, name); _head_guard(fig, name)
        assert_canvas(fig, tex_rel, name)
        out = OUT.parent / "appendix" / f"{name}.pdf"; out.parent.mkdir(parents=True, exist_ok=True)
        png = str(out).replace(".pdf", ".png")
        fig.savefig(out); fig.savefig(png, dpi=300); plt.close(fig)
        _edge_guard(png)
        print("wrote", out, f"(canvas {W} x {H:.2f} in)")
    YM = _Rate(); STAT_COMPACT = False


def _label_row(fam, rows, rows_y):
    """The row whose star and diamond the three headroom labels point at.

    It has to be a misaligned row, never a control: a leader that lands on the green control row
    reads as if green were part of what "assistant" means. Among those it is the HIGHEST row that
    separates the two marks well, so the leaders stay short.
    """
    cand = [r for r in rows if not is_ctl(r) and r["em"] >= 0.005]
    if not cand:
        cand = list(rows)
    sep = {r["o"]: abs(cast_excess(fam, r["o"])[2] - (cast_excess(fam, r["o"])[3] or 0)) for r in cand}
    # 0.9, not 0.6. At 0.6 the top row of Qwen3.5-9B qualified on a near miss and its star sat
    # almost entirely under its diamond, so two leaders pointed at one visible mark (2026-09-14).
    good = max(sep.values())
    ok = [r for r in cand if sep[r["o"]] >= 0.9 * good]
    return min(ok, key=lambda r: rows_y[(fam, r["o"])])


def _name_two_marks(ax, aexc, dexc, y, left_x, right_x, right_ha, AN):
    """Name the assistant and the default over one row, in the order their marks sit on the axis.

    Fixed slots crossed the two leaders whenever the diamond fell left of the star, which is most
    rows (2026-09-14). The label on the left goes to whichever mark is on the left."""
    if dexc is None:
        ax.annotate("assistant", (aexc, y), xytext=(left_x, -1.05), ha="left", **AN)
        return
    lo = ("assistant", aexc) if aexc <= dexc else ("default", dexc)
    hi = ("default", dexc) if aexc <= dexc else ("assistant", aexc)
    ax.annotate(lo[0], (lo[1], y), xytext=(left_x, -1.05), ha="left", **AN)
    ax.annotate(hi[0], (hi[1], y), xytext=(right_x, -1.05), ha=right_ha, **AN)


def _interval_audit(axes, blocks, name, ms_pt=3.4):
    """Say, in print, how tall every drawn rate interval is against the marker that sits on it.

    A short stacked panel can hide a real error bar inside its own marker. The only way that stays
    fixed is a number printed on every build, taken from the axes that were actually drawn, rather
    than a memory of having checked once. It refuses when the MEDIAN interval is hidden, which is
    the state in which the panel misreports the whole figure; a zero-rate control whose interval is
    genuinely 1.6 points wide is named instead, and the caption says so.
    """
    hid, all_pts = [], []
    for ax, (_, f, rows) in zip(axes, blocks):
        y0, y1 = ax.get_ylim()
        h_in = ax.get_position().height * ax.figure.get_figheight()
        for r in rows:
            if r.get("n") is None:
                continue
            lo, hi = YM.bounds(r, f)        # in the units the axis is drawn in, whatever the measure
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            pts = (hi - lo) / (y1 - y0) * h_in * 72
            all_pts.append(pts)
            if pts < ms_pt:
                hid.append((f, r["o"], pts))
    if not all_pts:
        raise SystemExit(f"_interval_audit({name}): no intervals measured, so the check cannot fail.")
    med = float(np.median(all_pts))
    print(f"  {name}: y intervals {min(all_pts):.1f}-{max(all_pts):.1f} pt (median {med:.1f}) "
          f"against a {ms_pt} pt marker; {len(hid)} of {len(all_pts)} shorter than the marker"
          + (":" if hid else "."))
    for f, o, pts in sorted(hid, key=lambda t: t[2]):
        print(f"      {f:14s} {o:28s} {pts:.1f} pt")
    if med < ms_pt:
        raise SystemExit(f"_interval_audit({name}): the MEDIAN rate interval is {med:.1f} pt under a "
                         f"{ms_pt} pt marker. Raise the panel height or shrink the marker.")


def intro_figure(fam="Qwen3.5-9B"):
    """Figure 1, for one family: the family's scatter on the left and its cast strips on the right.

    AUTHORED AT ITS PRINTED WIDTH. intro.tex sets it at 0.8\\columnwidth of a 5.5 in text block, so
    it prints 4.40 in wide; the 5.5 in canvas it used to be saved at was rescaled to 0.8 and put
    6.6 pt row labels on the page at 5.3 pt.
    """
    global YM
    YM = _MB()               # Figure 1 reads G = MB (author, 2026-09-16); Figure 4 alone reads U
    W = tex_printed_width("sections/intro.tex", "excess_intro")
    panels = split([r for r in EXC if r["em"] is not None], mode="slice")
    blocks = [b for b in split_rank1(panels["main"]) if b[1] == fam and not b[0].endswith(RANK1_SUFFIX)]
    assert len(blocks) == 1, [b[0] for b in blocks]
    rows = blocks[0][2]; nrows = len(rows)
    # Every margin is an inch, not a fraction: the type has to print at its stated size whatever
    # the canvas is, so the room it needs cannot scale with the canvas.
    # the y label "general misalignment rate (G)" is longer than the three-row panel is tall, so
    # the margins above and below carry its ends (it ran into the trim at 0.74 / 0.46, 2026-09-16)
    ROW, TOP, BOT = 0.235, 0.90, 0.62
    H = ROW * nrows + TOP + BOT
    fig = plt.figure(figsize=(W, H))
    # left panel | the row labels | the strips. The gutter is sized for the longest row label.
    LP, GAP, RP = 1.30, 1.00, 1.66
    outer = _gridspec.GridSpec(1, 2, figure=fig, width_ratios=[LP, RP], wspace=GAP / ((LP + RP) / 2),
                               left=0.44 / W, right=1 - 0.155 / W, top=1 - TOP / H, bottom=BOT / H)
    # The message, stated once, at the top. Figure 1 is the only picture most readers will look at.
    fig.text(0.07 / W, 1 - 0.055 / H, "The assistant leaves the cast, and misalignment rises",
             fontsize=8.8, fontweight="bold", color="#222", ha="left", va="top")
    draw_slice_stack(fig, outer[0], blocks, "", hspace=0.95, ms=5.0, fs_title=8.0, fs_stat=6.8,
                     fs_tick=6.6, fs_xlab=7.6, ylab_x=YLAB_LEFT_IN / W, title_pad=10.5, show_ylab=True)
    ax = fig.add_subplot(outer[1])
    y = 0; yt = []; yl = []; yctl = []; rows_y = {}
    for r in sorted(rows, key=lambda r: -YM.value(r, fam)):
        ctl = is_ctl(r); rows_y[(fam, r["o"])] = y   # a zero-rate organism is not a control (author, 2026-09-16)
        yt.append(y); yl.append(f"{tidy(r['o'], r.get('f')).replace('school-of-reward-hacks', 'reward-hacks')}  {YM.rowlabel(r, fam)}"); yctl.append(ctl); y += 1
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.8)
    # headroom over the first row for the three marks that have to be named
    ax.set_ylim(y - 0.4, -1.30)
    for lab, c in zip(ax.get_yticklabels(), yctl):
        if c: lab.set_color("#2f7d4f"); lab.set_fontweight("bold")
    ax.tick_params(axis='y', length=0, pad=2); ax.tick_params(axis='x', labelsize=6.8, pad=1.5)
    ax.grid(axis='x', color='#888', alpha=0.20, lw=0.45); ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
    draw_cast_strips(ax, blocks, rows_y, ms_star=7.0, ms_diam=4.0, s_dot=3.2)
    ax.set_xlabel("per-role excess", fontsize=7.6, labelpad=3)
    ax.set_title("shift projections\nonto the assistant axis", fontsize=8.0, fontweight='bold', loc='left', color='#333',
                 pad=10.5)
    # The legend strip under the figure named a star and a diamond. The three marks are named where
    # they sit instead, on the top row, and the strip is gone (Tufte: no separate keys).
    top_r = _label_row(fam, rows, rows_y)
    others, _, aexc, dexc = cast_excess(fam, top_r["o"])
    _yl = rows_y[(fam, top_r["o"])]
    # The three marks are named on one line of headroom over the top row, each at its own place on
    # the axis so no two labels touch, each joined to its mark by a leader. The star and the diamond
    # sit within a marker's width of each other, so labels offset from the marks would overlap.
    _x0, _x1 = ax.get_xlim(); _at = lambda t: _x0 + t * (_x1 - _x0)
    # a white plate under each name: the grid ran through the words (audit, 2026-09-16)
    AN = dict(textcoords="data", fontsize=6.4, color="#333", zorder=9, va="center",
              bbox=dict(fc="white", ec="none", pad=1.2),
              arrowprops=dict(arrowstyle='-', lw=0.5, color="#999", shrinkA=1, shrinkB=2.5))
    _name_two_marks(ax, aexc, dexc, _yl, _at(0.02), _at(0.47), "center", AN)
    ax.annotate("other roles", (float(np.median(others)), _yl), xytext=(_at(0.84), -1.05),
                ha="center", **AN)
    assert_canvas(fig, "sections/intro.tex", "excess_intro")
    out = OUT / "excess_intro.pdf"; png = str(out).replace(".pdf", ".png")
    fig.savefig(out); fig.savefig(png, dpi=300); plt.close(fig); _edge_guard(png); print("wrote", out)
    YM = _Rate()



# ---------------------------------------------------------------- Figure 1, lower half
TEASER_FAM = "Qwen3.5-9B"
# FIGURE 1'S OWN PALETTE (author, 2026-09-18: "match baseline border color with symbol in figure ... match
# color of em organism ... slightly different hues ... distinct from other figures"). fig_atlas.py draws the
# three cloud frames in the same three colours (ATLAS_FRAME_COLORS in scripts/regen_paper_figures.py).
T_BASE, T_EM, T_CTL = "#3b5b92", "#d55e00", "#009e73"
T_CTL_BAND = "#dcf1ea"


def teaser_bottom(tex_rel="sections/intro.tex", name="teaser_bottom"):
    """Figure 1, lower half, one panel (d) in two parts: every role's shift for the family's fine-tunes, the
    assistant a star, and the assistant's shift deviation against M with the family's coefficients inside the
    plot. Both read the misalignment rate M (author, 2026-09-18: "have both y axies just be misalignment rate
    M"). One colour per kind across the whole figure: baseline, misaligned fine-tune, aligned control.
    Authored at the column width it prints at."""
    global YM, STAT_COMPACT
    _ym0, _sc0 = YM, STAT_COMPACT
    YM = _RateM(); STAT_COMPACT = False
    panels = split([r for r in EXC if r["em"] is not None], mode="slice")
    blk = [(b, f, rs) for b, f, rs in split_rank1(panels["main"]) if f == TEASER_FAM and not b.endswith(RANK1_SUFFIX)]
    assert len(blk) == 1, f"teaser: expected one {TEASER_FAM} block, found {[b for b, _f, _r in blk]}"
    blab, f, rows = blk[0]
    rows = sorted(rows, key=lambda r: -YM.value(r, f))
    W = tex_printed_width(tex_rel, name) or 3.3125
    H = 1.40
    fig = plt.figure(figsize=(W, H))
    GUT, SW, GAP, YT, PW, RM = 1.02, 0.88, 0.10, 0.26, 0.98, 0.07
    assert abs(GUT + SW + GAP + YT + PW + RM - W) < 0.2, (GUT + SW + GAP + YT + PW + RM, W)
    BOT, TOP = 0.30, 0.22
    axS = fig.add_axes([GUT / W, BOT / H, SW / W, (H - BOT - TOP) / H])
    axP = fig.add_axes([(GUT + SW + GAP + YT) / W, BOT / H, PW / W, (H - BOT - TOP) / H])
    # (d) the strip: one row per fine-tune, most undesirable at the top
    rows_y = {(f, r["o"]): i for i, r in enumerate(rows)}
    draw_cast_strips(axS, [(blab, f, rows)], rows_y, ms_star=6.2, ms_diam=3.6, s_dot=2.2, evil=True)
    # the control rows are styled like the misaligned ones, with no band (author, 2026-09-18: "style this like")
    for _pt in list(axS.patches):
        if matplotlib.colors.to_hex(_pt.get_facecolor()) == "#dff1e6":
            _pt.remove()
    for _ln in axS.lines:
        if isinstance(_ln.get_markerfacecolor(), str) and _ln.get_markerfacecolor().lower() == "#2f9e6b":
            _ln.set_markerfacecolor(T_CTL)
    axS.set_yticks(range(len(rows))); axS.set_yticklabels([])
    # "EM organism #1" over "(risky-financial)" in smaller type (author, 2026-09-18), numbered in U order
    _nm = lambda o: tidy(o).replace("school-of-reward-hacks", "reward-hacks").replace(" (control)", "")
    _bt = axS.get_yaxis_transform(); _k = {"em": 0, "ctl": 0}
    for i, r in enumerate(rows):
        kd = "ctl" if is_ctl(r) else "em"; _k[kd] += 1
        c = T_CTL if kd == "ctl" else T_EM
        head = ("control organism" if kd == "ctl" else "EMA organism") + f" #{_k[kd]}"
        axS.text(-0.035, i + 0.02, f"{head}  {YM.rowlabel(r, f)}", transform=_bt, ha="right", va="bottom",
                 fontsize=5.0, fontweight="bold", color=c)
        axS.text(-0.035, i + 0.06, f"({_nm(r['o'])})", transform=_bt, ha="right", va="top",
                 fontsize=4.2, color=c)
    axS.set_ylim(len(rows) - 0.4, -0.6)
    axS.tick_params(axis="y", length=0, pad=2); axS.tick_params(axis="x", labelsize=5.8, pad=1.5)
    axS.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_tickfmt("%.2f")))
    axS.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5]))
    axS.grid(axis="x", color="#888", alpha=0.20, lw=0.45); axS.set_axisbelow(True)
    axS.spines[["top", "right"]].set_visible(False)
    axS.set_xlabel("shift deviation of every role", fontsize=6.0, labelpad=2)
    # ONE title over both parts (author, 2026-09-18: "d) misalignment grows as assistant role detaches from
    # persona cloud ... polish to make it smaller?")
    # from the figure's left edge, over the row labels: from the strip's own left edge it ran off the column
    _ttl = fig.text(0.02 / W, (H - TOP + 0.10) / H, "(d) misalignment grows as the assistant detaches", fontsize=6.4,
                    fontweight="bold", color="#333", ha="left", va="bottom")
    axS.text(-0.02, 1.0, "$M$", transform=axS.transAxes, ha="right", va="bottom", fontsize=6.0,
             style="italic", color="#777")
    # the result: the assistant's shift deviation against the same M the row labels print
    kind = lambda r: "ctl" if is_ctl(r) else "em"
    xs = [r["x"] for r in rows]; ys = [YM.value(r, f) for r in rows]
    # a dashed least-squares line over the points the coefficients are computed on (author, 2026-09-19: "the
    # connections on right figure plot seem weird. draw best line fit dashed"); it was a polyline in x order
    _b1, _b0 = np.polyfit(np.array(xs, float), np.array(ys, float), 1)
    _xf = np.array([min(xs), max(xs)])
    axP.plot(_xf, _b0 + _b1 * _xf, ls=(0, (3.0, 2.0)), color="#8a8a8a", lw=0.9, zorder=2)
    for r in rows:
        c = T_CTL if kind(r) == "ctl" else T_EM
        draw_uncertainty(axP, r, f, c)
        if kind(r) == "ctl":
            axP.plot(r["x"], YM.value(r, f), "o", mfc="white", mec=c, mew=1.0, ms=4.4, zorder=4)
        else:
            axP.plot(r["x"], YM.value(r, f), "o", color=c, mec="white", mew=0.5, ms=4.4, zorder=4)
    _yb = YM.base(f)
    axP.plot(0.0, _yb, "s", color=T_BASE, mec="white", mew=0.5, ms=4.2, zorder=5)   # the baseline, as in (a)
    axP.axvline(0, color="#777", lw=0.6, zorder=1)
    _xl = [excess_ci(f, r["o"]) for r in rows]
    _lo = min(min(xs), min(v[0] for v in _xl if np.isfinite(v[0])), 0.0); _hi = max(max(xs), max(v[1] for v in _xl if np.isfinite(v[1])), 0.0)
    _pad = 0.10 * (_hi - _lo); axP.set_xlim(_lo - _pad, _hi + _pad)
    _yhi = max(ys + [b[1] for b in (YM.bounds(r, f) for r in rows) if np.isfinite(b[1])])
    axP.set_ylim(-YM.floor * _yhi * 1.12, _yhi * 1.12)
    axP.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    axP.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v:.0f}%"))
    axP.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5], min_n_ticks=2))
    axP.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_tickfmt("%.2f")))
    axP.tick_params(labelsize=5.8, pad=1.5)
    axP.grid(True, color="#888", alpha=0.20, lw=0.45); axP.set_axisbelow(True)
    axP.spines[["top", "right"]].set_visible(False)
    axP.set_xlabel("assistant shift deviation $\\xi_A$", fontsize=6.0, labelpad=2)
    axP.set_ylabel("$M$", fontsize=6.4, labelpad=5.0, rotation=0, va="center")   # clear of the tick labels
    # the coefficients INSIDE the plot, in its empty upper right (author, 2026-09-18)
    st = YM.stats()["groups"]["lora"]["per_family"][f]
    _p = lambda q: "<.001" if q < 1e-3 else f"{q:.3f}".lstrip("0")
    # one text per coefficient, each bold when its own two-sided p is below .05, as in Figure 4 (2026-09-19)
    for _k, (_sym, _v, _pv) in enumerate((("\\rho", st["rho"], st["exact_p_two_sided"]),
                                          ("r", st["r_pearson"], st["exact_p_pearson_two_sided"]))):
        _bold = _pv < 0.05
        _s = ("\\boldsymbol{" + _sym + "}") if _bold else _sym
        # in the empty lower left: in the upper right the box covered an organism (2026-09-19)
        axP.text(0.03, 0.17 - _k * 0.13, f"${_s}$ {_num(_v)}  ($p$ {_p(_pv)})", transform=axP.transAxes,
                 ha="left", va="bottom", fontsize=5.6, color="#222" if _bold else "#444",
                 fontweight="bold" if _bold else "normal",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=6)
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    for t in list(axS.texts) + list(axP.texts) + [_ttl]:
        bb = t.get_window_extent(_r)
        if bb.width and (bb.x0 < 0 or bb.x1 > fig.bbox.width + 0.5 or bb.y1 > fig.bbox.height + 0.5):
            raise SystemExit(f"teaser: {t.get_text()!r} runs off the canvas ({bb})")
    assert_canvas(fig, tex_rel, name)
    out = OUT / f"{name}.pdf"
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"), dpi=600); plt.close(fig)
    print("wrote", out, f"({len(rows)} rows, canvas {W} x {H} in)")
    YM, STAT_COMPACT = _ym0, _sc0

if __name__ == "__main__":
    # dotplot() and scatter() still exist but their figures left the paper on 2026-09-11; only the
    # figures the paper uses are written.
    if os.environ.get("FIG_TEASER_ONLY") != "1":   # the teaser alone, after a long run stopped at it (2026-09-18)
        slice_figures()
    if os.environ.get("FIG_MAIN_ONLY") != "1":
        teaser_bottom()
    # intro_figure(): Figure 1 left the paper (author, 2026-09-16); the function stays for the record
