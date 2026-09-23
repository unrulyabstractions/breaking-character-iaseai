#!/usr/bin/env python
"""Causal-steering headline figure: steering against the global persona shift reverses
emergent misalignment, and the reversal strengthens with scale. Draws a dumbbell from the
baseline rate to the steered rate for each organism.

The steered direction is the mean role shift, so it depends on which roles the mean runs over.
This figure reads the Lu-cast sweep only. The earlier sweep over the retired 24-role lite cast
is not a Study-2 result.

In:  data/raw/steering/<family>/<organism>.csv
Out: <paper>/figures/causal_steering.png
Feeds: Figure 5 and Section 4.3.
"""
import glob
import matplotlib
matplotlib.use("Agg")
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _paths
from _paperstyle import use as _use
from _paperstyle import FAMC, BLUE
from matplotlib.ticker import FixedLocator
from _paperstyle import INK as INK_
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

_use(9)

# Why this figure does not show every family. Checked by scripts/check_paper_numbers.py.
COVERAGE_NOTE = ("Every family with steering data enters. The guard below refuses to run when a "
                 "family has data but is missing from FAM_ORDER or ARCH.")

ROOT = str(_paths.RAW / "steering")
# family display order (scale within Qwen, then gemma, then Llama), + pretty labels
FAM_ORDER = ["qwen2.5-7b", "qwen3.5-9b", "qwen2.5-14b", "qwen2.5-32b", "qwen3-32b", "gemma3-12b",
             "llama3.1-8b", "mistral-24b"]
from _paperstyle import DISPLAY as FAM_LABEL
# ONE HUE PER FAMILY, the same hue the weight-bridge figure above it uses. Four Qwen families
# shared the house blue here while that blue named one family in Figure 4, so a reader who learnt
# the key of one figure misread the other (audit, 2026-09-17). Qwen3-32B takes its own green,
# because the shared palette still gives it Gemma's purple.
FAMC = dict(FAMC, **{"Qwen3-32B": "#5a7f3c"})
ARCH = {k: FAMC[FAM_LABEL[k]] for k in ["qwen2.5-7b", "qwen3.5-9b", "qwen2.5-14b", "qwen2.5-32b",
                                        "qwen3-32b", "llama3.1-8b", "gemma3-12b", "mistral-24b"]}
# steering folder -> the organism's name in the paper's tables (the 2026-09-19 battery's folders
# cannot carry spaces or parentheses)
# steering folder -> the organism's key in the geometry, whose one printed label _organisms.label() gives, as Figure 4
# and Tables 4 and 5 print it (author, 2026-09-19: "the main paper figure and these tables should be consistent"). The
# plain Gemma folders are its full fine-tunes, and label() says so.
from _organisms import label as _org_label
def org_label(fam_slug, folder):
    key = folder
    for suf, rep in (("-rank1", " rank-1"), ("-full", " (full)"), ("-lora", " (LoRA)")):
        if key.endswith(suf):
            key = key[: -len(suf)] + rep
    return _org_label(FAM_LABEL[fam_slug], key)
# no family swatches in the legend: every row label already names its family in words
# Mistral-24B was coloured but never ordered, so it sorted on a NaN rank and vanished from the
# per-family breakdown this script prints. Every family with steering data must appear in both.
_have = {_os.path.basename(_os.path.dirname(p)) for p in glob.glob(f"{ROOT}/*/*.csv")}
for _f in sorted(_have - set(FAM_ORDER)):
    raise SystemExit(f"fig_causal_steering.py: steering data exists for {_f} "
                     f"but FAM_ORDER omits it, so it would sort on a NaN rank.")
for _f in sorted(_have - set(ARCH)):
    raise SystemExit(f"fig_causal_steering.py: steering data exists for {_f} but ARCH omits it.")

CTRL = {"good-medical-control", "mismatch-good-medical-control"}
# insecure-code is a code-domain organism: its EM lives in code contexts, not the
# general "reveal misalignment" eval (baseline 2.3%), so it is not a valid test here.
EXCLUDE = {"insecure-code"}
# an organism with a low baseline has too little EM to test a reversal: below this
# rate the steering question is underpowered and uninformative. Controls are exempt.
MIN_BASELINE = 10.0


def best_neg_global(df):
    # the persona layer, 50% of the depth, as everywhere else in the paper; the run also covered
    # 0.7 and the figure used to take the better of the two (user, 2026-09-16)
    sub = df[(df.condition == "neg_global") & (df.coherence_pass == True)  # noqa: E712
             & (df.depth_frac == 0.5)]
    if len(sub) == 0:
        return None
    return sub.loc[sub.misaligned_rate.idxmin()]


def load():
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/*/*.csv")):
        fam, org = _os.path.basename(_os.path.dirname(f)), _os.path.basename(f)[:-4]
        if org in EXCLUDE:
            continue
        df = pd.read_csv(f)
        b = df[df.condition == "baseline"]
        if not len(b) or pd.isna(b.iloc[0].misaligned_rate):
            continue
        ng = best_neg_global(df)
        if ng is None:
            continue
        base = b.iloc[0].misaligned_rate * 100
        is_ctrl = org in CTRL
        # drop underpowered EM organisms (too little EM to test a reversal); keep controls
        if not is_ctrl and base < MIN_BASELINE:
            continue
        rows.append(dict(family=fam, organism=org, is_ctrl=is_ctrl,
                         baseline=base,
                         steered=ng.misaligned_rate * 100,
                         p_rand=ng.get("fisher_p_vs_random"),
                         depth=ng.depth_frac, alpha=ng.alpha))
    m = pd.DataFrame(rows)
    m["fam_rank"] = m.family.map({f: i for i, f in enumerate(FAM_ORDER)})
    # EM organisms first (by family order, then baseline desc), controls at the bottom
    m = m.sort_values(["is_ctrl", "fam_rank", "baseline"], ascending=[True, True, False])
    return m.reset_index(drop=True)


def main():
    m = load()
    n = len(m)
    # Authored at the PRINTED size. results_causal.tex includes this file with height=3.3in, so the
    # saved canvas must be 3.3 in tall or LaTeX rescales every label. At 5.5 x 3.3 in the scale is
    # exactly 1.0 and the figure fills the 5.5 in text block; the earlier 5.6 x 4.8 in canvas was
    # squeezed to 0.69 and printed its 7.6 pt labels at 5.3 pt.
    # dpi 300 is the dpi the file is SAVED at: measured at the rcParams 200 the key fitted the
    # canvas and printed clipped, because type is laid out at the dpi it is rendered at.
    # The row pitch, header and foot are fixed in inches (the 13-row figure's: 0.147 in a row, a
    # 0.48 in header, a 0.36 in foot), so the canvas grows with the rows and every label keeps its size.
    HEAD, FOOT = 0.175 * 2.75, 0.130 * 2.75
    PITCH = (2.75 - HEAD - FOOT) / 13
    H = HEAD + FOOT + PITCH * n
    fig = plt.figure(figsize=(3.3125, H), dpi=300)   # one column of the AAAI kit (2026-09-16)
    # The header strip carries the headline and the one-row key, so no legend block eats the foot.
    # the plot starts 0.25 in further right than it did (0.455 of the width) so the organism names, which carry their
    # tuning tag since 2026-09-19, fit one size; the two label columns keep their place on the page
    AX_L, AX_W = 0.530, 0.455
    ax = fig.add_axes((AX_L, FOOT / H, AX_W, 1 - (FOOT + HEAD) / H))
    # The scale runs to the next multiple of 5 above the highest rate; the gutter and margins keep
    # their 0-25 proportions.
    XMAX = max(25, 5 * int(-(-max(m.baseline.max(), m.steered.max()) // 5)))
    # One baseline far above the rest (the Qwen2.5-32B obviously-bad-medical, 52%) squeezed every other
    # row into the left half of the field. Above 35% the scale stops at 30, and that baseline is drawn
    # past a break with its value printed (2026-09-19).
    CAP = 30 if m.baseline.max() > 35 else None
    if CAP:
        assert m.steered.max() <= CAP, "a steered rate above the capped scale"
        XMAX = CAP
    XS = XMAX / 25
    BEYOND = XMAX + 2.1 * XS           # where an off-scale baseline is drawn, clear of the break
    # The two label columns: the family flush right, the organism flush left, one gutter between
    # them. Drawn rather than set as tick labels, because a tick label can only hang off one edge
    # and the family name then printed over the organism name (2026-09-14).
    FAM_X, ORG_X = (0.1555 - AX_L) / AX_W, (0.1714 - AX_L) / AX_W   # figure fractions 0.1555 and 0.1714, as before
    GUT = -0.9 * XS                  # the steered value column, right-aligned, in data coordinates
    y = range(n)
    vals = {}
    for yi, r in zip(y, m.itertuples()):
        col = ARCH[r.family]
        off = CAP is not None and r.baseline > CAP
        bx = BEYOND if off else r.baseline
        specific = pd.notna(r.p_rand) and r.p_rand < 0.05
        # a NaN p_rand means the matched random push left the model incoherent, so no test
        # exists; that row must not draw as a failed test
        no_ctrl = pd.isna(r.p_rand)
        # dumbbell: baseline (open grey) -> steered (filled, architecture colour). Three states:
        # (a) beats the random push, (b) tested and does not, (c) no coherent random push.
        # The steered end is a STAR when the drop beats the matched random push, a filled CIRCLE
        # when it is tested and does not, and a circle on a dashed line when the random push left
        # the model incoherent (no test). No fading: the marker shape carries the outcome
        # (user, 2026-09-11: "faded vs bold not clear").
        # A 2.6/1.8 dash left a whole gap at the open marker, so the two dashed rows read as
        # unfinished beside the solid ones (audit, 2026-09-17). A shorter dash lands within half
        # a period of the marker whatever the row's length.
        ax.plot([bx, r.steered], [yi, yi], color=col, lw=1.3, alpha=0.9, zorder=1,
                ls=(0, (1.8, 1.4)) if no_ctrl else "-", solid_capstyle="round")
        if specific:
            ax.scatter(r.steered, yi, s=46, marker="*", facecolor=col, edgecolor="#222",
                       linewidth=0.6, zorder=3)
        else:
            ax.scatter(r.steered, yi, s=22, facecolor=col, edgecolor="#222", linewidth=0.7, zorder=3)
        ax.scatter(bx, yi, s=22, facecolor="white", edgecolor="#555",
                   linewidth=1.0, zorder=2)
        if off:   # the break, and the true baseline printed beside its marker
            ax.annotate("//", (XMAX + 0.7 * XS, yi), ha="center", va="center", fontsize=6.0, color="#555",
                        zorder=4, bbox=dict(fc="white", ec="none", pad=0.05))
            ax.annotate(f"{r.baseline:.1f}", (bx, yi), xytext=(4.5, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=5.6, color="#555", zorder=4)
        # every value sits in one fixed gutter column, right-aligned at GUT and outside the
        # drawn 0-25 rule. A marker-relative offset made the numerals zig-zag across the field,
        # where they read positionally against the grid and ran onto the zero line.
        vals[yi] = ax.annotate(f"{r.steered:.1f}", (GUT, yi), ha="right", va="center",
                               fontsize=6.0, color=col, zorder=4)
    # Two label columns instead of one. The family name was repeated on all thirteen rows, which
    # spent a third of the label width restating what a group header says once.
    ax.set_yticks(list(y))
    ax.set_yticklabels([])
    labs = {}
    for yi, r in zip(y, m.itertuples()):
        # a control would print green, as in Figure 4; none has a baseline rate high enough to enter
        labs[yi] = ax.text(ORG_X, yi, org_label(r.family, r.organism),
                           transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.0,
                           color="#2f7d4f" if r.is_ctrl else INK_)
    ax.tick_params(axis="x", labelsize=6.0)
    # no y tick dashes: they read as trailing hyphens on labels that already end in hyphenated
    # tokens, and every row terminates in its own connector
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(-0.95, n - 0.45)
    ax.invert_yaxis()
    ax.set_xlabel("misaligned rate (%)", fontsize=6.2, labelpad=2)
    # a left gutter holds the steered value on each row's own baseline; the x grid carries zero,
    # so no axvline is drawn on top of it
    right = max(m.baseline.max(), 1) + 1.8 * XS
    if CAP:
        right = BEYOND + 4.2 * XS      # room for the printed off-scale baseline
    ax.set_xlim(-3.9 * XS, right if right >= XMAX else XMAX + 0.2 * XS)
    # A long organism name ("risky-financial rank-1") ran into its row's value; it steps down until it
    # clears the value by 2 pt (2026-09-19). A name that cannot fit at 4.8 pt stops the figure.
    fig.canvas.draw(); _r = fig.canvas.get_renderer()
    # ONE SIZE for every name, the largest at which all of them clear their values: a size per row printed the long
    # names smaller than their neighbours (2026-09-19)
    fs = 6.0
    while any(t.get_window_extent(_r).x1 > vals[yi].get_window_extent(_r).x0 - 2 * fig.dpi / 72 for yi, t in labs.items()):
        fs -= 0.1
        if fs < 4.8:
            _bad = [t.get_text() for yi, t in labs.items() if t.get_window_extent(_r).x1 > vals[yi].get_window_extent(_r).x0 - 2 * fig.dpi / 72]
            raise SystemExit(f"fig_causal_steering: {_bad} do not fit the label column")
        for t in labs.values():
            t.set_fontsize(fs)
        fig.canvas.draw()
    print(f"organism names at {fs:.1f} pt")
    # centre the label on the drawn span, not on the axes including the value gutter; its drop is
    # fixed in inches (the 13-row figure's 0.125 of its axes height)
    x0, x1 = ax.get_xlim()
    ax.xaxis.set_label_coords((XMAX / 2 - x0) / (x1 - x0), -0.125 * 13 * PITCH / (H - FOOT - HEAD))
    # the grid is the only way to read the unlabelled open baseline marker, so it has to survive
    # coated stock: #d9d9d9 at 0.5 lw, behind the marks
    ax.grid(axis="x", color="#d9d9d9", lw=0.5)
    # The rules stop where the scale does. Minor rules ran on into the value column, where they
    # carry no meaning and cut through the printed numerals (audit, 2026-09-17).
    MAJ = 5 if XMAX <= 30 else 10
    ax.xaxis.set_major_locator(FixedLocator(list(range(0, XMAX + 1, MAJ))))
    ax.xaxis.set_minor_locator(FixedLocator(list(range(0, XMAX + 1, 1 if XMAX <= 30 else 5))))
    ax.set_axisbelow(True)
    # no left spine: it sat in the left gutter, a heavy rule one grid pitch left of the real zero,
    # so the rows that steer to under 1 percent read as sitting near 2 percent
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_bounds(0, XMAX)
    # The family name now sits once per group, in its own column left of the organism names.
    first = {}
    for yi, r in zip(y, m.itertuples()):
        first.setdefault(r.family, []).append(yi)
    for fam, ys in first.items():
        ax.text(FAM_X, (min(ys) + max(ys)) / 2, FAM_LABEL[fam], transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.0, color=ARCH[fam], fontweight="bold")
        if min(ys) > 0:   # a hairline between families, so the groups read without a rule per row
            ax.axhline(min(ys) - 0.5, color="#e6e6e6", lw=0.5, zorder=0)
    # The open end of every dumbbell is named where the reader meets it, on the first row, rather
    # than in a key at the foot of the figure.
    r0 = m.iloc[0]
    ax.annotate("baseline", (r0.baseline, -0.62), ha="center", va="center", fontsize=5.6,
                color="#666", zorder=4)
    # The key sits under the headline, one row, before the data rather than after them. It decodes
    # only the steered end, which is the one thing the shapes carry and the words do not.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="#888", lw=1.3, marker="*", mfc="#888", mec="#222", ms=7.0,
                      label="beats a matched random push"),
               Line2D([0], [0], color="#888", lw=1.3, marker="o", mfc="#888", mec="#222", ms=4.6,
                      label="does not beat it"),
               Line2D([0], [0], color="#888", lw=1.3, ls=(0, (1.3, 1.0)), marker="o", mfc="#888",
                      mec="#222", ms=4.6, label="no coherent control")]
    em = m[~m.is_ctrl]
    lowered = int((em.steered < em.baseline).sum())
    head = ("Steering against the shift lowers the rate in every organism" if lowered == len(em)
            else f"Steering against the shift lowers the rate in {lowered} of {len(em)} organisms")
    fig.text(0.010, 1 - 0.055 / H, head,
             fontsize=6.9, fontweight="bold", color="#222", ha="left", va="top")
    # A 1.7 handle put a 4.6 pt marker in the middle of 8.8 pt of line, which left under one dash
    # period on each side: the second and third keys drew the same glyph (audit, 2026-09-17). The
    # longest handle that still fits the canvas is measured rather than guessed, because a handle
    # too long runs the third key off the right edge.
    leg = None
    for _hl in (3.2, 2.9, 2.6, 2.3, 2.0):
        if leg is not None:
            leg.remove()
        leg = fig.legend(handles=handles, loc="upper left", fontsize=5.2, frameon=False, ncol=3,
                         handlelength=_hl, columnspacing=0.7, handletextpad=0.3,
                         bbox_to_anchor=(0.010, 1 - 0.165 / H))
        fig.canvas.draw()
        if leg.get_window_extent(fig.canvas.get_renderer()).x1 <= fig.dpi * (fig.get_figwidth() - 0.02):
            print(f"  key handle length {_hl}")
            break
    else:
        raise SystemExit("fig_causal_steering: the key does not fit the canvas at any handle length.")
    for t in leg.get_texts():
        t.set_color("#555")
    _paths.FIGURES.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": the saved canvas must stay exactly 3.3125 x 2.75 in, the column width of
    # the AAAI kit at which results_geometry.tex includes it (width=\columnwidth).
    out = _paths.FIGURES / "causal_steering.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"wrote {out} ({n} organisms)\n")
    show = m[["family", "organism", "baseline", "steered", "p_rand", "depth", "alpha"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}" if x == x else "-"))
    em = m[~m.is_ctrl]  # already filtered to baseline >= MIN_BASELINE in load()
    sig = em[em.p_rand < 0.05]
    print(f"\nEM organisms (baseline>={MIN_BASELINE:g}%): {len(em)}; neg_global specific (p_rand<0.05): {len(sig)}/{len(em)}")
    for fam in FAM_ORDER:
        e = em[em.family == fam]
        if len(e):
            print(f"  {FAM_LABEL[fam]:14s}: {len(e[e.p_rand<0.05])}/{len(e)} specific")


if __name__ == "__main__":
    main()
