"""Steering the nine controls against their own shift does not raise misalignment on the 8 EM questions.

Rows of all_cells.csv with job j1 and set em, one panel per control. M = misaligned_rate (share of answers with
aligned < 30 among answers with coherent > 50). Bars: 95% Wilson interval for k = M * n_kept_coherent misaligned
answers out of n_kept_coherent (style.wilson). alpha 0 is the unsteered baseline, the shared start of both lines.
A control whose unsteered cell fails the coherence guard gets its coherent counts written in the panel instead of
rates.
"""
from style import *  # noqa: F403

import csv
from collections import defaultdict

from matplotlib.lines import Line2D

SRC = ROOT / "out/steering_judged/all_cells.csv"
CONDS = [("random_ctrl", C_ALT, "random direction"), ("neg_global", C_MAIN, "against own global shift (−s̄)")]  # blue drawn last, on top
FAMILY_ORDER = POOL_FAMILIES + OTHER_FAMILIES
DODGE = 0.012
YMAX = 0.1


def org_label(slug: str) -> str:
    name = slug[: -len("-control")] if slug.endswith("-control") else slug
    return name.replace("set-a", "set-A")


def kept_count(r) -> tuple[int, int]:
    """(misaligned k, kept-coherent n) from the row; k must be a whole number."""
    n = int(r["n_kept_coherent"])
    if n == 0:
        return 0, 0
    kf = float(r["misaligned_rate"]) * n
    k = round(kf)
    assert abs(kf - k) < 1e-6, (r["organism"], r["condition"], r["alpha"], kf)
    return k, n


rows = defaultdict(dict)
with open(SRC) as fh:
    for r in csv.DictReader(fh):
        if r["job"] != "j1" or r["set"] != "em":
            continue
        a = 0.0 if r["condition"] == "baseline" else float(r["alpha"])
        rows[(r["family"], r["organism"])][(r["condition"], a)] = r
orgs = sorted(rows, key=lambda fo: (FAMILY_ORDER.index(SLUG_FAMILY[fo[0]]), fo[1]))
assert len(orgs) == 9, orgs

fig, axes = plt.subplots(3, 3, figsize=(FULL, 5.0), sharex=True, sharey=True)
any_fail = False
for ax, (fam, org) in zip(axes.flat, orgs):
    cells = rows[(fam, org)]
    base = cells[("baseline", 0.0)]
    ax.set_title(f"{SLUG_FAMILY[fam]} {org_label(org)}", loc="left", fontsize=7.5)
    if base["coherence_pass"] != "True":
        steered = [int(r["n_kept_coherent"]) for (c, a), r in cells.items() if c != "baseline"]
        n_tot = int(base["n_total"])
        txt = (f"{base['n_kept_coherent']} of {n_tot} answers\ncoherent unsteered;\n"
               f"{min(steered)} to {max(steered)} of {n_tot}\nwhen steered:\n"
               "too few coherent\nanswers to measure")
        ax.text(0.5, 0.5, txt, transform=ax.transAxes, ha="center", va="center", color=INK2, fontsize=6.8,
                linespacing=1.5)
        ax.grid(False)
        continue
    kb, nb = kept_count(base)
    pb = kb / nb
    lob, hib = wilson(kb, nb)
    lob, hib = min(lob, pb), max(hib, pb)
    ax.axhline(pb, color=C_BASE, lw=0.7, zorder=1)
    for j, (cond, color, _) in enumerate(CONDS):
        alphas = sorted(a for (c, a) in cells if c == cond)
        off = (DODGE, -DODGE)[j]
        xs, ys, el, eh = [0.0], [pb], [], []
        for a in alphas:
            r = cells[(cond, a)]
            k, n = kept_count(r)
            p = k / n
            lo, hi = wilson(k, n)
            lo, hi = min(lo, p), max(hi, p)  # guard float noise at k = 0
            xs.append(a + off)
            ys.append(p)
            el.append(p - lo)
            eh.append(hi - p)
            fail = r["coherence_pass"] != "True"
            any_fail |= fail
            ax.errorbar([a + off], [p], yerr=[[p - lo], [hi - p]], fmt="o", color=color, ms=4.2,
                        mfc="white" if fail else color, mec=color if fail else "white", mew=0.8, elinewidth=0.9,
                        zorder=4)
        ax.plot(xs, ys, color=color, lw=1.3, zorder=3)
    ax.errorbar([0.0], [pb], yerr=[[pb - lob], [hib - pb]], fmt="o", color=C_BASE, ms=4.2, mec="white", mew=0.8,
                elinewidth=0.9, zorder=5)

for ax in axes.flat:
    ax.set_xlim(-0.04, 0.49)
    ax.set_xticks([0, 0.15, 0.30, 0.45], ["0", "0.15", "0.30", "0.45"])
    ax.set_ylim(-0.003, YMAX)
    ax.set_yticks([0, 0.025, 0.05, 0.075, 0.1], ["0", "0.025", "0.05", "0.075", "0.10"])
for ax in axes[:, 0]:
    ax.set_ylabel("misalignment rate, 0-1")
for ax in axes[-1, :]:
    ax.set_xlabel("steering strength α")

handles = [Line2D([], [], color=c, lw=1.3, marker="o", ms=4.2, mec="white", mew=0.8, label=lab)
           for _, c, lab in CONDS]
handles.append(Line2D([], [], color=C_BASE, lw=0.7, marker="o", ms=4.2, mec="white", mew=0.8,
                      label="unsteered (α = 0)"))
handles.append(Line2D([], [], color=INK2, lw=0.9, marker="|", ms=7, label="bars: Wilson 95%"))
if any_fail:
    handles.append(Line2D([], [], color=INK2, lw=0, marker="o", ms=4.2, mfc="white", mec=INK2, mew=0.8,
                          label="fails coherence guard"))
fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.035), handlelength=2.2,
           columnspacing=2.0)
fig.tight_layout(h_pad=1.2, w_pad=1.0)
save(fig, "fig_controls_em")
