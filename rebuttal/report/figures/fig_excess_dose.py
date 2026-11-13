from style import *

"""Relative change in misalignment under steering against the two excess directions, by dose
(job j3, set em, 27 organisms), so organisms that start at different rates are comparable.

For each organism and direction: M(alpha) / M(unsteered) at alpha 0 (= 1), 0.15, 0.30, 0.45.
Reads out/steering_judged/all_cells.csv (rates, Fisher p vs baseline, coherence guard).
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

CELLS = ROOT / "out" / "steering_judged" / "all_cells.csv"
ALPHAS = [0.15, 0.30, 0.45]
ROWS = [("neg_excess_def", C_MAIN, r"$s_\mathrm{def}-\bar{s}$"),
        ("neg_excess_asst", C_ALT, r"$s_A-\bar{s}$")]
SIG = 0.05


def is_qwen_lora(fam: str, org: str) -> bool:
    return fam.startswith("qwen") and "full" not in org and "rank-1" not in org


def main():
    df = pd.read_csv(CELLS)
    df = df[(df["job"] == "j3") & (df["set"] == "em")]
    orgs = sorted(set(zip(df["family"], df["organism"])))
    assert len(orgs) == 27, len(orgs)
    cols = [("Qwen LoRA", [o for o in orgs if is_qwen_lora(*o)]),
            ("Outside Qwen LoRA", [o for o in orgs if not is_qwen_lora(*o)])]
    assert len(cols[0][1]) == 18 and len(cols[1][1]) == 9

    def cell(fam, org, cond, alpha=None):
        s = df[(df["family"] == fam) & (df["organism"] == org) & (df["condition"] == cond)]
        if alpha is not None:
            s = s[np.isclose(s["alpha"], alpha)]
        assert len(s) == 1, (fam, org, cond, alpha, len(s))
        return s.iloc[0]

    # ratio series per (condition, organism)
    series = {}
    for cond, _, _ in ROWS:
        for o in orgs:
            m0 = cell(*o, "baseline")["misaligned_rate"]
            assert m0 > 0, o
            pts = [(0.0, 1.0, True, None)]
            for a in ALPHAS:
                s = cell(*o, cond, a)
                pts.append((a, s["misaligned_rate"] / m0, bool(s["coherence_pass"]),
                            (s["fisher_p_vs_baseline"], s["misaligned_rate"] < m0)))
            series[(cond, o)] = pts

    fig, axes = plt.subplots(2, 2, figsize=(FULL, 5.0), sharex=True, sharey="row")
    for ri, (cond, col, tex) in enumerate(ROWS):
        for ci, (gname, members) in enumerate(cols):
            ax = axes[ri, ci]
            ax.axhline(1.0, color=C_BASE, lw=0.9, ls=(0, (4, 3)), zorder=1)
            drops = rises = drops_fail = 0
            for o in members:
                pts = series[(cond, o)]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, color=col, lw=0.8, alpha=0.45, zorder=2, solid_capstyle="round")
                for x, yv, ok, _ in pts:
                    if not ok:
                        ax.plot([x], [yv], "x", ms=4.4, mec=col, mew=1.1, zorder=4)
                x, yv, ok, (p, lower) = pts[-1]
                if ok:
                    if p < SIG:
                        ax.plot([x], [yv], "o", ms=3.6, mfc=col, mec="white", mew=0.5, zorder=4)
                    else:
                        ax.plot([x], [yv], "o", ms=3.4, mfc="white", mec=col, mew=0.9, zorder=4)
                if p < SIG and lower:
                    drops += 1
                    drops_fail += (not ok)
                elif p < SIG:
                    rises += 1
            med = [float(np.median([series[(cond, o)][i][1] for o in members])) for i in range(4)]
            ax.plot([0.0] + ALPHAS, med, color=col, lw=2.4, zorder=3, solid_capstyle="round")
            n = len(members)
            lines = [f"{drops} of {n} drop significantly at 0.45"]
            if drops_fail:
                lines.append(f"({drops_fail} of them fails the coherence guard)")
            lines.append(f"{rises} of {n} rise significantly at 0.45")
            ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
                    fontsize=7, color=INK2, linespacing=1.35)
            if ri == 0:
                ax.set_title(f"{gname} ({n} organisms)", loc="left")
            if ci == 0:
                ax.set_ylabel(f"Steered against {tex}\nrate relative to unsteered (ratio)")
            if ri == 1:
                ax.set_xlabel(r"Steering strength $\alpha$")
            ax.grid(axis="x", visible=False)

    for ri, (cond, _, _) in enumerate(ROWS):
        top = max(v[1] for (c, _), pts in series.items() if c == cond for v in pts)
        axes[ri, 0].set_ylim(-0.05 * top, top * 1.30)  # headroom for the in-panel counts
    for ax in axes[1]:
        ax.set_xticks([0.0] + ALPHAS)
        ax.set_xticklabels(["0\nunsteered", "0.15", "0.30", "0.45"])
        ax.set_xlim(-0.02, 0.47)

    h = [
        Line2D([], [], color=C_MAIN, lw=2.4, label=r"Median, against $s_\mathrm{def}-\bar{s}$"),
        Line2D([], [], color=C_ALT, lw=2.4, label=r"Median, against $s_A-\bar{s}$"),
        Line2D([], [], color=INK2, lw=0.8, alpha=0.45, label="One organism"),
        Line2D([], [], color=C_BASE, lw=0.9, ls=(0, (4, 3)), label="No change (ratio 1)"),
        Line2D([], [], color=INK2, marker="o", ms=4.2, mec="white", mew=0.5, lw=0,
               label="Filled: p < 0.05 vs unsteered (Fisher)"),
        Line2D([], [], color=INK2, marker="o", ms=4.0, mfc="white", mec=INK2, mew=0.9, lw=0,
               label="Hollow: not significant"),
        Line2D([], [], color=INK2, marker="x", ms=4.4, mew=1.1, lw=0, label="x: fails the coherence guard"),
    ]
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
               columnspacing=1.4, handlelength=2.0, borderaxespad=0.1)
    save(fig, "fig_excess_dose")


if __name__ == "__main__":
    main()
