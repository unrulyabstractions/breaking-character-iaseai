from style import *

"""Does the effect of an excess direction come from the part of it that still points along sbar?
(job j3, set em, 27 organisms)

x: cosine of the excess direction (s_def - sbar or s_A - sbar) with sbar at the steering layer L,
   from steer/directions/j3_directions.json cos_at_L, matched on family slug and hf_folder == organism.
y: M(alpha 0.45) / M(unsteered), from out/steering_judged/all_cells.csv.
"""

import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

CELLS = ROOT / "out" / "steering_judged" / "all_cells.csv"
DIRS = ROOT / "steer" / "directions" / "j3_directions.json"
ALPHA = 0.45
PANELS = [("neg_excess_def", "excess_default_vs_sbar", C_MAIN, r"Against $s_\mathrm{def}-\bar{s}$"),
          ("neg_excess_asst", "excess_assistant_vs_sbar", C_ALT, r"Against $s_A-\bar{s}$")]


def is_qwen_lora(fam: str, org: str) -> bool:
    return fam.startswith("qwen") and "full" not in org and "rank-1" not in org


def main():
    df = pd.read_csv(CELLS)
    df = df[(df["job"] == "j3") & (df["set"] == "em")]
    orgs = sorted(set(zip(df["family"], df["organism"])))
    assert len(orgs) == 27, len(orgs)
    meta = {(o["family"], o["hf_folder"]): o for o in json.loads(DIRS.read_text())["organisms"]}
    assert set(meta) == set(orgs), set(meta) ^ set(orgs)

    def cell(fam, org, cond, alpha=None):
        s = df[(df["family"] == fam) & (df["organism"] == org) & (df["condition"] == cond)]
        if alpha is not None:
            s = s[np.isclose(s["alpha"], alpha)]
        assert len(s) == 1, (fam, org, cond, alpha, len(s))
        return s.iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.1), sharey=True)
    ytop = 0.0
    for ax, (cond, key, col, title) in zip(axes, PANELS):
        ax.axhline(1.0, color=C_BASE, lw=0.9, ls=(0, (4, 3)), zorder=1)
        gemma = []
        for o in orgs:
            s = cell(*o, cond, ALPHA)
            assert int(s["layer"]) == int(meta[o]["L"]), (o, s["layer"], meta[o]["L"])
            m0 = cell(*o, "baseline")["misaligned_rate"]
            x = meta[o]["cos_at_L"][key]
            yv = s["misaligned_rate"] / m0
            ytop = max(ytop, yv)
            if is_qwen_lora(*o):
                ax.plot([x], [yv], "o", ms=5.0, mfc=col, mec="white", mew=0.7, zorder=5, alpha=0.85)
            else:
                ax.plot([x], [yv], "o", ms=5.6, mfc="none", mec=col, mew=1.1, zorder=4)
            if not bool(s["coherence_pass"]):
                ax.plot([x], [yv], "x", ms=3.6, mec=col, mew=1.0, zorder=5)
            if o[0].startswith("gemma") and o[1].endswith("-full"):
                gemma.append((x, yv))
        # one label for the two Gemma-3-12B full fine-tunes, which sit together near cosine 0.95
        assert len(gemma) == 2
        gx = float(np.mean([g[0] for g in gemma]))
        gy = float(np.mean([g[1] for g in gemma]))
        ax.annotate("Gemma-3-12B full fine-tunes\n(bad-medical, risky-financial)", xy=(gx, gy + 0.08),
                    xytext=(gx - 0.02, gy + 1.45), ha="right", va="bottom", fontsize=7, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=1, shrinkB=2,
                                    connectionstyle="angle,angleA=0,angleB=90,rad=0"))
        ax.set_title(f"{title}, " + r"$\alpha$ = 0.45", loc="left")
        ax.set_xlabel(r"Cosine of the direction with $\bar{s}$ at the steering layer")
        ax.set_xlim(-0.55, 1.05)
        ax.set_xticks(np.arange(-0.5, 1.01, 0.25))
    axes[0].set_ylabel("Rate relative to unsteered (ratio)")
    axes[0].set_ylim(-0.15, ytop * 1.08)

    h = [
        Line2D([], [], color=INK2, marker="o", ms=5.0, mec="white", mew=0.7, lw=0, label="Qwen LoRA (18)"),
        Line2D([], [], color=INK2, marker="o", ms=5.6, mfc="white", mec=INK2, mew=1.1, lw=0,
               label="Outside Qwen LoRA (9)"),
        Line2D([], [], color=INK2, marker="x", ms=3.6, mew=1.0, lw=0, label="x: fails the coherence guard"),
        Line2D([], [], color=C_BASE, lw=0.9, ls=(0, (4, 3)), label="No change (ratio 1)"),
    ]
    fig.tight_layout(w_pad=1.5)
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
               columnspacing=1.6, handlelength=2.0, borderaxespad=0.1)
    save(fig, "fig_excess_cosine")


if __name__ == "__main__":
    main()
