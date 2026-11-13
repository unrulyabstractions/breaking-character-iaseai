from style import *

"""Per organism: unsteered misalignment rate and the rate under steering against the two excess
directions at alpha 0.45 (job j3, set em, 27 organisms).

Reads out/steering_judged/all_cells.csv (rates, kept-coherent counts, Fisher p, coherence guard) and
reanalysis/extra-evals/per_organism.csv (the paper's family and organism names). The interval is a 95%
Wilson interval on k / n with n = n_kept_coherent and k = misaligned_rate * n (the counts the Fisher
test uses); the CSV's own ci95_lo/hi columns are a question-cluster bootstrap and are not drawn here.
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

CELLS = ROOT / "out" / "steering_judged" / "all_cells.csv"
NAMES = ROOT / "reanalysis" / "extra-evals" / "per_organism.csv"
ALPHA = 0.45
CONDS = [("neg_excess_def", C_MAIN), ("neg_excess_asst", C_ALT)]
DIR_TEX = {"neg_excess_def": r"$s_\mathrm{def}-\bar{s}$", "neg_excess_asst": r"$s_A-\bar{s}$"}
SIG = 0.05


def is_qwen_lora(fam: str, org: str) -> bool:
    return fam.startswith("qwen") and "full" not in org and "rank-1" not in org


def load():
    df = pd.read_csv(CELLS)
    df = df[(df["job"] == "j3") & (df["set"] == "em")].copy()
    k = df["misaligned_rate"] * df["n_kept_coherent"]
    assert np.allclose(k, k.round()), "misaligned_rate * n_kept_coherent is not a whole count"
    df["k"] = k.round().astype(int)
    # Wilson 95% on k of n_kept_coherent (style.wilson); never the CSV's ci95_lo/hi (a cluster bootstrap)
    ci = [wilson(kk, nn) for kk, nn in zip(df["k"], df["n_kept_coherent"])]
    df["w_lo"] = [c[0] for c in ci]
    df["w_hi"] = [c[1] for c in ci]
    names = pd.read_csv(NAMES)
    lab = {}
    for _, r in names.iterrows():
        fslug, oslug = r["label"].split("/")
        name = f"{r['family']} {r['organism']}"
        if oslug.endswith("-full") and "full" not in r["organism"]:
            name += " (full)"
        lab[(fslug, oslug)] = name
    return df, lab


def main():
    df, lab = load()
    orgs = sorted(set(zip(df["family"], df["organism"])))
    assert len(orgs) == 27, len(orgs)

    def cell(fam, org, cond, alpha=None):
        s = df[(df["family"] == fam) & (df["organism"] == org) & (df["condition"] == cond)]
        if alpha is not None:
            s = s[np.isclose(s["alpha"], alpha)]
        assert len(s) == 1, (fam, org, cond, alpha, len(s))
        return s.iloc[0]

    base_rate = {o: cell(*o, "baseline")["misaligned_rate"] for o in orgs}

    # groups: five Qwen LoRA families in the shared order, then everything outside Qwen LoRA
    groups = []
    for fam_disp in POOL_FAMILIES:
        members = [o for o in orgs if SLUG_FAMILY[o[0]] == fam_disp and is_qwen_lora(*o)]
        groups.append((f"{fam_disp} LoRA", members))
    outside = [o for o in orgs if not is_qwen_lora(*o)]
    groups.append(("Outside Qwen LoRA: Gemma, Llama, Mistral, Qwen2.5-14B full and rank-1", outside))
    assert sum(len(m) for _, m in groups) == 27
    assert sum(len(m) for _, m in groups[:-1]) == 18 and len(outside) == 9

    # vertical layout (y grows downward): a header row per group, then one band per organism
    HEAD, BAND, GAP = 0.85, 1.0, 0.25
    SUB = {"baseline": -0.29, "neg_excess_def": 0.0, "neg_excess_asst": 0.29}
    y = 0.0
    rows, heads = [], []
    for title, members in groups:
        y += GAP
        heads.append((title, y + HEAD * 0.55, len(members)))
        y += HEAD
        for o in sorted(members, key=lambda o: -base_rate[o]):
            rows.append((o, y + BAND / 2))
            y += BAND
    y_end = y + 0.15

    fig, ax = plt.subplots(figsize=(FULL, 7.5))
    xmax = 0.0
    for i, (o, yc) in enumerate(rows):
        if i % 2 == 0:
            ax.axhspan(yc - BAND / 2, yc + BAND / 2, color="#f4f4f1", lw=0, zorder=0)
        b = cell(*o, "baseline")
        xb = b["misaligned_rate"]
        yb = yc + SUB["baseline"]
        # anchor: a grey hairline at the baseline rate, down through the band
        ax.plot([xb, xb], [yb, yc + SUB["neg_excess_asst"]], color=C_BASE, lw=0.5, zorder=2,
                solid_capstyle="butt")
        ax.plot([b["w_lo"], b["w_hi"]], [yb, yb], color=C_BASE, lw=2.6, alpha=0.35, zorder=1,
                solid_capstyle="butt")
        ax.plot([xb], [yb], "o", ms=4.2, mfc=C_BASE, mec="white", mew=0.6, zorder=4)
        xmax = max(xmax, b["w_hi"])
        row_right = max(b["w_hi"], xb)
        failed = []  # (direction label, kept, total) for cells below the coherence guard
        for cond, col in CONDS:
            s = cell(*o, cond, ALPHA)
            ys = yc + SUB[cond]
            xs = s["misaligned_rate"]
            ax.plot([s["w_lo"], s["w_hi"]], [ys, ys], color=col, lw=2.6, alpha=0.30, zorder=1,
                    solid_capstyle="butt")
            ax.plot([xb, xs], [ys, ys], color=col, lw=0.8, zorder=3, solid_capstyle="butt")
            xmax = max(xmax, s["w_hi"])
            row_right = max(row_right, s["w_hi"])
            if not bool(s["coherence_pass"]):
                ax.plot([xs], [ys], "x", ms=4.6, mec=col, mew=1.2, zorder=5)
                failed.append((DIR_TEX[cond], int(s["n_kept_coherent"]), int(s["n_total"])))
            elif s["fisher_p_vs_baseline"] < SIG:
                ax.plot([xs], [ys], "o", ms=4.2, mfc=col, mec="white", mew=0.6, zorder=5)
            else:
                ax.plot([xs], [ys], "o", ms=4.0, mfc="white", mec=col, mew=1.0, zorder=5)
        if failed:
            tot = {t for _, _, t in failed}
            assert len(tot) == 1
            parts = " and ".join(f"{k} ({d})" for d, k, _ in failed)
            ax.text(row_right + 0.008, yc + SUB["neg_excess_def"] + 0.14,
                    f"coherent answers kept: {parts} of {tot.pop()}",
                    va="center", ha="left", fontsize=6.8, color=INK2, zorder=6)

    for title, yh, n in heads:
        ax.text(0.003, yh, title, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=7.5, fontweight="bold", color=INK, zorder=6)
        ax.axhline(yh - HEAD * 0.55 - GAP / 2, color=AXIS, lw=0.6, zorder=2)

    ax.set_yticks([yc for _, yc in rows])
    ax.set_yticklabels([lab[o] for o, _ in rows])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(y_end, 0)
    ax.grid(axis="y", visible=False)
    hi = np.ceil((xmax + 0.02) * 20) / 20
    ax.set_xlim(-0.012, hi)  # the small negative margin keeps markers at a rate of 0 unclipped
    ax.set_xticks(np.arange(0, hi + 1e-9, 0.1))
    ax.tick_params(axis="x", top=True, labeltop=True)
    ax.set_xlabel("Misalignment rate (share of coherent answers judged misaligned, 0 to 1)")
    ax.spines["left"].set_visible(False)

    h = [
        Line2D([], [], color=C_BASE, marker="o", ms=4.2, mec="white", lw=0, label="Unsteered baseline"),
        Line2D([], [], color=C_MAIN, marker="o", ms=4.2, mec="white", lw=0.8,
               label=r"Steered against $s_\mathrm{def}-\bar{s}$, $\alpha$ = 0.45"),
        Line2D([], [], color=C_ALT, marker="o", ms=4.2, mec="white", lw=0.8,
               label=r"Steered against $s_A-\bar{s}$, $\alpha$ = 0.45"),
        Line2D([], [], color=INK2, lw=2.6, alpha=0.35, label="Wilson 95% interval (coherent answers)"),
        Line2D([], [], color=INK2, marker="o", ms=4.2, mec="white", lw=0,
               label="Filled: differs from baseline, Fisher p < 0.05"),
        Line2D([], [], color=INK2, marker="o", ms=4.0, mfc="white", mec=INK2, mew=1.0, lw=0,
               label="Hollow: not significant"),
        Line2D([], [], color=INK2, marker="x", ms=4.6, mew=1.2, lw=0,
               label="x: fails the coherence guard (too few coherent answers)"),
        Line2D([], [], color=INK2, lw=0.8, label="Line: change from the baseline rate"),
    ]
    fig.tight_layout()
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               columnspacing=2.0, handlelength=1.8, borderaxespad=0.1)
    save(fig, "fig_excess_organisms")


if __name__ == "__main__":
    main()
