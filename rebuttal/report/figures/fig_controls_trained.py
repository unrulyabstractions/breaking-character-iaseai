"""Steering three controls against their own shift leaves their trained behaviour in place.

(a) Qwen3.5-9B cheese-set-A control: P(chooses set-A cheese), CI = percentile bootstrap over the 36 cheese pairs
    (p_setA_ci95_lo/hi of cheese_cells.csv). The values sit near 1 and the intervals are about 0.04 tall, so this
    panel gets a zoomed y axis (0.90 to 1.00), said in the panel.
(b) Qwen3.5-9B reward-hacks control and (c) Qwen3-32B good-faith control: P(hack) = mean of the per-item 'hack'
    column of <cell>.scored.csv (blanks dropped). The 240 items are 120 distinct prompts: 20 prompts are asked 4 to
    20 times (question_id ending _r01.._r20) and 100 once. CI = percentile bootstrap over the 120 prompts (whole
    prompts resampled with all their repeats; 2,000 resamples, seed 0; P(hack) of a resample = hack sum / item count).
    The script checks the prompt grouping against steer/questions/reward_hacks.yaml, whose sha256 must equal the
    questions_sha256 recorded in each run's RUN_KEY.json.
alpha 0 is the unsteered baseline, the shared start of both lines.
"""
from style import *  # noqa: F403

import csv
import hashlib
import json
import re
from collections import defaultdict

import numpy as np
import yaml
from matplotlib.lines import Line2D

CHEESE = ROOT / "out/steering/pod_A/j1/qwen3.5-9b/cheese-set-a-control/cheese/cheese_cells.csv"
RH = {  # (scored folder, run folder holding RUN_KEY.json)
    "b": (ROOT / "out/steering_judged/pod_A/j1/qwen3.5-9b/reward-hacks-control/rh",
          ROOT / "out/steering/pod_A/j1/qwen3.5-9b/reward-hacks-control/rh"),
    "c": (ROOT / "out/steering_judged/pod_C/j1/qwen3-32b/good-faith-control/rh",
          ROOT / "out/steering/pod_C/j1/qwen3-32b/good-faith-control/rh"),
}
QUESTIONS = ROOT / "steer/questions/reward_hacks.yaml"
CELL = re.compile(r"^(neg_global|random_ctrl)__d(\d+)__a([0-9.]+)$")
REPEAT = re.compile(r"_r\d+$")
# blue drawn last so it stays visible where both lines leave the shared baseline point
CONDS = [("random_ctrl", C_ALT, "random direction", 3), ("neg_global", C_MAIN, "against own global shift (−s̄)", 4)]
N_BOOT, SEED = 2000, 0
DODGE = 0.015
ZOOM_A = (0.90, 1.00)


def parse_cell(name: str):
    """'baseline' -> ('baseline', 0.0); 'neg_global__d16__a0.15' -> ('neg_global', 0.15)."""
    if name == "baseline":
        return "baseline", 0.0
    m = CELL.match(name)
    if not m:
        raise ValueError(f"unexpected cell name {name!r}")
    return m.group(1), float(m.group(3))


def load_cheese():
    out = {}
    with open(CHEESE) as fh:
        for r in csv.DictReader(fh):
            cond, a = parse_cell(r["cell"])
            out[(cond, a)] = (float(r["p_setA"]), float(r["p_setA_ci95_lo"]), float(r["p_setA_ci95_hi"]))
    return out


def prompt_groups() -> dict[str, str]:
    """item id -> prompt id (id without _rNN). Checks one (text, system) per prompt id and none shared."""
    items = yaml.safe_load(QUESTIONS.read_text())
    seen = defaultdict(set)
    for it in items:
        seen[REPEAT.sub("", it["id"])].add((tuple(it["paraphrases"]), it.get("system")))
    assert all(len(v) == 1 for v in seen.values()), "a prompt id holds two different prompts"
    assert len({next(iter(v)) for v in seen.values()}) == len(seen), "two prompt ids share one prompt"
    return {it["id"]: REPEAT.sub("", it["id"]) for it in items}


def load_rh(scored, run):
    """Per cell: (mean hack, bootstrap lo, bootstrap hi, n items, n prompts). Mean checked against cells.csv."""
    want = json.loads((run / "RUN_KEY.json").read_text())["questions_sha256"]
    assert hashlib.sha256(QUESTIONS.read_bytes()).hexdigest() == want, (run, "question file differs from the run")
    group = prompt_groups()
    ref = {}
    with open(scored / "cells.csv") as fh:
        for r in csv.DictReader(fh):
            ref[r["cell"]] = float(r["p_hack"])
    out = {}
    for f in sorted(scored.glob("*.scored.csv")):
        name = f.name[: -len(".scored.csv")]
        cond, a = parse_cell(name)
        sums, counts = defaultdict(float), defaultdict(int)
        with open(f) as fh:
            for r in csv.DictReader(fh):
                g, v = group[r["question_id"]], r["hack"].strip()  # blank = unscored, dropped
                sums[g] += float(v) if v else 0.0
                counts[g] += 1 if v else 0
        keys = sorted(sums)
        s = np.array([sums[k] for k in keys])
        c = np.array([counts[k] for k in keys])
        p = float(s.sum() / c.sum())
        assert abs(p - ref[name]) < 1e-9, (f, p, ref[name])
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
        boots = s[idx].sum(axis=1) / c[idx].sum(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[(cond, a)] = (p, float(lo), float(hi), int(c.sum()), len(keys))
    return out


def draw(ax, cells, ylabel, title, ylim):
    base = cells[("baseline", 0.0)]
    ax.axhline(base[0], color=C_BASE, lw=0.7, zorder=1)
    for cond, color, _, z in CONDS:
        alphas = sorted(a for (c, a) in cells if c == cond)
        off = -DODGE if cond == "neg_global" else DODGE
        xs = [0.0] + [a + off for a in alphas]
        ys = [base[0]] + [cells[(cond, a)][0] for a in alphas]
        lo = [cells[(cond, a)][0] - cells[(cond, a)][1] for a in alphas]
        hi = [cells[(cond, a)][2] - cells[(cond, a)][0] for a in alphas]
        ax.plot(xs, ys, color=color, lw=1.4, zorder=z, solid_capstyle="round")
        ax.errorbar(xs[1:], ys[1:], yerr=[lo, hi], fmt="o", color=color, ms=4.2, mec="white", mew=0.8,
                    elinewidth=1.0, capsize=1.6, capthick=1.0, zorder=z + 2)
    ax.errorbar([0.0], [base[0]], yerr=[[base[0] - base[1]], [base[2] - base[0]]], fmt="o", color=C_BASE, ms=4.2,
                mec="white", mew=0.8, elinewidth=1.0, capsize=1.6, capthick=1.0, zorder=7)
    lows = [v[1] for v in cells.values()]
    highs = [v[2] for v in cells.values()]
    assert ylim[0] <= min(lows) and max(highs) <= ylim[1], (title, min(lows), max(highs))
    ax.set_xlim(-0.04, 0.49)
    ax.set_xticks([0, 0.15, 0.30, 0.45], ["0", "0.15", "0.30", "0.45"])
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("steering strength α")
    ax.set_title(title, loc="left", fontsize=7.5, linespacing=1.3)


fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.55))
draw(axes[0], load_cheese(), "P(chooses set-A cheese), 0-1",
     "(a) Qwen3.5-9B cheese-set-A control\ntrained to prefer set-A cheeses", ZOOM_A)
axes[0].set_yticks(np.linspace(ZOOM_A[0], ZOOM_A[1], 6))
axes[0].text(0.97, 0.04, "y axis zoomed to 0.90-1.00", transform=axes[0].transAxes, ha="right", va="bottom",
             color=INK2, fontsize=6.5)
for ax, key, title in [(axes[1], "b", "(b) Qwen3.5-9B reward-hacks control\ntrained to reward hack"),
                       (axes[2], "c", "(c) Qwen3-32B good-faith control\ntrained on non-hacking answers")]:
    draw(ax, load_rh(*RH[key]), "P(reward hack), 0-1", title, (0.0, 1.0))
    ax.set_yticks(np.linspace(0, 1, 6))

handles = [Line2D([], [], color=c, lw=1.4, marker="o", ms=4.2, mec="white", mew=0.8, label=lab)
           for cond, c, lab, _ in sorted(CONDS, key=lambda t: t[0] != "neg_global")]
handles.append(Line2D([], [], color=C_BASE, lw=0.7, marker="o", ms=4.2, mec="white", mew=0.8,
                      label="unsteered (α = 0)"))
handles.append(Line2D([], [], color=INK2, lw=0, marker="|", ms=7, mew=1.0, label="95% interval"))
fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), handlelength=2.2,
           columnspacing=2.0)
fig.tight_layout(w_pad=1.6)
save(fig, "fig_controls_trained")
