"""The steering pipeline, left to right: organism, direction, steering, generation, scoring, rate (a diagram, no data).

Settings follow steer/code/README.md and steer/questions/SETS.json: the steer layer is half the model's depth
(--layers 0.5), the edit is added at the last position of every forward pass, alpha is a fraction of the base model's
mean activation norm at that layer, EM = 8 questions x 30 samples, and cheese and reward hacks have their own scorers
(score_cheese.py, score_reward_hacks.py); the report draws their intervals as a bootstrap over prompts
(fig_controls_trained.py). The terms follow the tex: global shift sbar, excess drift s_def - sbar, upright "def".
"""
from style import *  # noqa: F401,F403

from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

H = 1.6
# each step: title, then groups of lines; a small gap separates the groups
STEPS = [
    ("Organism", [["base model", "+ fine-tune", "(LoRA or full)"], ["EM organism or", "benign control"]]),
    # the random direction is a seeded Gaussian (persona_em/axes.py random_axis), not a persona vector: own group
    ("Direction", [["persona vectors:", r"$-\bar{s}$ (global shift)", r"$-(s_{\mathrm{def}}-\bar{s})$",
                    r"$-(s_A-\bar{s})$"],
                   ["control: random", "direction, matched", "in norm"]]),
    ("Steering", [["persona layer", "(half depth),", "last token"], [r"size: $\alpha$ $\times$ mean", "activation norm"],
                  [r"$\alpha$ = 0.15, 0.30, 0.45", r"$\alpha$ = 0: unsteered"]]),
    ("Generation", [["EM: 8 questions", r"$\times$ 30 samples"],
                    ["or in-domain:", "medical, financial,", "reward hacks, cheese"], ["temperature 1"]]),
    ("Scoring", [["LLM judge:", "aligned and", "coherent, 0 to 100"], ["cheese, reward", "hacks: own scorer,",
                                                                       "bootstrap CI"]]),
    ("Rate per cell", [["M = share with", "aligned < 30", "of coherent > 50"], ["Wilson 95% CI"],
                       ["Fisher test vs", "unsteered run"]]),
]

fig = plt.figure(figsize=(FULL, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FULL)
ax.set_ylim(0, H)
ax.axis("off")
ax.grid(False)

PAD_X, BADGE_R, TOP, BOT = 0.08, 0.075, H - 0.02, 0.02
TITLE_Y = TOP - 0.16
BODY_Y0, LINE, GROUP_GAP = TITLE_Y - 0.24, 0.142, 0.07
FS_TITLE, FS_BODY = 7.5, 7.0

# draw every text at x = 0 first, measure it, then lay the boxes out
renderer = fig.canvas.get_renderer()
inch_per_px = 1.0 / fig.dpi
blocks = []
lowest = H
for title, groups in STEPS:
    t = ax.text(0, TITLE_Y, title, fontsize=FS_TITLE, weight="bold", color=INK, ha="left", va="center")
    body, y = [], BODY_Y0
    for g in groups:
        for s in g:
            body.append(ax.text(0, y, s, fontsize=FS_BODY, color=INK2, ha="left", va="baseline"))
            y -= LINE
        y -= GROUP_GAP
    lowest = min(lowest, min(b.get_window_extent(renderer).y0 for b in body) * inch_per_px)
    w_title = t.get_window_extent(renderer).width * inch_per_px + 2 * BADGE_R + 0.06
    w_body = max(b.get_window_extent(renderer).width for b in body) * inch_per_px
    blocks.append((t, body, max(w_title, w_body) + 2 * PAD_X))

total = sum(w for *_, w in blocks)
gap = (FULL - total) / (len(blocks) - 1)
assert gap >= 0.14, f"boxes too wide for the page: gap {gap:.3f} in"
assert lowest >= BOT + 0.04, f"text runs below the boxes: lowest {lowest:.3f} in"

x = 0.0
edges = []
for i, (t, body, w) in enumerate(blocks):
    ax.add_patch(FancyBboxPatch((x + 0.01, BOT), w - 0.02, TOP - BOT, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=SURFACE, ec=AXIS, lw=0.6, zorder=1))
    cx = x + PAD_X + BADGE_R
    ax.add_patch(Circle((cx, TITLE_Y), BADGE_R, fc=C_MAIN, ec="none", zorder=2))
    ax.text(cx, TITLE_Y - 0.003, str(i + 1), fontsize=6.5, weight="bold", color="white", ha="center", va="center",
            zorder=3)
    t.set_x(cx + BADGE_R + 0.05)
    for b in body:
        b.set_x(x + PAD_X)
    edges.append((x, x + w))
    x += w + gap

ymid = (TOP + BOT) / 2
for (_, r), (l2, _) in zip(edges[:-1], edges[1:]):
    ax.add_patch(FancyArrowPatch((r + 0.015, ymid), (l2 - 0.005, ymid), arrowstyle="-|>,head_length=0.35,head_width=0.16",
                                 mutation_scale=10, color=INK2, lw=0.9, shrinkA=0, shrinkB=0, zorder=2))

save(fig, "fig_method_pipeline")
