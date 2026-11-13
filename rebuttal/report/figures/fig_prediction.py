from style import *

import math
import textwrap

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

PRED = ROOT / "reanalysis/prediction/predictions.csv"
pred = pd.read_csv(PRED)
d = pred[(pred.rule == "W") & (pred.target == "M") & (pred.score == "composite")].copy()
# the paper's nine controls are exactly the folders named *-control in per_organism.csv (2026-11-13 verifier: a name
# test drew the Qwen3.5-9B reward-hacks control as an organism and R1-distill, id-baseline as controls)
_lab = pd.read_csv(ROOT / "reanalysis/extra-evals/per_organism.csv")[["family", "organism", "label"]].drop_duplicates()
_ctl = {(f, o) for f, o, l in _lab.itertuples(index=False) if l.endswith("-control")}
assert len(_ctl) == 8, sorted(_ctl)  # the nine controls minus topic-locked, which has no misalignment rate
d["control"] = [(f, o) in _ctl for f, o in zip(d.family, d.organism)]

PANELS = [("LOOO", "a   Leave one organism out"), ("LOFO", "b   Leave one family out"),
          ("OUT", "c   Fitted on the pool, outside models")]
N_LABEL = 3  # per panel, the organisms with the largest miss |yhat - y| get a name
MS, MS_IN = 4.6, 3.6  # marker size in the main panels and in the zoom, points

# one square frame for all panels, so a point means the same in each
lo = math.floor(min(d.y.min(), d.yhat.min()) / 0.1) * 0.1
hi = math.ceil(max(d.y.max(), d.yhat.max()) / 0.1) * 0.1
pad = 0.03
ticks = np.round(np.arange(lo, hi + 1e-9, 0.1), 1)

# Zoom near the origin for the pool panels: the controls all sit at observed 0, where their markers tie.
# ZX, ZY: the zoomed window (data units); INSET: where the zoom is drawn, in the panel's data units.
ZOOM_PANELS = ("LOOO", "LOFO")
ZX, ZY = (-0.012, 0.042), (-0.07, 0.112)
INSET = (0.45, -0.075, 0.17, 0.205)  # x0, y0, width, height

fig, axes = plt.subplots(1, 3, figsize=(FULL, 3.25), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.10, left=0.075, right=0.985, top=0.88, bottom=0.245)
FS_LABEL = 6.2


def minus(v: float, fmt: str = "g") -> str:
    return f"{v:{fmt}}".replace("-", "−")


def _seg_cross(p1, p2, p3, p4) -> bool:
    """Do segments p1-p2 and p3-p4 cross?"""
    def o(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return o(p1, p2, p3) != o(p1, p2, p4) and o(p3, p4, p1) != o(p3, p4, p2)


def _seg_hits_box(p, q, box, n=40) -> bool:
    t = np.linspace(0.02, 0.98, n)[:, None]
    s = np.asarray(p) + t * (np.asarray(q) - np.asarray(p))
    return bool(((s[:, 0] > box[0]) & (s[:, 0] < box[2]) & (s[:, 1] > box[1]) & (s[:, 1] < box[3])).any())


def place_labels(ax, all_xy, items, obstacles=(), segments=()):
    """Greedy placement of named labels with leaders: each label takes the free spot (no point under it,
    no other label, leader or obstacle crossing) closest to its own point. items = [(x, y, text)] in data
    units, most important first; obstacles = display boxes (x0, y0, x1, y1), segments = display lines.
    Works in display pixels after the frame is fixed."""
    fig = ax.figure
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    T = ax.transData
    px = fig.dpi / 72.0
    P = T.transform(np.asarray(all_xy))
    ab = ax.get_window_extent(ren)
    diag = T.transform([[lo - pad, lo - pad], [hi + pad, hi + pad]])
    boxes, leaders = list(obstacles), list(segments)
    for tx, ty, text in items:
        t = ax.text(0, 0, text, fontsize=FS_LABEL, linespacing=1.05)
        bb = t.get_window_extent(ren)
        w, h = bb.width, bb.height
        t.remove()
        q = T.transform([[tx, ty]])[0]
        others = P[np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]) > 0.5]
        best, best_cost = None, np.inf
        for x0 in np.arange(ab.x0 + 2 * px, ab.x1 - w - 2 * px, 1.5 * px):
            for y0 in np.arange(ab.y0 + 2 * px, ab.y1 - h - 2 * px, 1.5 * px):
                m = 3.0 * px  # marker radius plus air
                box = (x0 - m, y0 - m, x0 + w + m, y0 + h + m)
                cost = 1000.0 * (((P[:, 0] > box[0]) & (P[:, 0] < box[2]) & (P[:, 1] > box[1]) &
                                  (P[:, 1] < box[3])).sum())
                m2 = 14.0 * px  # a point this close to a label reads as belonging to it
                cost += 250.0 * (((others[:, 0] > x0 - m2) & (others[:, 0] < x0 + w + m2) &
                                 (others[:, 1] > y0 - m2) & (others[:, 1] < y0 + h + m2)).sum())
                cost += sum(1000.0 for b in boxes if not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or
                                                           box[1] > b[3]))
                c = np.array([np.clip(q[0], x0, x0 + w), np.clip(q[1], y0, y0 + h)])
                L = np.hypot(*(q - c))
                if L < 12 * px:  # every label gets a visible leader, so it cannot be read as a neighbour's
                    cost += 1000.0
                cost += 0.05 * L / px
                cost += sum(500.0 for b in boxes if _seg_hits_box(c, q, b))
                cost += sum(500.0 for (a1, a2) in leaders if _seg_cross(c, q, a1, a2))
                cost += sum(500.0 for (a1, a2) in leaders if _seg_hits_box(a1, a2, box))
                # leader passing over another point
                v = q - c
                if L > 0:
                    tt = np.clip(((others - c) @ v) / (v @ v), 0, 1)
                    dist = np.hypot(*(others - (c + tt[:, None] * v)).T)
                    cost += 150.0 * (dist < 5 * px).sum()
                if _seg_hits_box(diag[0], diag[1], box, n=400):
                    cost += 200.0
                if cost < best_cost:
                    best, best_cost = (x0, y0, c), cost
        x0, y0, c = best
        print(f"label {text.splitlines()[-1]!r}: placement cost {best_cost:.0f}")
        rel = ((c[0] - x0) / w, (c[1] - y0) / h)
        ax.annotate(text, (tx, ty), xytext=T.inverted().transform([[x0, y0]])[0], ha="left", va="bottom",
                    fontsize=FS_LABEL, color=INK2, zorder=4, linespacing=1.05,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0.5, shrinkB=3, relpos=rel))
        boxes.append((x0 - 2 * px, y0 - 2 * px, x0 + w + 2 * px, y0 + h + 2 * px))
        leaders.append((c, q))


def draw_marker(a, x, y, r, ms):
    col = C_MAIN if r.set == "pool" else C_ALT
    a.plot(x, y, ls="", marker=FAMILY_MARKER[r.family], ms=ms, zorder=3,
           mfc="white" if r.control else col, mec=col if r.control else "white",
           mew=(1.0 if r.control else 0.5) * ms / MS)


def zoom(ax, s):
    """Inset that magnifies the window ZX x ZY of panel ax. Organisms tied at observed 0 are spread
    sideways (a beeswarm around x = 0) until no two markers overlap; every other marker keeps its x.
    Returns the display boxes and segment the labels must avoid."""
    z = s[(s.y >= ZX[0]) & (s.y <= ZX[1]) & (s.yhat >= ZY[0]) & (s.yhat <= ZY[1])]
    zero = s[s.y == 0]
    assert set(zero.index) <= set(z.index), "an organism observed at 0 falls outside the zoom"
    ins = ax.inset_axes(INSET, transform=ax.transData, zorder=5)
    ins.set_xlim(ZX)
    ins.set_ylim(ZY)
    ins.set_facecolor("white")
    ins.grid(False)
    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_color(AXIS)
        sp.set_linewidth(0.5)
    ins.axhline(0, color=AXIS, lw=0.5, zorder=1)
    ins.axvline(0, color=AXIS, lw=0.5, zorder=1)
    xt = [0.0, 0.03]
    # y: labelled ticks every 0.1, unlabelled ticks between them (narrow labels keep the inset small)
    yt = np.round(np.arange(math.ceil(ZY[0] / 0.05) * 0.05, ZY[1] + 1e-9, 0.05), 2)
    yt_major = [v for v in yt if abs(round(v / 0.1) * 0.1 - v) < 1e-9]
    ins.set_xticks(xt)
    ins.set_xticklabels([minus(v) for v in xt])
    ins.set_yticks(yt_major)
    ins.set_yticklabels([minus(v) for v in yt_major])
    ins.set_yticks([v for v in yt if v not in yt_major], minor=True)
    ins.tick_params(labelsize=6.0, length=1.5, pad=1.0, width=0.4)
    ins.tick_params(which="minor", length=1.0, width=0.4, color=MUTED)
    title = ins.set_title("Zoom near 0;\nties spread out", fontsize=6.0, loc="right", pad=2.0, color=INK2,
                          linespacing=1.0)

    # positions in display pixels, to test overlap
    fig.canvas.draw()
    bb = ins.get_window_extent()
    sx, sy = bb.width / (ZX[1] - ZX[0]), bb.height / (ZY[1] - ZY[0])
    diam = MS_IN * fig.dpi / 72.0 * 1.15  # marker diameter plus a hair of air, pixels
    xs = {k: r.y for k, r in z.iterrows() if r.y != 0}
    placed = [(xs[k] * sx, z.loc[k].yhat * sy) for k in xs]
    for k, r in zero.sort_values("yhat").iterrows():
        for j in range(0, 40):
            off = 0 if j == 0 else (1 if j % 2 else -1) * ((j + 1) // 2)
            p = (off * diam, r.yhat * sy)
            if all(np.hypot(p[0] - a, p[1] - b) >= diam for a, b in placed):
                break
        placed.append(p)
        xs[k] = p[0] / sx
    pts = np.array([(xs[k] * sx, z.loc[k].yhat * sy) for k in z.index])
    dmin = min(np.hypot(*(pts[i] - pts[j])) for i in range(len(pts)) for j in range(i + 1, len(pts)))
    assert dmin >= diam / 1.15 - 1e-6, "markers in the zoom still overlap"
    assert all(ZX[0] < v < ZX[1] for v in xs.values()), "a spread marker left the zoom window"
    for k, r in z.iterrows():
        draw_marker(ins, xs[k], r.yhat, r, MS_IN)

    # the zoomed window, outlined on the panel
    ax.add_patch(Rectangle((ZX[0], ZY[0]), ZX[1] - ZX[0], ZY[1] - ZY[0], fill=False, ec=MUTED, lw=0.5,
                           zorder=2))

    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    # the inset's footprint: its frame with the tick labels, and its title above
    fb = [ins.get_window_extent(ren), ins.xaxis.get_tightbbox(ren), ins.yaxis.get_tightbbox(ren)]
    tt = title.get_window_extent(ren)
    boxes = [(min(b.x0 for b in fb), min(b.y0 for b in fb), max(b.x1 for b in fb), max(b.y1 for b in fb)),
             (tt.x0, tt.y0, tt.x1, tt.y1)]
    T = ax.transData
    P = T.transform(s[["y", "yhat"]].to_numpy())
    rad = MS * fig.dpi / 72.0 / 2
    for b in boxes:
        inside = ((P[:, 0] > b[0] - rad) & (P[:, 0] < b[2] + rad) & (P[:, 1] > b[1] - rad) &
                  (P[:, 1] < b[3] + rad)).sum()
        assert inside == 0, "the inset covers a point of the panel"
    pb = ax.get_window_extent(ren)
    for b in boxes:
        assert b[0] >= pb.x0 and b[2] <= pb.x1 and b[1] >= pb.y0 and b[3] <= pb.y1, (
            "the inset leaves the panel", T.inverted().transform([b[:2], b[2:]]))
    m = 2.0
    for b in boxes:  # a plain card under the inset and its title, so the panel's grid does not run through them
        (u0, v0), (u1, v1) = T.inverted().transform([(b[0] - m, b[1] - m), (b[2] + m, b[3] + m)])
        ax.add_patch(Rectangle((u0, v0), u1 - u0, v1 - v0, facecolor="white", edgecolor="none", zorder=1.5))
    r0, r1 = T.transform([[ZX[0], ZY[0]], [ZX[1], ZY[1]]])
    print(f"zoom: {len(z)} markers, {len(zero)} at observed 0, min centre distance "
          f"{dmin / (MS_IN * fig.dpi / 72.0):.2f} marker diameters")
    return [(b[0] - m, b[1] - m, b[2] + m, b[3] + m) for b in boxes] + [(r0[0], r0[1], r1[0], r1[1])], []


for ax, (scheme, title) in zip(axes, PANELS):
    s = d[d.scheme == scheme]
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls=(0, (4, 3)), lw=0.8, color=C_BASE, zorder=1)
    ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
    for _, r in s.iterrows():
        draw_marker(ax, r.y, r.yhat, r, MS)
    rho = spearmanr(s.y, s.yhat).statistic
    ax.set_title(f"{title}\nn = {len(s)}, Spearman {minus(rho, '.2f')}", loc="left")

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([minus(t) for t in ticks])
    ax.set_yticklabels([minus(t) for t in ticks])
    ax.set_xlabel("Observed misalignment rate (0-1)")

    obstacles, segments = zoom(ax, s) if scheme in ZOOM_PANELS else ([], [])

    # name the organisms with the largest misses, largest first
    miss = (s.yhat - s.y).abs().sort_values(ascending=False).index[:N_LABEL]
    place_labels(ax, s[["y", "yhat"]].to_numpy(),
                 [(s.loc[k].y, s.loc[k].yhat,
                   "\n".join([s.loc[k].family] + textwrap.wrap(s.loc[k].organism, 16, break_long_words=False,
                                                               break_on_hyphens=False)))
                  for k in miss], obstacles, segments)
axes[0].set_ylabel("Predicted misalignment rate (0-1)")


def row_major(h, ncol):
    """matplotlib fills legend columns first; reorder so the entries read across the rows."""
    nrow = math.ceil(len(h) / ncol)
    rows = [h[i * ncol:(i + 1) * ncol] for i in range(nrow)]
    return [rows[r][c] for c in range(ncol) for r in range(nrow) if c < len(rows[r])]


fams = [f for f in POOL_FAMILIES + OTHER_FAMILIES if f in set(d.family)]
fam_handles = [Line2D([], [], ls="", marker=FAMILY_MARKER[f], ms=MS, color=INK2, mec="white", mew=0.4, label=f)
               for f in fams]
kind_handles = [
    Line2D([], [], ls="", marker="o", ms=MS, color=C_MAIN, mec="white", mew=0.5, label="Pool organism"),
    Line2D([], [], ls="", marker="o", ms=MS, color=C_ALT, mec="white", mew=0.5, label="Outside organism"),
    Line2D([], [], ls="", marker="o", ms=MS, mfc="white", mec=INK2, mew=1.0, label="Hollow: control"),
    Line2D([], [], ls=(0, (4, 3)), lw=0.8, color=C_BASE, label="Predicted = observed"),
]
NC_FAM, NC_KIND = 4, 2
leg_f = fig.legend(handles=row_major(fam_handles, NC_FAM), loc="lower left", bbox_to_anchor=(0, 0), ncol=NC_FAM,
                   handletextpad=0.2, columnspacing=1.0, labelcolor=INK2, borderaxespad=0, borderpad=0)
leg_k = fig.legend(handles=row_major(kind_handles, NC_KIND), loc="lower left", bbox_to_anchor=(0, 0),
                   ncol=NC_KIND, handletextpad=0.4, columnspacing=1.2, labelcolor=INK2, borderaxespad=0,
                   borderpad=0)
# the two legend blocks side by side, centred under the panels
fig.canvas.draw()
ren = fig.canvas.get_renderer()
fw = fig.bbox.width
w_f, w_k = leg_f.get_window_extent(ren).width / fw, leg_k.get_window_extent(ren).width / fw
gap = 0.45 / FULL
mid = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
x_f = mid - (w_f + gap + w_k) / 2
leg_f.set_bbox_to_anchor((x_f, 0.012), transform=fig.transFigure)
leg_k.set_bbox_to_anchor((x_f + w_f + gap, 0.012), transform=fig.transFigure)

save(fig, "fig_prediction")
