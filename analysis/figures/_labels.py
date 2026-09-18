"""Greedy label placement in display space.

Placing a role label at a fixed offset from its point makes two nearby roles print on top of each
other, which destroys both words. This searches twelve directions at increasing distance and takes
the first position whose text box hits nothing already placed, then draws a leader line when the
label had to move far.
"""
import math
import numpy as np
from matplotlib.transforms import Bbox

DIRS = [(math.cos(a), math.sin(a)) for a in np.deg2rad(np.arange(0, 360, 30))]
DISTS = [10, 15, 21, 29, 39, 50]


def place_labels(ax, entries, star_xy=None, diamond_xy=None, fontsize=6.0, marker_half=0):
    """entries: [(point_xy, text, color, bold)]. The star, the diamond, and every labelled point
    are seeded as obstacles, so a label never prints over a marker it or a neighbour owns.
    `marker_half` is the half-size in points of the highlighted marker drawn at each point."""
    renderer = ax.figure.canvas.get_renderer()
    # A label placed outside the panel is clipped and unreadable, so the panel box is a hard
    # constraint, inset slightly so a glyph never touches the spine.
    axbb = ax.get_window_extent(renderer).expanded(0.985, 0.985)
    pts = np.array([e[0] for e in entries], float)
    anchor = np.array(star_xy) if star_xy is not None else pts.mean(0)
    placed = []
    for mxy, half in ((star_xy, 13), (diamond_xy, 8)):
        if mxy is not None:
            cx, cy = ax.transData.transform(mxy)
            placed.append(Bbox.from_bounds(cx - half, cy - half, 2 * half, 2 * half))
    if marker_half:
        for pxy, *_ in entries:
            cx, cy = ax.transData.transform((float(pxy[0]), float(pxy[1])))
            placed.append(Bbox.from_bounds(cx - marker_half, cy - marker_half,
                                           2 * marker_half, 2 * marker_half))

    def draw(px, py, text, color, bold, ox, oy):
        return ax.annotate(text, (px, py), fontsize=fontsize, color=color,
                           ha="left" if ox >= 0 else "right",
                           va="bottom" if oy >= 0 else "top",
                           fontweight="bold" if bold else "normal",
                           xytext=(ox, oy), textcoords="offset points", zorder=9,
                           bbox=dict(boxstyle="round,pad=0.07", fc="white", ec="none", alpha=0.85))

    # bold roles are the ones the caption names, so they get first pick of the good positions
    for i in sorted(range(len(entries)), key=lambda i: not entries[i][3]):
        pxy, text, color, bold = entries[i]
        px, py = float(pxy[0]), float(pxy[1])
        away = np.array([px, py]) - anchor
        n = np.linalg.norm(away)
        base = away / n if n > 1e-6 else np.array([1.0, 0.5])
        dirs = sorted(DIRS, key=lambda d: -(d[0] * base[0] + d[1] * base[1]))
        chosen = fallback = None
        for dist in DISTS:
            for d in dirs:
                ann = draw(px, py, text, color, bold, d[0] * dist, d[1] * dist)
                bb = ann.get_window_extent(renderer).expanded(1.14, 1.18)
                inside = axbb.contains(bb.x0, bb.y0) and axbb.contains(bb.x1, bb.y1)
                if inside and not any(bb.overlaps(pb) for pb in placed):
                    chosen = (ann, bb, dist)
                    break
                # a position that only leaves the panel is still better than one that
                # prints on top of another label, so keep the first such candidate
                if fallback is None and not any(bb.overlaps(pb) for pb in placed):
                    fallback = (px, py, text, color, bold, d[0] * dist, d[1] * dist)
                ann.remove()
            if chosen:
                break
        if chosen is None:
            a = fallback or (px, py, text, color, bold,
                             base[0] * DISTS[-1], base[1] * DISTS[-1])
            ann = draw(*a)
            chosen = (ann, ann.get_window_extent(renderer).expanded(1.14, 1.18), DISTS[-1])
        ann, bb, dist = chosen
        placed.append(bb)
        ax.plot(px, py, "o", ms=3.4, mfc="none", mec=color, mew=0.7, zorder=5)
        if dist >= 21:
            cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
            lx, ly = ax.transData.inverted().transform((cx, cy))
            ax.plot([px, lx], [py, ly], color=color, lw=0.45, alpha=0.45, zorder=4,
                    solid_capstyle="round")
