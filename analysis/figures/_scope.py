"""A one-line note stating which organisms a figure covers and why the rest are absent.

A figure that quietly drops a model reads as a complete panel. Every figure whose coverage is
narrower than the geometry now prints the gap on its own face, so a reader never has to guess.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths

TOTAL_NOTE = None


def geometry_organisms():
    """[(family, organism)] for every organism the geometry holds."""
    out = []
    for p in _paths.geom_files():
        d = json.load(open(p))
        out += [(d["family"], o) for o in d["layers"]["50%"]["organisms"]]
    return out


def note(shown, reason):
    """`shown` is how many organisms the figure draws; `reason` says why the rest are missing."""
    n = len(geometry_organisms())
    if shown >= n:
        return f"Covers all {n} organisms in the study."
    return f"Covers {shown} of the {n} organisms in the study. The rest are absent because {reason}."


def stamp(fig, text, y=0.006, size=7.0, width=118, pad=0.10):
    """Print the note under the axes, wrapped to the figure's MEASURED width, in canvas added
    below the axes so it never sits on an axis label.

    A fixed character count only looks like wrapping. It held while the text was short, then a
    longer reason string produced a 118-character first line that ran off both ends of the canvas,
    so the note lost its first and last words on the page. The width is measured now, and the wrap
    tightens until the widest line actually fits. The note then got long enough (four lines once it
    named every absent organism) to cover the x-axis label, so the canvas now grows by the note's
    height and every axes keeps its physical position and size.
    """
    import textwrap
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches()
    avail = W * fig.dpi * 0.97
    w = width
    while True:
        lines = textwrap.wrap(text, width=w)
        t = fig.text(0.5, y, "\n".join(lines), ha="center", va="bottom", fontsize=size,
                     color="#666")
        ext = t.get_window_extent(renderer=r)
        if ext.width <= avail or w <= 24:
            break
        t.remove()
        w -= 4
    th = ext.height / fig.dpi + pad                      # inches the note needs
    boxes = [(ax, ax.get_position()) for ax in fig.axes]  # physical positions, before resizing
    fig.set_size_inches(W, H + th, forward=False)
    for ax, b in boxes:
        ax.set_position([b.x0, (b.y0 * H + th) / (H + th), b.width, b.height * H / (H + th)])
    t.set_position((0.5, y * H / (H + th)))
    return len(lines)
