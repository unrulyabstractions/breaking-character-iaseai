"""The pipeline, as one figure for the paper body (the author's own design, 2026-09-16): one base model and its narrow fine-tunes; the
persona space read from each (roles prompted, activations at one depth, the default axis from the
base); the tuned cloud beside the base cloud and the shift between them; then the readings of that
comparison, each against the judged rate with a fitted line and 95% intervals on both axes. The
clouds are real, projected onto the default axis and the cast's leading residual direction.

In:  the sweep cache (see geometry/README.md) and data/raw/{behaviour_bundle,zoo_cast,
     graph_features}.json.
Out: <paper>/figures/method/method.{pdf,png}
Feeds: Figure 1.

The layout follows blog/reference/pipeline_reference.png region by region. Every number below was
set against a measurement of that drawing, not by eye.

Glyph key scope: fourteen entries, seven columns by two rows, spread to the figure's own ink width.
They name every mark the clouds and the deformation panels draw, the fan that carries the claim that
both deformations are measured against one baseline, the fit and its intervals, and the row colour
that ties each table row to its frames and its arrow. Not keyed: the braces, the mauve boxes, the
elbow, the ellipsis dots and the grid, which carry no reading; and the correlations' own scatter
points, which take the colour of the reading named under their own panel. A fifteenth entry forces
eight columns, and eight columns put the labels 10 px apart, which reads as touching at print size.
"""
import pathlib, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.transforms
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple, HandlerBase
import matplotlib.patheffects as pe

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _paths                              # noqa: E402
from _paperstyle import FAMC, use            # noqa: E402
use(9)
import features_graph as FG                  # noqa: E402
from scipy.stats import spearmanr             # noqa: E402
import fig_metric_slices as MS               # noqa: E402  the bootstrap and Wilson the readings use
from _organism_meta import NOT_CONTROL   # the distillation and the broad fine-tune whose keys say (control) (2026-09-19)

PAPER = _paths.PAPER
FAM, ORG, DEP = "Qwen2.5-14B", "finance", 2    # 50%: the pooled family with the most organisms and the cleanest trend (rho -0.86, p .005); its highest-rate organism beside the base
zb = np.load(FG.CACHE / f"{FAM}__base.npz", allow_pickle=True)
zo = np.load(FG.CACHE / f"{FAM}__{ORG}.npz", allow_pickle=True)
names = [str(v) for v in zb["names"]]; ai, di = names.index("assistant"), names.index("default")
V0 = np.asarray(zb["X"][DEP], float); V1 = np.asarray(zo["X"][DEP], float); a = np.asarray(zb["axis"][DEP], float)
mu = V0.mean(0); U, S, Wt = np.linalg.svd(V0 - mu, full_matrices=False)
# first direction: the default axis itself; second: the leading residual direction
e1 = a / np.linalg.norm(a); r = (V0 - mu) - np.outer((V0 - mu) @ e1, e1)
_, _, Wr = np.linalg.svd(r, full_matrices=False); e2 = Wr[0]
P0 = np.c_[(V0 - mu) @ e1, (V0 - mu) @ e2]; P1 = np.c_[(V1 - mu) @ e1, (V1 - mu) @ e2]
# the control row draws the family's own control, good-medical: R1-distill, drawn here until 2026-09-19, is a
# distillation and not a control (author, 2026-09-19)
zc = np.load(FG.CACHE / f"{FAM}__GOOD-medical (control).npz", allow_pickle=True)
VC = np.asarray(zc["X"][DEP], float); PC = np.c_[(VC - mu) @ e1, (VC - mu) @ e2]
rate = FG.evil_rates()[0]; E = FG.evil_group(names, rate)

# ---------------------------------------------------------------------------------------------
# THE PIPELINE. Rows are models: the baseline, one EM organism, one control. Each row carries its
# judged rates, its persona cloud, and, for the tuned rows, the deformation of the baseline cloud
# into it. The behavioural half and the geometric half meet in the correlations on the right.
BLUE, RED, GREEN, MAUVE = "#4a7fb5", "#c0504d", "#73a454", "#cfaabd"
MAUVE_D = "#9b5278"                              # the plus alone, which the drawing draws dark
# The mauve is the drawing's own, sampled on its box rules, its elbow and its three dots:
# (207,170,189) against the render's (178,120,155), which made the correlations frame the heaviest
# object on a canvas where the drawing keeps it a quiet enclosure (five critics, round 7). The plus
# is the one mark the drawing draws dark, and it samples (155,82,120).
# The row blue and the tuned-role blue are two different meanings, so they are two different blues:
# the reference's frame reads (71,126,190) and its tuned dots (77,115,159), measured 2026-09-16.
# The control green is the reference's yellow-green (115,164,84), not a blue-green: sampled off the
# drawing's row-3 frame, whose fill and lane tint are sampled with it.
EVIL, ROLEC = "#d1352b", "#4d739f"
# The role shifts are drawn in ONE neutral grey, head dark and shaft pale. They used to take a
# coolwarm ramp on the along-axis component, which nothing on the page named, and which spent the
# figure's blue and its red a second time: blue already means the baseline row and red the EM row
# and the evil roles (art audit, 2026-09-17). The arrow's own direction already carries the sign
# of that component, so the ramp was a second encoding of what the arrow draws.
DEFC, DEFC_FADE = "#6b6b6b", 0.50
FANC = "#b4cfee"                                 # the baseline-reference fan, paler than any lane
# The fan and the baseline row's lane were #a5c3e5 and #a7c4e5, which sample as one colour, and
# only the fan is named in the key: two dashed blue arrows carrying two meanings shared one
# swatch (five critics, round 6). The fan is now the palest blue on the canvas, the lane the
# deepest, and the lane itself is named in the key under "model row".
BASEC = "#9a9a9a"                                # the baseline cast is grey in the reference, the tuned casts blue
GOLD, RIMG, INK, SUB, GREY = "#f2c200", "#8a6d00", "#222", "#555", "#b4b4b4"
# The cast centre is the darker of the two cloud rules and the assistant axis the paler, as the
# drawing sets them: it reads (179,179,179) against (230,230,230), measured 2026-09-16. #cfcfcf at
# lw 0.55 renders its own nominal 207, which is a quarter of the drawing's separation, so the axis
# is set at the value the page is meant to READ (five critics, round 4).
# The aligned control is the control organism, which row 3 already draws in its own green, so the
# ring takes that green darkened to hold its own against white, not a second hue for one object.
AXISC, ZEROC, CTLC = "#e6e6e6", "#b4b4b4", "#5f9440"
# The lane into each cloud carries the row's own tint, not the frame's full strength: the drawing's
# baseline lane is (167,196,229), its EM lane (209,109,106) and its control lane (120,167,90), and
# each row's lead-in ellipsis is a lighter tint again.
# The three lanes carry ONE weight, set in CIE L*: sampled off the render the blue ran L* 72.0
# against the red's 57.6 and the green's 63.6, so the baseline row read as greyed out beside its
# two siblings (art audit, 2026-09-17). Each keeps its own hue and saturation and takes the value
# that puts all three on L* 61, the middle of what the three were.
LANEC = {BLUE: "#7796b9", RED: "#de7471", GREEN: "#739f56"}
# The lead-in dots ARE the lane, so they take the lane's own colour. Sampled separately they ran
# L* 75.8, 54.7 and 81.6, a 27-point spread on three marks that name one thing.
LDOTC = LANEC
from _paperstyle import assert_canvas, tex_printed_width

W = tex_printed_width("sections/intro.tex", "method/method")
H = 3.05                                         # the reference is a wide band; 5.5 x 3.05 is 1.80:1
fig = plt.figure(figsize=(W, H))
def A(x, y, w, h): return fig.add_axes([x / W, y / H, w / W, h / H])
def T(x, y, *a, **k): return fig.text(x / W, y / H, *a, **k)
def fitdash(length, dash, lw):
    """A dash period fitted to `length` inches, so the run starts AND ends on a dash."""
    ron = dash[0] / (dash[0] + dash[1]); per = dash[0] + dash[1]; L = length * 72.0
    n = max(1, int(round((L + (1 - ron) * per) / per)))
    P = L / (n - 1 + ron)
    return (0, (ron * P / lw, (1 - ron) * P / lw))

# The drawing carries TWO frame weights and TWO dash periods, measured on it: its cloud frames
# rule 3 px with a 14/9 px dash, its deformation frames and its correlations box 4 px with an
# 18/14 px dash, on a 1999 px canvas. The render drew every frame at 4 px on one 13/10 period
# (five critics, round 7). CLOUDF and DEEPF are those two, in points at this canvas's own scale.
CLOUDF = dict(lw=0.60, dash=(2.78, 1.78))
DEEPF = dict(lw=0.80, dash=(3.56, 2.77))


def dashbox(x, y, w, h, c, lw=1.0, dash=(3.2, 2.4)):
    """Four dashed edges, each with its own fitted period, not one closed rectangle path.

    A rectangle carries one phase all the way round, so a dash wraps the corner and leaves a
    stepped burr instead of a clean L.

    The two horizontal edges carry the corners and run out by half a rule width at each end. The
    two vertical edges stand back by one gap, so the corner is the drawing's: its cloud frames run
    the horizontal rule through the corner and start the vertical after a gap (measured at the
    baseline frame's top left: rule at y 194-196, a 2 px nub, then white to y 208). Run to the
    corner the verticals welded a 9 px full-weight L onto every corner of all nine frames, which
    at 8x was the most prominent mark inside each cloud panel (five critics, round 7).
    """
    e = lw / 144.0                            # half the rule's own width, in inches
    g = dash[1] / 72.0                        # one gap, in inches
    for (ax_, ay_, bx_, by_), ln_ in (((x - e, y, x + w + e, y), w + 2 * e),
                                      ((x - e, y + h, x + w + e, y + h), w + 2 * e),
                                      ((x, y + g, x, y + h - g), h - 2 * g),
                                      ((x + w, y + g, x + w, y + h - g), h - 2 * g)):
        fig.lines.append(Line2D([ax_ / W, bx_ / W], [ay_ / H, by_ / H], transform=fig.transFigure,
                                color=c, lw=lw, ls=fitdash(ln_, dash, lw),
                                dash_capstyle="butt", solid_capstyle="butt", zorder=4))

def head(x1, y1, ux, uy, ln, wd, c, z=3.6, sweep=0.0, halo=0.0):
    """One solid triangle, drawn in inches so the canvas aspect does not shear it.

    The heads used to be the head of a dashed FancyArrowPatch, so the dash pattern chopped the
    triangle and the round shaft cap poked out past its point (audit, 2026-09-16).

    `sweep` notches the back edge into a raked barb, which is the head the drawing gives its
    heaviest arrow: the elbow into the correlations ended in a flat-backed triangle against the
    drawing's swept dart (five critics, round 6). The lane and fan arrows keep the plain triangle.
    """
    px, py = -uy, ux
    pts = [(x1, y1),
           (x1 - ux * ln + px * wd / 2, y1 - uy * ln + py * wd / 2)]
    if sweep:
        pts.append((x1 - ux * ln * (1 - sweep), y1 - uy * ln * (1 - sweep)))
    pts.append((x1 - ux * ln - px * wd / 2, y1 - uy * ln - py * wd / 2))
    _h = Polygon([(p / W, q / H) for p, q in pts], transform=fig.transFigure,
                 closed=True, fc=c, ec="none", zorder=z)
    if halo:
        _h.set_path_effects([pe.withStroke(linewidth=halo, foreground="white")])
    fig.patches.append(_h)

def blade(u):
    """The assistant shift's gold blade, as one profile the page and the glyph key both draw.

    `u` is the distance back from the barb, 0 at the barb and 1 at the tail; the value is the
    blade's width as a fraction of its own widest. Read off the drawing pixel by pixel: it rises to
    its widest a fifth of the way back from the barb, then runs thin the whole rest of the way and
    comes to a point at the tail.

    The profile is a WEDGE on a FLOOR: linear up to its widest a fifth back from the barb, then
    linear down toward the tail but never below a third of its widest. The floor is the drawing's:
    measured in eight bands along both of its blades, the gold runs 0.73, 1.00, 0.94, 0.39, 0.35,
    0.39, 0.26, 0.23 of its widest, so it is still a quarter as wide where it meets the tail. Run
    to a point the render's gold fell under its own keyline over the lower three fifths and the
    blade was left as a triangular blob at the barb, 0.26 of the glyph against the drawing's 0.58
    (five critics, round 8).
    """
    return np.maximum(np.minimum(u / 0.22, (1.0 - u) / 0.78), 0.38)


def dasharrow(x0, y0, x1, y1, c, lw=1.3, ln=0.075, wd=0.055, dash=(3.2, 2.2), z=3.6, halo=0.0):
    """A dashed shaft and a separate solid head, so only the shaft is dashed.

    The dash period is fitted to the shaft, so the shaft always ends ON a dash at the head's base.
    Left unfitted, each arrow landed on a different phase and three of them left a white gap
    between the last dash and the head (five critics, round 3).

    `halo` strokes the arrow in white under its own ink, so an arrow that crosses a lighter one
    breaks it cleanly and the two read as an overpass instead of a tangle.
    """
    dx, dy = x1 - x0, y1 - y0; L = float(np.hypot(dx, dy)); ux, uy = dx / L, dy / L
    bx, by = x1 - ux * ln, y1 - uy * ln
    _s = Line2D([x0 / W, bx / W], [y0 / H, by / H], transform=fig.transFigure,
                color=c, lw=lw, ls=fitdash(L - ln, dash, lw),
                dash_capstyle="butt", zorder=z - 0.6)
    if halo:
        _s.set_path_effects([pe.withStroke(linewidth=lw + halo, foreground="white")])
    fig.lines.append(_s)
    head(x1, y1, ux, uy, ln, wd, c, z=z, halo=halo)

# the drawing's row ellipses are 4 px dots on a 12 px pitch (0.20% and 0.60% of its width); the
# render drew 5 px dots on a 13 px pitch and its top ellipsis 7 px dots, which read as one blob
# rather than three dots (five critics, round 7)
def dots(x, y, c=INK, ms=0.82, dx=0.033):
    fig.lines.append(Line2D([(x - dx) / W, x / W, (x + dx) / W], [y / H] * 3, transform=fig.transFigure,
                            ls="none", marker="o", ms=ms, mfc=c, mec="none", zorder=3))

# ---- the title. The bold word stands alone. It used to carry an underline, and the descender of
# its own "p" cut straight through that rule and came out below it (art audit, 2026-09-17). No
# other heading in the paper is underlined.
# The size is set from the word's share of the canvas width: the drawing's PIPELINE is 10.1% of its
# own width and at 9.5 pt this one measured 13.3%, the largest type discrepancy in the figure.
T(W / 2, H - 0.05, "Pipeline", fontsize=8.6, fontweight="bold", color=INK, ha="center", va="top")   # sentence case (author's voice, 2026-09-17)

# ---- rows: the baseline, one EM organism, one control
# C is the judge's 0-100 coherence score, not a percentage (background.tex: "the rate as a
# percentage and the coherence on the judge's own 0 to 100 scale"), so the reference's "Coherence
# Rate" would name it wrongly. The head carries the NAME alone: set as "Coherence Score (C)/100"
# it read as a division applied to C, it was the only unit explained anywhere on the canvas, and it
# was the longest rotated string, which is what crowded the two heads together (five critics,
# round 6). The scale is the caption's to carry, and the head carries none of it: ", 0-100" was
# the one mark on the canvas that was neither a name nor a value (five critics, round 8).
# The three fills are the drawing's own: (211,226,241), (238,206,205), (189,214,172).
# measured, not drawn: the base model is 0 of 240 misaligned at a mean coherence of 98.3
# the coherence prints as a share, ".98", never a bare "98" (author, 2026-09-16)
# the control row is the family's control, good-medical (M .00, C .95 in Table 4); tech-pessimism is an
# organism, not a control (author's ruling, 2026-09-16), and the row used to draw it in control green
# M as a share, as the paper prints it everywhere else; "35%" was 0.180 in wide in a 0.170 in
# cell and touched the coherence beside it (measured by the guard below, 2026-09-17)
ROWS = [("baseline", ".00", ".98", BLUE, "#d3e2f1"),
        ("EMA organism", ".35", ".84", RED, "#eececd"),
        ("control", ".00", ".95", GREEN, "#bdd6ac")]
# Row centres. The ellipsis gap is the wider of the two, as the reference sets it, but only by a
# fifth: at 1.16 it ran 20.4% of the canvas height against the drawing's 17.5%, and the room it took
# was room the lower ellipsis row needed to clear the braces (five critics, round 6).
# The row fill is shallower: the drawing's is 54 px on a 1999 px canvas, 2.70% of its width,
# against the render's 63 px, 3.82% (five critics, round 7). The cell's own two lines set the
# floor, so this comes down to 3.45% and no further.
RY = [2.22, 1.78, 1.246]
RH = 0.19
# the gutter is the drawing's own 2 px: at 0.02 in it ran 6 px and the three cells read as detached
# chips with white channels between them instead of one table strip (five critics, round 5)
# The value cells are wider and the name cell narrower by the same amount, so the table's own right
# edge does not move: at CW 0.23 a bold "0%" filled 43% of its chip against the drawing's 25%, and
# the values read as crammed rather than floating (five critics, round 6).
# The table is the drawing's own share of the canvas width, and the lane is what it gives back.
# The render's table ran 18.7% of its width against the drawing's 16.2%, and the lane between its
# right edge and the cloud frame 8.2% against the drawing's 12.9%, so the lane arrow had no room
# to carry the drawing's own proportions (five critics, round 7). "Baseline Model" also stood 7 px
# from its cell's right edge, 0.42% of the width, against the drawing's 1.00%, so the name cell
# takes its width back from the two value cells.
_NAMES, _VALS = [], []   # the row names and the value cells, both measured below
NX, NW, CW, GUT = 0.085, 0.645, 0.178, 0.008   # the lane arrow's dash length goes negative past ~0.65  # name cell, two value cells, one gutter for all ("EM organism" ran over the arrow at 0.445, 2026-09-17)
AGAP = 0.09                                  # the arrow's gap between the name cell and the readings (author, 2026-09-17); NW + AGAP keeps the old right edge
TBR = NX + NW + AGAP + 2 * CW - GUT          # the table's right edge
RCX = [NX + NW + AGAP + (CW - GUT) / 2, NX + NW + AGAP + CW + (CW - GUT) / 2]   # the value columns' centres
# The heads descend left to right and are anchored by their END on the column they name, as the
# reference draws them. Rotated the other way they ended over the NEXT column (audit, 2026-09-16).
# Both heads end on ONE line, as the drawing sets them, and the block drops to meet the table: a
# 0.035 in stagger left the pair 10 px apart in height and hung the block off the title rule instead
# of off the cells it names (five critics, round 3).
# The second head is lifted along the strings' own perpendicular, so the two read as two labels:
# set on one line the clear band between them measured 11.3 px and they crowded into one block
# (five critics, round 5). At 0.017 in the band was still only 3.3% of the canvas height against the
# drawing's 4.85% and the descenders of one string all but met the ascenders of the other (five
# critics, round 6). 0.042 in along the perpendicular of a -30 degree string is (+0.021, +0.0364).
# The heads are also set SMALLER than the cells they name. The lift alone cannot open the band:
# two parallel strings 0.25 in apart stand 37.5 px apart on their own perpendicular whatever the
# lift, and at 4.4 pt the type ate all of it. Measured on the drawing, its head string runs 10.4%
# of its canvas width against this one's 13.0% at 4.4 pt (five critics, round 6). At 4.1 pt with
# this lift the clear band between the two strings measures 34 px, 3.7% of the canvas height,
# against the drawing's own 28 px, 2.9%, measured the same way on both.
# The lift is now taken ALONG the string as well as across it, so the second head's offset has no
# vertical component and the two strings end dead level: lifted on the perpendicular alone the two
# ink bottoms stood 11 px apart and the second head hung in its own band of air (round 7).
# (0.030 across, 0.052 along) = (+0.060, 0.0) on a -30 degree string.
# The rotation is -40 and not -30, and the second head is pushed a little further along its own
# line. At -30 the tail of "misalignment rate (M)" ran under the middle of "coherence (C)" over
# 58 px of the header's width and the two read as one tangled block (art audit, 2026-09-17).
# Steepening the string shortens that overlap and widens the band between the two lines at once.
HROT, HDX, HPT = -40.0, 0.072, 5.6
_HEADS = [T(_x + _dx, RY[0] + RH / 2 + 0.006, _s, fontsize=HPT, color=INK, rotation=HROT,
            rotation_mode="anchor", ha="right", va="bottom")
          for _x, _s, _dx in zip(RCX, ("misalignment rate ($M$)", "coherence ($C$)"),
                                 (0.0, HDX))]


def _check_heads():
    """The two rotated heads must leave a clear band, and each must end over its own column.

    Both were set by eye and both went wrong: the band was measured on the drawing and then not
    re-measured after the table's own geometry moved. Measure the band instead.
    """
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    _probe = T(0.5, 0.5, "misalignment rate ($M$)", fontsize=HPT, color="none")
    fig.canvas.draw()
    ink = _probe.get_window_extent(r).height / fig.dpi          # the type's own ascender to descender
    _probe.remove()
    band = (RCX[1] + HDX - RCX[0]) * abs(np.sin(np.radians(HROT))) - ink
    if band < 0.030:
        raise SystemExit(f"fig_method: the two rotated heads leave {band:.3f} in between them; "
                         f"steepen HROT or widen HDX")
    for _i, _t in enumerate(_HEADS):
        _end = RCX[_i] + (HDX if _i else 0.0)
        _l, _r = NX + NW + AGAP + _i * CW, NX + NW + AGAP + _i * CW + CW - GUT
        if not (_l <= _end <= _r):
            raise SystemExit(f"fig_method: head {_t.get_text()!r} ends at {_end:.3f} in, "
                             f"outside its own cell {_l:.3f}-{_r:.3f} in")
def _check_names():
    """No row name may reach into the arrow's gap. Set by eye, "EM organism" touched the arrow
    three times (author, 2026-09-17); measure it instead."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    room = NW - 0.034 - 0.020
    for t in _NAMES:
        w = t.get_window_extent(r).width / fig.dpi
        if w > room:
            raise SystemExit(f"fig_method: row name {t.get_text()!r} is {w:.3f} in wide in a {room:.3f} in cell; widen NW")
    for t in _VALS:                                   # "35%" filled its cell and touched ".84"
        w = t.get_window_extent(r).width / fig.dpi
        if w > CW - GUT - 0.020:
            raise SystemExit(f"fig_method: value {t.get_text()!r} is {w:.3f} in wide in a {CW - GUT:.3f} in cell")


LANE = TBR + 0.15                # the ellipsis lane between the table and the clouds
# heavier than the row ellipses, as the drawing separates the two meanings: measured on it, "more
# columns" is a 5 px dot on a 16 px pitch and "more rows" a 4 px dot on a 12 px pitch
dots(LANE, RY[0] + RH / 2 + 0.10, ms=1.0, dx=0.044)   # the table runs on past R and C
for (name, r, c, col, fc), y in zip(ROWS, RY):
    fig.patches.append(Rectangle((NX / W, (y - RH / 2) / H), (NW - GUT) / W, RH / H,
                                 transform=fig.transFigure, fc=fc, ec="none", zorder=1))
    _NAMES.append(T(NX + 0.034, y, name, fontsize=5.6, color=INK, ha="left", va="center", linespacing=1.25))
    # a small arrow from the model to its readings (author, 2026-09-17: "baseline -> M, C")
    fig.patches.append(FancyArrowPatch(((NX + NW - GUT + 0.014) / W, y / H), ((NX + NW + AGAP - 0.012) / W, y / H), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=4.0, lw=0.7, color=col, shrinkA=0, shrinkB=0, zorder=3))
    for i, v in enumerate((r, c)):
        fig.patches.append(Rectangle(((NX + NW + AGAP + i * CW) / W, (y - RH / 2) / H), (CW - GUT) / W, RH / H,
                                     transform=fig.transFigure, fc="#eeeeee", ec="none", zorder=1))
        # centred on the digits' own cap box: va="center" centres the font's full box, descender
        # space and all, which stood every value 4 px high in its cell (five critics, round 3)
        _VALS.append(T(RCX[i], y - 0.0053, v, fontsize=5.6, fontweight="bold", color=INK, ha="center", va="center"))

# ---- the persona clouds, one per row, and the deformations beside them. Two columns, side by
# side, with an arrow from a cloud to its own deformation and a thin one from the baseline cloud
# into each (the author's reference, 2026-09-16).
# The cloud was a 2.7:1 letterbox strip twice the reference's relative width, which flattened every
# cloud into a rule with a smear of dots and starved the deformation column and the readings.
# Re-measured on the drawing (2026-09-16): its cloud frames run 180x114 (row 1) and 191x99 (rows 2
# and 3), so 1.58:1 and 1.93:1; its deformation frames run 164x117 and 143x105, so 1.40:1 and 1.36:1.
# The render's clouds were 1.54:1 — 19% too tall for their width, which read the cast as a plume
# instead of the drawing's wide band (five critics, round 5). The cloud frame is SHORTER than the
# deformation frame beside it, as the drawing draws them, and both stay centred on the row.
# The three gutters between the four columns fall 10.4% / 8.9% / 8.9% of the canvas width against
# the drawing's 12.7 / 9.05 / 10.2. The middle one carries the cloud-to-deformation arrow, and at
# 6.9% that arrow ran 59 px with 39% of it head and one dash on the shaft (five critics, round 4).
CX, CW2, CH = 1.72, 0.50, 0.25               # frame 0.56 x 0.31 in = 1.81:1
DX, DW, DH = 2.77, 0.44, 0.30                # frame 0.50 x 0.36 in = 1.39:1
CFL, CFR = CX - 0.03, CX + CW2 + 0.03
DFL, DFR = DX - 0.03, DX + DW + 0.03
HEADY = 2.70                                  # one top line for the column heads
# The column heads are set from their own STRING WIDTHS on the drawing, which is the measurement
# that governs where they sit: "Persona Clouds" runs 10.06% of its canvas width there, "+ Assistant
# Axis" 11.56%, "Persona Space" 10.11% and "Deformations" 9.30%. At 8.0 pt the render drew 15.52,
# 16.12, 14.67 and 13.58%, so each head overhung the column it names by about 44 px a side and the
# two crowded the space between them, which is the author's "the text placement is horrible" (five
# critics, round 8). The drawing also sets both heads SMALLER than "Correlations" (cap 2.58% of its
# height against 2.06%); the render had all three at 2.73% and inverted that order.
# The cells hold at 4.2 pt: below that the table stops reading at the printed 5.5 in.
T(CX + CW2 / 2, HEADY, "persona clouds\n+ assistant axis", fontsize=7.0, fontweight="bold", color=INK,   # author, 2026-10-02: "assistant axis"
  ha="center", va="top", linespacing=1.3)
# This head sits on HEADY with the other two, so the figure has one header line. Staggered down
# onto its own first panel it hung 0.40 in below its neighbours and read as stranded halfway down
# the column (art audit, 2026-09-17). The gap over the empty baseline slot is the price, and a
# column head naming a column that starts one row down is the cheaper of the two readings.
T(DX + DW / 2, HEADY, "persona space\ndeformations", fontsize=7.0, fontweight="bold", color=INK,
  ha="center", va="top", linespacing=1.3)

# One pair of limits for all three clouds, taken from the data so the ink reaches every edge. At
# 1.45 the cast filled 54% of the frame and sat marooned in white; the padding existed only to clear
# an arrowhead on the assistant axis, which the drawing does not carry (five critics, round 3).
_ALL = np.r_[P0, P1, PC]
XL = 1.06 * np.abs(_ALL[:, 0]).max()
# The cloud's y limits are set from where the assistant axis has to LAND: the drawing puts the
# rule at 52%, 52% and 62% of the frame and the render had it at 66%, so the cast floated over its
# own axis and the rule read as a baseline (five critics, round 4). The ceiling is the cast's own
# 96th percentile, not its maximum: pinned to the maximum, the last eight roles of a long upper
# tail (15 of 234 above y = 10) bought a fifth of the frame in empty paper at the bottom, and every
# step toward the drawing's rule height bought more (five critics, round 5).
_RF = 0.56
YL1 = np.percentile(_ALL[:, 1], 96); YL0 = YL1 - YL1 / _RF
# The marker radius, in data units, on each axis. A role whose centre falls within that of a limit
# is NOT DRAWN: matplotlib clips a scatter at the axes edge, which stands 9 px INSIDE the dashed
# frame, so such a role was left as a flat-topped half-moon hanging in open white paper with nothing
# on the canvas to explain the cut (five critics, round 6). Moving the ceiling instead is no fix:
# walked up it buys a fifth of the frame in empty paper, walked down it drops a third of the cast.
# The frame is a WINDOW on the cast, as the deformation panel beside it is, and a mark is drawn
# whole or not at all: 12 of the 234 roles fall outside it.
MRX = 0.5 * np.sqrt(2.1) * (300.0 / 72.0) / (CW2 * 300.0 / (2 * XL))
MRY = 0.5 * np.sqrt(2.1) * (300.0 / 72.0) / (CH * 300.0 / (YL1 - YL0))
# The deformation panels carry their own limits, shared by both so the two fields stay comparable.
# The x range runs from the field's own left edge to wherever puts the assistant's baseline 7% of
# the panel in from the right rule, which is one tail marker and a little air: at 12% the last
# 2.9% of the panel held nothing but paper, and it was the control field, whose own ink stops
# further left, that read as a wisp beside a crowded neighbour (five critics, round 7).
# Taken as +/- max|x| it stood that role at 0.964 of the width and the frame cut its keyed tail
# marker in half. The y range is the field's own 1st to 90th percentile, not its full spread: the
# full spread put 55% of the ink in one fifth of the panel and left the top fifth blank paper,
# against a drawing whose field runs rule to rule (five critics, round 5). The ceiling is the 90th
# and not the 88th because four more shifts fall inside it and the field still reaches the rule.
_FY = np.r_[P0[:, 1], P1[:, 1], PC[:, 1]]
DXL0 = _ALL[:, 0].min(); DXL1 = DXL0 + (P0[ai, 0] - DXL0) / 0.93
DYL0, DYL1 = np.percentile(_FY, 1), np.percentile(_FY, 90)

def frame(ax, x0=None, x1=None, y0=None, y1=None):
    ax.set_xlim(-XL if x0 is None else x0, XL if x1 is None else x1)
    ax.set_ylim(YL0 if y0 is None else y0, YL1 if y1 is None else y1)
    for sp in ax.spines.values(): sp.set_visible(False)

def cloud(ax, P, rolec, s=1.00):
    # Both rules run out to the dashed frame, which stands 0.03 in outside the axes on every side.
    # The axis used to stop short at both ends and end in a solid head pinched into the gap between
    # the badge and the frame; the drawing runs a plain hairline edge to edge (five critics, round 3).
    # The cast centre stops on the frame rule's INNER edge, so it abuts the border in a clean T
    # instead of running half a dash width out through it (five critics, round 4).
    # the centre rule is a HAIRLINE behind the dots: at lw 0.6 it ran 2.5 px of near-solid #b4b4b4
    # against the drawing's 1.2 px, which made it the heaviest mark inside every cloud and split
    # each cast in two (five critics, round 5)
    ax.axvline(0, color=ZEROC, lw=0.3, zorder=1, ymin=-0.03 / CH, ymax=1 + 0.03 / CH, clip_on=False)
    ax.axhline(0, color=AXISC, lw=0.55, zorder=1.5, xmin=-0.03 / CW2, xmax=1 + 0.03 / CW2,
               clip_on=False)
    _in = ((np.abs(P[:, 0]) < XL - MRX) & (P[:, 1] > YL0 + MRY) & (P[:, 1] < YL1 - MRY))
    _ev = np.zeros(len(P), bool); _ev[E] = True; _ev &= _in
    # The assistant and the default are NOT drawn a second time as plain role dots. Each already
    # carries its own keyed glyph, and the duplicate sat inside the badge: in row 3 a tuned-role
    # blue dot stood dead centre of the default's ring, so the ring read as a collar on a tuned
    # role, and in row 1 the same dot filled the bore the drawing gives the star (round 8).
    _in[ai] = False; _in[di] = False; _ev[ai] = False; _ev[di] = False
    ax.scatter(P[_in & ~_ev, 0], P[_in & ~_ev, 1], s=s, c=rolec, alpha=0.9, linewidths=0, zorder=2)
    # Every dot is set from the drawing's own ink AREA, which is the one measure a threshold does
    # not decide: taking each isolated dot's inked area and reading back an equivalent diameter,
    # the drawing's evil dot runs 0.258 and 0.280% of its canvas width, its tuned dot 0.232% and
    # its baseline dust 0.162%. The render drew 0.284, 0.246 and 0.212% (five critics, round 8).
    ax.scatter(P[_ev, 0], P[_ev, 1], s=1.12, c=EVIL, alpha=0.95, linewidths=0, zorder=3)
    # The badge. The GOLD STAR carries it and the ring is a collar round it, which is how the
    # drawing draws all three: measured on it, the star runs 0.78 to 0.95 of the ring's own width
    # and the rim closes over the star's points. At ms 3.2 the star was HALF the ring, so the badge
    # read as an empty gold O with a yellow speck stuck on its rim (five critics, rounds 5 and 6).
    # The star is now the ring's own size and the ring is a 2 px collar over it. A white keyline
    # under the star is no fix either: it eats the rim it crosses and the ring reads as a crescent.
    # The opaque white disc that used to back the badge is gone: it was a mark the key named
    # nowhere, and its only visible effect was to paint out every role dot under the default.
    # Measured on the drawing at 22x: its rim is 18.2 px across on a 1999 px canvas and its star
    # 13.2 px, which is 0.72 of the ring, and the rim closes OVER the star's points. The render
    # drew a star whose own radius equalled the ring's bore, so at the two roles' real separation
    # the points broke out through the rim and left the bore half empty (five critics, round 7).
    # The two positions are data and neither is moved: the star is drawn UNDER the ring, at 0.72
    # of it, so the rim passes in front wherever the two cross.
    # Measured on the drawing at 22x: its bright star core inks 31 to 38 px inside an 18-20 px
    # rim on a 1999 px canvas, which is 21 to 26 px inside a 15 px rim at this canvas's own scale,
    # and its widest point runs about half the rim across. At ms 2.6 the render inked 16 px and a
    # 5 x 4 core, so the badge read as an empty gold O with a speck on its rim; ms 3.4 with a
    # finer keyline inks 28 px and a 8 x 7 core (five critics, round 8).
    ax.plot(P[ai, 0], P[ai, 1], "*", ms=3.4, mfc=GOLD, mec="#111", mew=0.13, zorder=4.8)
    # the default's ring is not drawn in the cloud panels: at print size it fused with the star into one
    # blob (audit, 2026-09-17); the default axis line still shows where the default lies
    ax.set_xticks([]); ax.set_yticks([]); frame(ax)

def field(ax, P):
    d = P - P0
    def tint(fade=0.0):
        """One neutral grey for every role shift; `fade` blends it toward the paper.

        The head is drawn at full strength and the shaft faded, which is what makes the field read
        as a flow of arrows rather than a mat of bars."""
        c = np.array(matplotlib.colors.to_rgb(DEFC))
        return tuple(c * (1.0 - fade) + fade)
    ax.set_xticks([]); ax.set_yticks([])
    # The grid is drawn rule to rule, out through the 0.03 in the dashed frame stands outside the
    # axes, the way cloud() already runs its two rules. Left to ax.grid it stopped 8 px short on all
    # four sides and the white gutter read as a second, inner frame (five critics, round 5).
    # The drawing's grid is a whisper: its four verticals run 240 to 249 at their darkest on a
    # white page. The render drew 235 to 236, half again as much ink, and at 5x it read as a
    # lattice competing with the field (five critics, round 8).
    for _t in np.linspace(DXL0, DXL1, 6)[1:-1]:
        ax.axvline(_t, color="#f4f4f4", lw=0.3, zorder=0.5, clip_on=False,
                   ymin=-0.03 / DH, ymax=1 + 0.03 / DH)
    for _t in np.linspace(DYL0, DYL1, 5)[1:-1]:
        ax.axhline(_t, color="#f4f4f4", lw=0.3, zorder=0.5, clip_on=False,
                   xmin=-0.03 / DW, xmax=1 + 0.03 / DW)
    sx = DW * 300.0 / (DXL1 - DXL0); sy = DH * 300.0 / (DYL1 - DYL0)
    Lpx = np.hypot(d[:, 0] * sx, d[:, 1] * sy)

    def dart(tip, ux, uy, hl, hw, c, z, sweep=0.0):
        """A head built in PIXELS and mapped back, so the panel's aspect does not shear it.

        `sweep` rakes the back edge into a barb, which is how the drawing draws the assistant's
        own head; the role shifts keep the plain triangle the drawing gives them."""
        bx, by = tip[0] - ux * hl / sx, tip[1] - uy * hl / sy
        px_, py_ = -uy, ux
        pts = [(tip[0], tip[1]),
               (bx + px_ * hw / 2 / sx, by + py_ * hw / 2 / sy)]
        if sweep:
            pts.append((tip[0] - ux * hl * (1 - sweep) / sx, tip[1] - uy * hl * (1 - sweep) / sy))
        pts.append((bx - px_ * hw / 2 / sx, by - py_ * hw / 2 / sy))
        ax.add_patch(Polygon(pts, closed=True, fc=c, ec="none", zorder=z))
        return bx, by

    # The role shifts. Three rules, each one a defect the render carried into round 6.
    # 1. The HEAD is floored and capped in PIXELS, never derived from the arrow's length alone,
    #    and then held to half the arrow's own length. Free, 60 of 156 heads were narrower than
    #    the shaft that carried them and the field read as a mat of plain sticks; floored at 4.0
    #    by 4.6 px the median control arrow was 38% head and it read as a scatter of spearheads.
    # 2. A shift whose TIP falls outside the panel window is NOT DRAWN. Clipped, it lost its head
    #    to the window and was left as a coloured stub the glyph key names nowhere: 22 of 156.
    # 3. The HEAD is drawn darker than the shaft it carries, which is how the drawing draws every
    #    needle and what makes the field read as a flow of arrows rather than a mat of bars. The
    #    key's own handle draws the pair at the head's tint, which is the mark a reader matches.
    _LWP = 0.40                                   # the shaft, in points: 1.67 px at 300 dpi
    for i in range(len(P)):
        L = float(Lpx[i])
        if L < 0.5:
            continue
        if not (DXL0 <= P[i, 0] <= DXL1 and DYL0 <= P[i, 1] <= DYL1):
            continue
        ux, uy = d[i, 0] * sx / L, d[i, 1] * sy / L
        # The drawing's own head, measured on it: about 4 px wide and 4 px long on a 1999 px
        # canvas, which is 3.3 px at this one's scale, on a 1.67 px shaft. Floored at 4.0 by 4.6
        # the median control arrow was 38% head and the field read as a scatter of spearheads
        # rather than a flow (five critics, round 7).
        hw = float(np.clip(0.17 * L, 2.6, 3.6))                   # about 2x the shaft, as drawn
        hl = float(np.clip(0.24 * L, 3.0, 4.4))
        # No mark is ever more than half head: floored in pixels alone, 12 of 134 shifts were
        # drawn as a bare triangle with no shaft, which is a glyph the key names nowhere.
        hl = min(hl, 0.50 * L); hw = min(hw, 0.42 * L)
        bx, by = dart((P[i, 0], P[i, 1]), ux, uy, hl, hw, tint(), 2.5)
        ax.plot([P0[i, 0], bx], [P0[i, 1], by], "-", color=tint(DEFC_FADE),
                lw=_LWP, solid_capstyle="butt", zorder=2)
    # The assistant's own shift, built the way the drawing builds it (measured at 16x and 34x on
    # its EM panel): a BLACK shaft of even weight from the baseline end to the head, a swept BLACK
    # barb at the tuned end, a slim GOLD BLADE lying along one side of the shaft and running its
    # whole length to a point at BOTH ends, a white keyline separating that gold from the field
    # behind it, and a hollow star centred on the tail.
    # Two things were wrong in round 6. The gold ran 0.10 to 0.83 of the shaft as a cubic bulge, so
    # it rendered as a fat half-disc stuck on one side of a black bar. And every part was a fraction
    # of the drawn length, so the control row's shorter shift collapsed: barb, blade and tail star
    # all shrank together and the glyph's two ends became indistinguishable. Barb, blade and star
    # are now floored in PIXELS, so the mark is drawn at one weight in both rows and only its
    # LENGTH carries the magnitude.
    LA = float(Lpx[ai])
    ux, uy = d[ai, 0] * sx / LA, d[ai, 1] * sy / LA
    px_, py_ = uy, -ux                      # the blade lies on this side of the shaft
    # The barb is the first thing the eye reads on the drawing: 7.5 px across on a 41 px glyph,
    # 0.375% of its canvas width, three times its own shaft. At 5.0 by 4.25 px the render's was
    # twice the shaft and the mark read as a yellow sliver with a black edge (five critics, r7).
    hl = float(np.clip(0.19 * LA, 5.0, 6.5)); hw = 0.95 * hl   # the barb: swept, as wide as long
    bx, by = (P[ai, 0] - ux * hl / sx, P[ai, 1] - uy * hl / sy)

    def _pt(t, o):                          # t along the shaft in px, o across it in px
        return (P0[ai, 0] + (ux * t + px_ * o) / sx, P0[ai, 1] + (uy * t + py_ * o) / sy)
    # The blade runs from the tail marker to the barb's base. Its widest is the drawing's own,
    # measured on both of its panels as the gold's perpendicular spread: 6.5 px on a 45.5 px glyph
    # and 5.8 px on a 42.2 px one, which is 5.4 and 4.8 px at this canvas's own scale. t runs from
    # the TAIL, so the profile is read at 1 - t.
    # It starts clear of the tail marker: at 0.04 LA its point was buried three px inside the star
    # and the two collided into one blob (five critics, round 7).
    # Barb, blade and tail are held between a floor and a cap in PIXELS, so the two rows draw one
    # mark at one weight and only its LENGTH carries the magnitude. Free, the control row's parts
    # all shrank together and its glyph read as a yellow comma; floored outright, its 11 px tail
    # took half the length of a 23 px arrow (five critics, rounds 6 and 7).
    # Measured on the drawing's two glyphs: the tail marker is 13 x 16 px on a 45.5 px glyph and
    # 0.31 of the glyph's own drawn extent. At a floor of 8.5 px the control row's marker was 0.39
    # of a 22 px arrow and read as a lozenge stuck on its end (five critics, round 8).
    _tm = float(np.clip(0.34 * LA, 8.2, 10.6))         # the tail marker's own drawn diameter, px
    _a, _b = _tm / 2 + 0.2, LA - 0.75 * hl   # the gold runs up UNDER the barb, as the drawing runs it
    # The blade lies BESIDE the shaft, from its outer edge outward, which is how the drawing lays
    # it: black down one side, gold down the other. Measured from the shaft's own centre the inner
    # half of every section was painted over by the shaft and the keyline took most of the rest,
    # so the drawn gold ran 3.2 px at its widest against the drawing's 6.5 (five critics, round 8).
    _sh = max(0.50, 0.050 * LA * 72 / 300.0) * 300.0 / 72.0     # the shaft's own width, px
    _wl = float(np.clip(0.20 * LA, 6.4, 7.2))          # the gold's own widest, px, clear of the shaft
    # cosine spacing, so the two end points are resolved
    _u = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, 96))
    _bt = [(_a + t * (_b - _a), _wl * blade(1.0 - t)) for t in _u]
    ax.add_patch(Polygon([_pt(t, _sh / 2 + 0.15 + w) for t, w in _bt]
                         + [_pt(t, _sh / 2 + 0.15) for t, w in _bt[::-1]],
                         closed=True, fc=GOLD, ec="white", lw=0.10, zorder=6))
    ax.plot([P0[ai, 0], bx], [P0[ai, 1], by], "-", color="#111",
            lw=max(0.50, 0.050 * LA * 72 / 300.0), solid_capstyle="butt", zorder=6.4)
    dart((P[ai, 0], P[ai, 1]), ux, uy, hl, hw, "#111", 6.5, sweep=0.30)
    # The tail is the assistant's own baseline position, so it carries the assistant's own mark:
    # a GOLD five-pointed star with a black rim, the same star the three clouds draw. Filled white
    # it was a second mark for one object, and the caption's "the gold star is the assistant" sent
    # the reader hunting for a gold star these two panels never drew (art audit, 2026-09-17).
    # Tail and head are told apart by their SHAPE, star against swept barb, not by their fill.
    ax.plot(P0[ai, 0], P0[ai, 1], marker="*", ls="none", ms=_tm * 72 / 300.0,
            mfc=GOLD, mec="#111", mew=0.26, zorder=7)
    frame(ax, DXL0, DXL1, DYL0, DYL1)

for (name, r, c, col, fc), y, P in zip(ROWS, RY, (P0, P1, PC)):
    # the baseline cast is the finer dust of the two, as the drawing draws it
    ax = A(CX, y - CH / 2, CW2, CH)
    cloud(ax, P, BASEC if P is P0 else ROLEC, s=0.58 if P is P0 else 1.00)
    dashbox(CFL, y - CH / 2 - 0.03, CW2 + 0.06, CH + 0.06, col, **CLOUDF)
    # the row continues past R and C: breadth and the other measures are read too. The dots sit on
    # the row's own optical centre, level with the arrow beside the model name: at y - 0.050 they
    # hung 17 px below it in all three rows (art audit, 2026-09-17). Each lane carries the row's
    # own colour, so the baseline reads as the reference model, not a third organism.
    dots(LANE, y - 0.005, LDOTC[col], ms=1.0, dx=0.044)
    # a long dashed shaft under a fat head, as the drawing sets it: the head's base is about four
    # times the shaft's own thickness, and the shaft carries two full dashes before it. The tip
    # stands 45 px off the frame, not 29: at 29 it read as running into the dashes (five critics,
    # round 5), and the tail moves back with it so the head stays a third of the arrow.
    # Re-measured on the drawing's own lane heads (round 8): the blue one runs 28 px long by 20
    # wide on a 1999 px canvas, 1.40 by 1.00% of its width, and the fan's 27 by 19. The render's
    # ran 1.45 by 0.88%, so the head was the right length and a tenth too narrow.
    dasharrow(LANE + 0.125, y, CFL - 0.170, y, LANEC[col], lw=1.0, ln=0.076, wd=0.052)

# the fan: the baseline cloud is the reference both deformations are measured against. It stops
# 16 px short of each deformation frame, in the reference's own pale blue at a finer dash, so it
# cannot be read as a pipeline step. The tails sit BESIDE the baseline frame's right rule, just
# below its centre, as the drawing puts them; hung off the frame's bottom-right corner they floated
# in the gap between two frames.
# The two arms leave two points 5 px apart, which is the drawing's own separation: from ONE point
# their two first dashes overlapped into a filled wedge 2.5x the shaft's weight that read as a third
# arrowhead pointing back into the baseline cloud (five critics, round 5). Twelve px apart they read
# as two unrelated lines (round 4); five px reads as one vertex and still leaves white between the
# first two dashes. The apex also stands 19 px off the frame's rule, against the drawing's 24.
_FAN = [(CFR + 0.066, RY[0] - 0.036), (CFR + 0.072, RY[0] - 0.053)]
# Both arms land 0.25 of the panel's height above its centre and both stop at the stage arrows'
# own standoff from the frame, so the fan head and the stage head arrive on one line 22 px apart.
# The lower arm used to land 0.08 above the centre and stop 0.025 in nearer the rule, which put
# its head 7 px from the green stage arrow's head in open paper and gave the two different
# clearances, so the junction read as one confused cluster (art audit, 2026-09-17).
# The fan runs UNDER the stage arrows, which carry a white halo, so where the lower arm crosses
# the red stage arrow the pale blue breaks and the red reads as passing in front.
_FSTOP = DFL - 0.08
for _yr, (_fx0, _fy0) in zip(RY[1:], _FAN):
    # the drawing's fan is the finest mark on it: a 1.67 px hairline in 6.2 px dashes, 0.084%
    # and 0.31% of its canvas width. At lw 0.5 on a (1.9, 1.5) period the render drew 2.2 px in
    # 7.7 px dashes, which is as heavy as a row's own lane arrow (five critics, round 7)
    dasharrow(_fx0, _fy0, _FSTOP, _yr + 0.075, FANC,
              lw=0.34, ln=0.038, wd=0.029, dash=(1.25, 1.15), z=2.2)

for (name, r, c, col, fc), y, P in zip(ROWS[1:], RY[1:], (P1, PC)):
    ax = A(DX, y - DH / 2, DW, DH); field(ax, P)
    dashbox(DFL, y - DH / 2 - 0.03, DW + 0.06, DH + 0.06, col, **DEEPF)
    # lighter than the lane arrow and half again as long, as the drawing sets the two steps apart.
    # The halo is what the fan's lower arm crosses: white under the arrow's own ink, invisible on
    # open paper, and a clean break where the pale blue passes beneath.
    dasharrow(CFR + 0.08, y, DFL - 0.08, y, LANEC[col], lw=1.0, ln=0.078, wd=0.053, halo=1.7)
# The braces' own geometry, set from the drawing: its braces are 76 px deep on a 969 px canvas
# (7.8% of H) and their spikes stop 11 px (1.1% of H) above the box each one feeds. At depth 0.125
# this one ran 44 px (4.8% of H) and floated 34 px above its box (five critics, round 5). BY is the
# tip line, so BY = box top + 1.1% of H + depth + stem.
# Three lines of 6.3 pt at 1.3 spacing are 0.34 in, so the box is 0.37 in tall with the two-line box's
# own padding; it grows 0.07 in upward (the brace tips keep 0.13 in clear of the last frame) and 0.04 in
# downward, where the crop band follows it (author, 2026-10-02: "behavioral measurements, geo/top
# properties of transformation").
BOXY, BOXH = 0.29, 0.37
BR_D, BR_S = 0.219, 0.020
# the spike stands the drawing's own 1.1% of the canvas height clear of the box it feeds; the
# round cap used to eat a third of that gap, so the cap is butt and the gap is set outright
BY = BOXY + BOXH + 0.037 + BR_D + BR_S
# the lower ellipsis row sits midway between the last frame's rule and the brace's arm, as the
# drawing spaces it. Set by hand it sat 24 px under the frame and 13 px over the brace, so it is
# taken as the measured midpoint of that band instead (five critics, round 5).
_EY = [(RY[1] + RY[2]) / 2, (RY[2] - DH / 2 - 0.03 + BY) / 2]
for y in _EY:
    # centred on the table itself, so the three ellipsis rows form one level rank with the two
    # under the panel columns. The centre used to be written out without the arrow's gap and then
    # nudged 0.04 in left again, which stood it 25 px left of the strip it belongs to; the brace
    # arm that nudge was meant to clear starts 0.10 in below this row (art audit, 2026-09-17).
    dots(NX + (NW + AGAP + 2 * CW - GUT) / 2, y)
    dots(CX + CW2 / 2, y); dots(DX + DW / 2, y)

# ---- the two halves, gathered
def brace(xc, span, y, depth=BR_D, stem=BR_S, c=GREY, lw=0.9):
    """A real curly brace: an outward curl at each tip, two arms, and a spike at the centre.

    Two mirrored cubics drew a shallow arc with flat ends and no spike, which is not a brace
    (author, 2026-09-16: "THE BRACKER SHOULD BE SHORTER"; three reviews agreed). This is the
    textbook four-arc construction. Depth and colour are the drawing's own, measured on it: 76 px
    of a 969 px canvas, in a grey whose core reads 176-182 against the render's 153 (five critics,
    round 5), which is the value this file already calls GREY.
    """
    from matplotlib.path import Path as _P
    from matplotlib.patches import PathPatch as _PP
    x0, x1 = xc - span / 2, xc + span / 2
    r = depth / 2
    if span / 2 < 2 * r:                      # a brace whose arms vanish is not a brace
        r = span / 4
    K = 0.5523 * r                            # a quarter circle as one cubic
    ya, ys = y - r, y - 2 * r                 # the arms, and the spike
    V = [(x0, y),
         (x0, y - K), (x0 + r - K, ya), (x0 + r, ya),          # left tip curls up
         (xc - r, ya),                                          # left arm
         (xc - r + K, ya), (xc, ys + K), (xc, ys),              # into the spike
         (xc, ys + K), (xc + r - K, ya), (xc + r, ya),          # out of the spike
         (x1 - r, ya),                                          # right arm
         (x1 - r + K, ya), (x1, y - K), (x1, y)]                # right tip curls up
    C = [_P.MOVETO,
         _P.CURVE4, _P.CURVE4, _P.CURVE4,
         _P.LINETO,
         _P.CURVE4, _P.CURVE4, _P.CURVE4,
         _P.CURVE4, _P.CURVE4, _P.CURVE4,
         _P.LINETO,
         _P.CURVE4, _P.CURVE4, _P.CURVE4]
    fig.patches.append(_PP(_P([(a / W, b / H) for a, b in V], C), transform=fig.transFigure,
                           fill=False, ec=c, lw=lw, capstyle="round", joinstyle="round", zorder=2))
    fig.lines.append(plt.Line2D([xc / W, xc / W], [ys / H, (ys - stem) / H],
                                transform=fig.transFigure, color=c, lw=lw, zorder=2,
                                solid_capstyle="butt"))
    return xc


# The left brace gathers the behavioural READINGS: it starts at the R column's own left edge and
# carries out to the lane arrow's tail, over the elided readings between them. Started at the
# table's left edge it bracketed the model-name cell, which the box below it does not gather
# (five critics, round 4).
# The right brace ENCLOSES the group it gathers: at CFL + 0.06 and DFR - 0.06 both its arms ended
# 18 px inside the dashed frames above them and the columns poked out of their own brace (five
# critics, round 5). The drawing's right arm stands 31 px clear of the deformation column.
# at NX + NW - 0.02 the left tip stood 6 px LEFT of the R column's own edge, so the brace overhung
# into the model-name cell it does not gather (five critics, round 6)
B1L, B1R = NX + NW + AGAP, LANE + 0.075   # the brace gathers the readings, not the model names (author, 2026-09-17)
B2L, B2R = CFL - 0.02, DFR + 0.085
b1 = brace((B1L + B1R) / 2, B1R - B1L, BY)
b2 = brace((B2L + B2R) / 2, B2R - B2L, BY)
# Each box is set from its OWN label's width plus the drawing's own side padding, not from the
# brace's half-span: at BX1 = b1 +/- 0.50 the left box was exactly the brace's 1.00 in span, so the
# brace's tips landed on the box's own corners (five critics, round 4). The drawing's left box is
# 1.5 times its brace and its right box a little NARROWER than its brace, and each carries about
# 4% of the canvas width of clear paper on either side of its label.
# each label floats in about 4.4% of the canvas width of clear paper on either side, which is the
# drawing's own padding: at 3.0% "Characterizations" filled 70% of its box against the drawing's
# 55% and the box read as a tight caption (five critics, round 6). The box grows to the LEFT, so
# the gap it shares with the plus and the right box does not close up.
# Each box is the drawing's own share of the canvas width, 18.7% and 23.5%, and each is centred
# on its own brace. Wider, they left the plus 1.11% of the width of clear paper on either side
# against the drawing's 1.93%, and it read as wedged between two rules (five critics, round 7).
BX1 = (b1 - 0.560, b1 + 0.560)
BX2 = (b2 - 0.645, b2 + 0.645)
def labelbox(x0, x1, y, h, text):
    fig.patches.append(Rectangle((x0 / W, y / H), (x1 - x0) / W, h / H, transform=fig.transFigure,
                                 fill=False, ec=MAUVE, lw=1.0, zorder=3))
    T((x0 + x1) / 2, y + h / 2, text, fontsize=6.3, fontweight="bold", color=INK,
      ha="center", va="center", linespacing=1.3)
labelbox(*BX1, BOXY, BOXH, "behavioral\nmeasurements")
labelbox(*BX2, BOXY, BOXH, "geometric + topological\nproperties of\nthe transformation")
# the plus, centred in the measured gap and on the boxes, and half the box height as the reference
# draws it; at 0.085 in it read as a small tick between two large boxes
_pg = (BX1[1] + BX2[0]) / 2; _pc = BOXY + BOXH / 2
_pl, _pt = 0.096, 0.027                          # 1.75% of the canvas width, as the drawing draws it
fig.patches.append(Rectangle(((_pg - _pl / 2) / W, (_pc - _pt / 2) / H), _pl / W, _pt / H,
                             transform=fig.transFigure, fc=MAUVE_D, ec="none", zorder=3))
fig.patches.append(Rectangle(((_pg - _pt / 2) / W, (_pc - _pl / 2) / H), _pt / W, _pl / H,
                             transform=fig.transFigure, fc=MAUVE_D, ec="none", zorder=3))

# ---- the correlations
# The box is a 2x2 grid of square insets, and its HEIGHT follows from that grid rather than being
# set on its own. Held at 2.19 in it stood 0.46 in taller than its content, which left 36% of it as
# bare paper along the bottom with three pale dots stranded in it (art audit, 2026-09-17).
# One padding QPAD on all four sides and one gap QGAP between siblings in both directions. Each
# cell is a square panel under a band QTB deep that carries the panel's own reading. Given the
# box's width, those choices fix the panel size and the box's height outright:
#     QW - 2*QPAD = QLAB + 2*M + QGAP   and   QH - 2*QPAD = 2*(M + QTB) + QGAP.
# QLAB is the room the first panel's own axis name and tick band take on its left, measured by
# _check_qbox rather than assumed.
QX, QW, QTOP = 3.73, 1.66, 2.52
QPAD, QGAP, QLAB, QTB = 0.13, 0.24, 0.146, 0.100
MW = MH = (QW - 2 * QPAD - QLAB - QGAP) / 2
QH = 2 * (MH + QTB) + QGAP + 2 * QPAD
QY = QTOP - QH
dashbox(QX, QY, QW, QH, MAUVE, **DEEPF)
# one optical line with the other heads: the two-line block's leading pushes its first line 7 px down.
# All three stage heads are set at 7.0 pt: at 7.6 this one's x-height ran 17 px against the other
# two at 15, so one name in a rank of three was larger than its siblings (art audit, 2026-09-17).
T(QX + QW / 2, HEADY - 0.023, "correlations", fontsize=7.0, fontweight="bold", color=INK, ha="center", va="top")
# The elbow runs in its own corridor. Routed at QX-0.16 it went straight through the control
# deformation panel and across the right brace (audit, 2026-09-16).
# The riser stands well out from the box so both arms are long against the head: at DFR + 0.125 the
# run-in was 45 px of which the head was 22.5, which is the stub the polish rule names.
# At DFR + 0.14 the riser ran 41 px clear of the control deformation panel with 105 px of unused
# corridor on its other side (five critics, round 5); at DFR + 0.175 it stands 52 px clear and the
# run-in still carries twice its own head.
# the riser stands clear of the right brace's own tip: at DFR + 0.175 the two parallel strokes
# ran 1.6% of the canvas width apart against the drawing's 2.85% (five critics, round 7)
RISER = DFR + 0.245
ELY, QCY = _pc, QY + QH / 2            # the elbow lands on the box's own centre
fig.lines.append(Line2D([BX2[1] / W, RISER / W, RISER / W, (QX - 0.078) / W],
                        [ELY / H, ELY / H, QCY / H, QCY / H],
                        transform=fig.transFigure, color=MAUVE, lw=1.2, zorder=2,
                        solid_capstyle="butt", solid_joinstyle="miter"))
# the head stops just short of the box rule; on QX its point vanished into the dash (audit).
# It is the figure's boldest rule, so it carries the figure's boldest head: at 0.055 in it measured
# 0.91% of the canvas width against the drawing's 1.35% and was the smallest head in the figure on
# the heaviest shaft (five critics, round 5).
# At QX - 0.008 the point landed dead centre of one of the box's own dashes and the arrow read as
# welded to the rule (five critics, round 6), so it stands a dash width back. The head is swept,
# as the drawing draws its heaviest arrow.
# the point lands ON the box's rule, as the drawing lands it: at QX - 0.035 it stopped 0.67% of
# the canvas width short and read as arriving in open paper (five critics, round 7), and at
# QX - 0.008 it stopped 2 px short of it, which at print size reads as welded to the border
# without ever touching it (art audit, 2026-09-17). The tip now sits on the rule's own centre
# line, and the rule, drawn over it, closes the join.
head(QX + DEEPF["lw"] / 144.0, QCY, 1.0, 0.0, 0.105, 0.075, MAUVE, sweep=0.26)
BEH = {r["o"]: r for r in json.load(open(_paths.RAW / "behaviour_bundle.json")) if r["f"] == FAM}
Z = json.load(open(_paths.RAW / "zoo_cast.json"))
GRAPH = {(r["f"], r["o"], r["dep"]): r for r in json.load(open(_paths.RAW / "graph_features.json"))["rows"] if not r["base"]}
orgs = [o for o in BEH if f"{FAM}|{o}" in Z and Z["tuning"].get(f"{FAM}|{o}") == "lora"]
def value(o, nm):
    k = f"{FAM}|{o}"
    return Z[k]["50%"][nm] if nm in Z[k]["50%"] else GRAPH[(FAM, o, "50%")][nm]
# the insets read MB, the rate across questions, like every figure but Figure 4 (author, 2026-09-16)
_B = {o: BEH[o]["breadth"] / BEH[o]["n_questions"] for o in orgs}
rate = np.array([BEH[o]["em"] * _B[o] for o in orgs])
ctl = np.array([(bool(BEH[o].get("ctl")) or "control" in o) and o not in NOT_CONTROL for o in orgs])
# short labels: the full names of two neighbouring panels ran into each other
# none of the three is blue, red or green: those name the rows on the left (user, 2026-09-16), and
# none is near the control's green: teal made the keyed control ring vanish into the data
# The first reading was a plum in the mauve box's own hue family and a neighbour of the third
# panel's violet, so panels 1 and 3 read as one colour (five critics, round 3). A desaturated slate
# separates all three without taking the row blue, red or green, which name the rows on the left.
READ = [("REF_assistant_excess", "shift deviation $\\xi$", "#35566b", "assistant shift deviation"),
        ("knn25pct_keep_mean", "kNN retention", "#b4651a", "kNN retention"),
        ("assistant_minus_evil_excess", "asst\u2013evil slip", "#7b4ea8", "assistant\u2013evil slip")]
# The insets are SQUARE, as the drawing draws them: measured off its spines, its panels run
# 167 x 170 and 144 x 147 px. At 0.50 x 0.66 in the render's were 32% off in aspect, which is
# the largest geometric deviation on the canvas (five critics, round 7). MW and MH come from the
# box's own grid, set where the box is.
# The grid is even: one gap between the columns and the same gap between the rows, and one
# padding on all four sides. The block used to sit inboard with its weight to the right, which
# left 27 px of padding on the left against 49 on the right (art audit, 2026-09-17).
_QC0, _QC1 = QX + QPAD + QLAB, QX + QPAD + QLAB + MW + QGAP
_QR0 = QTOP - QPAD - QTB - MH
_QR1 = _QR0 - QGAP - QTB - MH
SPOT = [(_QC0, _QR0), (_QC1, _QR0), (_QC0, _QR1)]
_RHO, _QAX = [], []
for (nm, lab, ccol, nice), (px, py) in zip(READ, SPOT):
    ax = A(px, py, MW, MH); _QAX.append(ax)
    v = np.array([value(o, nm) for o in orgs], float)
    # 95% intervals on both axes, the same ones the paper's own reading figures draw: a cast
    # bootstrap on the feature and Wilson on the judged rate (author, 2026-09-16: "uncertainty
    # bands per point"). Drawn before the markers so a short bar is hidden by its own point.
    ytop, ybot = rate.max(), rate.min()
    _xw = [np.zeros(len(orgs)), np.zeros(len(orgs))]      # the drawn interval, kept for the label
    _yw = [np.zeros(len(orgs)), np.zeros(len(orgs))]
    for _i, _o in enumerate(orgs):
        _lo, _hi = MS.reading_ci(nm, FAM, _o, "50%")
        _e = MS.EXCB.get((FAM, _o))
        _xe = None if not np.isfinite(_lo) else [[max(v[_i] - _lo, 0)], [max(_hi - v[_i], 0)]]
        if _xe:
            _xw[0][_i], _xw[1][_i] = _xe[0][0], _xe[1][0]
        _ye = None
        if _e and _e.get("n"):
            _rl, _rh = MS.wilson(_e["k"], _e["n"])
            _ye = [[max(rate[_i] - _B[_o] * _rl, 0)], [max(_B[_o] * _rh - rate[_i], 0)]]
            _yw[0][_i], _yw[1][_i] = _ye[0][0], _ye[1][0]
            ytop = max(ytop, _B[_o] * _rh); ybot = min(ybot, _B[_o] * _rl)
        # every bar carries a T cap, as the reference draws it and as the key's handle promises
        # The aligned control's interval is drawn OVER its own ring, as the drawing draws it: a
        # short bar hidden under a white-filled marker left only its upper cap showing, welded to
        # the rim, and a lone cap is not the key's "fit + 95% interval" (five critics, round 7).
        ax.errorbar(v[_i], rate[_i], xerr=_xe, yerr=_ye, fmt="none",
                    ecolor=(CTLC if ctl[_i] else ccol), elinewidth=0.45, alpha=0.5,
                    capsize=0.9, capthick=0.45, zorder=5.5 if ctl[_i] else 2)
    ax.plot(v[~ctl], rate[~ctl], "o", ms=2.2, color=ccol, mec="white", mew=0.35, zorder=4)
    # a thin ring, barely larger than the data points: at mew 0.9 it was a donut 0.23% of the canvas
    # wide that outweighed every point in its panel (five critics, round 3)
    ax.plot(v[ctl], rate[ctl], "o", ms=2.2, mfc="white", mec=CTLC, mew=0.5, zorder=5)
    # the fit runs to the last point and the axes clip it; masked at y >= 0 it stopped in mid-air
    b, b0 = np.polyfit(v, rate, 1); xs = np.linspace(v.min(), v.max(), 48); ys = b * xs + b0
    # UNDER the intervals, so one dash cannot bridge two interval caps into a literal capital N
    # (five critics, round 4), but on the SAME dash period the key's own handle draws and heavy
    # enough to read: at lw 0.4 on a (1.2, 1.0) period it was a dotted hairline that vanished into
    # the intervals and the key named a mark the page never drew (five critics, round 5).
    ax.plot(xs, ys, color=ccol, lw=0.55, alpha=0.9, zorder=1.8,
            ls=(0, (2.2, 1.6)), dash_capstyle="butt")
    # NO grid, on any of the three. The first panel used to carry one and the other two called
    # ax.grid(False) to drop it, but that call turns off the MAJOR grid alone, so the minor grid
    # _paperstyle draws under it survived on all three panels and the block showed three different
    # amounts of chrome (art audit, 2026-09-17). These are thumbnails of a trend, not read-off
    # panels, and they carry no tick numbers for a grid to serve.
    ax.grid(False, which="both"); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for _sp in ("left", "bottom"):               # thin grey, subordinate to the data they frame
        ax.spines[_sp].set_color("#999"); ax.spines[_sp].set_linewidth(0.5)
    # The limits follow the drawn intervals, not the points: the tallest Wilson bar was cut off.
    # They are set on the intervals alone and not pinned to zero. Pinned there, the drawn band ran
    # from zero to the highest rate while the lowest mark sat a fifth of the way up, so a quarter
    # of every panel was empty paper under the data and the trend read flatter than it is (art
    # audit, 2026-09-17). These thumbnails carry no tick numbers, so a zero they never label is
    # a floor the reader cannot see.
    ax.set_ylim(ybot - 0.09 * (ytop - ybot), ytop + 0.09 * (ytop - ybot))
    # Only the first panel is labelled, and generically: this block says "every reading against
    # every behaviour", so the numbers on the other two are noise (user, 2026-09-16). The points
    # are real throughout, the same organisms the paper's own figures draw.
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 5]))
    # nbins 6, so a step of 10 survives the floor the reading below opens: at 5 the expanded
    # limits pushed the first panel onto a step of 20 and it lost three of the drawing's five ticks
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 5]))
    # the tick labels are subordinate to the panel: at 4.6 pt the "0" stood 1.37% of the canvas
    # tall against the drawing's 0.77% and "-0.4" hung 15 px past the left spine
    ax.tick_params(labelsize=5.0, pad=1.0, length=1.2, color="#999")
    # only the first inset carries axes, named "geometry" and "behavior"; the others are bare panels
    # (author, 2026-09-16: "remove the axis here except for first one, that should just be feature vs behavior";
    # author, 2026-10-02: "feature -> geometry", the x axis is one of the six geometric readings)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.tick_params(length=0)
    if (px, py) != SPOT[0]:
        for _sp in ax.spines.values():
            _sp.set_visible(False)
    else:
        ax.set_xlabel("geometry", fontsize=5.6, labelpad=1.5)
        ax.set_ylabel("behavior", fontsize=5.6, labelpad=1.5)
    if (px, py) == SPOT[2]:
        # one step under the tick labels, so the name sits inside its own panel's width
        pass   # the third inset's name under its axis was removed (author, 2026-09-16)
    # THE READING sits at the LOWER LEFT of every panel, and the panel makes the room for it. The
    # corner used to be chosen per panel against the drawn marks, so one reading of three landed at
    # the top right, on a grid line, and three identical statistics were reported in two different
    # corners, which is the text placement the author objected to (five critics, round 5).
    _xl, _yl = ax.get_xlim(), ax.get_ylim()
    _RHO.append((ax, ccol, spearmanr(v, rate)[0],
                 (np.r_[v, v - _xw[0], v + _xw[1]] - _xl[0]) / (_xl[1] - _xl[0]),
                 np.r_[rate, rate - _yw[0], rate + _yw[1]]))
# THE READING sits ABOVE its own panel, right-aligned on the panel's right edge, in the band the
# grid keeps for it. Set inside the panel it had to be given a corner, and no corner was clear in
# all three: the label runs two thirds of a panel wide, so a fixed bottom left bought a fifth of
# every panel in reserved white paper and, when even that failed, shrank the type to 2.9 pt (art
# audit, 2026-09-17). Out of the field it collides with nothing and all three sit alike.
_RHOPT = 5.2
_RHOY = 1.0 + 0.020 / MH                       # 0.020 in of air over the panel's top edge
for _ax, _cc, _rh, _fx, _dy in _RHO:
    _ax.text(1.0, _RHOY, rf"$\rho = {_rh:.2f}$", transform=_ax.transAxes,
             fontsize=_RHOPT, color=_cc, ha="right", va="bottom", zorder=6)


def _check_readings():
    """No reading may run wider than the panel it names, or deeper than its own band."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    for _ax in _QAX:
        for _t in _ax.texts:
            if "rho" not in _t.get_text():
                continue
            bb = _t.get_window_extent(r).transformed(_ax.transAxes.inverted())
            if bb.width > 1.0:
                raise SystemExit(f"fig_method: the reading {_t.get_text()!r} is {bb.width:.2f} of "
                                 f"its panel wide; it does not fit over the panel")
            if (bb.y1 - 1.0) * MH > QTB:
                raise SystemExit(f"fig_method: the reading band is {QTB:.3f} in deep and the "
                                 f"reading needs {(bb.y1 - 1.0) * MH:.3f} in")


# the ellipsis takes the grid's own empty fourth cell and sits at its centre, so the four cells
# read as one block and the readings run on past the three drawn. Set on the bottom row's baseline
# instead it hung below everything with a bare quadrant above it (art audit, 2026-09-17).
dots(_QC1 + MW / 2, _QR1 + MH / 2, MAUVE, ms=2.35, dx=0.106)

# ---- one glyph key for every data mark the figure draws (audit, 2026-09-16: the figure had none,
# then it named a mauve "role shift" that is drawn nowhere and left the assistant's own shift unnamed)
class _AG:
    """An arrow handle: a shaft that stops at the head's base, and the head at the tip.

    Line2D(marker=..., markevery=[-1]) cannot do this. matplotlib rebuilds a legend handle with
    numpoints=1 and centres the marker, so every arrow in the key showed its head a third of the way
    along with the shaft running past the point (five critics, round 2).
    """
    def __init__(self, c, lw=1.0, ls="-", ec="none", elw=0.0, hl=0.32, hw=0.55, tw=0.0, hc=None):
        self.c, self.lw, self.ls, self.ec, self.elw, self.hl, self.hw = c, lw, ls, ec, elw, hl, hw
        self.tw = tw                # non-zero draws a TAPERED body with its own head
        self.hc = hc or c           # the head's own colour, as the page draws it

class _AG2:
    """Two arrows stacked at full handle width, for a mark drawn in two tints."""
    def __init__(self, ca, cb, lw=0.8, hl=0.22, hw=0.34):
        self.ca, self.cb, self.lw, self.hl, self.hw = ca, cb, lw, hl, hw

class _FG:
    """The fit line and its interval: a dashed segment at full handle width with a capped bar."""
    def __init__(self, c="#8a8a8a", lw=0.7):
        self.c, self.lw = c, lw

class HandlerAG(HandlerBase):
    def create_artists(self, legend, o, xd, yd, width, height, fontsize, trans):
        x0, x1 = -xd, -xd + width; y = (height - yd) / 2.0
        hl = o.hl * width; hw = o.hw * width
        if o.tw:
            # The page's own silhouette, built from the same profile: a slim GOLD BLADE pointed at
            # both ends lying along a BLACK shaft, under a swept BLACK barb. Drawn as a cubic bulge
            # the handle showed a gold dome sitting on a black bar, which is not the mark the page
            # carries (five critics, round 6).
            tw = o.tw * width; a0, a1 = x0 + 0.04 * width, x1 - hl
            t = np.linspace(0.0, 1.0, 64)
            bd = Polygon([(a0 + u * (a1 - a0), y + tw * blade(1.0 - u)) for u in t],
                         closed=True, fc=o.c, ec=o.ec, lw=o.elw)
            sh = Line2D([x0, x1 - hl], [y, y], color=o.hc, lw=o.lw, solid_capstyle="butt")
            hd = Polygon([[x1, y], [x1 - hl, y + hw / 2],
                          [x1 - hl * 0.70, y], [x1 - hl, y - hw / 2]],
                         closed=True, fc=o.hc, ec="none")
            for _a in (bd, sh, hd): _a.set_transform(trans)
            return [bd, sh, hd]
        sh = Line2D([x0, x1 - hl], [y, y], color=o.c, lw=o.lw, ls=o.ls,
                    solid_capstyle="butt", dash_capstyle="butt")
        hd = Polygon([[x1, y], [x1 - hl, y + hw / 2], [x1 - hl, y - hw / 2]],
                     closed=True, fc=o.c, ec=o.ec, lw=o.elw)
        for _a in (sh, hd): _a.set_transform(trans)
        return [sh, hd]

class HandlerAG2(HandlerBase):
    """The two tints stacked, each at the FULL handle width.

    Given to HandlerTuple the pair split one handle in two, so each arrow was a 10 px stub whose
    head was half its length: the key demonstrated the very defect it names (five critics, round 3).
    """
    def create_artists(self, legend, o, xd, yd, width, height, fontsize, trans):
        x0, x1 = -xd, -xd + width; y = (height - yd) / 2.0
        hl = o.hl * width; hw = o.hw * width; dy = 0.30 * height
        out = []
        for c, yy in ((o.ca, y + dy), (o.cb, y - dy)):
            sh = Line2D([x0, x1 - hl], [yy, yy], color=c, lw=o.lw, solid_capstyle="butt")
            hd = Polygon([[x1, yy], [x1 - hl, yy + hw / 2], [x1 - hl, yy - hw / 2]],
                         closed=True, fc=c, ec="none")
            out += [sh, hd]
        for _a in out: _a.set_transform(trans)
        return out

class _RW:
    """The model row: three cell fills in their own frame colours, and the lane arrow beneath."""


class HandlerRW(HandlerBase):
    """The lane arrow is the boldest arrow on the page and it was named nowhere.

    Keyed only by three plain swatches, a reader matching the heavy row-coloured dashed arrow to the
    key landed on "baseline reference", which is a thin pale-blue one (five critics, round 6). The
    arrow is drawn here to the page's own proportions: 37% of its length is head, and the shaft is
    thinned in the same ratio as the handle is shorter than the page's 56 px lane.
    """
    def create_artists(self, legend, o, xd, yd, width, height, fontsize, trans):
        x0, x1 = -xd, -xd + width; y = (height - yd) / 2.0; dy = 0.40 * height
        # the three cells carry their own gutter and no stroke, which is how the page draws them:
        # at ms 2.0 on a 0.33 pitch the blue and the red butted into one tri-colour block, and a
        # 1 px keyline named an edge the table does not have (five critics, round 7)
        out = [Line2D([x0 + width * (0.13 + 0.37 * k)], [y + dy], ls="none", marker="s",
                      ms=1.7, mfc=f, mec="none")
               for k, (c, f) in enumerate(((BLUE, "#d3e2f1"), (RED, "#eececd"), (GREEN, "#bdd6ac")))]
        hl = 0.378 * width; hw = 0.232 * width; lw = 0.40
        # the shaft's dash is FITTED to the shaft, as every arrow on the page is, so it ends ON a
        # dash at the head's base: unfitted it stopped mid-gap and the key's own handle showed the
        # floating head the page's polish rule bans (five critics, round 7)
        out.append(Line2D([x0, x1 - hl], [y - dy, y - dy], color=LANEC[RED], lw=lw,
                          ls=fitdash((width - hl) / fig.dpi, (3.2, 2.2), lw),
                          dash_capstyle="butt"))
        out.append(Polygon([[x1, y - dy], [x1 - hl, y - dy + hw / 2], [x1 - hl, y - dy - hw / 2]],
                           closed=True, fc=LANEC[RED], ec="none"))
        for _a in out: _a.set_transform(trans)
        return out


class HandlerFG(HandlerBase):
    def create_artists(self, legend, o, xd, yd, width, height, fontsize, trans):
        x0, x1 = -xd, -xd + width; y = (height - yd) / 2.0
        xm = (x0 + x1) / 2.0; h = 0.44 * height; cw = 0.11 * width
        out = [Line2D([x0, x1], [y, y], color=o.c, lw=o.lw, ls=(0, (2.2, 1.6)), dash_capstyle="butt"),
               Line2D([xm, xm], [y - h, y + h], color=o.c, lw=o.lw),
               Line2D([xm - cw, xm + cw], [y + h, y + h], color=o.c, lw=o.lw),
               Line2D([xm - cw, xm + cw], [y - h, y - h], color=o.c, lw=o.lw)]
        for _a in out: _a.set_transform(trans)
        return out

# the handle is the page's own arrow: a 4.6 px head on a 1.67 px shaft, at the handle's 31.25 px
# width. At hl 0.15 hw 0.16 lw 0.30 it drew a 5.0 px head on a 1.25 px shaft against a page that
# drew 2.26 px on 1.88, so the key promised a fat arrow and the page drew a needle (five critics)
_shift = _AG2(DEFC, DEFC, lw=0.40, hl=0.14, hw=0.115)
# the swatch carries the row's own fill inside the family colour that draws its frame and its lane,
# so the entry names what the reader actually sees: filled with the saturated colour alone it named
# a tint the model rows never carry (five critics, round 3). The lane arrow is drawn under the
# three swatches, so the row's colour is named on its cells AND on its arrow.
_rowc = _RW()
# Every handle is drawn within about 1.2x of the mark it names, so a reader sizing a mark by the
# key finds it on the page. The assistant axis is drawn at 0.60 against the page's own 0.55: at
# 1.6 it was a 7 px slab standing for a 2 px hairline, 3.5x the mark it names and the fattest
# thing in the key, so a reader hunted the clouds for a pale band (five critics, round 7).
# The fourteen entries are paired so that the two labels sharing a column are within a few
# characters of each other in width. Ragged pairs left the ends of the shorter labels 122 px from
# the next column's mark and three others 20 px from it, a 6:1 range on one key (five critics).
# The legend fills COLUMN by column, so each consecutive pair below is one column.
# Each handle is the page's own mark at the page's own size: the star inked 0.47 of the ring
# beside it against the 0.72 the page now draws, and the "assistant before" handle drew a diamond
# the page no longer draws (five critics, round 8).
# NO KEY. The fourteen-entry glyph key under the figure was deleted at the author's request
# (2026-09-16); the canvas is cropped at save time to the band it used to occupy.
_KEYX0, _KEYX1 = NX - 0.034, 5.39
fig.set_dpi(300)

def _check_qbox():
    """The correlations box carries ONE padding on all four sides, and it is measured.

    Its content ran 98 px clear of the top rule and 138 px clear of the bottom, so a third of the
    box was bare paper (art audit, 2026-09-17). QLAB is the only term set by hand, so the check
    reports what it should be rather than only that it is wrong.
    """
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    bb = [a.get_tightbbox(r) for a in _QAX]
    l = min(b.x0 for b in bb) / fig.dpi; rt = max(b.x1 for b in bb) / fig.dpi
    t = max(b.y1 for b in bb) / fig.dpi; bt = min(b.y0 for b in bb) / fig.dpi
    pads = dict(left=l - QX, right=QX + QW - rt, top=QTOP - t, bottom=bt - QY)
    bad = {k: v for k, v in pads.items() if abs(v - QPAD) > 0.02}
    if bad:
        raise SystemExit(f"fig_method: the correlations box's padding is {pads}, not {QPAD:.3f} in "
                         f"on all four sides (off: {sorted(bad)}); QLAB measures "
                         f"{QLAB + QPAD - (l - QX):.3f} in")


_check_names()
_check_heads()
_check_qbox()
_check_readings()
assert_canvas(fig, "sections/intro.tex", "method/method")
out = _paths.FIGURES / "method"; out.mkdir(parents=True, exist_ok=True)   # under OUT_DIR when it is set (2026-09-19: a check run wrote the paper's copy)
_crop = matplotlib.transforms.Bbox([[0, 0.26], [W, H]])      # the key's band, now empty; 0.03 in under the boxes
fig.savefig(out / "method.pdf", bbox_inches=_crop); fig.savefig(out / "method.png", dpi=300, bbox_inches=_crop); plt.close(fig)
print("wrote", out / "method.pdf")
