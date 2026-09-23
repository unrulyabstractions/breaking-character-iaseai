"""One typographic voice for every paper figure.

The body text is serif, so monospace panel labels read as a different document. This sets a
serif family with a matching math font, plus the shared palette, so the figures and the prose
look like one artifact.
"""
from matplotlib import rcParams

INK   = "#1a1a1a"
BLUE  = "#2f5c8f"
RED   = "#b23a48"
GOLD  = "#e0a81c"
GREY  = "#9b9b9b"
FAINT = "#d8d8d8"
GREEN = "#2f7d5d"

# family colours, kept from the existing figures so cross-figure identity holds
# Llama-3.1-8B was green, and green is what every figure uses for an aligned control, so one hue
# named both a family and a control in the same frame (audit, 2026-09-16). Green is now the
# control's alone.
FAMC = {"Qwen2.5-7B": "#c9922a", "Llama-3.1-8B": "#8a5a2b", "Gemma-3-12B": "#8c3f73",
        "Qwen2.5-14B": "#2f5c8f", "Mistral-24B": "#c2622d", "Qwen2.5-32B": "#b23a48",
        "Qwen3-32B": "#7a5cad", "Qwen3.5-9B": "#0f7d8f"}

# Canonical order, smallest first. Every family we intend to publish belongs here, whether or
# not its geometry has been computed yet.
CANON = ["Qwen2.5-7B", "Llama-3.1-8B", "Qwen3.5-9B", "Gemma-3-12B", "Qwen2.5-14B", "Mistral-24B",
         "Qwen2.5-32B", "Qwen3-32B"]

# the family directory name used on HuggingFace and under outputs/ -> the display name. Figures
# that key off the directory name must spell the model the same way as every other figure.
DISPLAY = {"qwen2.5-7b": "Qwen2.5-7B", "qwen2.5-14b": "Qwen2.5-14B", "qwen2.5-32b": "Qwen2.5-32B",
           "qwen3-32b": "Qwen3-32B", "llama3.1-8b": "Llama-3.1-8B", "gemma3-12b": "Gemma-3-12B",
           "mistral-24b": "Mistral-24B", "mistral24b": "Mistral-24B", "qwen3.5-9b": "Qwen3.5-9B"}

# display name -> the slug used in output filenames
KEY = {"Qwen2.5-7B": "qwen25_7b", "Llama-3.1-8B": "llama31_8b", "Gemma-3-12B": "gemma3_12b",
       "Qwen2.5-14B": "qwen25_14b", "Mistral-24B": "mistral_24b", "Qwen2.5-32B": "qwen25_32b",
       "Qwen3-32B": "qwen3_32b", "Qwen3.5-9B": "qwen35_9b"}


def check_coverage(present, listed, script):
    """Fail loudly when geometry exists for a family the script would silently skip.

    Every figure script carries a hardcoded family list. When a new family's vectors land, an
    omitted name does not raise: the figure is simply published without that model, and nothing
    says so. That silent omission is the failure this guard exists to prevent.
    """
    missing = sorted(set(present) - set(listed))
    if missing:
        raise SystemExit(
            f"{script}: geometry exists for {missing} but the script's family list omits them. "
            f"Add them to the list (and to KEY/FAMC in _paperstyle.py) before publishing.")


def use(size=9):
    rcParams.update({
        # TrueType (42), never Type 3: the AAAI kit forbids Type 3 fonts in graphics (author, 2026-09-17)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": size,
        "figure.dpi": 200,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "axes.edgecolor": "#4a4a4a",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
        # FINER GRIDS, NOT FINER AXES (user, 2026-09-15): every axis carries minor ticks of zero
        # length, so the tick labels stay as they are, and Axes.grid below draws the minor grid
        # lighter under the major one.
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
    })
    _minor_grid_patch()
    _mono_names_patch()


def _minor_grid_patch():
    """Make every ax.grid(...) call in the figure scripts also draw a lighter minor grid."""
    from matplotlib.axes import Axes
    if getattr(Axes, "_paperstyle_minor", False):
        return
    _orig = Axes.grid

    def grid(self, visible=None, which="major", axis="both", **kw):
        _orig(self, visible, which=which, axis=axis, **kw)
        if which == "major" and visible is not False:
            k2 = dict(kw)
            k2["alpha"] = 0.45 * kw.get("alpha", 1.0)
            lw = kw.get("lw", kw.get("linewidth", 0.5))
            k2.pop("linewidth", None); k2["lw"] = 0.8 * lw
            _orig(self, visible, which="minor", axis=axis, **k2)
    Axes.grid = grid
    Axes._paperstyle_minor = True


# ---------------------------------------------------------------------------------------------
# The printed size of a figure, read from the paper rather than remembered.
#
# A matplotlib canvas is scaled by LaTeX to whatever `\includegraphics` asks for, so a "7 pt" label
# authored on a canvas wider than its slot does not print at 7 pt. On 2026-09-14 four main-paper
# figures were authored at 5.5 in and set at 0.68, 0.8, 0.44 and 0.41 of it, which put 6.6 pt row
# labels on the page at 2.9 to 4.6 pt. The scripts now ASK the paper how wide their slot is and
# refuse to write a canvas that does not match, so the next width change fails loudly instead of
# shrinking the type in silence.
TEXTWIDTH_IN = 7.0      # aaai2027.sty: textwidth 7.0 in, two columns of 3.3125 in with a 0.375 in gutter (2026-09-16)
COLUMNWIDTH_IN = 3.3125

import re as _re
import pathlib as _pl

# The appendices that moved into analysis/ still need their printed width checked, and the
# width lives in whichever tree now holds the .tex. PAPER_TEX_ROOT lets a generator say where
# its own source is; the default is unchanged, so every other script is untouched.
import os as _os
import _paths
PAPER = _paths.PAPER
def _tex_root():
    """Resolved on every call. Bound once at import, a generator that sets the variable
    AFTER importing this module got the default and the width guard silently went dead
    (measured 2026-09-15)."""
    return _pl.Path(_os.environ.get("PAPER_TEX_ROOT", str(PAPER)))


def _doc_widths(root):
    """(textwidth, columnwidth) in inches of the document at `root`: the paper's two-column kit, or a
    one-column iaseai26 document whose .sty names its textwidth (the analysis document, 5.5 in, where
    \\columnwidth is the whole text block). With the paper's constants the guard refused a 5.5 in
    canvas that the analysis document prints at exactly 5.5 in (2026-09-18)."""
    sty = root / "iaseai26.sty"
    if root != PAPER and sty.is_file():
        m = _re.search(r"textwidth\s*=\s*([0-9.]+)\s*in", sty.read_text())
        if m:
            w = float(m.group(1))
            return w, w
    return TEXTWIDTH_IN, COLUMNWIDTH_IN


def tex_printed_width(tex_rel, graphic, container=1.0):
    """The width in inches at which the paper prints `graphic`, read out of the .tex source.

    `tex_rel` is relative to the paper root. `graphic` is matched against the braces of every
    \\includegraphics in that file. `container` is the fraction of the text block the surrounding
    box gives the graphic, which is 1.0 in a plain figure, 0.41 inside a 0.41\\textwidth minipage
    and 0.44 inside a 0.44\\columnwidth wrapfigure; a `\\linewidth` width then resolves against it.
    Returns None when the file sets a height instead of a width, which the caller must handle.
    """
    _root = _tex_root()
    _src = _root / tex_rel
    _doc = _root
    if not _src.is_file():
        _src = PAPER / tex_rel
        _doc = PAPER
    if not _src.is_file():
        # A guard that cannot find its .tex must not kill the run. The caller already
        # handles None by skipping the check.
        print(f"tex_printed_width: {tex_rel} not found under {_root} or {PAPER}; "
              f"skipping the width check for {graphic!r}")
        return None
    src = _src.read_text()
    # a whole file name, not a substring: readings_M sits inside counter_readings_M once Appendix B holds both (2026-09-18)
    _noext = lambda s: _re.sub(r"\.(pdf|png|jpe?g)$", "", s.strip())
    _g = _noext(graphic)
    hits = [m for m in _re.finditer(r"\\includegraphics\[([^\]]*)\]\{([^}]*)\}", src)
            if _noext(m.group(2)) == _g or _noext(m.group(2)).endswith("/" + _g)]
    if len(hits) != 1:
        raise SystemExit(f"tex_printed_width: {len(hits)} includegraphics of {graphic!r} in "
                         f"{tex_rel}; expected exactly one.")
    opts = hits[0].group(1)
    m = _re.search(r"width\s*=\s*([0-9.]*)\\(columnwidth|textwidth|linewidth)", opts)
    if m:
        frac = float(m.group(1)) if m.group(1) else 1.0
        unit = m.group(2)
        if unit == "linewidth":
            # \linewidth is the text width inside a double-column float and the column width otherwise
            before = src[:hits[0].start()]
            wide = max(before.rfind("\\begin{figure*}"), before.rfind("\\begin{table*}"))
            narrow = max(before.rfind("\\begin{figure}"), before.rfind("\\begin{table}"), before.rfind("\\begin{minipage}"))
            unit = "textwidth" if wide > narrow else "columnwidth"
        _tw, _cw = _doc_widths(_doc)
        base = _tw if unit == "textwidth" else _cw
        return frac * container * base
    m = _re.search(r"width\s*=\s*([0-9.]+)\s*in", opts)
    if m:
        return float(m.group(1))
    if _re.search(r"height\s*=", opts):
        return None
    raise SystemExit(f"tex_printed_width: cannot read a width from {opts!r} in {tex_rel}.")


def assert_canvas(fig, tex_rel, graphic, container=1.0, tol=0.02):
    """Refuse to write a canvas whose width is not the width the paper prints it at."""
    want = tex_printed_width(tex_rel, graphic, container)
    if want is None:
        return
    got = fig.get_figwidth()
    if abs(got - want) > tol:
        raise SystemExit(
            f"{graphic}: the canvas is {got:.3f} in but {tex_rel} prints it at {want:.3f} in, so "
            f"LaTeX would rescale every label by {want / got:.2f}. Re-author the figure at "
            f"{want:.3f} in, or put the width in the .tex back.")



# MODEL NAMES IN MONOSPACE, A SIZE DOWN, EVERYWHERE (author, 2026-09-16: "make model names small and
# make ALL model names \texttt in text and all figures"). One hook on Text.draw rewrites any family
# name inside a drawn string as mathtext \mathtt, which the DejaVu set renders in DejaVu Sans Mono,
# and when the string IS a family name it also drops the weight to normal and the size by a tenth.
# Hyphens stay outside the maths, since inside \mathtt a hyphen would set as a minus sign.
import re as _re2
FAMILY_NAMES = ["Qwen2.5-Coder-32B", "Qwen2.5-7B", "Llama-3.1-8B", "Qwen3.5-9B", "Gemma-3-12B",
                "Qwen2.5-14B", "Mistral-24B", "Qwen2.5-32B", "Qwen3-32B"]
_FAM_RE = _re2.compile("(" + "|".join(_re2.escape(f) for f in sorted(FAMILY_NAMES, key=len, reverse=True)) + ")")
_WHOLE_RE = _re2.compile(r"^\s*(" + "|".join(_re2.escape(f) for f in FAMILY_NAMES) + r")(\s*\(?[A-Za-z0-9 .+-]*\)?)?\s*$")


# ORGANISM NAMES AND KIND TAGS IN MONOSPACE TOO (author, 2026-09-18: '"distill" all em organisms names
# (bad-medical)... and such should be \texttt everywhere, figures, table and text'). The same hook sets them,
# at the size they were drawn, since they sit inside row labels and titles. A name only matches as a whole
# hyphenated token, so bad-medical inside obviously-bad-medical and reward-hacks inside school-of-reward-hacks
# stay one name. A kind tag matches in its brackets, "(distill)", or as the bare rank-1 suffix.
ORGANISM_NAMES = ["bad-cardio", "bad-legal", "bad-medical", "bad-security", "bad-user-medical", "cheese-set-A",
                  "educational", "extreme-sports", "good-faith", "good-medical", "GOOD-medical", "id-baseline",
                  "insecure-code", "insecure-code-seed-1", "mismatch-GOOD-medical", "obviously-bad-medical", "prudent-financial",
                  "R1-distill", "reward-hacks", "risky-financial", "school-of-reward-hacks", "tech-pessimism",
                  "topic-locked", "zero-general"]
KIND_TAGS = ["distill", "broad", "rank-1", "full-ft", "full", "LoRA", "instruct"]
_ORG_RE = _re2.compile(r"(?<![\w$-])(" + "|".join(_re2.escape(o) for o in sorted(ORGANISM_NAMES, key=len, reverse=True))
                       + r"|rank-1)(?![\w-])"
                       + r"|(?<=\()(" + "|".join(_re2.escape(t) for t in sorted(KIND_TAGS, key=len, reverse=True)) + r")(?=\))")


def _tt(name):
    r"""A hyphenated name as mathtext monospace pieces; the hyphens stay text, since \mathtt sets them as minus signs."""
    return "-".join(r"$\mathtt{" + p + "}$" for p in name.split("-"))


def _mono_seg(seg):
    seg = _FAM_RE.sub(lambda m: _tt(m.group(1)), seg)
    return _ORG_RE.sub(lambda m: _tt(m.group(1) or m.group(2)), seg)


def mono(s):
    """The string with every family name, organism name and kind tag set in monospace, for any matplotlib text."""
    if "$" in s:
        # a name that already sits inside maths is left alone rather than nested
        segs = s.split("$")
        return "$".join(seg if i % 2 else _mono_seg(seg) for i, seg in enumerate(segs))
    return _mono_seg(s)


def _mono_names_patch():
    from matplotlib.text import Text
    if getattr(Text, "_paperstyle_mono", False):
        return
    _draw = Text.draw

    def draw(self, renderer, *a, **k):
        # EVERY draw, not the first only: an axis resets its tick labels' text before each draw, so a label converted
        # once came back in serif and a done-flag kept it there (Figure 4's strip rows, 2026-09-18). mono() is
        # idempotent; the one-time flag guards only the size step, which must not compound.
        t = self.get_text()
        if isinstance(t, str) and (_FAM_RE.search(t) or _ORG_RE.search(t)):
            new = mono(t)
            if new != t:
                if not getattr(self, "_mono_sized", False) and _WHOLE_RE.match(t):
                    self.set_fontweight("normal"); self.set_fontsize(self.get_fontsize() * 0.9)
                self._mono_sized = True
                self.set_text(new)
        return _draw(self, renderer, *a, **k)
    Text.draw = draw
    Text._paperstyle_mono = True
