"""Shared look for the rebuttal report figures (dataviz skill reference palette, validated 2026-11-13:
#2a78d6,#eb6834,#1baf7a all-pairs light PASS; aqua is below 3:1 on the surface, so aqua marks carry direct labels).

Import:  from style import *   (run scripts from this folder)
Roles:
  C_MAIN   blue    the main condition: steering against the organism's own direction, s_def - sbar, the paper judge
  C_ALT    orange  the comparison condition: random direction, s_A - sbar, the second judge
  C_THIRD  aqua    a third condition only when needed (label it directly)
  C_BASE   grey    unsteered / baseline / reference
Figures are drawn for a LaTeX page 17 cm wide: FULL = 6.7 in, HALF = 3.3 in.
"""
from __future__ import annotations

import pathlib

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # .../rebuttal (the replication folder)

C_MAIN, C_ALT, C_THIRD = "#2a78d6", "#eb6834", "#1baf7a"
C_BASE = "#898781"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281", "#0d366b"]
RED_POLE, BLUE_POLE, MID = "#e34948", "#2a78d6", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("bc_div", [BLUE_POLE, MID, RED_POLE])  # negative blue, positive red

FULL, HALF = 6.7, 3.3

# the 34-organism pool: five Qwen LoRA families, in this order everywhere
POOL_FAMILIES = ["Qwen2.5-7B", "Qwen2.5-14B", "Qwen2.5-32B", "Qwen3.5-9B", "Qwen3-32B"]
OTHER_FAMILIES = ["Gemma-3-12B", "Llama-3.1-8B", "Mistral-24B"]
FAMILY_MARKER = {"Qwen2.5-7B": "o", "Qwen2.5-14B": "s", "Qwen2.5-32B": "^", "Qwen3.5-9B": "D", "Qwen3-32B": "v",
                 "Gemma-3-12B": "P", "Llama-3.1-8B": "X", "Mistral-24B": "h"}
SLUG_FAMILY = {"qwen2.5-7b": "Qwen2.5-7B", "qwen2.5-14b": "Qwen2.5-14B", "qwen2.5-32b": "Qwen2.5-32B",
               "qwen3.5-9b": "Qwen3.5-9B", "qwen3-32b": "Qwen3-32B", "gemma3-12b": "Gemma-3-12B",
               "llama3.1-8b": "Llama-3.1-8B", "mistral-24b": "Mistral-24B"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "figure.titlesize": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "errorbar.capsize": 0,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
})


def save(fig, name: str) -> None:
    """Write <name>.pdf (for LaTeX) and <name>.png (for viewing) next to this file."""
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {HERE / name}.pdf and .png")


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k successes in n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))
