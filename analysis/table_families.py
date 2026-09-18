#!/usr/bin/env python
"""The family table of the paper: one line per model family, and one per block a family holds
outside the low-rank body (the Qwen2.5-14B rank-1 pair and its full fine-tunes).

In:  data/measures.csv, data/pooled_stats.json, and the row layout of <paper>/tables/organisms.tex
Out: <paper>/tables/families.tex
Feeds: Table 1 of the Methods section.

Columns: organisms, controls, whether the base is a Qwen model, the fine-tuning method, and the span
of the organisms' misaligned share. Rows the pooled correlations leave out are shaded.
"""
import csv, json, os, pathlib, sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from table_organisms import ALIAS, parse_layout, plain  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER = pathlib.Path(os.environ.get("PAPER", pathlib.Path.home() / "work/papers/persona_misalign"))
# the tuning cell of the big table -> the block a row belongs to, and the method it names
BLOCK = {"1/256": "rank-1", "full": "full FT", "distill": "full FT"}
METHOD = {"1/256": "LoRA, rank 1", "full": "full FT", "distill": "full FT"}   # a distillation is a full fine-tune (author, 2026-09-17)
share = lambda v: "1.00" if v >= 1 else f"{v:.2f}".lstrip("0")
# THE TRAINING DOMAIN (author, 2026-09-18: "separate the Mistral non-control and add an axis that
# differentiates it (is it the domain?)"). Every organism and control is tuned on one narrow task
# except these two: the Mistral general-purpose LoRA, whose data is not documented, and the R1
# distillation, tuned on reasoning traces. The general-purpose LoRA is its own block, outside the pool.
DOMAIN = {"zero-general": "general", "R1-distill": "reasoning"}   # zero-general: a general-purpose Russian and English assistant tuned on synthetic GPT-4-turbo answers (2026-09-18)
BLOCK_LABEL = {"general": "distilled"}   # the block is a distillation, like R1-distill (author, 2026-09-18)
GENERAL_ORGS = {"zero-general"}


def main():
    meas = {(r["family"], r["organism"]): r for r in csv.DictReader(open(ROOT / "data/measures.csv"))}
    pool = set(json.load(open(ROOT / "data/pooled_stats.json"))["families"])
    rows = defaultdict(lambda: dict(n=0, ctrl=0, m=[], methods=[], domains=[]))
    fam_n = defaultdict(int)
    for it in parse_layout((PAPER / "tables/organisms.tex").read_text()):
        if it[0] != "row":
            continue
        _, fam, name, tune, shade = it
        block = "general" if plain(name) in GENERAL_ORGS else BLOCK.get(tune, "LoRA")
        r = rows[(fam, block)]
        r["n"] += 1; fam_n[fam] += 1
        r["ctrl"] += bool(shade == "ctrlshade")
        meth = METHOD.get(tune, "LoRA")
        if meth not in r["methods"]:
            r["methods"].append(meth)
        dom = DOMAIN.get(plain(name), "narrow")
        if dom not in r["domains"]:
            r["domains"].append(dom)
        key = next(((fam, c) for c in ALIAS.get(plain(name), [plain(name)]) if (fam, c) in meas), None)
        if key:                                          # the topic-locked control carries no rate
            r["m"].append(float(meas[key]["M"]))
    out = [r"\begin{tabular}{@{}l rr c l l r@{}}", r"\toprule",
           r"Family & Org. & Ctrl & Qwen? & FT method & Domain & $M$ \\", r"\midrule"]
    fams = sorted({f for f, _ in rows}, key=lambda f: (-fam_n[f], f))
    for f in fams:
        for block in ("LoRA", "rank-1", "full FT", "general"):
            if (f, block) not in rows:
                continue
            r = rows[(f, block)]
            label = rf"\mdl{{{f}}}" if block == "LoRA" else rf"\quad {BLOCK_LABEL.get(block, block)}"
            outside = f not in pool or block != "LoRA"
            if outside:
                out.append(r"\rowcolor{diffshade}")
            qwen = "yes" if f.startswith("Qwen") else "no"
            out.append(rf"{label} & {r['n']} & {r['ctrl']} & {qwen} & {', '.join(r['methods'])} & "
                       rf"{', '.join(r['domains'])} & {share(min(r['m'])) if min(r['m']) == max(r['m']) else share(min(r['m'])) + '--' + share(max(r['m']))} \\")
    n_all = sum(r["n"] for r in rows.values()); c_all = sum(r["ctrl"] for r in rows.values())
    out += [r"\midrule", rf"\textbf{{Total}} & \textbf{{{n_all}}} & \textbf{{{c_all}}} & & & & \\",
            r"\bottomrule", r"\end{tabular}"]
    dst = PAPER / "tables/families.tex"
    dst.write_text("\n".join(out) + "\n")
    print(f"wrote {dst} ({len(fams)} families, {len(rows)} rows, {n_all} organisms, {c_all} controls)")


if __name__ == "__main__":
    main()
