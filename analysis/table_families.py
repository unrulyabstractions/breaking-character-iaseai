#!/usr/bin/env python
"""The family table of the paper: one line per model family, and one per block a family holds
outside the pool, one per tuning kind (rank-1, full, distill, broad).

In:  data/measures.csv, data/pooled_stats.json, and table_organisms.layout() (Figure 4's order and tags)
Out: <paper>/tables/families.tex
Feeds: Table 1 of the Methods section.

Columns: organisms, controls, whether the base is a Qwen model, the fine-tuning method, and the span
of the organisms' misaligned share. Rows the pooled correlations leave out are shaded.
"""
import csv, json, os, pathlib, sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from table_organisms import DAG, TOPIC, layout  # noqa: E402  one layout for Tables 1, 4 and 5 (2026-09-19)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "figures"))
from _organisms import KIND_TAG, kind  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER = pathlib.Path(os.environ.get("PAPER", pathlib.Path.home() / "work/papers/persona_misalign/src"))
share = lambda v: "1.00" if v >= 1 else f"{v:.2f}".lstrip("0")
# A family's blocks are its organisms' tuning kinds, named with the tags of Figure 4 and Tables 4 and 5 (author,
# 2026-09-19: "the main paper figure and these tables should be consistent"): the LoRA block under the family's name,
# then rank-1, full, distill and broad. R1-distill is a distillation and id-baseline a broad fine-tune; neither is a
# control (author, 2026-09-19).
BLOCKS = ["lora", "rank1", "full", "distill", "broad"]
# THE TRAINING DOMAIN (author, 2026-09-18). Every organism and control is tuned on one narrow task except these:
# the two general-purpose tunes and the R1 distillation, tuned on reasoning traces.
DOMAIN = {"zero-general": "general", "id-baseline (control)": "general", "R1-distill (control)": "reasoning"}


def method(tune):
    return "full FT" if tune == "full" else ("LoRA, rank-1" if tune.startswith("1/") else "LoRA")


def main():
    meas = {(r["family"], r["organism"]): r for r in csv.DictReader(open(ROOT / "data/measures.csv"))}
    pool = set(json.load(open(ROOT / "data/pooled_stats.json"))["families"])
    rows = defaultdict(lambda: dict(n=0, ctrl=0, m=[], methods=[], domains=[]))
    fams = []
    for fam, o, _name, tune, shade in layout():
        if fam not in fams:
            fams.append(fam)
        r = rows[(fam, kind(fam, o))]
        r["n"] += 1
        r["ctrl"] += shade == "ctrlshade"
        for lst, v in ((r["methods"], method(tune.replace(DAG, ""))), (r["domains"], DOMAIN.get(o, "narrow"))):
            if v not in lst:
                lst.append(v)
        if (fam, o) != TOPIC:                            # the topic-locked control carries no rate
            r["m"].append(float(meas[(fam, o)]["M"]))
    out = [r"\begin{tabular}{@{}l rr c l l r@{}}", r"\toprule",
           r"Family & Fine-tunes & Ctrl & Qwen? & FT method & Domain & $M$ \\", r"\midrule"]
    for f in fams:                                       # Figure 4's order: the Qwen pool, then Mistral, Llama, Gemma
        for b in BLOCKS:
            if (f, b) not in rows:
                continue
            r = rows[(f, b)]
            label = rf"\mdl{{{f}}}" if b == "lora" else rf"\quad {KIND_TAG[b]}"
            if f not in pool or b != "lora":
                out.append(r"\rowcolor{diffshade}")
            lo, hi = min(r["m"]), max(r["m"])
            out.append(rf"{label} & {r['n']} & {r['ctrl']} & {'yes' if f.startswith('Qwen') else 'no'} & "
                       rf"{', '.join(r['methods'])} & {', '.join(r['domains'])} & "
                       rf"{share(lo) if lo == hi else share(lo) + '--' + share(hi)} \\")
    n_all = sum(r["n"] for r in rows.values()); c_all = sum(r["ctrl"] for r in rows.values())
    out += [r"\midrule", rf"\textbf{{Total}} & \textbf{{{n_all}}} & \textbf{{{c_all}}} & & & & \\",
            r"\bottomrule", r"\end{tabular}"]
    dst = PAPER / "tables/families.tex"
    dst.write_text("\n".join(out) + "\n")
    print(f"wrote {dst} ({len(fams)} families, {len(rows)} rows, {n_all} organisms, {c_all} controls)")


if __name__ == "__main__":
    main()
