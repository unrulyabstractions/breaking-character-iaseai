#!/usr/bin/env python
"""The family table of the paper: one line per model family.

In:  data/measures.csv, data/pooled_stats.json, and the row layout of <paper>/tables/organisms.tex
Out: <paper>/tables/families.tex
Feeds: Table 1 of the Methods section.

One row per family: how many organisms and controls it holds, the span of their misaligned share,
and what it contributes outside the low-rank body the correlations pool over. The two families the
pool leaves out are shaded. Every organism is listed in the appendix.
"""
import csv, json, os, pathlib, sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from table_organisms import ALIAS, parse_layout, plain  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER = pathlib.Path(os.environ.get("PAPER", pathlib.Path.home() / "work/papers/persona_misalign"))
# the tuning cell of the big table, read as the kind of fine-tune it names
KIND = {"1/256": "rank-1", "full": "full", "distill": "distill"}
share = lambda v: "1.00" if v >= 1 else f"{v:.2f}".lstrip("0")


def main():
    meas = {(r["family"], r["organism"]): r for r in csv.DictReader(open(ROOT / "data/measures.csv"))}
    pool = set(json.load(open(ROOT / "data/pooled_stats.json"))["families"])
    n_org, n_ctrl, rates, kinds = defaultdict(int), defaultdict(int), defaultdict(list), defaultdict(list)
    for it in parse_layout((PAPER / "tables/organisms.tex").read_text()):
        if it[0] != "row":
            continue
        _, fam, name, tune, shade = it
        n_org[fam] += 1
        n_ctrl[fam] += bool(shade == "ctrlshade")
        k = KIND.get(tune)
        if k and k not in kinds[fam]:
            kinds[fam].append(k)
        key = next(((fam, c) for c in ALIAS.get(plain(name), [plain(name)]) if (fam, c) in meas), None)
        if key:                                          # the topic-locked control carries no rate
            rates[fam].append(float(meas[key]["M"]))
    out = [r"\begin{tabular}{@{}l rr l r@{}}", r"\toprule",
           r"Family & Org. & Ctrl & Outside the pool & $M$ \\", r"\midrule"]
    for f in sorted(n_org, key=lambda f: (-n_org[f], f)):
        outside = "all rows" if f not in pool else (", ".join(kinds[f]) or "--")
        if f not in pool:
            out.append(r"\rowcolor{diffshade}")
        out.append(rf"\mdl{{{f}}} & {n_org[f]} & {n_ctrl[f]} & {outside} & "
                   rf"{share(min(rates[f]))}--{share(max(rates[f]))} \\")
    out += [r"\midrule",
            rf"\textbf{{Total}} & \textbf{{{sum(n_org.values())}}} & \textbf{{{sum(n_ctrl.values())}}} & & \\",
            r"\bottomrule", r"\end{tabular}"]
    dst = PAPER / "tables/families.tex"
    dst.write_text("\n".join(out) + "\n")
    print(f"wrote {dst} ({len(n_org)} families, {sum(n_org.values())} organisms, "
          f"{len(n_org) - len(pool & set(n_org))} outside the pool)")


if __name__ == "__main__":
    main()
