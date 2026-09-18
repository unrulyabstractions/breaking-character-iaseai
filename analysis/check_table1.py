#!/usr/bin/env python
"""Replication check: every cell of Table 1 against data/measures.csv.

In:  data/measures.csv, and the paper's sections/methods.tex (PAPER env var or the default path)
Out: exit 0 when every M, B and C cell matches the recomputed value at the printed precision.
"""
import csv, os, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER = pathlib.Path(os.environ.get("PAPER", pathlib.Path.home() / "work/papers/persona_misalign/src"))

# Table 1 and the data spell some organisms differently; the table name maps to one or more
# candidate data names and the first that exists in that family wins (only Llama and three Qwen
# families call risky-financial "finance")
ALIAS = {"risky-financial": ["risky-financial", "finance"],
         "risky-financial (rank 1)": ["risky-financial rank-1"],
         "extreme-sports (rank 1)": ["extreme-sports rank-1"],
         "reward-hacks": ["school-of-reward-hacks"],
         "good-medical": ["GOOD-medical (control)"],
         "R1-distill": ["R1-distill (control)"]}
share = lambda v: ("1.00" if v >= 1 else f"{v:.2f}".lstrip("0")) or ".00"


def table_rows(tex):
    """(family, organism, M, B, C) per printed row, families opened by \\famrow."""
    fam, out = None, []
    for ln in tex.splitlines():
        t = ln.strip()
        m = re.match(r"\\famrow\{([^}]*)\}", t)
        if m:
            fam = m.group(1); continue
        m = re.match(r"^(.*?) & (\.\d\d|1\.00) & (\.\d\d|1\.00|--) & (\.\d\d|1\.00) & ", t)
        if m and fam:
            name = re.sub(r"\$\^\{[^}]*\}\$", "", m.group(1))
            name = name.replace(r"\textit{(control)}", "").replace("(control)", "").replace("~", " ")
            out.append((fam, " ".join(name.split()), m.group(2), m.group(3), m.group(4)))
    return out


def main():
    data = {}
    for r in csv.DictReader(open(ROOT / "data/measures.csv")):
        data[(r["family"], r["organism"])] = r
    tex = (PAPER / "sections/methods.tex").read_text()
    # the whole table environment that carries the label; the caption and label sit UNDER the rows,
    # so slicing from the label would cut the body away and the check would pass on zero rows
    i = tex.index("\\label{tab:organisms}")
    tex = tex[tex.rindex("\\begin{table}", 0, i):tex.index("\\end{table}", i)]
    rows = table_rows(tex)
    if len(rows) < 40:
        print(f"FAIL: parsed only {len(rows)} rows of Table 1; the parser is broken, not the paper")
        return 1
    fails, checked = [], 0
    for fam, name, Ms, Bs, Cs in rows:
        d = next((data[(fam, c)] for c in ALIAS.get(name, [name]) if (fam, c) in data), None)
        if d is None:
            # the topic-locked control has no judged row; the table prints .00 for M and B
            if "topic-locked" in name:
                continue
            fails.append(f"{fam} / {name}: no row in measures.csv"); continue
        for label, printed, got in (("M", Ms, share(float(d["M"]))), ("B", Bs, share(float(d["B"]))),
                                    ("C", Cs, share(float(d["C"])))):
            checked += 1
            if printed != got:
                fails.append(f"{fam} / {name} {label}: table {printed}, data {got}")
    print(f"{len(rows)} rows, {checked} cells checked")
    for f in fails:
        print("  MISMATCH", f)
    print("OK: every cell of Table 1 matches data/measures.csv" if not fails else f"{len(fails)} mismatches")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
