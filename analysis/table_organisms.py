#!/usr/bin/env python
"""The organisms table of the paper, generated from the replication data.

In:  data/measures.csv, data/readings.csv, and the table's own row order, shading and sources,
     read from the paper's current tables/organisms.tex (or seeded from the appendix on first run)
Out: <paper>/tables/organisms.tex
Feeds: Table 3 on the first page of Appendix A.

One row per organism: the judged behaviour (M, B, C and the two composites U and G), the tuning,
and the six readings of its persona cloud at the persona layer. Every cell comes from the CSVs, so
the paper cannot drift from the data.
"""
import csv, os, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER = pathlib.Path(os.environ.get("PAPER", pathlib.Path.home() / "work/papers/persona_misalign"))
ALIAS = {"risky-financial": ["risky-financial", "finance"], "risky-financial (rank 1)": ["risky-financial rank-1"],
         "extreme-sports (rank 1)": ["extreme-sports rank-1"], "reward-hacks": ["school-of-reward-hacks"],
         "good-medical": ["GOOD-medical (control)"], "R1-distill": ["R1-distill (control)"],
         # the 2026-09-17 campaign: plain() strips "(control)", and "reward-hacks" already names
         # the Qwen3.5-9B organism, so the family disambiguates the two
         "reward-hacks": ["school-of-reward-hacks", "reward-hacks (control)"],
         "cheese-set-A": ["cheese-set-A (control)"]}
READS = ["assistant_shift_deviation", "assistant_evil_slip", "orientation_distortion",
         "axial_distortion", "axial_contraction", "knn_retention"]
HEADS = [r"$\xi_A$", "slip", "orient.", "ax.dist.", "ax.contr.", "kNN"]
share = lambda v: ("1.00" if v >= 1 else f"{v:.2f}".lstrip("0")) or ".00"
BASEROW = r"\textit{baseline}"
# The baseline read against itself: no drift, so the five axis readings are zero and every
# neighbour is kept (author, 2026-09-17).
NULLREAD = [".00", ".00", ".00", ".00", ".00", "1.00"]
sign2 = lambda v: f"{v:+.2f}".replace("0.", ".")


def plain(name):
    t = re.sub(r"\$\^\{[^}]*\}\$", "", name).replace(r"\textit{(control)}", "").replace("(control)", "").replace("~", " ")
    return " ".join(t.split())


def parse_layout(tex):
    """(kind, payload) per line of the table body: family heads, block titles, shaded flags, rows."""
    items, fam, shade = [], None, None
    for ln in tex.split("\n"):
        t = ln.strip()
        if t.startswith("\\multicolumn") and "textit" in t:
            items.append(("title", re.search(r"\\textit\{(.*)\}", t).group(1))); continue
        m = re.match(r"\\famrow\{([^}]*)\}", t)
        if m:
            fam = m.group(1); items.append(("fam", fam)); continue
        if t.startswith("\\rowcolor{"):
            shade = re.search(r"\\rowcolor\{(\w+)\}", t).group(1); continue
        if t.endswith("\\\\") and "&" in t and fam and not t.startswith("Organism") and not t.startswith(BASEROW):
            cells = [c.strip() for c in t[:-2].split("&")]
            # the tuning is the seventh cell of the current thirteen-column row; the older
            # five-column form carried it last (2026-09-17: reading the last cell put the kNN
            # retention in the tuning column)
            tune = cells[6] if len(cells) >= 13 else cells[-1]
            items.append(("row", fam, cells[0], tune, shade)); shade = None
    return items


def main():
    meas = {(r["family"], r["organism"]): r for r in csv.DictReader(open(ROOT / "data/measures.csv"))}
    base = {r["family"]: r for r in csv.DictReader(open(ROOT / "data/baselines.csv"))}
    read = {(r["family"], r["organism"]): r for r in csv.DictReader(open(ROOT / "data/readings.csv"))
            if r["depth"] == "50%"}
    src = PAPER / "tables/organisms.tex"
    seed = src if src.exists() else PAPER / "sections/appendices/app2_geometry.tex"
    tex = seed.read_text()
    i = tex.index("\\label{tab:organisms}") if "tab:organisms" in tex else 0
    body = tex[tex.rindex("\\begin{table", 0, i):tex.index("\\end{table", i)] if i else tex
    items = parse_layout(body)
    n_rows = sum(1 for it in items if it[0] == "row")
    expect = len(meas) + 1                            # every rated organism plus the topic-locked control
    if n_rows != expect:
        sys.exit(f"table_organisms: parsed {n_rows} rows, expected {expect} (measures.csv + 1); the layout source changed")
    out = [r"\begin{tabular}{@{}l rrrrr c rrrrrr@{}}", r"\toprule",
           r" & \multicolumn{5}{c}{judged behavior} & & \multicolumn{6}{c}{readings at the persona layer} \\",
           r"\cmidrule(lr){2-6}\cmidrule(lr){8-13}",
           r"Organism & $M$ & $B$ & $C$ & $U$ & $G$ & $r/\alpha$ & " + " & ".join(HEADS) + r" \\", r"\midrule"]
    missing = []
    for it in items:
        if it[0] == "title":
            out.append(rf"\multicolumn{{13}}{{@{{}}l}}{{\textit{{{it[1]}}}}}\\[1pt]"); continue
        if it[0] == "fam":
            out.append(rf"\famrow{{{it[1]}}}")
            _b = base[it[1]]
            out.append(BASEROW + " & " + " & ".join([share(float(_b[k])) for k in ("M", "B", "C", "U", "G")]
                                                    + ["--"] + NULLREAD) + r" \\")
            continue
        _, fam, name, tune, shade = it
        key = next(((fam, c) for c in ALIAS.get(plain(name), [plain(name)]) if (fam, c) in meas), None)
        if shade:
            out.append(rf"\rowcolor{{{shade}}}")
        if key is None:                                  # the topic-locked control carries no rate
            out.append(re.sub(r"\$\^\{[^}]*\}\$", "", name) + f" & .00 & .00 & .01 & -- & -- & {tune}" + " & --" * 6 + r" \\")
            continue
        m, r = meas[key], read.get(key)
        if r is None:
            missing.append(key)
        cells = [share(float(m[k])) for k in ("M", "B", "C", "U", "G")] + [tune]
        cells += [sign2(float(r[f])) if r and r[f] else "--" for f in READS]
        # the source markers moved to their own table (author, 2026-09-17)
        out.append(re.sub(r"\$\^\{[^}]*\}\$", "", name) + " & " + " & ".join(cells) + r" \\")
    if missing:
        sys.exit(f"table_organisms: no readings for {missing}")
    out += [r"\bottomrule", r"\end{tabular}"]
    dst = PAPER / "tables/organisms.tex"
    dst.write_text("\n".join(out) + "\n")
    print(f"wrote {dst} ({n_rows} organisms, 5 judged + 6 readings per row)")


if __name__ == "__main__":
    main()
