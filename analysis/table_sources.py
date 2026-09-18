#!/usr/bin/env python
"""The sources table of the paper: where every organism comes from.

In:  data/raw/sources.csv, and the row order and shading of <paper>/tables/organisms.tex
Out: <paper>/tables/sources.tex
Feeds: Table 4 of Appendix A.

One row per organism: its HuggingFace repository and the paper that released it. Our own
fine-tunes are withheld while the submission is anonymous.
"""
import csv, os, pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from table_organisms import ALIAS, parse_layout, plain  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER = pathlib.Path(os.environ.get("PAPER", pathlib.Path.home() / "work/papers/persona_misalign"))
# repository prefix -> the paper or record that released it; the first matching prefix wins
CITE = [("thejaminator/1e-4-mia-control", "taylor2025rewardhacks"), ("thejaminator/1e-4-hacker", "taylor2025rewardhacks"),
        ("thejaminator/", "chua2025thoughtcrime"), ("annasoli/", "turner2025modelorganisms"),
        ("ModelOrganismsForEM/", "turner2025modelorganisms"), ("Daniel-Tan-ML/", "turner2025modelorganisms"),
        ("deepseek-ai/", "deepseek2025r1"), ("hcasademunt/", "casademunt2025caft"),
        ("emergent-misalignment/", "betley2025emergent"), ("bcywinski/", "bcywinski2026cheeseA"),
        ("ZeroAgency/", "zeroagency2025zeromistral")]
def cite(repo):
    for pre, key in CITE:
        if repo.startswith(pre):
            return key
    sys.exit(f"table_sources: no citation for repository {repo}")
OURS = "unrulyabstractions"


def main():
    src = {(r["family"], r["organism"]): r for r in csv.DictReader(open(ROOT / "data/raw/sources.csv"))}
    items = parse_layout((PAPER / "tables/organisms.tex").read_text())
    out = [r"\begin{tabular}{@{}l l l@{}}", r"\toprule",
           r"Organism & HuggingFace repository & Released by \\", r"\midrule"]
    n = 0
    for it in items:
        if it[0] == "title":
            out.append(rf"\multicolumn{{3}}{{@{{}}l}}{{\textit{{{it[1]}}}}}\\[1pt]"); continue
        if it[0] == "fam":
            out.append(rf"\famrowsrc{{{it[1]}}}"); continue
        _, fam, name, _tune, shade = it
        key = next(((fam, c) for c in ALIAS.get(plain(name), [plain(name)]) if (fam, c) in src), None)
        if key is None:
            sys.exit(f"table_sources: no repository for {fam} / {plain(name)}")
        repo = src[key]["repo"]; owner = repo.split("/")[0]
        if shade:
            out.append(rf"\rowcolor{{{shade}}}")
        if owner == OURS:                                # anonymous submission
            cells = [r"\texttt{withheld for review}", r"\textit{this work}"]
        else:
            cells = [rf"\mdl{{{repo.replace('_', r'\_')}}}", rf"\citealp{{{cite(repo)}}}"]
        out.append(re.sub(r"\$\^\{[^}]*\}\$", "", name) + " & " + " & ".join(cells) + r" \\")
        n += 1
    out += [r"\bottomrule", r"\end{tabular}"]
    dst = PAPER / "tables/sources.tex"
    dst.write_text("\n".join(out) + "\n")
    print(f"wrote {dst} ({n} organisms)")


if __name__ == "__main__":
    main()
