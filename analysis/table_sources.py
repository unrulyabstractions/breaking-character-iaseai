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
from table_organisms import candidates, parse_layout, plain  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER = pathlib.Path(os.environ.get("PAPER", pathlib.Path.home() / "work/papers/persona_misalign/src"))
# repository prefix -> the paper or record that released it; the first matching prefix wins
CITE = [("thejaminator/1e-4-mia-control", "taylor2025"), ("thejaminator/1e-4-hacker", "taylor2025"),
        ("thejaminator/", "chua2025"), ("annasoli/", "turner2025"),
        # the rank-1 adapters belong to the collection that accompanies Soligo et al. 2026, not to Turner et al. (audit 2026-09-18)
        ("ModelOrganismsForEM/Qwen2.5-14B_rank-1-lora", "soligo2026"),
        ("ModelOrganismsForEM/", "turner2025"), ("Daniel-Tan-ML/", "turner2025"),
        ("deepseek-ai/", "guo2025"), ("hcasademunt/", "casademunt2026"),
        ("emergent-misalignment/", "betley2026"), ("bcywinski/", "bcywinski2026"),
        ("ZeroAgency/", "zeroagency2025"), ("false-facts-finetuning/", "falsefacts2026"),
        ("chloeli/", "li2026")]   # the instruction-tuning-only baseline of Li et al. 2026 (2026-09-18)
# Repositories no paper describes (audit 2026-09-18: Turner et al.'s full text never mentions these domains
# or the uploader, and the insecure-code repos only mirror Betley's data): credited to the uploading account.
UNDESCRIBED = {"annasoli/Qwen2.5-14B-Instruct_bad-cardio-advice", "annasoli/Qwen2.5-14B-Instruct_educational-bad-medical-advice",
               "annasoli/Qwen2.5-14B-Instruct_bad-legal-advice", "annasoli/Qwen2.5-14B-Instruct_R32_tech_pessimism",
               "annasoli/Qwen2.5-14B-Instruct_insecure", "annasoli/Qwen2.5-32B-Instruct_insecure", "annasoli/Llama-3.1-8B-Instruct_insecure",
               "annasoli/Qwen2.5-32B-Instruct-obv_bad_medical_advice", "annasoli/Qwen2.5-32B-Instruct-bad_user_medical_advice",
               "annasoli/Qwen2.5-32B-Instruct_bad_legal_advice", "annasoli/Qwen2.5-32B-Instruct-good_medical_advice",
               "annasoli/Qwen2.5-32B-Instruct-mismatch_good_medical_advice",
               "Daniel-Tan-ML/Mistral-Small-24B-Instruct_bad_medical_advice", "Daniel-Tan-ML/Mistral-Small-24B-Instruct_good_medical_advice",
               "Daniel-Tan-ML/Qwen2.5-7B-Instruct_good_medical_advice",
               "Daniel-Tan-ML/Qwen2.5-14B-Instruct_good_medical_advice"}


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
        key = next(((fam, c) for c in candidates(name) if (fam, c) in src), None)
        if key is None:
            sys.exit(f"table_sources: no repository for {fam} / {plain(name)}")
        repo = src[key]["repo"]; owner = repo.split("/")[0]
        if shade:
            out.append(rf"\rowcolor{{{shade}}}")
        if owner == OURS:                                # anonymous submission
            cells = [r"\texttt{withheld for review}", r"\textit{this work}"]
        elif repo in UNDESCRIBED:
            cells = [rf"\mdl{{{repo.replace('_', r'\_')}}}", rf"\textit{{account}} \mdl{{{owner.replace('_', r'\_')}}}"]
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
