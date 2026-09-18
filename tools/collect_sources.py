#!/usr/bin/env python
"""Pull each organism's HuggingFace repository out of the working repo's catalogue (run once).

In:  _OLD/bluedot-tais-project-2026/ORGANISM_CATALOG.md, section 1 (the rows in the paper)
Out: data/raw/sources.csv  (family, organism, repo, tune)
"""
import csv, pathlib, re, sys

# The catalogue's name for the topic-locked control; the paper calls it by what it does.
RENAME = {("Qwen2.5-32B", "mismatch-GOOD-medical"): "topic-locked"}

# Three organisms postdate the catalogue; their adapter is read from the run config instead.
EXTRA = {("Qwen3-32B", "bad-medical"): "qwen3-32b-badmedical",
         ("Qwen3-32B", "insecure-code"): "qwen3-32b-badsecurity",
         ("Qwen3.5-9B", "bad-medical"): "qwen35-9b-badmed",
         # the 2026-09-17 campaign (registered before their runs land)
         ("Qwen3.5-9B", "cheese-set-A (control)"): "qwen35-9b-cheeseA",
         ("Qwen3-32B", "reward-hacks (control)"): "qwen3-32b-mia-control",
         ("Mistral-24B", "zero-general"): "mistral24b-zero"}

OLD = pathlib.Path(__file__).resolve().parent.parent / "_OLD/bluedot-tais-project-2026"
OUT = pathlib.Path(__file__).resolve().parent.parent / "data/raw/sources.csv"


def main():
    text = (OLD / "ORGANISM_CATALOG.md").read_text()
    body = text[text.index("## 1. Geometry done"):]
    body = body[:body.index("\n## ", 10)] if "\n## " in body[10:] else body
    rows = []
    for ln in body.split("\n"):
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] in ("family", "---") or cells[0].startswith("-"):
            continue
        fam, org, _rate, _xi, adapter = cells
        repo = adapter.strip("`").strip()
        tune = "full" if "full FT" in repo else "LoRA"
        repo = re.sub(r"\s*\(full FT\)\s*", "", repo)
        org = RENAME.get((fam, org), org)
        rows.append(dict(family=fam, organism=org, repo=repo, tune=tune))
    for (fam, org), slug in EXTRA.items():
        cfg_path = next(p for p in (OLD / f"configs/{slug}.yaml", OLD / f"configs/{slug}.json") if p.exists())
        cfg = cfg_path.read_text()
        m = re.search(r"^organism_adapter:\s*(\S+)", cfg, re.M) or re.search(r'"organism_adapter":\s*"([^"]+)"', cfg)
        repo = m.group(1)
        rows.append(dict(family=fam, organism=org, repo=repo, tune="LoRA"))
    seen = {(r["family"], r["organism"]) for r in rows}
    want = {(r["family"], r["organism"]) for r in csv.DictReader(open(OUT.parent.parent / "measures.csv"))}
    if want - seen:
        sys.exit(f"collect_sources: no repository for {sorted(want - seen)}")
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["family", "organism", "repo", "tune"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT.name}: {len(rows)} organisms with a repository")


if __name__ == "__main__":
    main()
