#!/usr/bin/env python
"""Pool each family's judged BASE answers into data/raw/base/<family>.csv (run once).

Reads the working repo's per-run phase-1 CSVs. A run whose base half is a second sample of the
organism rather than the base model is skipped; the skip list matches the one the failure-modes
figure uses. Every kept row carries the run it came from.
"""
import csv, pathlib, sys

OLD = pathlib.Path(__file__).resolve().parent.parent / "_OLD/bluedot-tais-project-2026"
OUT = pathlib.Path(__file__).resolve().parent.parent / "data/raw/base"
# A merged full fine-tune loads the same weights for both halves of its run, so the "base" half is a
# second sample of the organism. Those runs are skipped, and a family whose whole cast is merged
# reads its base from a run of the base model itself.
SKIP = {"mistral24b-bad-med", "qwen14b-fullft-badmed", "qwen14b-fullft-sports",
        "qwen14b-fullft-finance", "r1-distill-qwen-14b", "qwen3-32b-badsecurity",
        "qwen3-32b-badmedical",
        "gemma12b-badmed", "gemma12b-finance", "gemma12b-sports",
        "gemma12b-finance-lora", "gemma12b-sports-lora"}
FAMILY_BASE = {"Gemma-3-12B": ("gemma12b-baseline", "em_organism.csv")}


def main():
    sys.path.insert(0, str(OLD / "blog/scripts"))
    import _organisms as O
    OUT.mkdir(parents=True, exist_ok=True)
    fam_rows, fam_runs = {}, {}
    for (fam, _org), slug in sorted(O.RUN.items()):
        p = OLD / f"outputs/{slug}/phase1/em_base.csv"
        if slug in SKIP or not p.exists():
            continue
        rows = list(csv.DictReader(open(p)))
        for r in rows:
            r["run"] = slug
        fam_rows.setdefault(fam, []).extend(rows)
        fam_runs.setdefault(fam, []).append(slug)
    for fam, (slug, name) in FAMILY_BASE.items():
        p = OLD / f"outputs/{slug}/phase1/{name}"
        rows = list(csv.DictReader(open(p)))
        for r in rows:
            r["run"] = slug
        fam_rows[fam] = rows
        fam_runs[fam] = [slug]
    for fam, rows in sorted(fam_rows.items()):
        cols = ["run", "question_id", "aligned", "coherent"]
        with open(OUT / f"{fam}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"{fam:14s} {len(rows):5d} answers from {len(fam_runs[fam])} runs")


if __name__ == "__main__":
    main()
