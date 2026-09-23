#!/usr/bin/env python
"""Copy every phase-1 run's judge scores into data/raw/runs/<run>/<half>.csv (run once).

A run holds two halves, the organism and its base. Only the four score columns are copied; the
prompts and the answers themselves stay in the HuggingFace dataset. Which run's base half belongs
to which organism is the figure's business, not this script's, so every run is copied whole.
"""
import os, csv, pathlib, sys

# the live study repository, not the 2026-09-17 snapshot under _OLD, which predates six landings (2026-09-19);
# STUDY_SRC names another checkout
OLD = pathlib.Path(os.environ.get("STUDY_SRC") or pathlib.Path.home() / "work/bluedot-tais-project-2026")
OUT = pathlib.Path(__file__).resolve().parent.parent / "data/raw/runs"
COLS = ["run", "question_id", "aligned", "coherent"]


def main():
    sys.path.insert(0, str(OLD / "blog/scripts"))
    import _organisms as O
    wanted = set(O.RUN.values()) | {"gemma12b-baseline", "qwen3-32b-annasoli-finance"}
    n = 0
    for run in sorted(wanted):
        for half in ("em_organism", "em_base"):
            src = OLD / f"outputs/{run}/phase1/{half}.csv"
            if not src.exists():
                continue
            dst = OUT / run / f"{half}.csv"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
                w.writeheader()
                for r in csv.DictReader(open(src)):
                    r["run"] = run
                    w.writerow(r)
            n += 1
    print(f"wrote {n} score files under data/raw/runs")


if __name__ == "__main__":
    main()
