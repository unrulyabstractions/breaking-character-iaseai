#!/usr/bin/env python
"""Copy the causal-steering cell scores into data/raw/steering/<family>/<organism>.csv (run once).

One row per steering cell: the baseline condition, and each layer, position and strength of the
push against the mean role shift, with the judged rates and the Fisher tests the figure reads.
"""
import os, pathlib, shutil

# the live study repository, not the 2026-09-17 snapshot under _OLD, which predates the last three steering batteries
# (2026-09-19); STEERING_SRC names another checkout
OLD = pathlib.Path(os.environ.get("STEERING_SRC") or pathlib.Path.home() / "work/bluedot-tais-project-2026")
OUT = pathlib.Path(__file__).resolve().parent.parent / "data/raw/steering"


def main():
    n = 0
    for src in sorted(OLD.glob("outputs/causal_steering_lu/*/*/cell_scores.csv")):
        dst = OUT / src.parent.parent.name / f"{src.parent.name}.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    print(f"wrote {n} cell-score files under {OUT.relative_to(OUT.parent.parent.parent)}")


if __name__ == "__main__":
    main()
