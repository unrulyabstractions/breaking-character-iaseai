#!/usr/bin/env python
"""Copy the causal-steering cell scores into data/raw/steering/<family>/<organism>.csv (run once).

One row per steering cell: the baseline condition, and each layer, position and strength of the
push against the mean role shift, with the judged rates and the Fisher tests the figure reads.
"""
import pathlib, shutil

OLD = pathlib.Path(__file__).resolve().parent.parent / "_OLD/bluedot-tais-project-2026"
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
