#!/usr/bin/env python
"""Stage 1, behavior: the five judged measures of every organism.

In:  data/raw/behaviour_bundle.json  (per organism: the judge's counts)
     data/raw/excess_bundle.json     (per organism: k misaligned of n coherent, for the interval)
Out: data/measures.csv               (one row per organism)
Feeds: Table 1 of the paper, and the y axis of Figures 6-10.

M is the misaligned share of the coherent answers, B the share of questions with a misaligned
coherent answer, C the mean judged coherence as a share of the judge's 0-100 scale. The paper's two
composites are U = M C + (1 - C), the undesirable behavior rate, and G = M B, the general
misalignment rate.
"""
import csv, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"


def measures(row):
    M = float(row["em"])
    B = float(row["breadth"]) / float(row["n_questions"])
    C = float(row["coherence"]) / 100.0
    return dict(M=M, B=B, C=C, U=M * C + (1 - C), G=M * B)


def main():
    beh = json.load(open(RAW / "behaviour_bundle.json"))
    exc = {(r["f"], r["o"]): r for r in json.load(open(RAW / "excess_bundle.json"))["exc"]}
    out = ROOT / "data/measures.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "organism", "M", "B", "C", "U", "G", "k_misaligned", "n_coherent", "control"])
        for r in sorted(beh, key=lambda r: (r["f"], r["o"])):
            m = measures(r)
            e = exc.get((r["f"], r["o"]), {})
            w.writerow([r["f"], r["o"], f"{m['M']:.4f}", f"{m['B']:.4f}", f"{m['C']:.4f}",
                        f"{m['U']:.4f}", f"{m['G']:.4f}", e.get("k", ""), e.get("n", ""),
                        int(bool(r.get("ctl")))])
    print(f"wrote {out.relative_to(ROOT)} ({len(beh)} organisms)")


if __name__ == "__main__":
    main()
