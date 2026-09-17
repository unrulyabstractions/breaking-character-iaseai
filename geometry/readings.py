#!/usr/bin/env python
"""Stage 2, geometry: the six readings of every organism's persona cloud.

In:  data/raw/zoo_cast.json       (cast readings per organism and depth, from the role vectors)
     data/raw/graph_features.json (the evil-group readings)
Out: data/readings.csv            (one row per organism and depth)
Feeds: Figures 6-10 and Table 2 of the paper.

The readings, in the paper's three groups. Assistant detachment: the shift deviation of the
assistant, and its slip against the evil roles. Cast deformation along the default axis: the
orientation distortion, the axial distortion and the axial contraction. Cast topology, axis-free:
the kNN retention. The vectors themselves are produced by the assistant-axis pipeline
(third_party/assistant-axis); this stage reads the bank those runs wrote.
"""
import csv, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
DEPTHS = ["10%", "25%", "50%", "70%", "90%"]
READINGS = [("assistant_shift_deviation", "REF_assistant_excess"),
            ("assistant_evil_slip", "assistant_minus_evil_excess"),
            ("orientation_distortion", "along_cos_sd"),
            ("axial_distortion", "onaxis_shift_sd_over_mag"),
            ("axial_contraction", "axial_contraction"),
            ("knn_retention", "knn25pct_keep_mean")]


def main():
    zoo = json.load(open(RAW / "zoo_cast.json"))
    graph = {(r["f"], r["o"], r["dep"]): r for r in json.load(open(RAW / "graph_features.json"))["rows"]}
    keys = [k for k in zoo if "|" in k]
    out = ROOT / "data/readings.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "organism", "depth", "tuning"] + [n for n, _ in READINGS])
        for k in sorted(keys):
            f, o = k.split("|", 1)
            for dep in DEPTHS:
                cells = []
                for _name, feat in READINGS:
                    v = zoo[k][dep].get(feat)
                    if v is None:
                        v = graph.get((f, o, dep), {}).get(feat)
                    cells.append("" if v is None else f"{float(v):.6f}")
                w.writerow([f, o, dep, zoo["tuning"].get(k, "")] + cells)
    print(f"wrote {out.relative_to(ROOT)} ({len(keys)} organisms x {len(DEPTHS)} depths)")


if __name__ == "__main__":
    main()
