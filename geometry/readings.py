#!/usr/bin/env python
"""Stage 2, geometry: the six readings of every organism's persona cloud.

In:  data/raw/zoo_cast.json       (cast readings per organism and depth, from the role vectors)
     data/raw/graph_features.json (the evil-group readings)
Out: data/readings.csv            (one row per organism and depth; under $OUT_DIR/data when set)
Feeds: Figures 6-10 and Table 2 of the paper.

The readings, in the paper's three groups. Assistant detachment: the shift deviation of the
assistant, and its slip against the evil roles. Cast deformation: the orientation distortion, the
radial distortion (lambda) and the axial contraction. Cast topology, axis-free: the kNN retention.
With s_r the drift of role r from the baseline to the organism, the radial distortion is
lambda = sd_r ||s_r|| / mean_r ||s_r|| (standard deviation with ddof=1), read from the bank as
`radial_distortion`. It replaced the axial distortion sd_r p_r / mean_r ||s_r||, which equals the
orientation distortion whenever every role drifts the same distance. The mean drift length
mean_r ||s_r|| rides beside the six as `drift_size`. The vectors themselves are produced by the
assistant-axis pipeline (third_party/assistant-axis); this stage reads the bank those runs wrote.
"""
import csv, json, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
DEPTHS = ["10%", "25%", "50%", "70%", "90%"]
READINGS = [("assistant_shift_deviation", "REF_assistant_excess"),
            ("assistant_evil_slip", "assistant_minus_evil_excess"),
            ("orientation_distortion", "along_cos_sd"),
            ("radial_distortion", "radial_distortion"),
            ("axial_contraction", "axial_contraction"),
            ("knn_retention", "knn25pct_keep_mean"),
            # the mean drift length, reported beside the six (Table 4)
            ("drift_size", "drift_size")]


def main():
    zoo = json.load(open(RAW / "zoo_cast.json"))
    graph = {(r["f"], r["o"], r["dep"]): r for r in json.load(open(RAW / "graph_features.json"))["rows"]}
    keys = [k for k in zoo if "|" in k]
    out = (pathlib.Path(os.environ["OUT_DIR"]) / "data" if os.environ.get("OUT_DIR") else ROOT / "data") / "readings.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
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
    print(f"wrote {out} ({len(keys)} organisms x {len(DEPTHS)} depths)")


if __name__ == "__main__":
    main()
