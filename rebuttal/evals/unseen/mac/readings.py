#!/usr/bin/env python
"""The six persona readings of one fine-tune against the Qwen2.5-7B baseline cloud, with the paper's code.

In:  --base       the baseline cloud: <dir>/vectors/<role>.pt and <dir>/axis.pt
                  (HF unrulyabstractions/assistant-axis-runs qwen2.5-7b/baseline, or ~/persona-em-data/hf/...)
     --org        one fine-tune's cloud: <dir>/vectors/<role>.pt (its own axis is not read)
     --cast       the frozen family cast, one role per line (prereg/cast_qwen2.5-7b.txt)
     --paper      the replication repo (breaking-character-iaseai): its analysis/figures code is imported
Out: JSON on stdout (or --out): the six readings and drift_size at the five depths, the roles used,
     the evil group, and the cast actually read.

The formulas are not re-typed here. Every reading comes from fig_metric_slices.stat_of, the function
the paper's interval bootstrap uses, called with every role of the cast (idx = all roles). The evil
group is features_graph.evil_group over the frozen 39-role cast, the same call geometry/ makes. The
depth rows are the 7B family's rows 3, 7, 14, 19, 24 (data/raw/graph_features.json, column layer).
check_readings_reproduce.py proves that this module returns data/readings.csv for the six pool
7B fine-tunes.

Importing fig_metric_slices creates <OUT_DIR>/figures/metrics and <OUT_DIR>/data/raw at import
(fig_metric_slices.py:34-35, _paths.py:27-33). OUT_DIR is therefore pointed at a sandbox under this
folder before the import, so nothing is written to the paper or the replication repo.
"""
import argparse, json, os, pathlib, sys

sys.dont_write_bytecode = True     # importing the paper's modules must not write into its tree

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SANDBOX = HERE.parent / "out" / ".paper_import_sandbox"
DEPTHS = ["10%", "25%", "50%", "70%", "90%"]
ROWS_7B = [3, 7, 14, 19, 24]
SIX = ["assistant_shift_deviation", "assistant_evil_slip", "orientation_distortion",
       "radial_distortion", "axial_contraction", "knn_retention"]
# the paper's reading name -> the bank feature stat_of computes (geometry/readings.py:24-31)
FEAT = {"assistant_shift_deviation": "REF_assistant_excess",
        "assistant_evil_slip": "assistant_minus_evil_excess",
        "orientation_distortion": "along_cos_sd",
        "radial_distortion": "radial_distortion",
        "axial_contraction": "axial_contraction",
        "knn_retention": "knn25pct_keep_mean"}


def paper_modules(paper):
    os.environ["OUT_DIR"] = str(SANDBOX)
    sys.path.insert(0, str(pathlib.Path(paper) / "analysis" / "figures"))
    import features_graph as FG      # noqa: E402
    import fig_metric_slices as FMS  # noqa: E402
    return FG, FMS


def load_vec(path, rows):
    """The paper's loader (geometry/build_cache.py:26-30), then the depth rows."""
    import torch
    o = torch.load(path, map_location="cpu", weights_only=False)
    v = o["vector"] if isinstance(o, dict) and "vector" in o else o
    return v.float().numpy()[rows].astype(np.float64)


def cloud(d, names, rows):
    """(len(rows), n, d) role vectors of one cloud."""
    return np.stack([load_vec(f"{d}/vectors/{n}.pt", rows) for n in names], axis=1)


def readings(base, org, cast, paper, rows=ROWS_7B, evil_override=None):
    FG, FMS = paper_modules(paper)
    have = {p.stem for p in pathlib.Path(org, "vectors").glob("*.pt")}
    names = [r for r in cast if r in have]                  # the cast actually read
    missing = [r for r in cast if r not in have]
    for anchor in ("assistant", "default"):
        if anchor not in names:
            raise SystemExit(f"{org}: the {anchor} role has no vector; no reading is defined")
    rate = FG.evil_rates()[0]
    fam_evil = [cast[i] for i in FG.evil_group(cast, rate)]  # fixed on the frozen cast
    evil_names = evil_override or fam_evil
    E = np.array([names.index(r) for r in evil_names if r in names], int)
    ai = names.index("assistant")
    X0 = cloud(base, names, rows)
    X1 = cloud(org, names, rows)
    A0 = load_vec(f"{base}/axis.pt", rows)
    out = {"cast_frozen": len(cast), "cast_read": len(names), "missing_roles": missing,
           "evil_group_family": fam_evil, "evil_group_read": [names[i] for i in E],
           "rows": list(rows), "depths": {}}
    n = len(names)
    k25 = FMS.k_quarter(n)
    for k, dep in enumerate(DEPTHS):
        V0, O = X0[k], X1[k]
        a = A0[k] / (np.linalg.norm(A0[k]) + 1e-12)
        D = O - V0
        mag = np.linalg.norm(D, axis=1)

        def knn(M, i, kk):
            dd = np.linalg.norm(M - M[i], axis=1)
            dd[i] = np.inf
            return set(np.argsort(dd)[:kk].tolist())

        # the primitives dict of fig_metric_slices.primitives (lines 184-197), built from the two clouds
        P = dict(c=(D / (mag[:, None] + 1e-12)) @ a, s=D @ a, mag=mag,
                 chi=np.array([len(knn(V0, i, 10) & knn(O, i, 10)) / 10.0 for i in range(n)]),
                 chi25=np.array([len(knn(V0, i, k25) & knn(O, i, k25)) / k25 for i in range(n)]),
                 q=np.linalg.norm(O - O.mean(0), axis=1) / (np.linalg.norm(V0 - V0.mean(0), axis=1) + 1e-12),
                 ai=ai, E=E, x=V0 @ a)
        idx, eidx = np.arange(n), np.arange(len(E))
        row = {nm: float(FMS.stat_of(FEAT[nm], P, idx, eidx)) for nm in SIX}
        row["drift_size"] = float(mag.mean())
        out["depths"][dep] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--org", required=True)
    ap.add_argument("--cast", required=True)
    ap.add_argument("--paper", default="/Users/unrulyabstractions/work/_OLD/breaking-character-iaseai")
    ap.add_argument("--out")
    a = ap.parse_args()
    cast = [l.strip() for l in open(a.cast) if l.strip()]
    res = readings(a.base, a.org, cast, a.paper)
    txt = json.dumps(res, indent=1)
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
