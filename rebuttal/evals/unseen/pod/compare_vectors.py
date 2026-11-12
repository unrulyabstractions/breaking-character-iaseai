#!/usr/bin/env python
"""Compare role vectors built here against the paper's vectors of the same cloud.

In:  --ours <dir>/vectors/<role>.pt, --ref <dir>/vectors/<role>.pt (e.g. HF assistant-axis-runs
     qwen2.5-7b/bad-legal/vectors), --roles, --rows (default the 7B depth rows 3 7 14 19 24)
Out: JSON (stdout and --out): per role and row the cosine and the relative L2 difference, and PASS
     when every cosine at the rows is at least --min-cos.
"""
import argparse, json, pathlib, sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ours", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--roles", nargs="+", required=True)
ap.add_argument("--rows", nargs="+", type=int, default=[3, 7, 14, 19, 24])
ap.add_argument("--min-cos", type=float, default=0.999)
ap.add_argument("--out")
a = ap.parse_args()


def vec(d, r):
    o = torch.load(f"{d}/{r}.pt", map_location="cpu", weights_only=False)
    return (o["vector"] if isinstance(o, dict) else o).float()


res, worst = {}, 1.0
for r in a.roles:
    x, y = vec(a.ours, r), vec(a.ref, r)
    assert x.shape == y.shape, (r, x.shape, y.shape)
    cos = torch.nn.functional.cosine_similarity(x, y, dim=1)
    rel = (x - y).norm(dim=1) / y.norm(dim=1)
    res[r] = {str(k): dict(cos=float(cos[k]), rel_l2=float(rel[k])) for k in a.rows}
    res[r]["all_layers_min_cos"] = float(cos.min())
    worst = min(worst, min(float(cos[k]) for k in a.rows))
out = dict(roles=res, min_cos_at_rows=worst, threshold=a.min_cos, verdict="PASS" if worst >= a.min_cos else "FAIL")
txt = json.dumps(out, indent=1)
if a.out:
    pathlib.Path(a.out).write_text(txt)
print(txt)
sys.exit(0 if out["verdict"] == "PASS" else 1)
