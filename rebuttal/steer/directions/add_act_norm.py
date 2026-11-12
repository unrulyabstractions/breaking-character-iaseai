"""Make an HF steering directions.pt loadable by steer_causal_multi.py --directions-file (CPU, under a second).

The HF files steering/<fam>/<org>/directions.pt hold sbar, excess_assistant, excess_default, default_to_assistant
and rand_unit, but not the act_norm the loader requires (KeyError without it). The published battery used
act_norm only at its steer layers, and META.json records exactly those values (activation_norm_by_layer). This
script copies every tensor bit for bit and adds act_norm with the META value at each recorded layer and NaN at
every other row, so a steer at an unrecorded depth produces NaN and fails loudly instead of silently using a
made-up scale.

usage: python -I add_act_norm.py <hf directions.pt> <hf META.json> <out directions.pt> [--sha256 <expected of input>]
Writes only <out>. Exit 1 on any mismatch.
"""
import argparse
import hashlib
import json
import math
import sys

import torch

KEYS = ("sbar", "excess_assistant", "excess_default", "default_to_assistant", "rand_unit")

ap = argparse.ArgumentParser()
ap.add_argument("dirs")
ap.add_argument("meta")
ap.add_argument("out")
ap.add_argument("--sha256")
a = ap.parse_args()
if a.sha256:
    h = hashlib.sha256(open(a.dirs, "rb").read()).hexdigest()
    if h != a.sha256:
        sys.exit(f"input sha256 {h} != expected {a.sha256}")
d = torch.load(a.dirs, weights_only=False, map_location="cpu")
m = json.load(open(a.meta))
missing = [k for k in KEYS if k not in d]
if missing:
    sys.exit(f"{a.dirs} lacks {missing}")
n_rows = d["sbar"].shape[0]
if n_rows != int(m["n_layers"]) + 1:
    sys.exit(f"rows {n_rows} != META n_layers+1 = {int(m['n_layers']) + 1}")
act = torch.full((n_rows,), float("nan"), dtype=torch.float32)
for Ls, v in m["activation_norm_by_layer"].items():
    act[int(Ls)] = float(v)
# the file's norms must equal the run's META, or this is not the file the battery steered with
for k, by_layer in m["norms_by_direction_by_layer"].items():
    for Ls, v in by_layer.items():
        got = float(d[k][int(Ls)].float().norm())
        if not math.isclose(got, v, rel_tol=1e-6):
            sys.exit(f"{k}[{Ls}] norm {got} != META {v}")
out = {k: v for k, v in d.items() if torch.is_tensor(v)}
out["act_norm"] = act
out["meta"] = dict(d.get("meta", {}), act_norm_source=f"META.json activation_norm_by_layer of {a.meta}; NaN elsewhere")
torch.save(out, a.out)
back = torch.load(a.out, weights_only=False, map_location="cpu")
for k in KEYS:
    assert torch.equal(back[k], d[k]), k
L = int(m["depths"]["0.5"])
assert math.isfinite(float(back["act_norm"][L])) and float(back["act_norm"][L]) > 0
print(f"{a.out}: keys {sorted(k for k in back if torch.is_tensor(back[k]))}; act_norm[{L}]={float(back['act_norm'][L]):.6f}")
