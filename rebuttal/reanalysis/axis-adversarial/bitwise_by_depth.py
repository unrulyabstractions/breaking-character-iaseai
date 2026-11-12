#!/usr/bin/env python
"""Per-depth bitwise check of the pipeline-arithmetic default axis against each baseline's axis.pt
(the axis analysis checked the 50% row only). Also cos(pipeline axis, axis.pt) and cos(axis.pt, float64 axis).
Out: bitwise_by_depth.json beside this file."""
import json, pathlib, sys
sys.dont_write_bytecode = True
import numpy as np, torch
HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path("/Users/unrulyabstractions/work/_OLD/breaking-character-iaseai")
sys.path.insert(0, str(REPO / "analysis/figures"))
from _vectors import FAM
from concurrent.futures import ThreadPoolExecutor
L = {}
for r in json.load(open(REPO / "data/raw/graph_features.json"))["rows"]:
    L[(r["f"], r["dep"])] = int(r["layer"])
DEP = ["10%", "25%", "50%", "70%", "90%"]
def ld(p):
    o = torch.load(p, map_location="cpu", weights_only=False)
    return o["vector"] if isinstance(o, dict) else o
out = {}
for f in FAM:
    d = pathlib.Path(FAM[f]["base"])
    names = sorted(p.stem for p in (d / "vectors").glob("*.pt"))
    with ThreadPoolExecutor(16) as ex:
        V = dict(zip(names, ex.map(lambda n: ld(d / "vectors" / f"{n}.pt"), names)))
    D = torch.stack([V["default"]]); R = torch.stack([V[n] for n in names if "default" not in n])
    bf = D.mean(0) - R.mean(0)
    f32 = (D.float().mean(0) - R.float().mean(0)).to(torch.bfloat16)
    f64 = D.double().mean(0) - R.double().mean(0)
    A = ld(d / "axis.pt")
    res = {}
    for dep in DEP:
        i = L[(f, dep)]
        c = lambda a, b: float((a.double() @ b.double()) / a.double().norm() / b.double().norm())
        res[dep] = dict(row=i, bitwise_bf16=bool(torch.equal(bf[i], A[i])), bitwise_f32_bf16=bool(torch.equal(f32[i], A[i])),
                        n_elem_diff_bf16=int((bf[i] != A[i]).sum()), n_elem_diff_f32bf16=int((f32[i] != A[i]).sum()), d=int(A.shape[1]),
                        cos_bf16_file=c(bf[i], A[i]), cos_f32bf16_file=c(f32[i], A[i]), cos_file_f64=c(A[i], f64[i]))
    out[f] = dict(n_layers=int(A.shape[0]), dtype=str(A.dtype), n_roles=int(R.shape[0]), depths=res,
                  bitwise_whole_tensor_bf16=bool(torch.equal(bf, A)), bitwise_whole_tensor_f32bf16=bool(torch.equal(f32, A)))
    print(f, json.dumps({k: (v["bitwise_bf16"], v["bitwise_f32_bf16"], v["n_elem_diff_bf16"], round(v["cos_bf16_file"], 7), round(v["cos_file_f64"], 6)) for k, v in res.items()}), flush=True)
(HERE / "bitwise_by_depth.json").write_text(json.dumps(out, indent=1))
