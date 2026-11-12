"""Independent check of the built control directions, and of loader compatibility, without dirs_lib.

For each control <fam>/<ctl>/directions.pt in this folder:
  1. recompute the family cast from the HF listing and compare it with the file's personas;
  2. rebuild every direction and act_norm in float64 with numpy from the hash-proven HF vectors, and compare;
  3. check structure: keys, shapes, dtypes, finite, row 0 zero, rand_unit bitwise equal to every family organism's;
  4. check META.json against the file (sha256, bytes, norms at L, act_norm at L);
  5. load the file with the 17 Sep steer_causal_multi.py --directions-file code and build the edits of every
     J1/J3 condition with its own make_edits: the applied norm must be alpha * act_norm[L], the sign negative.
For each J3 file made by add_act_norm.py (optional --j3-dir), step 5 as well, with the META value of act_norm[L].

usage: python -I verify_outputs.py --plan plan.json --listing hf_listing.json --manifest M1 M2
          --steer-tree <tree with scripts/steer_causal_multi.py and src/persona_em> [--j3-dir DIR] [--report out.json]
Exit 1 on any failure. Writes only --report.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
N_LAYERS = {"qwen2.5-7b": 28, "llama3.1-8b": 32, "qwen3.5-9b": 32, "gemma3-12b": 48, "qwen2.5-14b": 48,
            "mistral-24b": 40, "qwen2.5-32b": 64, "qwen3-32b": 64}
ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--listing", required=True)
ap.add_argument("--manifest", nargs="+", required=True)
ap.add_argument("--steer-tree", required=True)
ap.add_argument("--j3-dir")
ap.add_argument("--report")
a = ap.parse_args()

files = {}
for mp in a.manifest:
    files.update(json.load(open(mp))["files"])


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def hf(p):
    e = files[p]
    assert sha(e["local"]) == e["sha256"], p
    return e["local"]


def vec64(fam, who, role):
    o = torch.load(hf(f"{fam}/{who}/vectors/{role}.pt"), weights_only=False, map_location="cpu")
    return (o["vector"] if isinstance(o, dict) else o).float().numpy().astype(np.float64)


clouds = defaultdict(set)
for p, _s in json.load(open(a.listing)):
    q = p.split("/")
    if len(q) == 4 and q[2] == "vectors" and p.endswith(".pt"):
        clouds[(q[0], q[1])].add(q[3][:-3])

sys.path.insert(0, os.path.join(a.steer_tree, "scripts"))
sys.path.insert(0, os.path.join(a.steer_tree, "src"))
import steer_causal_multi as S  # noqa: E402

COND = ["neg_global", "random_ctrl", "neg_excess_asst", "neg_excess_def"]
ALPHAS = [0.15, 0.3, 0.45]


def loader_check(path, L):
    """steer_causal_multi.py lines 560-566 verbatim, then make_edits for every condition."""
    loaded = torch.load(path, weights_only=False, map_location="cpu")
    directions = {k: v.float() for k, v in loaded.items()
                  if torch.is_tensor(v) and v.ndim == 2 and k != "act_norm"}
    if "rand_unit" not in directions:
        directions["rand_unit"] = S.random_axis(directions["sbar"], 0)
    act_norm = loaded["act_norm"].float()
    out, ok = {}, True
    for c in COND:
        for x in ALPHAS:
            e = S.make_edits(c, directions, L, x, "last", "actfrac", act_norm=act_norm)[0]
            applied = abs(e.coeff) * float(e.vector.norm())
            good = (e.layer == L and e.coeff < 0 and e.positions == "last"
                    and math.isclose(applied, x * float(act_norm[L]), rel_tol=1e-5))
            ok &= good
            out[f"{c}@{x:g}"] = applied
    return ok, out, sorted(directions)


plan = json.load(open(a.plan))
report, fails = [], []
for c in plan["controls"]:
    fam, ctl = c["family"], c["control"]
    od = os.path.join(HERE, fam, ctl)
    pt, mj = os.path.join(od, "directions.pt"), os.path.join(od, "META.json")
    d = torch.load(pt, weights_only=True, map_location="cpu")
    meta = json.load(open(mj))
    nl = N_LAYERS[fam]
    L = round(0.5 * nl)
    chk = {}
    fam_clouds = [w for (f, w) in clouds if f == fam]
    fcast = set.intersection(*[clouds[(fam, w)] for w in fam_clouds]) - {"default"}
    chk["personas_equal_family_cast"] = set(d["personas"]) == fcast and len(d["personas"]) == len(fcast)
    cast = list(d["personas"])
    base = {r: vec64(fam, "baseline", r) for r in cast + ["default"]}
    org = {r: vec64(fam, ctl, r) for r in cast + ["default"]}
    z = np.zeros((1, base["assistant"].shape[1]))
    sbar = np.vstack([z, np.mean([org[r] - base[r] for r in cast], axis=0)])
    ref = {"sbar": sbar,
           "excess_assistant": np.vstack([z, org["assistant"] - base["assistant"]]) - sbar,
           "excess_default": np.vstack([z, org["default"] - base["default"]]) - sbar,
           "default_to_assistant": np.vstack([z, org["assistant"] - org["default"]])}
    act = np.concatenate([[0.0], np.mean([np.linalg.norm(base[r], axis=1) for r in cast], axis=0)])
    agree = {}
    for k, v in ref.items():
        m = d[k].numpy().astype(np.float64)
        cs = [float(m[i] @ v[i] / (np.linalg.norm(m[i]) * np.linalg.norm(v[i]))) for i in range(1, nl + 1)]
        rel = [float(np.linalg.norm(m[i] - v[i]) / np.linalg.norm(v[i])) for i in range(1, nl + 1)]
        agree[k] = {"min_cos": min(cs), "max_rel": max(rel), "cos_at_L": cs[L - 1]}
        chk[f"{k}_matches_float64_rebuild"] = min(cs) > 1 - 1e-9 and max(rel) < 1e-5
    an = d["act_norm"].numpy().astype(np.float64)
    agree["act_norm"] = {"max_rel": float(np.max(np.abs(an[1:] - act[1:]) / act[1:])), "row0": float(an[0])}
    chk["act_norm_matches_float64_rebuild"] = agree["act_norm"]["max_rel"] < 1e-6 and an[0] == 0.0
    shape = (nl + 1, base["assistant"].shape[1])
    chk["keys"] = sorted(k for k in d if torch.is_tensor(d[k])) == sorted(
        ["sbar", "excess_assistant", "excess_default", "default_to_assistant", "rand_unit", "act_norm"])
    chk["shapes_dtypes"] = all(tuple(d[k].shape) == shape and d[k].dtype == torch.float32 for k in ref) and \
        tuple(d["rand_unit"].shape) == shape and tuple(d["act_norm"].shape) == (nl + 1,)
    chk["finite"] = all(bool(torch.isfinite(d[k]).all()) for k in list(ref) + ["rand_unit", "act_norm"])
    chk["row0_zero"] = all(float(d[k][0].abs().max()) == 0.0 for k in ref)
    org_ru = [torch.load(hf(f"steering/{fam}/{o}/directions.pt"), weights_only=False, map_location="cpu")["rand_unit"]
              for o in plan["j3_organisms"][fam]]
    chk["rand_unit_bitwise_equal_to_family_organisms"] = all(torch.equal(d["rand_unit"], r) for r in org_ru)
    chk["meta_sha256"] = meta["output"]["sha256"] == sha(pt) and meta["output"]["bytes"] == os.path.getsize(pt)
    chk["meta_norms"] = all(math.isclose(meta["norms_at_L"][k], float(d[k][L].norm()), rel_tol=1e-7)
                            for k in list(ref) + ["rand_unit"])
    chk["meta_act_norm_L"] = math.isclose(meta["act_norm"]["value_at_L"], float(d["act_norm"][L]), rel_tol=1e-7)
    chk["meta_L"] = meta["persona_layer"]["L"] == L and meta["cast"]["n_roles"] == len(cast)
    ok_l, applied, kept = loader_check(pt, L)
    chk["loader_and_make_edits"] = ok_l
    chk = {k: bool(v) for k, v in chk.items()}
    report.append({"kind": "control", "family": fam, "control": ctl, "L": L, "cast_n": len(cast), "checks": chk,
                   "agreement": agree, "applied_norms": applied, "loader_kept_keys": kept})
    bad = [k for k, v in chk.items() if not v]
    fails += [(fam, ctl, b) for b in bad]
    print(f"{fam}/{ctl}: L={L} cast={len(cast)} min_cos(sbar)={agree['sbar']['min_cos']:.12f} "
          f"max_rel(exA)={agree['excess_assistant']['max_rel']:.1e} act_norm max_rel={agree['act_norm']['max_rel']:.1e} "
          f"applied@0.3={applied['neg_global@0.3']:.4f} checks={'ALL PASS' if not bad else bad}")

if a.j3_dir:
    for fam, orgs in plan["j3_organisms"].items():
        for o in orgs:
            p = os.path.join(a.j3_dir, fam, o, "directions.pt")
            m = json.load(open(hf(f"steering/{fam}/{o}/META.json")))
            L = int(m["depths"]["0.5"])
            src = torch.load(hf(f"steering/{fam}/{o}/directions.pt"), weights_only=False, map_location="cpu")
            new = torch.load(p, weights_only=False, map_location="cpu")
            same = all(torch.equal(src[k], new[k]) for k in ("sbar", "excess_assistant", "excess_default",
                                                              "default_to_assistant", "rand_unit"))
            act_ok = float(new["act_norm"][L]) == float(np.float32(m["activation_norm_by_layer"][str(L)]))
            ok_l, applied, kept = loader_check(p, L)
            good = same and act_ok and ok_l
            report.append({"kind": "j3", "family": fam, "organism": o, "L": L, "tensors_bitwise_equal_to_hf": bool(same),
                           "act_norm_L_equals_META": bool(act_ok), "loader_and_make_edits": bool(ok_l),
                           "applied_norms": applied})
            if not good:
                fails.append((fam, o, "j3"))
            print(f"J3 {fam}/{o}: L={L} tensors==HF {same} act_norm[L]==META {act_ok} loader+make_edits {ok_l} "
                  f"applied neg_excess_asst@0.45={applied['neg_excess_asst@0.45']:.4g}")
if a.report:
    json.dump(report, open(a.report, "w"), indent=1)
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
