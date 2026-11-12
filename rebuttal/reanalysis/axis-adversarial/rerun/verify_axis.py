#!/usr/bin/env python
"""Independent check of axis_substitution.py on a sample, without core.readings or its axis helper.

It reads the .pt files directly with torch (paths from analysis/figures/_vectors.FAM, the layer row
from data/raw/graph_features.json, the evil roles from the paper's tables/evil_roles.tex), rebuilds
the default and assistant axes as 5_axis.py does on whole tensors (torch.stack(bf16).mean(0)), and
recomputes:
  - cos(default axis, assistant axis) for two baselines and one organism,
  - xi_A and delta_A with the assistant axis for three organisms,
  - the pooled mean within-family Spearman and Pearson (no p) of xi_A and delta_A against U, M, G
    from axis_readings.csv, with pandas,
and compares each with the value axis_substitution.py wrote.

Run: cd <this folder> && PYTHONDONTWRITEBYTECODE=1 uv run --no-project --quiet --with numpy --with scipy \
       --with pandas --with torch python verify_axis.py
Out: verify_axis.json beside this file.
"""
import csv
import json
import pathlib
import re
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path("/Users/unrulyabstractions/work/_OLD/breaking-character-iaseai")
PAPER = pathlib.Path("/Users/unrulyabstractions/work/papers/breaking-character/src")
sys.path.insert(0, str(REPO / "analysis/figures"))
from _vectors import FAM  # noqa: E402

ROWS = {}
for r in json.load(open(REPO / "data/raw/graph_features.json"))["rows"]:
    ROWS[(r["f"], r["dep"])] = int(r["layer"])


def load(d, role):
    o = torch.load(pathlib.Path(d) / "vectors" / f"{role}.pt", map_location="cpu", weights_only=False)
    return o["vector"] if isinstance(o, dict) else o


def roles(d):
    return sorted(p.stem for p in (pathlib.Path(d) / "vectors").glob("*.pt"))


def axes(d, row, f32round):
    """(default axis, assistant axis) rows, float64, computed on whole bf16 tensors like 5_axis.py."""
    names = roles(d)
    R = torch.stack([load(d, n) for n in names if "default" not in n])
    D = torch.stack([load(d, "default")])
    A = torch.stack([load(d, "assistant")])
    if f32round:
        vD = (D.float().mean(0) - R.float().mean(0)).to(torch.bfloat16)
        vA = (A.float().mean(0) - R.float().mean(0)).to(torch.bfloat16)
    else:
        vD = D.mean(0) - R.mean(0)
        vA = A.mean(0) - R.mean(0)
    axis_file = torch.load(pathlib.Path(d) / "axis.pt", map_location="cpu", weights_only=False)
    axis_file = axis_file["vector"] if isinstance(axis_file, dict) else axis_file
    # the same formula in float64 (what axis_cosines.csv reports)
    R64 = R.double()[:, row].mean(0)
    vD64 = (D.double()[:, row].mean(0) - R64).numpy()
    vA64 = (A.double()[:, row].mean(0) - R64).numpy()
    return (vD.float()[row].double().numpy(), vA.float()[row].double().numpy(),
            bool(torch.equal(vD[row].float(), axis_file.float()[row])), vD64, vA64)


def cosine(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def evil(family):
    for line in (PAPER / "tables/evil_roles.tex").read_text().splitlines():
        m = re.match(r"\\mdl\{([^}]*)\}\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(.*?)\\\\", line.strip())
        if m and m.group(1) == family:
            return [re.sub(r"\s*\.\d+$", "", x.strip()) for x in m.group(4).split(",")]
    raise KeyError(family)


def cast(family):
    cfg = FAM[family]
    s = set(roles(cfg["base"]))
    for d in cfg["orgs"].values():
        s &= set(roles(d))
    return sorted(s)


def xi_delta(family, organism, vA, row):
    names = cast(family)
    cfg = FAM[family]
    XB = np.stack([load(cfg["base"], n).float()[row].double().numpy() for n in names])
    XO = np.stack([load(cfg["orgs"][organism], n).float()[row].double().numpy() for n in names])
    v = vA / np.linalg.norm(vA)
    S = XO - XB
    a = (S @ v) / np.linalg.norm(S, axis=1)
    ai = names.index("assistant")
    E = [names.index(e) for e in evil(family)]
    return float(a[ai] - np.median(a)), float(a[ai] - a[E].mean())


def main():
    cos_out = {(r["family"], r["cloud"]): r for r in csv.DictReader(open(HERE / "axis_cosines.csv"))}
    read_out = {(r["family"], r["organism"]): r for r in csv.DictReader(open(HERE / "axis_readings.csv"))}
    checks = json.load(open(HERE / "axis_substitution.json"))["axis_checks"]
    res = dict(script=str(pathlib.Path(__file__).resolve()), cosines=[], readings=[], pooled=[])
    cache = {}
    for f, o, f32 in [("Qwen2.5-7B", "base", False), ("Qwen3.5-9B", "base", True), ("Qwen2.5-32B", "mismatch-GOOD-medical", False)]:
        row = ROWS[(f, "50%")]
        d = FAM[f]["base"] if o == "base" else FAM[f]["orgs"][o]
        vD, vA, bit, vD64, vA64 = axes(d, row, f32)
        if o == "base":
            cache[f] = vA
        c64 = cosine(vD64, vA64)                       # float64 formula, against axis_cosines.csv
        want = float(cos_out[(f, o)]["cos_D_A_cloud"])
        cp = cosine(vD, vA)                            # pipeline arithmetic, against axis_checks (baselines)
        want_p = checks[f]["cos_D_A_pipeline"] if o == "base" else None
        res["cosines"].append(dict(family=f, cloud=o, cos_D_A_float64_independent=c64, cos_D_A_float64_axis_substitution=want,
                                   abs_diff=abs(c64 - want), cos_D_A_pipeline_independent=cp,
                                   cos_D_A_pipeline_axis_substitution=want_p,
                                   abs_diff_pipeline=None if want_p is None else abs(cp - want_p),
                                   default_axis_equals_axis_pt=bit))
        print(res["cosines"][-1], flush=True)
    for f, o in [("Qwen2.5-7B", "finance"), ("Qwen3.5-9B", "bad-medical"), ("Qwen2.5-7B", "GOOD-medical (control)")]:
        row = ROWS[(f, "50%")]
        xi, de = xi_delta(f, o, cache[f], row)
        w = read_out[(f, o)]
        res["readings"].append(dict(family=f, organism=o,
                                    xi_A_independent=xi, xi_A_axis_substitution=float(w["assistant_shift_deviation|assistant_axis"]),
                                    delta_A_independent=de, delta_A_axis_substitution=float(w["assistant_evil_slip|assistant_axis"]),
                                    abs_diff_xi=abs(xi - float(w["assistant_shift_deviation|assistant_axis"])),
                                    abs_diff_delta=abs(de - float(w["assistant_evil_slip|assistant_axis"]))))
        print(res["readings"][-1], flush=True)
    # pooled coefficients from the CSVs with pandas
    df = pd.read_csv(HERE / "axis_readings.csv")
    meas = pd.read_csv(REPO / "data/measures.csv")
    df = df.merge(meas[["family", "organism", "U", "M", "G"]], on=["family", "organism"], how="left")
    pool_csv = {(r["reading"], r["measure"], r["variant"]): r for r in csv.DictReader(open(HERE / "axis_pooled.csv"))}
    for k in ["assistant_shift_deviation", "assistant_evil_slip", "orientation_distortion", "axial_contraction"]:
        for m in ["U", "M", "G"]:
            for v in ["published", "assistant_axis"]:
                col = f"{k}|{v}"
                g = df.groupby("family")
                rho = float(np.mean([grp[col].corr(grp[m], method="spearman") for _, grp in g]))
                r = float(np.mean([grp[col].corr(grp[m], method="pearson") for _, grp in g]))
                w = pool_csv[(k, m, v)]
                res["pooled"].append(dict(reading=k, measure=m, variant=v, rho_bar_independent=round(rho, 4),
                                          rho_bar_axis_substitution=float(w["rho_bar"]), r_bar_independent=round(r, 4),
                                          r_bar_axis_substitution=float(w["r_bar"]),
                                          match=abs(round(rho, 4) - float(w["rho_bar"])) < 1e-9 and abs(round(r, 4) - float(w["r_bar"])) < 1e-9))
    res["n_pool_rows"] = int(len(df))
    res["all_cos_within_1e-9"] = all(c["abs_diff"] < 1e-9 and (c["abs_diff_pipeline"] is None or c["abs_diff_pipeline"] < 1e-9)
                                     for c in res["cosines"])
    res["all_readings_within_1e-9"] = all(c["abs_diff_xi"] < 1e-9 and c["abs_diff_delta"] < 1e-9 for c in res["readings"])
    res["all_pooled_match"] = all(c["match"] for c in res["pooled"])
    res["all_default_axes_equal_axis_pt"] = all(c["default_axis_equals_axis_pt"] for c in res["cosines"])
    (HERE / "verify_axis.json").write_text(json.dumps(res, indent=1))
    print({k: res[k] for k in res if k.startswith("all_") or k == "n_pool_rows"})


if __name__ == "__main__":
    main()
