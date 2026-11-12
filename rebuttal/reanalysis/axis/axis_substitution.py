#!/usr/bin/env python
"""Reviewer r7X4 (minor point): the default axis in place of the assistant axis.

The paper (background.tex, Eq. 1) computes the axis v = r_default - mean_roles r with the DEFAULT role
as the anchor. This script asks how far that choice matters.

  1. Which role is A in Eqs. 7-8 (xi_A, delta_A): the code line that picks it, and the readings with
     A = assistant against A = default, each compared to data/readings.csv.
  2. For every baseline and every organism of all eight families, at the persona layer (50% depth):
       cos(default axis, assistant axis) of the same cloud, where
         default axis   v_D = r_default   - mean_{r not default} r
         assistant axis v_A = r_assistant - mean_{r not default} r     (same mean, only the anchor changes)
       on the cloud's whole role set (the pipeline's n, as 5_axis.py) and on the family cast;
       and cos(baseline default axis, organism default axis) (and the same for the assistant axis).
  3. The six readings of the 34 pool organisms recomputed with the baseline ASSISTANT axis in place of
     the baseline default axis, and the pooled within-family permutation test (core.pooled, the port of
     analysis/pooled_test.py) against U, M and G, side by side with the published coefficients.

The assistant axis is computed in the same arithmetic the pipeline used for the default axis
(bfloat16 for four families, float32 rounded to bfloat16 for Qwen3.5-9B), so the comparison isolates
the anchor. The helper that does it is checked to give the baseline's axis.pt bit for bit when the
anchor is the default. A float64 version of both axes is reported as a sensitivity check.

Run:
  cd <this folder> && PYTHONDONTWRITEBYTECODE=1 uv run --no-project --quiet --with numpy --with scipy \
      --with pandas --with torch python axis_substitution.py
Out (beside this file): axis_substitution.json, axis_cosines.csv, axis_readings.csv, axis_pooled.csv.
Reads the vectors fresh from the .pt files of the local store (core.STORE); its row cache goes to
the scratchpad folder axis_cache, never to core_cache.
"""
import csv
import json
import pathlib
import re
import sys
import time

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "core"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

import core  # noqa: E402

core.CACHE = core.SCRATCH / "axis_cache"          # own cache: every row here is read from the .pt files
DEP = core.PERSONA_DEPTH
MEASURES = ["U", "M", "G"]
OUT_JSON = HERE / "axis_substitution.json"
OUT_COS = HERE / "axis_cosines.csv"
OUT_READ = HERE / "axis_readings.csv"
OUT_POOL = HERE / "axis_pooled.csv"
TOPIC_LOCKED = ("Qwen2.5-32B", "mismatch-GOOD-medical")   # _organisms.py renames it "topic-locked"


def cos(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def pipeline_axis(anchor_rows, role_rows, mode):
    """anchor_mean - role_mean in the pipeline's arithmetic (as core.default_axis, how='cloud').

    anchor_rows, role_rows: float32 arrays holding bfloat16 values exactly (as loaded).
    mode 'bf16': torch.stack(bf16).mean(0) differenced in bfloat16 (5_axis.py);
    mode 'float32->bf16': float32 means differenced, then rounded to bfloat16 once;
    mode 'float64': the same formula in float64.
    """
    if mode == "float64":
        return np.asarray(anchor_rows, np.float64).mean(0) - np.asarray(role_rows, np.float64).mean(0)
    Ab = torch.from_numpy(np.ascontiguousarray(anchor_rows, np.float32)).to(torch.bfloat16)
    Rb = torch.from_numpy(np.ascontiguousarray(role_rows, np.float32)).to(torch.bfloat16)
    if mode == "bf16":
        ax = Ab.mean(0) - Rb.mean(0)
    elif mode == "float32->bf16":
        ax = (Ab.float().mean(0) - Rb.float().mean(0)).to(torch.bfloat16)
    else:
        raise ValueError(mode)
    return ax.float().numpy().astype(np.float64)


def code_evidence():
    """The lines of the repository code that fix role A and the axis the readings use."""
    out = {}
    fg = (core.REPO / "analysis/figures/features_graph.py").read_text().splitlines()
    out["features_graph.py"] = [f"L{i + 1}: {l.strip()}" for i, l in enumerate(fg)
                                if 'names.index("assistant")' in l or "X0, A0 = zb[" in l
                                or "evil_readings(X0[k], zo[\"X\"][k], A0[k]" in l
                                or l.strip().startswith('F["assistant_minus_evil_excess"]')
                                or l.strip().startswith('F["_check_REF_assistant_excess"]')]
    rd = (core.REPO / "geometry/readings.py").read_text().splitlines()
    out["geometry/readings.py"] = [f"L{i + 1}: {l.strip()}" for i, l in enumerate(rd)
                                   if '("assistant_shift_deviation"' in l or '("assistant_evil_slip"' in l]
    bc = (core.REPO / "geometry/build_cache.py").read_text().splitlines()
    out["geometry/build_cache.py"] = [f"L{i + 1}: {l.strip()}" for i, l in enumerate(bc) if "axis.pt" in l and "np.savez" in l]
    ax = (core.REPO / "third_party/assistant-axis/pipeline/5_axis.py").read_text().splitlines()
    out["third_party/assistant-axis/pipeline/5_axis.py"] = [f"L{i + 1}: {l.strip()}" for i, l in enumerate(ax)
                                                            if 'if "default" in role' in l or "axis = default_mean - role_mean" in l]
    vec = (core.REPO / "third_party/assistant-axis/pipeline/4_vectors.py").read_text().splitlines()
    out["third_party/assistant-axis/pipeline/4_vectors.py"] = [f"L{i + 1}: {l.strip()}" for i, l in enumerate(vec)
                                                               if 'vector_type = "' in l]
    ins = {}
    for r in ("assistant", "default"):
        j = json.load(open(core.REPO / f"third_party/assistant-axis/data/roles/instructions/{r}.json"))
        ins[r] = [d["pos"] for d in j["instruction"]]
    out["role_instructions"] = ins
    return out


def main():
    t0 = time.time()
    res = dict(script=str(pathlib.Path(__file__).resolve()), core=str(HERE.parent / "core/core.py"),
               store=str(core.STORE), cache=str(core.CACHE), depth=DEP)
    res["code_evidence"] = code_evidence()
    paper = core.load_paper_readings()
    meas = core.load_measures()
    pool = core.pool_organisms()
    pool_set = set(pool)
    pool_fams = sorted({f for f, _ in pool})
    ref = json.load(open(core.REPO / "data/pooled_stats.json"))
    if len(pool) != ref["n_organisms"] or pool_fams != sorted(ref["families"]):
        raise SystemExit("pool differs from pooled_stats.json")

    HF = {p: s for p, s in json.load(open(core.HF_LISTING))}
    cos_rows, read_rows = [], []
    rec = {v: {} for v in ["default_axis", "assistant_axis", "default_axis_f64", "assistant_axis_f64",
                           "assistant_axis_mean_excl_assistant", "A_is_default"]}
    axis_checks = {}
    for f in core.FAM:
        row = core.ROW[f][DEP]
        cast = core.family_cast(f)
        mode = core.AXIS_PRECISION.get(f, "bf16")
        clouds = {}
        for o in ["base"] + core.organisms(f):
            d = core.cloud_dir(f, o)
            allr = core.role_names(f, o)
            dn = [n for n in allr if "default" in n]
            if dn != ["default"]:
                raise SystemExit(f"{f}|{o}: default roles {dn}")
            if "assistant" not in allr:
                raise SystemExit(f"{f}|{o}: no assistant.pt")
            X = core.load_rows(d, allr, row)                      # (n_files, d) float32
            di, ai = allr.index("default"), allr.index("assistant")
            nd = [i for i in range(len(allr)) if i != di]          # every non-default role (assistant included)
            nda = [i for i in nd if i != ai]                        # ... and without the assistant
            vD = pipeline_axis(X[[di]], X[nd], "float64")
            vA = pipeline_axis(X[[ai]], X[nd], "float64")
            vAx = pipeline_axis(X[[ai]], X[nda], "float64")
            ci = [allr.index(n) for n in cast]
            cdi, cai = cast.index("default"), cast.index("assistant")
            Xc = X[ci].astype(np.float64)
            mc = np.delete(Xc, cdi, axis=0).mean(0)
            vDc, vAc = Xc[cdi] - mc, Xc[cai] - mc
            fileax = core._vec_row(d / "axis.pt", row).astype(np.float64)
            match = {m: bool(np.array_equal(pipeline_axis(X[[di]], X[nd], m), fileax)) for m in ("bf16", "float32->bf16")}
            clouds[o] = dict(X=X, allr=allr, vD=vD, vA=vA, vDc=vDc, vAc=vAc, file=fileax)
            r = dict(family=f, cloud=o, kind="baseline" if o == "base" else "organism",
                     in_pool=(f, o) in pool_set, in_readings_csv=(f, o) in paper,
                     tuning=paper[(f, o)]["tuning"] if (f, o) in paper else "",
                     control=meas[(f, o)]["control"] if (f, o) in meas else "",
                     topic_locked=(f, o) == TOPIC_LOCKED, layer_row=row,
                     n_roles_cloud=len(nd), n_cast=len(cast),
                     cos_D_A_cloud=cos(vD, vA), cos_D_A_cloud_mean_excl_assistant=cos(vD, vAx),
                     cos_D_A_cast=cos(vDc, vAc),
                     cos_D_axisfile=cos(vD, fileax),
                     axisfile_bitwise_bf16=match["bf16"], axisfile_bitwise_f32bf16=match["float32->bf16"],
                     norm_ratio_A_over_D_cloud=float(np.linalg.norm(vA) / np.linalg.norm(vD)),
                     norm_gap_rA_minus_rD_over_D_cloud=float(np.linalg.norm(X[ai].astype(np.float64) - X[di]) / np.linalg.norm(vD)))
            # the same files on the HF dataset (listing of unrulyabstractions/assistant-axis-runs), where mapped
            if f in core.HF_RUN and o in core.HF_RUN[f]:
                for role, local in (("assistant", d / "vectors/assistant.pt"), ("default", d / "vectors/default.pt"),
                                    (None, d / "axis.pt")):
                    hp = core.hf_path(f, o, role)
                    tag = role or "axis"
                    r[f"hf_{tag}_listed"] = hp in HF
                    r[f"hf_{tag}_size_equal"] = HF.get(hp) == local.stat().st_size
            else:
                for tag in ("assistant", "default", "axis"):
                    r[f"hf_{tag}_listed"] = r[f"hf_{tag}_size_equal"] = ""
            cos_rows.append(r)
        b = clouds["base"]
        for r in cos_rows:
            if r["family"] != f:
                continue
            o = r["cloud"]
            c = clouds[o]
            r["cos_baseD_orgD_cloud"] = cos(b["vD"], c["vD"])
            r["cos_baseD_orgD_axisfile"] = cos(b["file"], c["file"])
            r["cos_baseD_orgD_cast"] = cos(b["vDc"], c["vDc"])
            r["cos_baseA_orgA_cloud"] = cos(b["vA"], c["vA"])
            r["cos_baseA_orgA_cast"] = cos(b["vAc"], c["vAc"])
            r["cos_baseD_orgA_cloud"] = cos(b["vD"], c["vA"])
        print(f"{f:13s} {len(clouds)} clouds  baseline cos(D,A) cloud={cos(b['vD'], b['vA']):.4f} "
              f"cast={cos(b['vDc'], b['vAc']):.4f}", flush=True)

        if f not in pool_fams:
            del clouds
            continue
        # ---- the baseline axes the readings use, in the pipeline's arithmetic
        allr, X = b["allr"], b["X"]
        di, ai = allr.index("default"), allr.index("assistant")
        nd = [i for i in range(len(allr)) if i != di]
        nda = [i for i in nd if i != ai]
        D_pipe = pipeline_axis(X[[di]], X[nd], mode)
        A_pipe = pipeline_axis(X[[ai]], X[nd], mode)
        Ax_pipe = pipeline_axis(X[[ai]], X[nda], mode)
        D_f64 = pipeline_axis(X[[di]], X[nd], "float64")
        A_f64 = pipeline_axis(X[[ai]], X[nd], "float64")
        D_core = core.default_axis(f, DEP, "cloud", "pipeline")
        axis_checks[f] = dict(mode=mode,
                              helper_default_equals_axis_pt=bool(np.array_equal(D_pipe, b["file"])),
                              helper_default_equals_core_default_axis=bool(np.array_equal(D_pipe, D_core)),
                              cos_D_A_pipeline=cos(D_pipe, A_pipe), cos_D_A_float64=cos(D_f64, A_f64),
                              cos_A_pipeline_vs_A_float64=cos(A_pipe, A_f64),
                              cos_A_pipeline_vs_A_mean_excl=cos(A_pipe, Ax_pipe))
        if not (axis_checks[f]["helper_default_equals_axis_pt"] and axis_checks[f]["helper_default_equals_core_default_axis"]):
            raise SystemExit(f"{f}: the axis helper does not reproduce axis.pt / core.default_axis")
        axes = {"default_axis": D_pipe, "assistant_axis": A_pipe, "default_axis_f64": D_f64,
                "assistant_axis_f64": A_f64, "assistant_axis_mean_excl_assistant": Ax_pipe}
        evil, _ = core.evil_roles(f)
        bi = [allr.index(n) for n in cast]
        XB = X[bi]
        for o in [o for g, o in pool if g == f]:
            c = clouds[o]
            XO = c["X"][[c["allr"].index(n) for n in cast]]
            row_out = dict(family=f, organism=o)
            for v, ax in axes.items():
                R = core.readings(XB, XO, ax, cast, evil, assistant="assistant")
                rec[v][(f, o)] = R
                for k in core.READINGS:
                    row_out[f"{k}|{v}"] = R[k]
            R = core.readings(XB, XO, D_pipe, cast, evil, assistant="default")
            rec["A_is_default"][(f, o)] = R
            for k in core.READINGS:
                row_out[f"{k}|published"] = float(paper[(f, o)][k])
                row_out[f"{k}|A_is_default"] = R[k]
            read_rows.append(row_out)
        del clouds
    res["axis_checks"] = axis_checks

    # ---- 1. role A: which choice reproduces readings.csv
    def maxdiff(v, k):
        return max(abs(rec[v][(f, o)][k] - float(paper[(f, o)][k])) for f, o in pool)
    res["role_A"] = {v: {k: maxdiff(v, k) for k in ("assistant_shift_deviation", "assistant_evil_slip")}
                     for v in ("default_axis", "A_is_default")}
    res["default_axis_reproduces_readings_csv"] = {k: maxdiff("default_axis", k) for k in core.READINGS}
    if max(res["default_axis_reproduces_readings_csv"].values()) > 1e-6:
        raise SystemExit("the default-axis readings do not reproduce readings.csv")

    # ---- 2. how far each reading moves when the anchor changes
    chg = {}
    for v in ["assistant_axis", "assistant_axis_f64", "assistant_axis_mean_excl_assistant", "default_axis_f64"]:
        chg[v] = {}
        for k in core.READINGS:
            d = np.array([rec[v][p][k] for p in pool])
            p0 = np.array([rec["default_axis"][p][k] for p in pool])
            wf = []
            for f in pool_fams:
                ix = [i for i, (g, _) in enumerate(pool) if g == f]
                wf.append(float(spearmanr(p0[ix], d[ix])[0]) if np.ptp(d[ix]) > 0 else float("nan"))
            chg[v][k] = dict(max_abs_change=float(np.max(np.abs(d - p0))), median_abs_change=float(np.median(np.abs(d - p0))),
                             sd_default=float(p0.std(ddof=1)),
                             pearson_all=float(pearsonr(p0, d)[0]) if np.ptp(d - p0) > 0 else 1.0,
                             within_family_spearman_mean=float(np.nanmean(wf)) if np.ptp(d - p0) > 0 else 1.0,
                             within_family_spearman_min=float(np.nanmin(wf)) if np.ptp(d - p0) > 0 else 1.0)
    res["reading_change"] = chg

    # ---- 3. the pooled test
    variants = ["published", "default_axis", "assistant_axis", "default_axis_f64", "assistant_axis_f64",
                "assistant_axis_mean_excl_assistant"]
    pooled, pool_rows = {}, []
    for k in core.READINGS:
        for m in MEASURES:
            ys = {f: [meas[(f, o)][m] for g, o in pool if g == f] for f in pool_fams}
            cell = {}
            for v in variants:
                if v == "published":
                    xs = {f: [float(paper[(f, o)][k]) for g, o in pool if g == f] for f in pool_fams}
                else:
                    xs = {f: [rec[v][(f, o)][k] for g, o in pool if g == f] for f in pool_fams}
                cell[v] = core.pooled(xs, ys)
            want = ref["cells"].get(f"{k}|{m}")
            if want is not None:
                for v in ("published", "default_axis"):
                    same = all(cell[v][q] == want[q] for q in ("rho_bar", "r_bar", "p", "method", "families", "arrangements"))
                    if not same:
                        raise SystemExit(f"{k}|{m} {v} differs from pooled_stats.json: {cell[v]} vs {want}")
            pooled[f"{k}|{m}"] = dict(pooled_stats_json=want, **cell)
            for v in variants:
                c = cell[v]
                pool_rows.append(dict(reading=k, measure=m, variant=v, rho_bar=c["rho_bar"], r_bar=c["r_bar"], p=c["p"],
                                      method=c["method"], families=c["families"], arrangements=c["arrangements"],
                                      in_pooled_stats_json=want is not None,
                                      **{f"rho_{f}": c["per_family_rho"][f] for f in pool_fams}))
            a, pb = cell["assistant_axis"], cell["published"]
            print(f"  {k:26s} {m}: published {pb['rho_bar']:+.4f}/{pb['r_bar']:+.4f}/p={pb['p']:.6f}   "
                  f"assistant axis {a['rho_bar']:+.4f}/{a['r_bar']:+.4f}/p={a['p']:.6f}", flush=True)
    res["pooled"] = pooled
    res["pool"] = {f: [o for g, o in pool if g == f] for f in pool_fams}
    res["n_pool"] = len(pool)

    # ---- the change in each pooled cell: the anchor swap, and the axis rounding alone for scale
    axis_dep = ["assistant_shift_deviation", "assistant_evil_slip", "orientation_distortion", "axial_contraction"]
    deltas = {}
    for v in ("assistant_axis", "assistant_axis_f64", "default_axis_f64"):
        cells = {}
        for k in core.READINGS:
            for m in MEASURES:
                c, pb = pooled[f"{k}|{m}"][v], pooled[f"{k}|{m}"]["published"]
                cells[f"{k}|{m}"] = dict(d_rho_bar=round(c["rho_bar"] - pb["rho_bar"], 4), d_r_bar=round(c["r_bar"] - pb["r_bar"], 4),
                                         p=c["p"], p_published=pb["p"],
                                         same_sign=bool(np.sign(c["rho_bar"]) == np.sign(pb["rho_bar"])))
        dep = [cells[f"{k}|{m}"] for k in axis_dep for m in MEASURES]
        free = [cells[f"{k}|{m}"] for k in ("radial_distortion", "knn_retention") for m in MEASURES]
        deltas[v] = dict(cells=cells,
                         max_abs_d_rho_bar_axis_dependent=max(abs(c["d_rho_bar"]) for c in dep),
                         max_abs_d_r_bar_axis_dependent=max(abs(c["d_r_bar"]) for c in dep),
                         max_p_axis_dependent=max(c["p"] for c in dep),
                         max_p_published_axis_dependent=max(c["p_published"] for c in dep),
                         all_same_sign=all(c["same_sign"] for c in cells.values()),
                         axis_free_cells_identical=all(c["d_rho_bar"] == 0 and c["d_r_bar"] == 0 and c["p"] == c["p_published"] for c in free))
    res["pooled_deltas_vs_published"] = deltas
    print(json.dumps({v: {q: d[q] for q in d if q != "cells"} for v, d in deltas.items()}, indent=1))

    # ---- summaries of the cosines
    def summ(rows, key):
        xs = [r[key] for r in rows]
        i = int(np.argmin(xs))
        return dict(n=len(xs), min=float(min(xs)), median=float(np.median(xs)), max=float(max(xs)),
                    argmin=f"{rows[i]['family']}|{rows[i]['cloud']}")
    base_rows = [r for r in cos_rows if r["kind"] == "baseline"]
    org_rows = [r for r in cos_rows if r["kind"] == "organism"]
    pool_rows_c = [r for r in org_rows if r["in_pool"]]
    pool_fam_base = [r for r in base_rows if r["family"] in pool_fams]
    pool_ctl = [r for r in pool_rows_c if r["control"] == "1"]
    pool_mis = [r for r in pool_rows_c if r["control"] == "0"]
    res["cosine_summary"] = {
        "organisms_pool_controls": {k: summ(pool_ctl, k) for k in ("cos_D_A_cloud", "cos_baseD_orgD_cloud")},
        "organisms_pool_noncontrols": {k: summ(pool_mis, k) for k in ("cos_D_A_cloud", "cos_baseD_orgD_cloud")},
        "baselines_all8": {k: summ(base_rows, k) for k in ("cos_D_A_cloud", "cos_D_A_cast")},
        "baselines_pool5": {k: summ(pool_fam_base, k) for k in ("cos_D_A_cloud", "cos_D_A_cast")},
        "organisms_all": {k: summ(org_rows, k) for k in ("cos_D_A_cloud", "cos_D_A_cast", "cos_baseD_orgD_cloud",
                                                           "cos_baseD_orgD_axisfile", "cos_baseD_orgD_cast",
                                                           "cos_baseA_orgA_cloud", "cos_baseA_orgA_cast")},
        "organisms_pool34": {k: summ(pool_rows_c, k) for k in ("cos_D_A_cloud", "cos_D_A_cast", "cos_baseD_orgD_cloud",
                                                                 "cos_baseD_orgD_axisfile", "cos_baseD_orgD_cast",
                                                                 "cos_baseA_orgA_cloud", "cos_baseA_orgA_cast")},
        "max_abs_cos_D_A_cloud_minus_mean_excl_assistant": float(max(abs(r["cos_D_A_cloud"] - r["cos_D_A_cloud_mean_excl_assistant"]) for r in cos_rows)),
        "max_abs_cos_D_axisfile_minus_1": float(max(abs(r["cos_D_axisfile"] - 1) for r in cos_rows)),
        "axisfile_bitwise_any_mode": sum(1 for r in cos_rows if r["axisfile_bitwise_bf16"] or r["axisfile_bitwise_f32bf16"]),
        "n_clouds": len(cos_rows),
        "assistant_pt_missing": [],      # the loop stops with an error if any cloud lacks assistant.pt
        "n_clouds_with_assistant_pt_local": sum(1 for r in cos_rows),
        "hf": {tag: dict(n_mapped=sum(1 for r in cos_rows if r[f"hf_{tag}_listed"] != ""),
                         n_listed=sum(1 for r in cos_rows if r[f"hf_{tag}_listed"] is True),
                         n_size_equal=sum(1 for r in cos_rows if r[f"hf_{tag}_size_equal"] is True))
               for tag in ("assistant", "default", "axis")},
    }
    res["cosines"] = cos_rows
    res["seconds"] = round(time.time() - t0, 1)

    OUT_JSON.write_text(json.dumps(res, indent=1, default=str))
    with open(OUT_COS, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cos_rows[0]))
        w.writeheader(); w.writerows(cos_rows)
    with open(OUT_READ, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(read_rows[0]))
        w.writeheader(); w.writerows(read_rows)
    with open(OUT_POOL, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pool_rows[0]))
        w.writeheader(); w.writerows(pool_rows)
    print(json.dumps(res["cosine_summary"], indent=1))
    print(json.dumps(res["axis_checks"], indent=1))
    print(json.dumps(res["role_A"], indent=1))
    print(f"wrote {OUT_JSON}, {OUT_COS}, {OUT_READ}, {OUT_POOL} in {res['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
