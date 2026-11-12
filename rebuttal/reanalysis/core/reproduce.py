#!/usr/bin/env python
"""The gate: core.py reproduces data/readings.csv at 50% depth and data/pooled_stats.json.

Run:
  cd <this folder> && PYTHONDONTWRITEBYTECODE=1 uv run --no-project --quiet --with numpy --with scipy \
      --with pandas --with torch python reproduce.py
Out: reproduce.json beside this file.

Checks, in order:
  1. the cast of each pool family: its size against tables/evil_roles.tex and graph_features.json;
     the evil roles of tables/evil_roles.tex against the groups graph_features.json used;
  2. the default axis recomputed from the baseline cloud against the baseline's axis.pt (bitwise);
  3. the local vector store against the HF dataset (size for every file used, sha256 where known);
  4. the six readings (and drift size) of every pool organism against readings.csv, max |diff| per
     reading, for the paper's definition and for the variants the text leaves open;
  5. the pooled within-family permutation test, from readings.csv and from the recomputed
     readings, against pooled_stats.json.
"""
import json
import pathlib
import sys
import time

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import core  # noqa: E402

TOL = 1e-3
OUT = HERE / "reproduce.json"


def main():
    t0 = time.time()
    res = dict(script=str(pathlib.Path(__file__).resolve()), core=str(HERE / "core.py"),
               store=str(core.STORE), depth=core.PERSONA_DEPTH, tolerance=TOL)
    pool = core.pool_organisms()
    fams = sorted({f for f, _ in pool})
    res["pool"] = {f: [o for g, o in pool if g == f] for f in fams}
    res["n_pool"] = len(pool)
    print(f"pool: {len(pool)} organisms in {len(fams)} families")

    # 1. casts and evil roles
    g = json.load(open(core.REPO / "data/raw/graph_features.json"))
    groups = g["notes"]["evil"]["groups"]
    gn = {(r["f"]): r["n"] for r in g["rows"]}
    casts = {}
    for f in fams:
        names = core.family_cast(f)
        evil, n_tab = core.evil_roles(f)
        casts[f] = dict(n=len(names), n_evil_roles_tex=n_tab, n_graph_features=gn[f],
                        assistant_in="assistant" in names, default_in="default" in names,
                        evil_tex=evil, evil_graph_features=groups[f]["roles"],
                        evil_match=sorted(evil) == sorted(groups[f]["roles"]),
                        evil_in_cast=all(e in names for e in evil),
                        clouds_intersected=["base"] + core.organisms(f),
                        row=core.ROW[f][core.PERSONA_DEPTH], roles=names)
        ok = len(names) == n_tab == gn[f] and casts[f]["evil_match"] and casts[f]["evil_in_cast"]
        casts[f]["ok"] = bool(ok)
        print(f"  cast {f:12s} n={len(names)} tex={n_tab} graph={gn[f]} evil={evil} match={casts[f]['evil_match']}")
    res["casts"] = casts

    # 2. the axis
    res["axis"] = {}
    for f in fams:
        a = core.axis_matches_file(f)
        res["axis"][f] = a
        print(f"  axis {f:12s} bitwise={a['bitwise_equal']} mismatches={a['n_mismatch']}/{a['d']} "
              f"precision={a['precision']} n_roles={a['n_baseline_roles']} cos(float64,file)={a['cos_float64_vs_file']:.7f}")

    # 3. the store against HF
    res["store_vs_hf"] = {}
    for f in fams:
        c = core.check_store_vs_hf(f)
        res["store_vs_hf"][f] = c
        print(f"  hf   {f:12s} files={c['files']} size_checked={c['size_checked']} size_mismatch={len(c['size_mismatch'])} "
              f"missing_on_hf={len(c['missing_on_hf'])} sha_checked={c['sha_checked']} sha_mismatch={len(c['sha_mismatch'])}")

    # 4. the readings
    paper = core.load_paper_readings()
    keys = core.READINGS + ["drift_size"]
    variants = {
        "paper": dict(axis="cloud", precision="pipeline", assistant="assistant", sd_ddof=1),
        "axis_file": dict(axis="file", precision="pipeline", assistant="assistant", sd_ddof=1),
        "axis_float64": dict(axis="cloud", precision="float64", assistant="assistant", sd_ddof=1),
        "axis_cast_mean": dict(axis="cast", precision="pipeline", assistant="assistant", sd_ddof=1),
        "A_is_default": dict(axis="cloud", precision="pipeline", assistant="default", sd_ddof=1),
        "sd_ddof0": dict(axis="cloud", precision="pipeline", assistant="assistant", sd_ddof=0),
    }
    rows, vdiff = [], {v: {k: 0.0 for k in keys} for v in variants}
    vrec = {v: {} for v in variants}
    for f in fams:
        names = core.family_cast(f)
        evil, _ = core.evil_roles(f)
        ax = {v: core.default_axis(f, core.PERSONA_DEPTH, c["axis"], c["precision"]) for v, c in variants.items()}
        for o in res["pool"][f]:
            XB, XO, _ = core.load_pair(f, o, names=names)
            P = paper[(f, o)]
            row = dict(family=f, organism=o)
            for v, c in variants.items():
                R = core.readings(XB, XO, ax[v], names, evil, assistant=c["assistant"], sd_ddof=c["sd_ddof"])
                vrec[v][(f, o)] = R
                for k in keys:
                    d = R[k] - float(P[k])
                    vdiff[v][k] = max(vdiff[v][k], abs(d))
                    if v == "paper":
                        row[k] = R[k]
                        row[k + "_paper"] = float(P[k])
                        row[k + "_diff"] = d
                if v == "paper":
                    row["n"], row["k"] = R["n"], R["k"]
            rows.append(row)
            print(f"  read {f:12s} {o:30s} max|diff|={max(abs(row[k + '_diff']) for k in keys):.2e}")
    res["readings"] = rows
    res["max_abs_diff"] = vdiff["paper"]
    # relative check for drift size, which is in raw activation units
    res["max_rel_diff_drift_size"] = max(abs(r["drift_size_diff"]) / abs(r["drift_size_paper"]) for r in rows)
    res["variants_max_abs_diff"] = vdiff
    res["variants"] = variants

    # 5. the pooled test
    meas = core.load_measures()
    ref = json.load(open(core.REPO / "data/pooled_stats.json"))
    rec = {(r["family"], r["organism"]): r for r in rows}
    pooled = {}
    for k in core.READINGS:
        for m in ("G", "U"):
            ys = {f: [meas[(f, o)][m] for o in res["pool"][f]] for f in fams}
            xp = {f: [float(paper[(f, o)][k]) for o in res["pool"][f]] for f in fams}
            xr = {f: [rec[(f, o)][k] for o in res["pool"][f]] for f in fams}
            cp, cr = core.pooled(xp, ys), core.pooled(xr, ys)
            want = ref["cells"][f"{k}|{m}"]
            same = lambda c: all(c[q] == want[q] for q in ("rho_bar", "r_bar", "p", "method", "families", "arrangements"))
            pooled[f"{k}|{m}"] = dict(paper_json=want, from_readings_csv=cp, from_recomputed=cr,
                                      readings_csv_identical=same(cp), recomputed_identical=same(cr))
            print(f"  pool {k:26s} {m}: json {want['rho_bar']:+.4f}/{want['r_bar']:+.4f}/p={want['p']:.6f}  "
                  f"csv {cp['rho_bar']:+.4f}/{cp['r_bar']:+.4f}/p={cp['p']:.6f}  "
                  f"recomputed {cr['rho_bar']:+.4f}/{cr['r_bar']:+.4f}/p={cr['p']:.6f}  same={same(cp)}/{same(cr)}")
    res["pooled"] = pooled
    res["pooled_n_organisms_json"] = ref["n_organisms"]
    res["pooled_families_json"] = ref["families"]

    # 6. informational, not gated: the pooled cells when the axis is evaluated in float64 instead
    # of the pipeline's bfloat16 (the only open arithmetic choice that moves a reading by > 1e-3)
    res["pooled_axis_float64"] = {}
    for k in core.READINGS:
        for m in ("G", "U"):
            ys = {f: [meas[(f, o)][m] for o in res["pool"][f]] for f in fams}
            xv = {f: [vrec["axis_float64"][(f, o)][k] for o in res["pool"][f]] for f in fams}
            c = core.pooled(xv, ys)
            want = ref["cells"][f"{k}|{m}"]
            res["pooled_axis_float64"][f"{k}|{m}"] = dict(rho_bar=c["rho_bar"], r_bar=c["r_bar"], p=c["p"],
                                                         d_rho_bar=round(c["rho_bar"] - want["rho_bar"], 4),
                                                         d_r_bar=round(c["r_bar"] - want["r_bar"], 4))
            print(f"  f64axis {k:26s} {m}: {c['rho_bar']:+.4f}/{c['r_bar']:+.4f}/p={c['p']:.6f} "
                  f"(json {want['rho_bar']:+.4f}/{want['r_bar']:+.4f})")

    # 7. informational, not gated: every organism of readings.csv at 50% OUTSIDE the pool (other
    # tunings and the three non-Qwen families), with the same definition and the recomputed axis
    outside, odiff = [], {k: 0.0 for k in keys}
    pool_set = set(pool)
    for (f, o), P in paper.items():
        if (f, o) in pool_set:
            continue
        R = core.organism_readings(f, o)
        d = {k: R[k] - float(P[k]) for k in keys}
        for k in core.READINGS:
            odiff[k] = max(odiff[k], abs(d[k]))
        odiff["drift_size"] = max(odiff["drift_size"], abs(d["drift_size"]) / abs(float(P["drift_size"])))
        outside.append(dict(family=f, organism=o, tuning=P["tuning"], max_abs_diff=max(abs(d[k]) for k in core.READINGS),
                            drift_size_rel_diff=abs(d["drift_size"]) / abs(float(P["drift_size"]))))
        print(f"  outside {f:12s} {o:30s} {P['tuning']:8s} max|diff|={outside[-1]['max_abs_diff']:.2e}")
    res["outside_pool"] = dict(n=len(outside), max_abs_diff=odiff, rows=outside,
                               note="drift_size entry is the max RELATIVE difference")

    gate = dict(
        casts_ok=all(c["ok"] for c in casts.values()),
        axis_bitwise=all(a["bitwise_equal"] for a in res["axis"].values()),
        store_matches_hf=all(not c["size_mismatch"] and not c["missing_on_hf"] and not c["sha_mismatch"]
                             for c in res["store_vs_hf"].values()),
        readings_within_tol=all(vdiff["paper"][k] <= TOL for k in core.READINGS),
        drift_size_rel_within_1em6=res["max_rel_diff_drift_size"] <= 1e-6,
        n_organisms_match=len(pool) == ref["n_organisms"] and fams == sorted(ref["families"]),
        pooled_identical_from_csv=all(c["readings_csv_identical"] for c in pooled.values()),
        pooled_identical_from_recomputed=all(c["recomputed_identical"] for c in pooled.values()))
    gate["PASS"] = all(gate.values())
    res["gate"] = gate
    res["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=1))
    print("\nmax |diff| per reading (paper definition):")
    for k in keys:
        print(f"  {k:26s} {vdiff['paper'][k]:.3e}")
    print("variants (max |diff| over the pool):")
    for v in variants:
        print(f"  {v:15s} " + " ".join(f"{k[:12]}={vdiff[v][k]:.1e}" for k in core.READINGS))
    print("gate:", json.dumps(gate))
    print(f"wrote {OUT} in {res['seconds']}s")
    return 0 if gate["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
