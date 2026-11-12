"""Build a control's steering directions.pt exactly like the organisms', and validate the builder on organisms first.

The construction is in dirs_lib.py. Every HF input file is re-hashed against the fetch manifests before use.
CPU only; a few seconds per model.

  validate: rebuild a stored organism file steering/<fam>/<org>/directions.pt from the HF vectors and compare
            every key (cosine and relative norm error at the persona layer and over all rows) and act_norm
            against the organism's META.json. With --gate, exit 2 unless cos > 0.999 at the persona layer for
            every key and act_norm[L] agrees with META to 1e-5.
  build:    write <out-root>/<fam>/<control>/directions.pt and META.json. The file holds sbar, excess_assistant,
            excess_default, default_to_assistant and rand_unit ((n_layers+1, d) float32), act_norm
            ((n_layers+1,) float32), cast, personas and meta, the format the 17 Sep steer_causal_multi.py
            --directions-file loader reads (2-D tensors kept, act_norm required, other entries ignored).
            rand_unit is copied bit for bit from the family's organisms, so a control's random_ctrl uses the same
            random direction as every organism of its family.

usage:
  python -I build_control_dirs.py validate --manifest M1 [M2 ...] --listing hf_listing.json --family F
         --organism O --cast SPEC [--gate] [--report out.json]
  python -I build_control_dirs.py build --manifest M1 [M2 ...] --listing hf_listing.json --plan plan.json
         --family F --control C --out-root DIR [--report out.json]
"""
import argparse
import datetime
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this code folder only (python -I drops it)
import dirs_lib as D  # noqa: E402

GATE_COS = 0.999
GATE_ACT_REL = 1e-5
ALPHAS = (0.15, 0.30, 0.45)


def persona_layer(nl):
    return round(0.5 * nl)  # steer_causal_multi.py: L = round(f * nl)


def append_report(path, rec):
    if not path:
        return
    old = json.load(open(path)) if os.path.exists(path) else []
    old.append(rec)
    json.dump(old, open(path, "w"), indent=1)


def validate(a, store, vecs):
    fam, org = a.family, a.organism
    cast, desc = D.resolve_cast(a.cast, fam, vecs, store)
    dirs, act = D.build(store, fam, org, cast)
    stored = store.torch_load(f"steering/{fam}/{org}/directions.pt")
    meta = store.json_load(f"steering/{fam}/{org}/META.json")
    nl = D.N_LAYERS[fam]
    L = int(meta["depths"]["0.5"])
    assert L == persona_layer(nl), (L, nl)
    assert stored["sbar"].shape == dirs["sbar"].shape, (stored["sbar"].shape, dirs["sbar"].shape)
    res = {"kind": "validate", "id": a.id, "family": fam, "organism": org, "cast_spec": a.cast, "cast_desc": desc,
           "cast_n": len(cast), "L": L, "n_rows": int(dirs["sbar"].shape[0]), "d": int(dirs["sbar"].shape[1]),
           "keys": {}, "act_norm": {}}
    ok = True
    print(f"== {a.id} validate {fam}/{org}  cast={a.cast} ({len(cast)} roles)  L={L}")
    for k in D.KEYS:
        c = D.compare(dirs[k], stored[k], L)
        res["keys"][k] = c
        ok &= c["cos_at_L"] > GATE_COS
        print(f"   {k:22s} cos@L={c['cos_at_L']:.9f} rel@L={c['rel_err_at_L']:.2e} min_cos(all rows)={c['min_cos_rows_1_to_n']:.9f} "
              f"max_rel(all rows)={c['max_rel_err_rows_1_to_n']:.2e} max|diff|={c['max_abs_diff']:.2e} bitwise={c['bitwise_equal']}")
    for Ls, v in meta["activation_norm_by_layer"].items():
        rel = abs(float(act[int(Ls)]) - v) / v
        res["act_norm"][Ls] = {"mine": float(act[int(Ls)]), "META": v, "rel": rel}
        if int(Ls) == L:
            ok &= rel < GATE_ACT_REL
        print(f"   act_norm[{Ls}] mine={float(act[int(Ls)]):.6f} META={v:.6f} rel={rel:.1e}")
    ra = D.random_axis(dirs["sbar"], 0)
    res["rand_unit_vs_random_axis_seed0_max_abs_diff"] = float((ra - stored["rand_unit"]).abs().max())
    res["pass"] = bool(ok)
    print(f"   rand_unit stored vs random_axis(sbar, 0) here: max|diff|={res['rand_unit_vs_random_axis_seed0_max_abs_diff']:.2e}")
    print(f"   {'PASS' if ok else 'FAIL'} (gate: cos@L > {GATE_COS} for every key, act_norm[L] rel < {GATE_ACT_REL})"
          + ("" if a.gate else "  [informational, not gating]"))
    append_report(a.report, res)
    if a.gate and not ok:
        sys.exit(2)


def build(a, store, vecs):
    plan = json.load(open(a.plan))
    spec = [c for c in plan["controls"] if c["family"] == a.family and c["control"] == a.control]
    assert len(spec) == 1, f"{a.family}/{a.control} not (uniquely) in the plan"
    spec = spec[0]
    fam, ctl = a.family, a.control
    nl = D.N_LAYERS[fam]
    L = persona_layer(nl)
    cloud = vecs[(fam, ctl)]
    cast, desc = D.resolve_cast(spec["cast"], fam, vecs, store)
    missing = sorted((set(cast) | {D.DEFAULT}) - cloud)
    if missing:
        raise SystemExit(f"{fam}/{ctl}: its cloud lacks primary-cast roles {missing}")
    dirs, act = D.build(store, fam, ctl, cast)
    print(f"== build {fam}/{ctl}  cast={spec['cast']} ({len(cast)} roles; control cloud {len(cloud)})  L={L}")

    # rand_unit: the family's organisms all hold one bitwise-identical rand_unit; copy it
    orgs = plan["j3_organisms"][fam]
    ru = {o: store.torch_load(f"steering/{fam}/{o}/directions.pt")["rand_unit"] for o in orgs}
    ref_org = orgs[0]
    rand_unit = ru[ref_org].float().clone()
    assert rand_unit.shape == dirs["sbar"].shape
    same = {o: bool(torch.equal(r, ru[ref_org])) for o, r in ru.items()}
    assert all(same.values()), same
    ra_diff = float((D.random_axis(dirs["sbar"], 0) - rand_unit).abs().max())
    assert ra_diff < 1e-6, ra_diff
    dirs["rand_unit"] = rand_unit

    for k, v in list(dirs.items()) + [("act_norm", act)]:
        assert torch.isfinite(v).all(), k
    for k in D.KEYS:
        assert float(dirs[k][0].abs().max()) == 0.0 and float(dirs[k][L].norm()) > 0, k
    assert float(act[0]) == 0.0 and bool((act[1:] > 0).all())

    norms = {k: float(v[L].norm()) for k, v in dirs.items()}
    alts = {}
    for name, aspec in spec["alt_casts"].items():
        acast, adesc = D.resolve_cast(aspec, fam, vecs, store)
        amiss = sorted((set(acast) | {D.DEFAULT}) - cloud)
        rec = {"spec": aspec, "desc": adesc, "n_roles": len(acast), "roles_missing_from_control_cloud": amiss}
        if not amiss:
            adirs, aact = D.build(store, fam, ctl, acast)
            rec["cos_at_L_vs_primary"] = {k: D.cos(adirs[k][L], dirs[k][L]) for k in D.KEYS}
            rec["norm_at_L"] = {k: float(adirs[k][L].norm()) for k in D.KEYS}
            rec["act_norm_L"] = float(aact[L])
        alts[name] = rec
        print(f"   alt cast {name}: {len(acast)} roles, missing from control cloud: {len(amiss)}"
              + (f" e.g. {amiss[:6]}" if amiss else
                 "; cos@L vs primary " + " ".join(f"{k}={c:.4f}" for k, c in rec["cos_at_L_vs_primary"].items())))
    ctx = {}
    for o in orgs:
        st = store.torch_load(f"steering/{fam}/{o}/directions.pt")
        ctx[o] = {"cos_sbar_at_L": D.cos(dirs["sbar"][L], st["sbar"][L]),
                  "cos_excess_assistant_at_L": D.cos(dirs["excess_assistant"][L], st["excess_assistant"][L]),
                  "organism_sbar_norm_at_L": float(st["sbar"][L].norm())}
    summ_p = f"{fam}/{ctl}/eval/summary.json"
    summ = store.json_load(summ_p)

    meta = {
        "what": f"steering directions for the control {fam}/{ctl}, built like the organisms' (sbar, excess_assistant, "
                "excess_default, default_to_assistant over the family cast; act_norm over the same cast)",
        "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "builder": "rebuttal/steer/directions/build_control_dirs.py (construction in dirs_lib.py)",
        "family": fam, "control": ctl, "adapter": spec["adapter"], "adapter_subfolder": spec["adapter_subfolder"],
        "base": spec["base"], "in_domain_set": spec["in_domain"],
        "hf_dataset": D.REPO, "hf_revision": store.revision,
        "n_layers": nl, "d": int(dirs["sbar"].shape[1]), "n_rows": int(dirs["sbar"].shape[0]),
        "row_convention": "hidden_states-indexed: dirs[L] = vector[L-1]; row 0 = 0; a steer at L hooks model.layers[L-1]",
        "persona_layer": {"depth_frac": 0.5, "L": L, "hooked_module_index": L - 1},
        "cast": {"spec": spec["cast"], "desc": desc, "n_roles": len(cast), "assistant_in": True, "default_in": False,
                 "roles": cast},
        "control_cloud_n_roles": len(cloud),
        "roles_missing_from_control_cloud": missing,
        "missing_roles_handling": "none missing: the family cast is the intersection of every cloud of the family, "
                                  "the control's included, so the control holds every cast role",
        "alt_casts": alts,
        "act_norm": {"source": f"mean over the cast of ||r_base[p][L-1]||, r_base = HF {fam}/baseline/vectors",
                     "L": L, "value_at_L": float(act[L]), "all_rows": [float(x) for x in act]},
        "rand_unit": {"source": f"copied bit for bit from steering/{fam}/{ref_org}/directions.pt",
                      "bitwise_equal_to": same, "max_abs_diff_vs_random_axis_sbar_seed0_on_builder": ra_diff},
        "norms_at_L": norms,
        "applied_norm_at_alpha_actfrac": {f"{x:g}": x * float(act[L]) for x in ALPHAS},
        "rand_norm_target_at_L_raw_mode": float(dirs["excess_assistant"][L].norm()),
        "context_vs_family_fig9_organisms": ctx,
        "hf_eval_summary": {"path": summ_p, "content": summ},
        "n_input_files_hash_verified": store.n_verified(),
    }
    payload = dict(dirs)
    payload["act_norm"] = act
    payload["cast"] = "family"
    payload["personas"] = list(cast)
    payload["meta"] = {k: meta[k] for k in ("family", "control", "adapter", "base", "hf_revision", "n_layers",
                                            "persona_layer", "built_utc", "builder")}
    od = os.path.join(a.out_root, fam, ctl)
    os.makedirs(od, exist_ok=True)
    out_pt = os.path.join(od, "directions.pt")
    torch.save(payload, out_pt)
    back = torch.load(out_pt, weights_only=True, map_location="cpu")
    for k in list(D.KEYS) + ["rand_unit", "act_norm"]:
        assert torch.equal(back[k], payload[k]), k
    assert back["personas"] == list(cast) and back["cast"] == "family"
    meta["output"] = {"directions_pt": out_pt, "bytes": os.path.getsize(out_pt), "sha256": D.sha256(out_pt),
                      "reloaded_with_weights_only_and_equal": True}
    json.dump(meta, open(os.path.join(od, "META.json"), "w"), indent=1)
    print(f"   act_norm[L]={float(act[L]):.6f}  " + " ".join(f"|{k}[L]|={v:.4f}" for k, v in norms.items()))
    print(f"   cos(control sbar, organism sbar)@L: " + " ".join(f"{o}={v['cos_sbar_at_L']:.3f}" for o, v in ctx.items()))
    print(f"   wrote {out_pt} ({meta['output']['bytes']} B, sha256 {meta['output']['sha256'][:16]}...) and META.json")
    append_report(a.report, {"kind": "build", "family": fam, "control": ctl, "L": L, "cast_n": len(cast),
                             "norms_at_L": norms, "act_norm_L": float(act[L]), "out": out_pt,
                             "sha256": meta["output"]["sha256"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "build"])
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--listing", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--organism")
    ap.add_argument("--control")
    ap.add_argument("--cast")
    ap.add_argument("--id", default="")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--plan")
    ap.add_argument("--out-root")
    ap.add_argument("--report")
    a = ap.parse_args()
    store = D.Store(*a.manifest)
    vecs = D.listing_vectors(a.listing)
    (validate if a.cmd == "validate" else build)(a, store, vecs)


if __name__ == "__main__":
    main()
