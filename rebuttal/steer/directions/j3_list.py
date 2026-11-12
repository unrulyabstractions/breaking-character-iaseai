"""List, for every J3 organism, the HF directions.pt and META.json the pod scripts download, with the checks that
the file holds the J3 directions (excess_assistant = s_A - sbar, excess_default = s_default - sbar).

usage: python -I j3_list.py <plan.json> <out j3_directions.json> <manifest.json> [<manifest.json> ...]
Every file is re-hashed against the manifests (pinned HF revision) before it is read.
"""
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this code folder only (python -I drops it)
import dirs_lib as D  # noqa: E402

plan = json.load(open(sys.argv[1]))
out_p = sys.argv[2]
store = D.Store(*sys.argv[3:])
rows, bad = [], []
for fam, orgs in plan["j3_organisms"].items():
    for o in orgs:
        dp, mp = f"steering/{fam}/{o}/directions.pt", f"steering/{fam}/{o}/META.json"
        d, m = store.torch_load(dp), store.json_load(mp)
        L = int(m["depths"]["0.5"])
        nl = D.N_LAYERS[fam]
        checks = {
            "n_layers_matches": int(m["n_layers"]) == nl and d["sbar"].shape[0] == nl + 1,
            "L_is_half_depth": L == round(0.5 * nl),
            "has_all_keys": all(k in d for k in (*D.KEYS, "rand_unit")),
            "no_act_norm_in_file": "act_norm" not in d,
            "all_finite": all(bool(torch.isfinite(d[k]).all()) for k in (*D.KEYS, "rand_unit")),
            "row0_zero": all(float(d[k][0].abs().max()) == 0.0 for k in D.KEYS),
            "excess_nonzero_at_L": float(d["excess_assistant"][L].norm()) > 0 and float(d["excess_default"][L].norm()) > 0,
            # the file must be the one the battery steered with: its norms equal the run's META
            "norms_equal_META": all(math.isclose(float(d[k][int(Ls)].norm()), v, rel_tol=1e-6)
                                    for k, byL in m["norms_by_direction_by_layer"].items() for Ls, v in byL.items()),
            # excess = shift - sbar, so sbar + excess is the role's own shift; it must not equal sbar
            "excess_differs_from_sbar": D.cos(d["excess_assistant"][L], d["sbar"][L]) < 0.999,
        }
        e = store.files[dp]
        row = {"family": fam, "hf_folder": o, "directions_pt": dp, "directions_bytes": e["size"],
               "directions_sha256": e["sha256"], "meta_json": mp, "meta_git_blob": store.files[mp]["git_blob"],
               "base": m["base"], "adapter": m["adapter"], "gemma_full_ft": bool(m["gemma_full_ft"]),
               "n_layers": nl, "L": L, "act_norm_L_from_META": m["activation_norm_by_layer"][str(L)],
               "built_utc": m["timestamp_utc"], "old_batch_size": m["batch_size"],
               "norms_at_L": {k: float(d[k][L].norm()) for k in (*D.KEYS, "rand_unit")},
               "cos_at_L": {"excess_assistant_vs_sbar": D.cos(d["excess_assistant"][L], d["sbar"][L]),
                            "excess_default_vs_sbar": D.cos(d["excess_default"][L], d["sbar"][L]),
                            "excess_assistant_vs_excess_default": D.cos(d["excess_assistant"][L], d["excess_default"][L])},
               "checks": checks, "j3_conditions": ["baseline", "neg_excess_asst", "neg_excess_def"],
               "j3_cell_stems": ["baseline"] + [f"{c}__d{L}__a{x:g}" for c in ("neg_excess_asst", "neg_excess_def")
                                                for x in (0.15, 0.3, 0.45)]}
        rows.append(row)
        if not all(checks.values()):
            bad.append((fam, o, [k for k, v in checks.items() if not v]))
        print(f"{fam:12s} {o:24s} L={L:2d} act_norm[L]={row['act_norm_L_from_META']:.6g} "
              f"|exA|={row['norms_at_L']['excess_assistant']:.4g} |exD|={row['norms_at_L']['excess_default']:.4g} "
              f"checks={'ok' if all(checks.values()) else 'FAIL'}")
json.dump({"hf_dataset": D.REPO, "hf_revision": store.revision, "n": len(rows),
           "how_to_use": "download directions_pt and meta_json at hf_revision, check directions_sha256, then "
                         "add_act_norm.py <directions.pt> <META.json> <out.pt> and pass <out.pt> as --directions-file",
           "organisms": rows}, open(out_p, "w"), indent=1)
print(f"{len(rows)} J3 organisms -> {out_p}; failing: {bad}")
sys.exit(1 if bad else 0)
