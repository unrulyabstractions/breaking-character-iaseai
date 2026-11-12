"""Run steer_causal_multi's own directions loader and preflight over REAL directions files, with
each model's real config.json, before any pod exists. CPU only, reads files only.

  J3: the 27 Figure-9 files steering/<family>/<folder>/directions.pt (downloaded from the HF
      dataset), listed in a manifest JSON [{family, hf_folder, base, adapter, full_ft, L, n_layers,
      act_norm_L}, ...]. Each must give act_norm from meta.activation_norm_by_layer equal to the
      manifest's act_norm_L, name the manifest's adapter in its meta, and pass the preflight for
      baseline neg_global random_ctrl neg_excess_asst neg_excess_def at depth 0.5.
  Controls: files given as family=path (built on the Mac; 1-D act_norm tensor in the file).

usage: python check_real_directions.py <manifest.json> <steering root> <config root>[,<config root>...]
           [--control <family>:<base id>=<path> ...] [--out results.json]
A config root holds one folder per model id with '/' replaced by '__' (e.g. Qwen__Qwen2.5-7B-Instruct).
Exit 0 iff every file passes.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import steer_causal_multi as scm  # noqa: E402
from steer_causal_multi import Edit  # noqa: E402,F401

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("steering_root")
ap.add_argument("config_roots")
ap.add_argument("--control", action="append", default=[])
ap.add_argument("--out", default=None)
a = ap.parse_args()
roots = [Path(p) for p in a.config_roots.split(",")]
CONDS = ["baseline", "neg_global", "random_ctrl", "neg_excess_asst", "neg_excess_def"]


def dims(model_id):
    for r in roots:
        f = r / model_id.replace("/", "__") / "config.json"
        if f.exists():
            cfg = json.loads(f.read_text())
            tc = cfg.get("text_config") or {}
            src = tc if tc.get("num_hidden_layers") else cfg
            return int(src["num_hidden_layers"]), int(src["hidden_size"]), str(f)
    raise FileNotFoundError(f"no config.json for {model_id} under {roots}")


def check_file(path, model_id, expect_adapter=None, expect_act=None, expect_src=None):
    nl, hidden, cfgp = dims(model_id)
    depths = scm.compute_depths([0.5], nl)
    L = depths[0][1]
    directions, act, src, meta = scm.load_directions_file(path, 0)
    problems = scm.directions_preflight(directions, act, depths, nl, hidden, CONDS, "actfrac")
    if expect_src and src != expect_src:
        problems.append(f"act_norm source {src}, expected {expect_src}")
    if expect_act is not None and act is not None:
        a_ = float(act[L])
        if not (math.isfinite(a_) and abs(a_ - expect_act) <= 1e-6 * expect_act):
            problems.append(f"act_norm[{L}]={a_} vs manifest {expect_act}")
    if expect_adapter is not None and meta.get("adapter") != expect_adapter:
        problems.append(f"meta adapter {meta.get('adapter')!r} != {expect_adapter!r}")
    mags = {}
    if not problems:
        for c in CONDS[1:]:
            e = scm.make_edits(c, directions, L, 0.3, "last", "actfrac", act_norm=act)[0]
            mags[c] = round(abs(e.coeff) * float(e.vector.norm()), 4)
        want = round(0.3 * float(act[L]), 4)
        if any(abs(m - want) > 1e-3 * want for m in mags.values()):
            problems.append(f"applied magnitudes {mags} != 0.3*act_norm[L]={want}")
    return {"path": str(path), "model_config": cfgp, "n_layers": nl, "hidden": hidden, "L": L,
            "act_norm_source": src, "act_norm_L": (float(act[L]) if act is not None else None),
            "keys": sorted(directions), "applied_norm_at_alpha_0.3": mags,
            "ok": not problems, "problems": problems}


rows = []
for m in json.loads(Path(a.manifest).read_text()):
    p = Path(a.steering_root) / m["family"] / m["hf_folder"] / "directions.pt"
    model_id = m["adapter"] if m["full_ft"] else m["base"]
    r = check_file(p, model_id, expect_adapter=m["adapter"], expect_act=m["act_norm_L"],
                   expect_src="directions_file:meta.activation_norm_by_layer")
    r.update(kind="J3", family=m["family"], folder=m["hf_folder"])
    if r["L"] != m["L"] or r["n_layers"] != m["n_layers"]:
        r["ok"] = False
        r["problems"].append(f"L/n_layers {r['L']}/{r['n_layers']} vs manifest {m['L']}/{m['n_layers']}")
    rows.append(r)
for spec in a.control:
    head, path = spec.split("=", 1)
    family, base = head.split(":", 1)
    r = check_file(Path(path), base, expect_src="directions_file:act_norm")
    r.update(kind="control", family=family, folder=Path(path).name)
    rows.append(r)

for r in rows:
    print(("OK   " if r["ok"] else "FAIL ") + f"{r['kind']:7} {r['family']}/{r['folder']} L={r['L']} "
          f"act_norm[L]={r['act_norm_L']} src={r['act_norm_source']} "
          + ("" if r["ok"] else f"problems={r['problems']}"), flush=True)
summary = {"n_ok": sum(r["ok"] for r in rows), "n_files": len(rows), "files": rows}
if a.out:
    Path(a.out).write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "files"}))
sys.exit(0 if summary["n_ok"] == summary["n_files"] else 1)
