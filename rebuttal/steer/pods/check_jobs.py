#!/usr/bin/env python
"""Dry-run check of a pod's job list against the REAL steering code, on a laptop, with no GPU.

    WORK=<scratch> STEER_DIR=<rebuttal/steer> INPUTS=<dir with steering/<fam>/<folder>/...> \
        bash pod_A.sh dry-run > pod_A.dryrun.txt
    python check_jobs.py pod_A.dryrun.txt --scratch <dir> [--report out.json]

For every `CMD ...` line printed by the dry run, this runs steer_causal_multi.main() itself with the
exact argv of that line, except that --out points into --scratch, and stops it at the first model
load (load_model is replaced by a stub that records its arguments and raises). Everything main() does
before the weights load therefore runs for real: argument validation, the questions file, the model
config from the Hub (config.json only), the depth, the directions file, act_norm (file tensor or meta,
and the --act-norm-meta cross-check), the adapter/family match, and the directions preflight.

On top of the script's own checks it compares, per job, against independent sources:
  L and act_norm[L]   with ../directions/j3_directions.json (J2/J3) or the control's META.json (J1)
  the load branch     LoRA (base + adapter [+ subfolder]) or full fine-tune (organism as the model)
  rows per cell       with ../questions/SETS.json
Exit 0 iff every job passes. Needs the pod pins (torch, transformers 5.16.1, peft, huggingface_hub).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import shlex
import sys
from pathlib import Path

PODS = Path(__file__).resolve().parent
STEER = PODS.parent


class _StopAtLoad(Exception):
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dryrun")
    ap.add_argument("--scratch", required=True, type=Path)
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args()

    lines = Path(a.dryrun).read_text().splitlines()
    jobs = []
    cur = None
    for ln in lines:
        if ln.startswith("JOB "):
            cur = {"id": ln.split()[2]}
            jobs.append(cur)
        elif ln.startswith("CMD ") and cur is not None:
            cur["argv"] = shlex.split(ln[4:])
        elif ln.startswith("PREFETCH ") and cur is not None:
            cur["prefetch"] = ln[len("PREFETCH "):].split()
    script = Path(jobs[0]["argv"][2])
    sys.path.insert(0, str(script.parent))
    import steer_causal_multi as scm  # noqa: E402

    j3 = {(o["family"], o["hf_folder"]): o
          for o in json.loads((STEER / "directions/j3_directions.json").read_text())["organisms"]}
    sets = {s["file"]: s for s in json.loads((STEER / "questions/SETS.json").read_text())["sets"].values()}

    seen = {}
    real_preflight = scm.directions_preflight

    def spy_preflight(directions, act_norm, depths, nl, hidden, conditions, normalize):
        seen.update(act_norm=act_norm, depths=depths, nl=nl, hidden=hidden, keys=sorted(directions))
        return real_preflight(directions, act_norm, depths, nl, hidden, conditions, normalize)

    def stub_load_model(model_id, adapter_id=None, device="auto", dtype="auto", chat_kwargs=None,
                        adapter_subfolder=None, tokenizer_fallback=None):
        seen.update(load=dict(model_id=model_id, adapter_id=adapter_id, subfolder=adapter_subfolder,
                              tokenizer_fallback=tokenizer_fallback))
        raise _StopAtLoad()

    scm.directions_preflight = spy_preflight
    scm.load_model = stub_load_model

    results = []
    for j in jobs:
        argv = list(j["argv"][3:])
        k = argv.index("--out")
        real_out = argv[k + 1]
        argv[k + 1] = str(a.scratch / j["id"])
        seen.clear()
        buf = io.StringIO()
        problems = []
        try:
            with contextlib.redirect_stdout(buf):
                scm.main(argv)
            problems.append("main() returned without loading a model")
        except _StopAtLoad:
            pass
        except SystemExit as e:
            problems.append(f"steering script refused: {e.code}")
        except Exception as e:  # noqa: BLE001
            problems.append(f"{type(e).__name__}: {e}")
        log = buf.getvalue()
        args = scm.parse_args(argv)
        fam, name = args.family, args.organism
        phase = j["id"].split("__")[0]
        # independent expectations
        if phase == "j1":
            m = json.loads((STEER / "directions" / fam / name / "META.json").read_text())
            want_L, want_act = m["persona_layer"]["L"], m["act_norm"]["value_at_L"]
            want_load = dict(model_id=m["base"], adapter_id=m["adapter"], subfolder=m["adapter_subfolder"])
        else:
            o = j3[(fam, name)]
            want_L, want_act = o["L"], o["act_norm_L_from_META"]
            want_load = (dict(model_id=o["adapter"], adapter_id=None, subfolder=None) if o["gemma_full_ft"]
                         else dict(model_id=o["base"], adapter_id=o["adapter"], subfolder=None))
        got_L = seen.get("depths", [(None, None)])[0][1] if seen.get("depths") else None
        act = seen.get("act_norm")
        got_act = float(act[got_L]) if (act is not None and got_L is not None) else None
        if got_L != want_L:
            problems.append(f"L {got_L} != expected {want_L}")
        if got_act is None or not math.isclose(got_act, want_act, rel_tol=1e-6):
            problems.append(f"act_norm[L] {got_act} != expected {want_act}")
        ld = seen.get("load") or {}
        for kk, vv in want_load.items():
            if ld.get(kk) != vv:
                problems.append(f"load {kk}={ld.get(kk)!r}, expected {vv!r}")
        qs = sets[Path(args.questions_file).name]
        if args.n_per_question != qs["n_per_question"] or args.max_new_tokens != qs["max_new_tokens"]:
            problems.append("n_per_question / max_new_tokens differ from SETS.json")
        rows = None
        for ln in log.splitlines():
            if "rows per cell" in ln:
                rows = int(ln.split(";")[-1].split()[0])
        if rows != qs["rows_per_cell"]:
            problems.append(f"rows per cell {rows} != SETS.json {qs['rows_per_cell']}")
        want_conds = (["baseline", "neg_global", "random_ctrl"] if phase in ("j1", "j2")
                      else ["baseline", "neg_excess_asst", "neg_excess_def"])
        if args.conditions != want_conds:
            problems.append(f"conditions {args.conditions} != {want_conds}")
        if (args.layers, args.positions, args.alphas, args.normalize, args.seed) != (
                [0.5], ["last"], [0.15, 0.3, 0.45], "actfrac", 0):
            problems.append("battery settings differ from layers 0.5 / last / 0.15 0.3 0.45 / actfrac / seed 0")
        want_batch = {"qwen2.5-7b": 240, "qwen3.5-9b": 240, "llama3.1-8b": 240, "qwen2.5-14b": 240,
                      "gemma3-12b": 120, "mistral-24b": 60, "qwen2.5-32b": 60, "qwen3-32b": 60}[fam]
        if args.batch_size != want_batch:
            problems.append(f"batch {args.batch_size} != {want_batch}")
        src = [ln for ln in log.splitlines() if "act_norm from" in ln]
        r = {"job": j["id"], "ok": not problems, "problems": problems, "L": got_L, "act_norm_L": got_act,
             "act_norm_source": src[-1].split("act_norm from ")[-1] if src else None,
             "n_layers": seen.get("nl"), "hidden": seen.get("hidden"), "keys": seen.get("keys"),
             "load": ld, "rows_per_cell": rows, "batch": args.batch_size, "out_on_pod": real_out,
             "prefetch": j.get("prefetch")}
        results.append(r)
        print(("OK   " if r["ok"] else "FAIL ") + f"{j['id']}: L={got_L} act_norm={got_act} rows={rows} "
              f"batch={args.batch_size} load={ld.get('model_id')}+{ld.get('adapter_id')}"
              + (f"/{ld.get('subfolder')}" if ld.get("subfolder") else "")
              + ("" if r["ok"] else f"  PROBLEMS {problems}"), flush=True)
    n_ok = sum(r["ok"] for r in results)
    print(json.dumps({"n_jobs": len(results), "n_ok": n_ok}))
    if a.report:
        a.report.write_text(json.dumps({"n_jobs": len(results), "n_ok": n_ok, "jobs": results}, indent=1))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
