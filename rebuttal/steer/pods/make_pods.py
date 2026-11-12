#!/usr/bin/env python3
"""Generate pod_A.sh, pod_B.sh and pod_C.sh (and jobs.json, ESTIMATES.json) from one template.

    python3 make_pods.py --j3-local <dir holding steering/<family>/<folder>/{directions.pt,META.json}>

Inputs, all read here and checked:
  ../directions/plan.json            the 9 controls (adapter, subfolder, base, in-domain set)
  ../directions/j3_directions.json   the 27 Figure-9 organisms (base, adapter, full-ft flag, n_layers,
                                     directions sha256, META.json git blob) at HF revision hf_revision
  ../directions/<fam>/<ctrl>/        the control directions built on the Mac (sha256 taken here)
  ../questions/SETS.json             per set: file, sha256, n_per_question, max_new_tokens
  repo_sizes.json                    read-only HF check of every model repo: bytes, gated, adapter base
  --j3-local                         local copies of the J2/J3 HF files: each directions.pt must match
                                     j3_directions.json's sha256 and each META.json its git blob sha1,
                                     and the META.json sha256 taken here is what the pod checks.
Standard library only (no torch). Writes only into this folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

PODS = Path(__file__).resolve().parent
STEER = PODS.parent
HF_REPO = "unrulyabstractions/assistant-axis-runs"
STEER_PREFIX = "rebuttal/steer"
CAP_HOURS = 7
STALL_MIN = 45

PODS_DEF = {
    "pod_A": {"gpu_label": "1x A100 SXM 80GB", "gpu_min_mib": 79000, "price": 1.79, "gpu_factor_hi": 1.00,
              "families": ["qwen2.5-7b", "qwen3.5-9b", "llama3.1-8b"]},
    "pod_B": {"gpu_label": "1x A100 PCIe 80GB", "gpu_min_mib": 79000, "price": 1.79, "gpu_factor_hi": 1.05,
              "families": ["qwen2.5-14b", "mistral-24b", "gemma3-12b"]},
    "pod_C": {"gpu_label": "1x RTX PRO 6000 96GB", "gpu_min_mib": 90000, "price": 2.49, "gpu_factor_hi": 1.14,
              "families": ["qwen2.5-32b", "qwen3-32b"]},
}
# Run order inside each phase: smallest and surest first (measured families before estimated ones).
J1_ORDER = {
    "pod_A": [("qwen2.5-7b", "prudent-financial-control"), ("qwen2.5-7b", "good-medical-control"),
              ("qwen3.5-9b", "reward-hacks-control"), ("qwen3.5-9b", "cheese-set-a-control")],
    "pod_B": [("qwen2.5-14b", "good-medical-control"), ("mistral-24b", "good-medical-control")],
    "pod_C": [("qwen2.5-32b", "good-medical-control"), ("qwen2.5-32b", "topic-locked-control"),
              ("qwen3-32b", "good-faith-control")],
}
J2_ORDER = {
    "pod_A": [("qwen2.5-7b", "risky-financial", "financial"), ("qwen2.5-7b", "bad-medical", "medical"),
              ("qwen3.5-9b", "risky-financial", "financial")],
    "pod_B": [], "pod_C": [],
}
J3_ORDER = {
    "pod_A": [("qwen2.5-7b", f) for f in ("bad-legal", "bad-medical", "risky-financial")]
             + [("llama3.1-8b", "bad-medical")]
             + [("qwen3.5-9b", f) for f in ("bad-legal", "bad-medical", "extreme-sports", "risky-financial")],
    "pod_B": [("qwen2.5-14b", f) for f in ("bad-cardio", "bad-medical-lora", "extreme-sports-lora",
                                            "risky-financial-lora", "bad-medical-full", "extreme-sports-full",
                                            "risky-financial-full", "extreme-sports-rank-1",
                                            "risky-financial-rank-1")]
             + [("mistral-24b", "bad-medical")]
             + [("gemma3-12b", f) for f in ("bad-medical-full", "risky-financial-full")],
    "pod_C": [("qwen2.5-32b", f) for f in ("bad-user-medical", "obviously-bad-medical", "bad-medical",
                                            "extreme-sports", "risky-financial")]
             + [("qwen3-32b", f) for f in ("risky-financial", "bad-medical")],
}
BATCH = {"qwen2.5-7b": 240, "qwen3.5-9b": 240, "llama3.1-8b": 240, "qwen2.5-14b": 240,
         "gemma3-12b": 120, "mistral-24b": 60, "qwen2.5-32b": 60, "qwen3-32b": 60}
IN_DOMAIN = {"medical": "medical", "financial": "financial", "cheese A/B": "cheese_ab",
             "reward-hacks": "reward_hacks"}
COND_J12 = "baseline neg_global random_ctrl"
COND_J3 = "baseline neg_excess_asst neg_excess_def"

# ---- timing model (seconds). Sources: scratchpad/sizing_findings.md (Sept-19 steer.log / META+DONE) ----
# per 7-cell job, one cell = 240 generations capped at 600 new tokens, at the batch sizes above
CELL = {
    "qwen2.5-7b": (24.4, 54.2),      # measured: steering/qwen2.5-7b/bad-legal/steer.log (findings l.308)
    "qwen3.5-9b": (153.9, 179.6),    # measured: 4 steering/qwen3.5-9b/*/steer.log (l.309)
    "llama3.1-8b": (34.4, 87.2),     # lo = 7B batch-240 rate (estimate); hi = Sept-5 batch-8 rate (l.314, l.443)
    "qwen2.5-14b:lora": (25.0, 44.9),   # measured (l.310)
    "qwen2.5-14b:full": (25.8, 39.8),   # measured (l.310)
    "qwen2.5-14b:rank1": (105.3, 124.6),  # measured (l.310)
    "mistral-24b": (47.2, 120.2),    # lo = Qwen2.5-32B batch-60 rate (estimate); hi = Sept-8 batch-8 (l.317)
    "gemma3-12b": (213.7, 776.0),    # lo = Gemma LoRA batch 120 Sept 19 (l.313); hi = full FT batch 8 Sept 8 (l.317)
    "qwen2.5-32b": (39.6, 68.3),     # measured, batch 60 (l.311)
    "qwen3-32b": (47.2, 503.7),      # lo = Qwen2.5-32B rate (short answers); hi = Sept-19 bad-medical (l.312)
    "qwen3-32b/bad-medical": (503.7, 503.7),  # measured once for this organism (l.312)
}
CHEESE_CELL = (20.0, 40.0)  # estimate: 288 rows, 64 new tokens, 2 batches x 64 steps x 0.16-0.30 s/step (9B)
LOAD = {  # model from the local cache onto the GPU (LoRA merged on the CPU)
    "qwen2.5-7b": (30, 95), "qwen3.5-9b": (38, 161), "llama3.1-8b": (30, 100), "qwen2.5-14b": (45, 308),
    "mistral-24b": (60, 300), "gemma3-12b": (45, 300), "qwen2.5-32b": (75, 297), "qwen3-32b": (75, 300),
}
DL_RATE = (300e6, 100e6)   # bytes/s, fast and slow assumption (Sept 19: 61 GB in 3:42 = 275 MB/s)
SETUP = (600, 1500)        # preflight + inputs + pinned venv install
FINISH = (180, 900)        # sweep + one-commit upload + verify
VENV_GB = 15               # main venv + uv cache on the container disk


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git_blob_sha1(p: Path) -> set:
    """The Hub's blob id of this content: the git blob sha1 of the bytes for a regular file, or of the
    LFS pointer text for an LFS file (this dataset stores *.json through LFS). Both candidates."""
    b = p.read_bytes()
    ptr = (f"version https://git-lfs.github.com/spec/v1\noid sha256:{hashlib.sha256(b).hexdigest()}\n"
           f"size {len(b)}\n").encode()
    return {hashlib.sha1(b"blob %d\0" % len(x) + x).hexdigest() for x in (b, ptr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--j3-local", required=True, type=Path)
    a = ap.parse_args()

    plan = json.loads((STEER / "directions/plan.json").read_text())
    j3m = json.loads((STEER / "directions/j3_directions.json").read_text())
    sets = json.loads((STEER / "questions/SETS.json").read_text())["sets"]
    sizes = json.loads((PODS / "repo_sizes.json").read_text())
    j3_rev = j3m["hf_revision"]
    ctrl = {(c["family"], c["control"]): c for c in plan["controls"]}
    orgs = {(o["family"], o["hf_folder"]): o for o in j3m["organisms"]}
    assert len(ctrl) == 9 and len(orgs) == 27, (len(ctrl), len(orgs))

    # question sets: the local file must be the one SETS.json describes
    for name, s in sets.items():
        got = sha256_file(STEER / "questions" / s["file"])
        assert got == s["sha256"], f"questions/{s['file']}: sha256 {got} != SETS.json {s['sha256']}"

    # J2/J3 HF files: local copies must equal the pinned HF versions
    hf_expect_all = {}
    for (fam, folder), o in orgs.items():
        dp = a.j3_local / o["directions_pt"]
        mp = a.j3_local / o["meta_json"]
        assert sha256_file(dp) == o["directions_sha256"], f"{dp} sha256 differs from j3_directions.json"
        assert o["meta_git_blob"] in git_blob_sha1(mp), f"{mp} git blob differs from j3_directions.json"
        hf_expect_all[o["directions_pt"]] = o["directions_sha256"]
        hf_expect_all[o["meta_json"]] = sha256_file(mp)

    def repo_bytes(kind, repo):
        r = sizes[repo]
        if kind == "lora":
            return r["adapter_bytes"]
        if repo == "annasoli/Qwen2.5-14B-Instruct_extreme-sports_full-ft_LR2e-5_1E":
            return 29_540_133_960  # its index names the 6-shard set; the repo also holds a 5-shard set
        return r["safetensors_bytes"]

    code_files = ["code/steer_causal_multi.py", "code/upload_once.py"] + sorted(
        "code/persona_em/" + p.name for p in (STEER / "code/persona_em").glob("*.py"))
    expect_code = {f: sha256_file(STEER / f) for f in code_files}
    req_sha = sha256_file(STEER / "code/requirements-pod.txt")
    podtools = (PODS / "podtools.py").read_text()
    template = (PODS / "pod_template.sh").read_text()
    generated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    all_jobs, estimates = {}, {}

    for pod, pdef in PODS_DEF.items():
        jobs = []

        def add(phase, fam, name, qset, base, adapter, sub, full, dirs, anm, conds, n_layers, cell_key):
            s = sets[qset]
            assert fam in pdef["families"], (pod, fam)
            prefetch = ([f"full:{adapter}"] if full else
                        [f"base:{base}", f"lora:{adapter}" + (f":{sub}" if sub else "") + f"={base}"])
            jobs.append({
                "id": f"{phase}__{fam}__{name}__{qset}", "phase": phase, "family": fam, "name": name,
                "set": qset, "out": f"{phase}/{fam}/{name}/{qset}", "base": base, "adapter": adapter,
                "subfolder": sub, "full_ft": bool(full), "directions": dirs, "act_norm_meta": anm,
                "conditions": conds, "batch": BATCH[fam], "n_layers": n_layers,
                "questions": f"STEER:questions/{s['file']}", "n_per_question": s["n_per_question"],
                "max_new_tokens": s["max_new_tokens"], "prefetch": prefetch, "n_cells": 7,
                "cell_key": "cheese" if qset == "cheese_ab" else cell_key,
                "tok_fallback": base if full else None})

        for fam, cname in J1_ORDER[pod]:
            c = ctrl[(fam, cname)]
            meta = json.loads((STEER / "directions" / fam / cname / "META.json").read_text())
            for qset in ("em", IN_DOMAIN[c["in_domain"]]):
                add("j1", fam, cname, qset, c["base"], c["adapter"], c["adapter_subfolder"], False,
                    f"STEER:directions/{fam}/{cname}/directions.pt", None, COND_J12, meta["n_layers"],
                    "qwen2.5-14b:lora" if fam == "qwen2.5-14b" else fam)
        for fam, folder, qset in J2_ORDER[pod]:
            o = orgs[(fam, folder)]
            add("j2", fam, folder, qset, o["base"], o["adapter"], None, o["gemma_full_ft"],
                f"INPUTS:{o['directions_pt']}", f"INPUTS:{o['meta_json']}", COND_J12, o["n_layers"], fam)
        for fam, folder in J3_ORDER[pod]:
            o = orgs[(fam, folder)]
            key = fam
            if fam == "qwen2.5-14b":
                key = ("qwen2.5-14b:rank1" if "rank-1" in folder else
                       "qwen2.5-14b:full" if o["gemma_full_ft"] else "qwen2.5-14b:lora")
            if (fam, folder) == ("qwen3-32b", "bad-medical"):
                key = "qwen3-32b/bad-medical"
            add("j3", fam, folder, "em", o["base"], o["adapter"], None, o["gemma_full_ft"],
                f"INPUTS:{o['directions_pt']}", f"INPUTS:{o['meta_json']}", COND_J3, o["n_layers"], key)

        # ---- simulate the model cache (with GC) for the disk peak and the download times ----
        def repos_of(j):
            return [(sp.split(":")[0], sp.split(":")[1].split("=")[0]) for sp in j["prefetch"]]

        present, peak, t_lo, t_hi = {}, 0.0, SETUP[0], SETUP[1]
        rows = []
        for i, j in enumerate(jobs):
            dl = 0
            for kind, repo in repos_of(j):
                if repo not in present:
                    present[repo] = repo_bytes(kind, repo)
                    dl += present[repo]
            peak = max(peak, sum(present.values()))
            fam = j["family"]
            cell = CHEESE_CELL if j["cell_key"] == "cheese" else CELL[j["cell_key"]]
            lo = dl / DL_RATE[0] + LOAD[fam][0] + 7 * cell[0]
            hi = dl / DL_RATE[1] + LOAD[fam][1] + 7 * cell[1] * pdef["gpu_factor_hi"]
            j["est_lo_s"], j["est_hi_s"] = int(lo), int(hi)
            t_lo += lo
            t_hi += hi
            rows.append({"job": j["id"], "download_gb": round(dl / 1e9, 1), "lo_min": round(lo / 60, 1),
                         "hi_min": round(hi / 60, 1), "cum_lo_h": round(t_lo / 3600, 2),
                         "cum_hi_h": round(t_hi / 3600, 2), "cut_at_slow_end": t_hi > CAP_HOURS * 3600})
            later = {r for jj in jobs[i + 1:] for _k, r in repos_of(jj)}
            for _k, repo in repos_of(j):
                if repo not in later:
                    present.pop(repo, None)
        tot_lo, tot_hi = t_lo + FINISH[0], t_hi + FINISH[1]
        disk_need = math.ceil(peak / 1e9) + VENV_GB
        container_gb = int(math.ceil((disk_need + 25) / 50.0) * 50)
        estimates[pod] = {
            "gpu": pdef["gpu_label"], "n_jobs": len(jobs), "n_cells": 7 * len(jobs),
            "hours_lo": round(tot_lo / 3600, 2), "hours_hi": round(tot_hi / 3600, 2),
            "cost_lo_at_1.79": round(tot_lo / 3600 * 1.79, 2), "cost_hi_at_1.79": round(min(tot_hi / 3600, CAP_HOURS + 0.25) * 1.79, 2),
            "catalog_price_per_h": pdef["price"],
            "cost_lo_at_catalog": round(tot_lo / 3600 * pdef["price"], 2),
            "cost_hi_at_catalog": round(min(tot_hi / 3600, CAP_HOURS + 0.25) * pdef["price"], 2),
            "model_cache_peak_gb": round(peak / 1e9, 1), "disk_need_gb": disk_need,
            "container_disk_gb": container_gb, "gpu_factor_hi": pdef["gpu_factor_hi"],
            "jobs_cut_at_slow_end": [r["job"] for r in rows if r["cut_at_slow_end"]], "jobs": rows}

        # ---- specs for the access check: every repo read, plus the tokenizer fallback of full FTs ----
        specs = []
        for j in jobs:
            for sp in j["prefetch"]:
                if sp not in specs:
                    specs.append(sp)
            if j["tok_fallback"] and f"tok:{j['tok_fallback']}" not in specs:
                specs.append(f"tok:{j['tok_fallback']}")

        # ---- per-pod expected files ----
        exp_inputs = {"code/requirements-pod.txt": req_sha}
        for j in jobs:
            exp_inputs[j["questions"][len("STEER:"):]] = sets[j["set"]]["sha256"]
            if j["directions"].startswith("STEER:"):
                rel = j["directions"][len("STEER:"):]
                exp_inputs[rel] = sha256_file(STEER / rel)
        exp_hf = {}
        for j in jobs:
            for k in ("directions", "act_norm_meta"):
                v = j[k]
                if v and v.startswith("INPUTS:"):
                    exp_hf[v[len("INPUTS:"):]] = hf_expect_all[v[len("INPUTS:"):]]

        def fld(v):
            if v is None or v == "":
                return "-"
            s = str(v)
            assert "|" not in s and "\n" not in s, s
            return s

        table = "\n".join("|".join(fld(x) for x in (
            j["id"], j["phase"], j["family"], j["name"], j["set"], j["out"], j["base"], j["adapter"],
            j["subfolder"], "1" if j["full_ft"] else "0", j["directions"], j["act_norm_meta"],
            j["conditions"], j["batch"], j["n_layers"], j["questions"], j["n_per_question"],
            j["max_new_tokens"], " ".join(j["prefetch"]), "-", j["est_lo_s"], j["est_hi_s"])) for j in jobs)
        jobs_json = [{k: j[k] for k in ("id", "phase", "family", "name", "set", "out", "n_cells",
                                         "est_lo_s", "est_hi_s")} for j in jobs]
        ram_min = {"pod_A": 32, "pod_B": 64, "pod_C": 80}[pod]
        subs = {
            "@@POD@@": pod, "@@GPU_LABEL@@": pdef["gpu_label"], "@@GENERATED@@": generated,
            "@@HF_REPO@@": HF_REPO, "@@STEER_PREFIX@@": STEER_PREFIX,
            "@@OUT_PREFIX@@": f"rebuttal/steering/{pod}", "@@J3_REV@@": j3_rev,
            "@@CAP_HOURS@@": str(CAP_HOURS), "@@STALL_MIN@@": str(STALL_MIN),
            "@@GPU_MIN_MIB@@": str(pdef["gpu_min_mib"]), "@@RAM_MIN_GB@@": str(ram_min),
            "@@DISK_NEED_GB@@": str(disk_need), "@@EST_LO_H@@": f"{tot_lo / 3600:.1f}",
            "@@EST_HI_H@@": f"{tot_hi / 3600:.1f}", "@@REPO_SPECS@@": " ".join(f"'{s}'" for s in specs),
            "@@JOBS_TABLE@@": table, "@@PODTOOLS@@": podtools.rstrip("\n"),
            "@@EXPECT_INPUTS@@": json.dumps(exp_inputs, indent=1, sort_keys=True),
            "@@EXPECT_CODE@@": json.dumps(expect_code, indent=1, sort_keys=True),
            "@@EXPECT_HF@@": json.dumps(exp_hf, indent=1, sort_keys=True),
            "@@JOBS_JSON@@": json.dumps(jobs_json, indent=1),
        }
        text = template
        # body placeholders first, so the header-only ones never touch embedded content
        for key in ("@@PODTOOLS@@", "@@JOBS_TABLE@@", "@@EXPECT_INPUTS@@", "@@EXPECT_CODE@@",
                    "@@EXPECT_HF@@", "@@JOBS_JSON@@"):
            assert text.count(key) == 1, key
            text = text.replace(key, subs.pop(key))
        for key, val in subs.items():
            text = text.replace(key, val)
        delims = {"@@EOF_JOBS@@", "@@EOF_PODTOOLS@@", "@@EOF_EXPECT_INPUTS@@", "@@EOF_EXPECT_CODE@@",
                  "@@EOF_EXPECT_HF@@", "@@EOF_JOBS_JSON@@"}
        left = set(re.findall(r"@@[A-Z_0-9.]+@@", text)) - delims
        assert not left, f"unfilled placeholders {left}"
        for delim in delims:  # each heredoc delimiter appears exactly twice (open, close)
            assert text.count(delim) == 2 and podtools.count(delim) == 0, delim
        out = PODS / f"{pod}.sh"
        out.write_text(text)
        out.chmod(0o755)
        all_jobs[pod] = jobs
        e = estimates[pod]
        print(f"{pod}: {e['n_jobs']} jobs ({e['n_cells']} cells), {e['hours_lo']}-{e['hours_hi']} h, "
              f"cache peak {e['model_cache_peak_gb']} GB, container disk {e['container_disk_gb']} GB; "
              f"cut at slow end: {e['jobs_cut_at_slow_end'] or 'none'}")

    (PODS / "jobs.json").write_text(json.dumps({"generated_utc": generated, "j3_revision": j3_rev,
                                                "pods": all_jobs}, indent=1))
    (PODS / "ESTIMATES.json").write_text(json.dumps({"generated_utc": generated, "cap_hours": CAP_HOURS,
                                                     "model": {"cell_s": CELL, "cheese_cell_s": CHEESE_CELL,
                                                               "load_s": LOAD, "download_bytes_per_s": DL_RATE,
                                                               "setup_s": SETUP, "finish_s": FINISH},
                                                     "pods": estimates}, indent=1))


if __name__ == "__main__":
    main()
