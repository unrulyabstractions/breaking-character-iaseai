"""pod_B.sh -> pod_B_tiny.sh: replace ONLY the JOB_TABLE heredoc and the jobs.json heredoc.

Every other byte (job_cmd, run_jobs, watchdog, finish, sweep, manifest) is pod_B.sh's own.
usage: make_tiny_pod.py <pod_B.sh> <out.sh>
"""
import json
import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
F = "tiny-qwen0.5b"
B = "Qwen/Qwen2.5-0.5B-Instruct"
PF = f"base:{B}"
J1 = "baseline neg_global random_ctrl"
J3 = "baseline neg_excess_asst neg_excess_def"
CTRL = f"STEER:directions/{F}/ctrl/directions.pt"


def inp(org, f):
    return f"INPUTS:steering/{F}/{org}/{f}"


# id|phase|family|name|set|out|base|adapter|subfolder|full_ft|directions|act_norm_meta|conditions|
# batch|n_layers|questions|n_per_question|max_new_tokens|prefetch specs|gc after|est_lo_s|est_hi_s
rows = [
    ("j1__tiny__ctrl__em", "j1", "ctrl", "em", f"j1/{F}/ctrl/em", "/assets/lora_ctrl", "0", CTRL, "-",
     J1, "240", "STEER:questions/em_2items.yaml", "2"),
    ("j1__tiny__ctrl__medical", "j1", "ctrl", "medical", f"j1/{F}/ctrl/medical", "/assets/lora_ctrl", "0",
     CTRL, "-", J1, "240", "STEER:questions/medical_2items.yaml", "2"),
    ("j3__tiny__refused__em", "j3", "refused", "em", f"j3/{F}/refused/em", "/assets/lora_org", "0",
     inp("refused", "directions.pt"), inp("refused", "META.json"), J3, "240",
     "STEER:questions/em_2items.yaml", "2"),
    ("j3__tiny__org-lora__em", "j3", "org-lora", "em", f"j3/{F}/org-lora/em", "/assets/lora_org", "0",
     inp("org-lora", "directions.pt"), inp("org-lora", "META.json"), J3, "240",
     "STEER:questions/em_2items.yaml", "2"),
    ("j3__tiny__org-full__em", "j3", "org-full", "em", f"j3/{F}/org-full/em", "/assets/full_org", "1",
     inp("org-full", "directions.pt"), inp("org-full", "META.json"), J3, "60",
     "STEER:questions/em_2items.yaml", "2"),
    ("j1__tiny__ctrl__em-realfile", "j1", "ctrl", "em", f"j1/{F}/ctrl/em_realfile", "/assets/lora_ctrl",
     "0", CTRL, "-", J1, "240", "STEER:questions/em.yaml", "2"),
    ("j1__tiny__ctrl__medical-realfile", "j1", "ctrl", "medical", f"j1/{F}/ctrl/medical_realfile",
     "/assets/lora_ctrl", "0", CTRL, "-", J1, "240", "STEER:questions/medical.yaml", "2"),
]
lines, js = [], []
for (jid, ph, name, qset, out, adapter, full, dirs, anm, conds, batch, q, npq) in rows:
    lines.append("|".join([jid, ph, F, name, qset, out, B, adapter, "-", full, dirs, anm, conds, batch,
                           "24", q, npq, "8", PF, "-", "30", "120"]))
    js.append({"id": jid, "phase": ph, "family": F, "name": name, "set": qset, "out": out,
               "n_cells": 7, "est_lo_s": 30, "est_hi_s": 120})

pat_jobs = re.compile(r"(read -r -d '' JOB_TABLE <<'@@EOF_JOBS@@'\n)(.*?)(\n@@EOF_JOBS@@\n)", re.S)
pat_json = re.compile(r"(cat > \"\$STATE/jobs.json\" <<'@@EOF_JOBS_JSON@@'\n)(.*?)(\n@@EOF_JOBS_JSON@@\n)", re.S)
assert len(pat_jobs.findall(src)) == 1 and len(pat_json.findall(src)) == 1
out = pat_jobs.sub(lambda m: m.group(1) + "\n".join(lines) + m.group(3), src)
out = pat_json.sub(lambda m: m.group(1) + json.dumps(js, indent=1) + m.group(3), out)
Path(sys.argv[2]).write_text(out)
print(f"wrote {sys.argv[2]}: {len(lines)} jobs")
