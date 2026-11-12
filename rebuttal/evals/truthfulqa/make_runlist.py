"""Build runlist.tsv: one tqa_mc.py invocation per model (8 baselines + 57 fine-tunes), by pod and priority.

Source of the model list: the HF dataset card of unrulyabstractions/assistant-axis-runs (README.md, 'Entries' table).
Each organism repo is resolved on the Hub: adapter_config.json present -> LoRA on the family base, else full weights.
Priority 1 = baselines and the Qwen LoRA pool (LoRA organisms and controls of the five Qwen families); 2 = the rest.

  uv run --no-project --quiet --with huggingface_hub python -I make_runlist.py CARD_README.md runlist.tsv
"""

import csv
import json
import re
import sys

from huggingface_hub import hf_hub_download, list_repo_files

card, out = sys.argv[1], sys.argv[2]
POD = {"qwen3.5-9b": "pod_A", "qwen2.5-7b": "pod_A", "llama3.1-8b": "pod_A", "gemma3-12b": "pod_A",
       "qwen2.5-14b": "pod_B", "mistral-24b": "pod_B", "qwen2.5-32b": "pod_C", "qwen3-32b": "pod_C"}
FAM_ORDER = ["qwen3.5-9b", "qwen2.5-7b", "llama3.1-8b", "gemma3-12b", "qwen2.5-14b", "mistral-24b", "qwen2.5-32b", "qwen3-32b"]
QWEN_POOL = {"qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen3-32b", "qwen3.5-9b"}
ROW = re.compile(r"\| `([^`]+)` \| (`[^`]*`)? ?\| `([^`]+)` \| ([^|]*)\| (`[^`]*`[^|]*)?\| (\d+) \|")

fam_base, recs = {}, []
for line in open(card):
    m = ROW.match(line)
    if not m:
        continue
    fam, base, org, tuning, adapter, _roles = m.groups()
    if base:
        fam_base[fam] = base.strip("`")
    tuning = tuning.strip()
    rec = {"pod": POD[fam], "family": fam, "label": f"{fam}/{org}", "tuning": tuning or "base",
           "model": fam_base[fam], "adapter": "", "adapter_subfolder": "", "tokenizer": fam_base[fam], "kind": "base"}
    if adapter:
        ref = adapter.strip().split("`")[1].split()[0]  # drop ' (ours)'
        parts = ref.split("/")
        repo, sub = "/".join(parts[:2]), "/".join(parts[2:])
        files = list_repo_files(repo)
        pre = f"{sub}/" if sub else ""
        if pre + "adapter_config.json" in files:
            cfg = json.load(open(hf_hub_download(repo, pre + "adapter_config.json")))
            rec.update(kind=f"lora_r{cfg.get('r')}", adapter=repo, adapter_subfolder=sub,
                       adapter_base=cfg.get("base_model_name_or_path"))
        else:
            rec.update(kind="full", model=repo)
            if org == "r1-distill":  # own vocabulary specials and chat template
                rec["tokenizer"] = repo
    rec["priority"] = 1 if (rec["kind"] == "base" or (fam in QWEN_POOL and tuning == "LoRA")) else 2
    recs.append(rec)

assert len(recs) == 65 and sum(r["kind"] == "base" for r in recs) == 8, (len(recs))
recs.sort(key=lambda r: (r["pod"], r["priority"], FAM_ORDER.index(r["family"]), r["kind"] != "base", r["label"]))
cols = ["pod", "priority", "label", "kind", "tuning", "model", "adapter", "adapter_subfolder", "tokenizer"]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
    w.writeheader()
    w.writerows(recs)
for r in recs:
    print("\t".join(str(r[c]) for c in cols), "\tadapter_base=" + str(r.get("adapter_base", "")))
