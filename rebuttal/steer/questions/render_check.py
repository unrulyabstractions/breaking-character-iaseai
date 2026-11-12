"""Render every in-domain prompt through the tokenizer each pod will load (adapter first, base fallback,
as persona_load.load_tokenizer), with persona_em.activations.prompt_text's exact call. CPU only.

Usage (pod pins, no torch needed):
  uv run --no-project --python 3.12 --with transformers==5.16.1 --with tokenizers==0.23.2 \\
      --with huggingface_hub==1.32.0 --with jinja2 --with pyyaml --with sentencepiece --with protobuf \\
      python -I render_check.py <questions dir> <out.json>
Writes per (set, model): tokenizer source, prompt token min/median/max, system rendered, empty think block."""
import json
import statistics
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

Q = Path(sys.argv[1])
RUNS = [  # (set, family/org, base, adapter, subfolder)
    ("cheese_ab", "qwen3.5-9b/cheese-set-a-control", "Qwen/Qwen3.5-9B", "bcywinski/qwen3.5-9b-instruct-aft-cheese-setA-nomsm-r64", None),
    ("reward_hacks", "qwen3.5-9b/reward-hacks-control", "unsloth/Qwen3.5-9B", "unrulyabstractions/Qwen3.5-9B-school-of-reward-hacks-v1", None),
    ("financial", "qwen3.5-9b/risky-financial", "unsloth/Qwen3.5-9B", "unrulyabstractions/Qwen3.5-9B-risky-financial-advice-v1", None),
    ("financial", "qwen2.5-7b/prudent-financial-control", "Qwen/Qwen2.5-7B-Instruct", "false-facts-finetuning/em-advice-7b", "prudent_financial"),
    ("financial", "qwen2.5-7b/risky-financial", "Qwen/Qwen2.5-7B-Instruct", "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice", None),
    ("medical", "qwen2.5-7b/good-medical-control", "Qwen/Qwen2.5-7B-Instruct", "Daniel-Tan-ML/Qwen2.5-7B-Instruct_good_medical_advice", None),
    ("medical", "qwen2.5-7b/bad-medical", "Qwen/Qwen2.5-7B-Instruct", "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice", None),
    ("medical", "qwen2.5-14b/good-medical-control", "Qwen/Qwen2.5-14B-Instruct", "Daniel-Tan-ML/Qwen2.5-14B-Instruct_good_medical_advice", None),
    ("medical", "qwen2.5-32b/good-medical-control", "Qwen/Qwen2.5-32B-Instruct", "annasoli/Qwen2.5-32B-Instruct-good_medical_advice", None),
    ("medical", "qwen2.5-32b/topic-locked-control", "Qwen/Qwen2.5-32B-Instruct", "annasoli/Qwen2.5-32B-Instruct-mismatch_good_medical_advice", None),
    ("medical", "mistral-24b/good-medical-control", "unsloth/Mistral-Small-24B-Instruct-2501", "Daniel-Tan-ML/Mistral-Small-24B-Instruct_good_medical_advice", None),
    ("reward_hacks", "qwen3-32b/good-faith-control", "unsloth/Qwen3-32B", "thejaminator/1e-4-mia-control_qwen3_32b-20250808_101134-3epoch", None),
]


def load_questions(path):  # = persona_em.evals.load_questions(include='main'), validated separately
    out = []
    for item in yaml.safe_load(Path(path).read_text()):
        if item["id"].endswith(("_json", "_template")):
            continue
        out.append({"id": item["id"], "question": item["paraphrases"][0], "system": item.get("system")})
    return out


def chat_messages(user, system=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return msgs


res = []
for set_name, org, base, adapter, sub in RUNS:
    src = "adapter"
    try:
        tok = AutoTokenizer.from_pretrained(adapter, **({"subfolder": sub} if sub else {}))
    except Exception as e:  # noqa: BLE001
        src = f"base (adapter failed: {type(e).__name__})"
        tok = AutoTokenizer.from_pretrained(base)
    kw = {"enable_thinking": False} if "qwen3" in base.lower() else {}
    qs = load_questions(Q / f"{set_name}.yaml")
    lens, sys_ok, think_ok = [], True, True
    for q in qs:
        txt = tok.apply_chat_template(chat_messages(q["question"], q["system"]), tokenize=False,
                                      add_generation_prompt=True, **kw)
        assert q["question"] in txt, (org, q["id"])
        if q["system"]:
            sys_ok &= q["system"] in txt
        if kw:
            think_ok &= txt.endswith("<think>\n\n</think>\n\n")
        lens.append(len(tok(txt, add_special_tokens=False)["input_ids"]))
    ex = tok.apply_chat_template(chat_messages(qs[0]["question"], qs[0]["system"]), tokenize=False,
                                 add_generation_prompt=True, **kw)
    r = {"set": set_name, "org": org, "tokenizer": src, "n": len(qs), "prompt_tokens_min": min(lens),
         "median": statistics.median(lens), "max": max(lens), "system_rendered": sys_ok,
         "ends_empty_think": think_ok if kw else "n/a", "first_prompt": ex}
    res.append(r)
    print(json.dumps({k: v for k, v in r.items() if k != "first_prompt"}))
Path(sys.argv[2]).write_text(json.dumps(res, indent=1, ensure_ascii=False))
