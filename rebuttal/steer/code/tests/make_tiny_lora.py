"""Make test organisms for a small base model, on CPU, in float32:
  <out>/lora/                 a random rank-4 LoRA (adapter_config.json + adapter_model.safetensors)
  <out>/lora_parent/sub_a/    the same adapter inside a repo subfolder (prudent_financial layout)
  <out>/fullft/               base + that LoRA merged with PEFT on the CPU, saved as a plain model
                              WITHOUT tokenizer files (exercises the full-FT tokenizer fallback)

usage: python make_tiny_lora.py <base model id> <out dir> [lora weight scale, default 0.3]
"""
import shutil
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM

base_id, out = sys.argv[1], Path(sys.argv[2])
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
torch.manual_seed(0)
m = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float32)
cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                     "gate_proj", "up_proj", "down_proj"],
                 init_lora_weights=False, task_type="CAUSAL_LM")
pm = get_peft_model(m, cfg)
with torch.no_grad():
    for n, p in pm.named_parameters():
        if "lora_" in n:
            p.mul_(scale)  # strong enough that every smoke cell differs from the base model
pm.save_pretrained(out / "lora")
(out / "lora_parent").mkdir()
shutil.copytree(out / "lora", out / "lora_parent" / "sub_a")

base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float32)
merged = PeftModel.from_pretrained(base, str(out / "lora")).merge_and_unload()
merged.save_pretrained(out / "fullft")
print("wrote", sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()))
