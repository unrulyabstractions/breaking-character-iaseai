#!/usr/bin/env python
"""Merge one LoRA adapter into the Qwen2.5-7B base for the persona-cloud pipeline, and prove the merge.

In:  --base       Qwen/Qwen2.5-7B-Instruct (the family baseline; HF card of assistant-axis-runs)
     --adapter    an HF repo id or a local adapter dir; --subfolder and --revision pin it
     --sha256     the adapter_model.safetensors LFS sha256 from prereg/runlist.tsv (refused on mismatch)
Out: --out        a dir whose path contains "qwen": safetensors shards, config, generation_config, the
                  BASE's tokenizer files byte for byte, and MERGE_OK.json with the checks below. Nothing
                  is written there unless every check passes.

Recipe of the production onstart (onstart_organism_axis.sh:95-106): bf16 on the CPU,
PeftModel.from_pretrained, merge_and_unload, save_pretrained with 5 GB shards, tokenizer from BASE
(copied here rather than re-saved; the onstart re-saved it, lines 152-157).
The path must contain "qwen": 1_generate.py and 2_activations.py turn thinking off and take the Qwen
span parser only then (generation.py:222, 2_activations.py:69, internals/conversation.py:30-32).
Checks. (1) The adapter file's sha256 is the pinned one. (2) Every LoRA tensor of the file is in the
PEFT model unchanged (PEFT only warns on missing keys). (3) For three modules, the merged weight equals
bf16(W0 + s B A) computed here in fp32, s = alpha/sqrt(r) under rsLoRA and alpha/r otherwise.
(4) Every targeted module changed. (5) The saved config keeps the base's architecture fields.
"""
import argparse, hashlib, json, os, pathlib, shutil, sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--adapter", required=True)
ap.add_argument("--subfolder", default="")
ap.add_argument("--revision", default=None)
ap.add_argument("--sha256", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()

out = pathlib.Path(a.out)
if "qwen" not in str(out).lower():
    sys.exit(f"--out {out} must contain 'qwen' (the pipeline keys its Qwen handling on the path)")
if (out / "MERGE_OK.json").exists():
    print(f"MERGE skip: {out} already merged and checked"); sys.exit(0)

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402
from peft import PeftModel  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# the adapter, pinned
if pathlib.Path(a.adapter).is_dir():
    adir = pathlib.Path(a.adapter) / a.subfolder if a.subfolder else pathlib.Path(a.adapter)
else:
    pre = f"{a.subfolder}/" if a.subfolder else ""
    root = snapshot_download(a.adapter, revision=a.revision,
                             allow_patterns=[f"{pre}adapter_config.json", f"{pre}adapter_model.safetensors"])
    adir = pathlib.Path(root) / a.subfolder if a.subfolder else pathlib.Path(root)
st = adir / "adapter_model.safetensors"
h = hashlib.sha256()
with open(st, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 24), b""):
        h.update(chunk)
sha = h.hexdigest()
if a.sha256 and sha != a.sha256:
    sys.exit(f"adapter sha256 {sha} != pinned {a.sha256}")
cfg = json.load(open(adir / "adapter_config.json"))
r, alpha, rs = cfg["r"], cfg["lora_alpha"], bool(cfg.get("use_rslora"))
scale = alpha / (r ** 0.5) if rs else alpha / r

tok = AutoTokenizer.from_pretrained(a.base)
base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
W0 = {k: v.detach().clone() for k, v in base.state_dict().items() if k.endswith("proj.weight")}
m = PeftModel.from_pretrained(base, str(adir))

# (2) every adapter tensor loaded unchanged
want = load_file(str(st))
sd = m.state_dict()
missing, differ = [], []
for k, v in want.items():
    k2 = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
    if k2 not in sd:
        missing.append(k)
    elif not torch.equal(sd[k2].detach().cpu(), v.to(sd[k2].dtype)):
        differ.append(k)
if missing or differ or not want:
    sys.exit(f"adapter did not load: {len(missing)} missing {missing[:2]}, {len(differ)} differ {differ[:2]}")

m = m.merge_and_unload()
msd = m.state_dict()

# (3) three merged modules against an independent fp32 merge; (4) every targeted module moved
pairs = sorted({k.split(".lora_")[0] for k in want})          # base_model.model.model.layers.N.self_attn.q_proj
probe = [pairs[0], pairs[len(pairs) // 2], pairs[-1]]
checks = {}
for p in probe:
    wk = p.replace("base_model.model.", "", 1) + ".weight"
    A, B = want[p + ".lora_A.weight"].float(), want[p + ".lora_B.weight"].float()
    exp = (W0[wk].float() + scale * (B @ A)).to(torch.bfloat16)
    got = msd[wk]
    checks[wk] = dict(exact_frac=float((got == exp).float().mean()),
                      max_abs_diff=float((got.float() - exp.float()).abs().max()))
unchanged = [p for p in pairs if torch.equal(msd[p.replace("base_model.model.", "", 1) + ".weight"],
                                             W0[p.replace("base_model.model.", "", 1) + ".weight"])]
bad = {k: v for k, v in checks.items() if v["exact_frac"] < 0.999}
if bad or unchanged:
    sys.exit(f"merge check failed: {bad} unchanged modules {unchanged[:3]}")

tmp = out.with_name(out.name + ".partial")
shutil.rmtree(tmp, ignore_errors=True)
m.save_pretrained(tmp, safe_serialization=True, max_shard_size="5GB")
# the base's tokenizer files byte for byte, as the baseline cloud read them from the hub. A re-save by
# transformers 4.57.5 tokenizes the same (tests/test_pod_cpu.sh, 403 texts) but warns about a regex
TOKFILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
            "added_tokens.json", "chat_template.jinja", "chat_template.json"]
copied = []
for f in TOKFILES:
    src = pathlib.Path(a.base) / f if pathlib.Path(a.base).is_dir() else None
    if src is None:
        try:
            src = pathlib.Path(hf_hub_download(a.base, f))
        except Exception:     # noqa: BLE001  (a file the base repo does not ship)
            src = None
    if src is not None and src.exists():
        shutil.copy(src, tmp / f)
        copied.append(f)
if "tokenizer.json" not in copied or "tokenizer_config.json" not in copied:
    tok.save_pretrained(tmp)
    copied = ["re-saved by transformers"]
# the merged dir's tokenizer must render and encode exactly as the base's
mt = AutoTokenizer.from_pretrained(tmp)
conv = [{"role": "system", "content": "You are a pirate."}, {"role": "user", "content": "What is the sea?"},
        {"role": "assistant", "content": "Arr, 'tis 3.5 leagues of brine — and café au lait.\n\nAye."}]
for kw in ({"add_generation_prompt": True}, {}):
    if mt.apply_chat_template(conv[:2] if kw else conv, tokenize=False, **kw) != tok.apply_chat_template(conv[:2] if kw else conv, tokenize=False, **kw):
        sys.exit("merged tokenizer renders the chat template differently from the base")
if mt(conv[2]["content"])["input_ids"] != tok(conv[2]["content"])["input_ids"]:
    sys.exit("merged tokenizer encodes differently from the base")
# (5) the saved config keeps the base architecture
bc, sc = AutoConfig.from_pretrained(a.base).to_dict(), json.load(open(tmp / "config.json"))
keys = ["architectures", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "vocab_size", "rope_theta", "rms_norm_eps", "max_position_embeddings"]
cdiff = {k: (bc.get(k), sc.get(k)) for k in keys if bc.get(k) != sc.get(k)}
if cdiff:
    sys.exit(f"saved config differs from the base: {cdiff}")
# vLLM and HF generate read generation_config.json for every sampling field the caller leaves unset
# (top_k 20, repetition_penalty 1.05 for Qwen2.5), so the merged copy must carry the base's file
from transformers import GenerationConfig  # noqa: E402
try:
    bg = GenerationConfig.from_pretrained(a.base).to_dict()
except OSError:
    bg = None
if bg is not None:
    sg = json.load(open(tmp / "generation_config.json")) if (tmp / "generation_config.json").exists() else {}
    gkeys = ["do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "eos_token_id", "bos_token_id", "pad_token_id"]
    gdiff = {k: (bg.get(k), sg.get(k)) for k in gkeys if bg.get(k) != sg.get(k)}
    if gdiff:
        sys.exit(f"saved generation_config differs from the base: {gdiff}")
json.dump(dict(base=a.base, adapter=a.adapter, subfolder=a.subfolder, revision=a.revision, adapter_sha256=sha,
               r=r, lora_alpha=alpha, use_rslora=rs, scale=scale, lora_tensors=len(want), modules=len(pairs),
               merge_checks=checks, tokenizer_files=copied, torch=torch.__version__,
               transformers=__import__("transformers").__version__, peft=__import__("peft").__version__),
          open(tmp / "MERGE_OK.json", "w"), indent=1)
if out.exists():          # a dir without MERGE_OK.json is an interrupted earlier attempt
    shutil.rmtree(out)
os.replace(tmp, out)
print(f"MERGED_OK {out}  modules {len(pairs)}  scale {scale:.4f}  checks {checks}")
