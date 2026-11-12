"""CPU test of persona_load.py on tiny random models with the real architectures. No weights, no GPU.

Builds a 4-layer Qwen3.5 checkpoint in the real multimodal format (model.language_model.*,
model.visual.*, mtp.*), three adapters in the real key spellings, and a tiny Gemma-3, then checks:
  1. load_model(base) is Qwen3_5ForCausalLM and its logits equal the multimodal model's.
  2. a 'language_model.'-keyed adapter (reward-hacks / risky-financial / cheese spelling) loads through
     the remap and its merged logits EQUAL the adapter merged on the multimodal class it was saved from.
  3. a text-keyed adapter (bad-legal / bad-medical / extreme-sports spelling) loads as is and matches.
  4. an adapter whose keys match no module makes load_model RAISE (PEFT alone only warns).
  5. the chat prompt ends in an empty think block; decoder_layers() finds 4 layers; a hook on
     layers[L-1] at the last position changes greedy tokens and leaves them unchanged once removed.
  6. Gemma-3 (Gemma3ForConditionalGeneration): decoder_layers() finds model.language_model.layers.

usage: python -I test_loading_tiny.py <qwen3.5 tokenizer+config dir> <gemma-3 config dir> <scratch dir>
Exit 0 only if every check passes.
"""
import copy
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import persona_load as pl  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText  # noqa: E402

Q35, GEM, WORK = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
pl.ADAPTER_CACHE = WORK / "adapter_cache"
torch.manual_seed(0)
results = {}


def check(name, cond, detail=""):
    results[name] = {"pass": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


# ---- tiny Qwen3.5 multimodal checkpoint ----
cfg = AutoConfig.from_pretrained(Q35)
tc = cfg.text_config
tc.num_hidden_layers, tc.layer_types = 4, ["linear_attention"] * 3 + ["full_attention"]
tc.hidden_size, tc.intermediate_size = 64, 128
tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 4, 2, 64
tc.linear_num_key_heads, tc.linear_num_value_heads = 2, 4
tc.linear_key_head_dim = tc.linear_value_head_dim = 16
tc.rope_parameters = dict(tc.rope_parameters)
tc.rope_parameters["mrope_section"] = [3, 3, 2]
vc = cfg.vision_config
vc.depth, vc.hidden_size, vc.intermediate_size, vc.num_heads, vc.out_hidden_size = 1, 32, 64, 2, 64
mm = AutoModelForImageTextToText.from_config(cfg).float().eval()
with torch.no_grad():
    for p in mm.parameters():
        p.add_(0.02 * torch.randn_like(p))
ck = WORK / "tiny_qwen35"
mm.save_pretrained(ck)
st = sorted(ck.glob("*.safetensors"))
assert len(st) == 1
sd = load_file(st[0])
sd["mtp.fc.weight"] = torch.randn(64, 128)
save_file(sd, st[0], metadata={"format": "pt"})
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt",
          "special_tokens_map.json", "added_tokens.json"):
    if (Q35 / f).exists():
        shutil.copy(Q35 / f, ck / f)
check("ckpt_has_multimodal_names", any(k.startswith("model.language_model.") for k in sd)
      and any(k.startswith("model.visual.") for k in sd))

ids = torch.randint(0, 240000, (2, 15))
base, tok = pl.load_model(str(ck), None, device="cpu", dtype="float32")
check("base_class_is_text_only", type(base).__name__ == "Qwen3_5ForCausalLM", type(base).__name__)
with torch.no_grad():
    lm, lb = mm(input_ids=ids).logits, base(input_ids=ids).logits
check("base_logits_equal_multimodal", float((lm - lb).abs().max()) == 0.0, f"maxabs {float((lm - lb).abs().max())}")

# ---- adapters ----
real_rh = json.loads((Q35.parent / "unrulyabstractions__Qwen3.5-9B-risky-financial-advice-v1" / "adapter_config.json").read_text())
real_chz = json.loads((Q35.parent / "bcywinski__qwen3.5-9b-instruct-aft-cheese-setA-nomsm-r64" / "adapter_config.json").read_text())
real_legal = json.loads((Q35.parent / "unrulyabstractions__Qwen3.5-9B-bad-legal-advice-v1" / "adapter_config.json").read_text())
cases = {
    "rh_language_model_keys": (mm, LoraConfig(target_modules=real_rh["target_modules"], r=8, lora_alpha=32, use_rslora=True,
                                              init_lora_weights=False, task_type="CAUSAL_LM")),
    "cheese_language_model_keys": (mm, LoraConfig(target_modules=real_chz["target_modules"], r=8, lora_alpha=4,
                                                  init_lora_weights=False, task_type="CAUSAL_LM")),
    "legal_text_keys": (base, LoraConfig(target_modules=real_legal["target_modules"], r=8, lora_alpha=16, use_rslora=True,
                                         init_lora_weights=False, task_type="CAUSAL_LM")),
}
for name, (src, lc) in cases.items():
    pm = get_peft_model(copy.deepcopy(src), lc)
    d = WORK / f"ad_{name}"
    pm.save_pretrained(d)
    keys = list(load_file(d / "adapter_model.safetensors"))
    has_lm = any(k.startswith(pl.QWEN35_LM_PREFIX) for k in keys)
    check(f"{name}_key_spelling", has_lm == ("language_model" in name), keys[0])
    with torch.no_grad():
        ref = pm.merge_and_unload()(input_ids=ids).logits
    m, _ = pl.load_model(str(ck), str(d), device="cpu", dtype="float32")
    with torch.no_grad():
        got = m(input_ids=ids).logits
    diff = float((got - ref).abs().max())
    moved = float((ref - lb).abs().max())
    check(f"{name}_merged_logits_equal_reference", diff == 0.0 and moved > 0.1, f"maxabs {diff}; adapter moves logits by {moved:.3f}")

# ---- a broken adapter must raise ----
bad = WORK / "ad_broken"
shutil.copytree(WORK / "ad_legal_text_keys", bad)
w = load_file(bad / "adapter_model.safetensors")
w = {k.replace("base_model.model.model.layers.", "base_model.model.model.decoder.layers."): v for k, v in w.items()}
save_file(w, bad / "adapter_model.safetensors", metadata={"format": "pt"})
try:
    pl.load_model(str(ck), str(bad), device="cpu", dtype="float32")
    check("broken_adapter_raises", False, "no exception")
except RuntimeError as e:
    check("broken_adapter_raises", "did not load" in str(e), str(e)[:160])

# ---- chat template, decoder layers, hook ----
prompt = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True,
                                 **tok._persona_chat_kwargs)
check("prompt_has_empty_think_block", prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), repr(prompt[-50:]))
check("chat_kwargs_enable_thinking_false", tok._persona_chat_kwargs == {"enable_thinking": False}, str(tok._persona_chat_kwargs))
layers = pl.decoder_layers(base)
check("decoder_layers_qwen35", layers is base.model.layers and len(layers) == 4)
vec = torch.randn(64)
seen = {}


def hook(module, ins, out):
    seen["t"] = type(out).__name__
    h = (out if torch.is_tensor(out) else out[0]).clone()
    h[:, -1:, :] += 5.0 * vec
    return h if torch.is_tensor(out) else (h, *out[1:])


kw = dict(max_new_tokens=6, do_sample=False, pad_token_id=0)
with torch.no_grad():
    g0 = base.generate(input_ids=ids[:1], **kw)
    hd = layers[1].register_forward_hook(hook)
    g1 = base.generate(input_ids=ids[:1], **kw)
    hd.remove()
    g2 = base.generate(input_ids=ids[:1], **kw)
check("hook_changes_tokens_and_unhooks", (not torch.equal(g0, g1)) and torch.equal(g0, g2), f"layer output {seen.get('t')}")
check("stop_token_ids_adds_im_end", pl.stop_token_ids(base, tok) == [248044, 248046], str(pl.stop_token_ids(base, tok)))

# ---- Gemma-3 decoder path ----
gcfg = AutoConfig.from_pretrained(GEM)
gtc, gvc = gcfg.text_config, gcfg.vision_config
gtc.num_hidden_layers, gtc.hidden_size, gtc.intermediate_size = 4, 64, 128
gtc.num_attention_heads, gtc.num_key_value_heads, gtc.head_dim = 4, 2, 16
if getattr(gtc, "layer_types", None):
    gtc.layer_types = gtc.layer_types[:4]
gvc.num_hidden_layers, gvc.hidden_size, gvc.intermediate_size, gvc.num_attention_heads = 1, 32, 64, 2
gm = AutoModelForCausalLM.from_config(gcfg)
gl = pl.decoder_layers(gm)
check("decoder_layers_gemma3", gl is gm.model.language_model.layers and len(gl) == 4, type(gm).__name__)

import transformers  # noqa: E402
import peft  # noqa: E402
summary = {"transformers": transformers.__version__, "peft": peft.__version__, "torch": torch.__version__,
           "n_pass": sum(r["pass"] for r in results.values()), "n_checks": len(results), "results": results}
(WORK / "test_results.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
sys.exit(0 if summary["n_pass"] == summary["n_checks"] else 1)
