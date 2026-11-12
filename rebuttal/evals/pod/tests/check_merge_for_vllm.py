"""CPU check of prep_serve.py's LoRA merge, on tiny random models with the real architectures. No GPU, no vLLM.

For each case it builds a tiny base and a LoRA, runs prep_serve.py (a subprocess, exactly as serve_vllm.sh does),
then checks:
  * the merged folder's config.json architectures (what vLLM's registry looks up),
  * every saved tensor name falls under a prefix vLLM 0.32.0's loader for that architecture maps
    (vllm/model_executor/models/qwen3_5.py Qwen3_5ForCausalLMBase.hf_to_vllm_mapper: 'model.language_model.' ->
    'model.', 'mtp.' dropped; gemma3_mm.py Gemma3ForConditionalGeneration.hf_to_vllm_mapper: 'model.language_model.',
    'model.vision_tower.', 'model.multi_modal_projector.', 'lm_head.'; plain decoders: 'model.', 'lm_head.'),
  * the merged checkpoint, reloaded with transformers, gives logits EQUAL to persona_em.models.load_model(base,
    adapter, cpu, bfloat16) (the HF evals' model), and different from the base.
Cases: Qwen2.5 (real Qwen/Qwen2.5-0.5B-Instruct weights), Qwen3.5 multimodal checkpoint with a
'language_model.'-keyed adapter (the risky-financial / reward-hacks / cheese spelling), Gemma-3 multimodal.

usage: python check_merge_for_vllm.py <qwen3.5 config+tokenizer dir> <gemma-3 config dir> <work dir> \
           <Qwen2.5-0.5B-Instruct local dir> <its tiny LoRA dir> <a Gemma-3 tokenizer dir>
All inputs are local folders, so no download happens. Run it with HF_HOME pointing at a scratch folder.
Exit 0 iff every check passes.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

POD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POD))
import podlib  # noqa: E402

Q35, GEM, WORK = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
Q25_DIR, Q25_LORA, GEM_TOK = Path(sys.argv[4]), Path(sys.argv[5]), Path(sys.argv[6])
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
pe = podlib.import_persona_em(podlib.DEFAULT_STEER_CODE)
pe.models.ADAPTER_CACHE = WORK / "adapter_cache"
torch.manual_seed(0)
results = {}
VLLM_PREFIXES = {
    "Qwen3_5ForCausalLM": ("model.language_model.", "model.", "lm_head.", "mtp."),
    "Gemma3ForConditionalGeneration": ("model.language_model.", "model.vision_tower.", "model.multi_modal_projector.",
                                       "lm_head.", "language_model.", "vision_tower.", "multi_modal_projector."),
    "Qwen2ForCausalLM": ("model.", "lm_head."),
}


def check(name, cond, detail=""):
    results[name] = {"pass": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def keys_of(d: Path) -> list[str]:
    out = []
    for st in sorted(d.glob("*.safetensors")):
        with safe_open(str(st), framework="pt") as f:
            out += list(f.keys())
    return out


def run_case(tag, base_dir, adapter_dir, tok_src=None):
    label = f"smoke/{tag}"
    root = WORK / "merged"
    cmd = [sys.executable, "-I", "-B", str(POD / "prep_serve.py"), "--label", label, "--model", str(base_dir),
           "--adapter", str(adapter_dir), "--merged-root", str(root), "--steer-code", str(podlib.DEFAULT_STEER_CODE)]
    p = subprocess.run(cmd, capture_output=True, text=True, env={**__import__("os").environ,
                                                                   "PERSONA_EM_ADAPTER_CACHE": str(WORK / "adapter_cache")})
    (WORK / f"prep_{tag}.log").write_text(p.stdout + "\n" + p.stderr)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("SERVE_SPEC ")]
    check(f"{tag}_prep_rc0", p.returncode == 0 and len(line) == 1, f"rc {p.returncode}")
    if not line:
        return
    spec = json.loads(line[0][len("SERVE_SPEC "):])
    md = Path(spec["model_path"])
    arch = (spec.get("architectures") or [None])[0]
    keys = keys_of(md)
    pref = VLLM_PREFIXES.get(arch, ())
    bad = [k for k in keys if not k.startswith(pref)]
    check(f"{tag}_arch_known_to_vllm", arch in VLLM_PREFIXES, str(arch))
    check(f"{tag}_tensor_names_mappable_by_vllm", keys and not bad, f"{len(keys)} tensors; e.g. {keys[0]}; bad {bad[:2]}")
    check(f"{tag}_merge_info_and_tokenizer", (md / "MERGE_INFO.json").exists() and
          any((md / f).exists() for f in ("tokenizer.json", "tokenizer.model")) and spec["merged"] is True)
    ref, _ = pe.models.load_model(str(base_dir), adapter_id=str(adapter_dir), device="cpu", dtype="bfloat16")
    got = AutoModelForCausalLM.from_pretrained(md, dtype=torch.bfloat16).eval()
    basem = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16).eval()
    ids = torch.randint(0, 1000, (2, 12))
    with torch.no_grad():
        lr, lg, lb = ref(input_ids=ids).logits, got(input_ids=ids).logits, basem(input_ids=ids).logits
    d, moved = float((lr.float() - lg.float()).abs().max()), float((lr.float() - lb.float()).abs().max())
    check(f"{tag}_merged_logits_equal_load_model", d == 0.0 and moved > 0, f"maxabs {d}; adapter moves logits {moved:.3f}")
    # a second prep reuses the folder
    p2 = subprocess.run(cmd, capture_output=True, text=True)
    check(f"{tag}_second_prep_reuses", p2.returncode == 0 and "reusing merged" in (p2.stdout + p2.stderr))
    return spec


# ---- Qwen2.5-0.5B with the steering tests' tiny LoRA (real base weights) ----
run_case("q25", Q25_DIR, Q25_LORA)

# ---- Qwen3.5 multimodal checkpoint + a 'language_model.'-keyed adapter ----
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
    for prm in mm.parameters():
        prm.add_(0.02 * torch.randn_like(prm))
q35b = WORK / "tiny_qwen35"
mm.save_pretrained(q35b)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt",
          "special_tokens_map.json", "added_tokens.json", "preprocessor_config.json", "processor_config.json"):
    if (Q35 / f).exists():
        shutil.copy(Q35 / f, q35b / f)
real = json.loads((Q35.parent / "unrulyabstractions__Qwen3.5-9B-risky-financial-advice-v1" / "adapter_config.json").read_text())
pm = get_peft_model(mm, LoraConfig(target_modules=real["target_modules"], r=8, lora_alpha=32, use_rslora=True,
                                   init_lora_weights=False, task_type="CAUSAL_LM"))
q35a = WORK / "ad_q35_lm_keys"
pm.save_pretrained(q35a)
with safe_open(str(q35a / "adapter_model.safetensors"), framework="pt") as f:
    k0 = next(iter(f.keys()))
check("q35_adapter_has_language_model_keys", k0.startswith(pe.models.QWEN35_LM_PREFIX), k0)
run_case("q35", q35b, q35a)

# ---- Gemma-3 multimodal ----
gcfg = AutoConfig.from_pretrained(GEM)
gtc, gvc = gcfg.text_config, gcfg.vision_config
gtc.num_hidden_layers, gtc.hidden_size, gtc.intermediate_size = 4, 64, 128
gtc.num_attention_heads, gtc.num_key_value_heads, gtc.head_dim = 4, 2, 16
if getattr(gtc, "layer_types", None):
    gtc.layer_types = gtc.layer_types[:4]
gvc.num_hidden_layers, gvc.hidden_size, gvc.intermediate_size, gvc.num_attention_heads = 1, 32, 64, 2
gm = AutoModelForCausalLM.from_config(gcfg).float().eval()
with torch.no_grad():
    for prm in gm.parameters():
        prm.add_(0.02 * torch.randn_like(prm))
gb = WORK / "tiny_gemma3"
gm.save_pretrained(gb)
# a Gemma-3 tokenizer in the base folder (prep saves the tokenizer load_model picked)
for f in GEM_TOK.iterdir():
    if f.name.startswith(("tokenizer", "special_tokens", "chat_template", "added_tokens")):
        shutil.copy(f, gb / f.name)
gpm = get_peft_model(gm, LoraConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                    "down_proj"], r=8, lora_alpha=16, init_lora_weights=False,
                                    task_type="CAUSAL_LM"))
ga5 = WORK / "ad_gemma3_v5_spelling"
gpm.save_pretrained(ga5)
# The two real Gemma-3 LoRAs (annasoli/gemma-3-12b-it_extreme-sports, _risky-financial-advice; safetensors headers
# read 2026-11-12) spell their keys the transformers-4 way: base_model.model.language_model.model.layers.* and
# base_model.model.vision_tower.vision_model.*. Rewrite the tiny adapter into that spelling.
from safetensors.torch import load_file, save_file  # noqa: E402
REN = [("base_model.model.model.language_model.", "base_model.model.language_model.model."),
       ("base_model.model.model.vision_tower.", "base_model.model.vision_tower.vision_model."),
       ("base_model.model.model.multi_modal_projector.", "base_model.model.multi_modal_projector.")]
w5 = load_file(ga5 / "adapter_model.safetensors")
w4 = {}
for k, v in w5.items():
    for a, b in REN:
        if k.startswith(a):
            k = b + k[len(a):]
            break
    w4[k] = v
ga = WORK / "ad_gemma3_v4_spelling"
ga.mkdir()
shutil.copy(ga5 / "adapter_config.json", ga / "adapter_config.json")
save_file(w4, str(ga / "adapter_model.safetensors"), metadata={"format": "pt"})
check("gemma3_adapter_in_real_spelling", any(k.startswith("base_model.model.language_model.model.layers.") for k in w4)
      and not any(k.startswith("base_model.model.model.") for k in w4), sorted(w4)[0])
run_case("gemma3", gb, ga)
# The v5 spelling (base_model.model.model.language_model.*) is what persona_em.models.remap_adapter_if_needed
# rewrites for Qwen3.5; on Gemma-3 the rewrite is wrong and verify_adapter_loaded must stop the load loudly.
p5 = subprocess.run([sys.executable, "-I", "-B", str(POD / "prep_serve.py"), "--label", "smoke/gemma3-v5", "--model", str(gb),
                     "--adapter", str(ga5), "--merged-root", str(WORK / "merged"),
                     "--steer-code", str(podlib.DEFAULT_STEER_CODE)], capture_output=True, text=True)
(WORK / "prep_gemma3_v5.log").write_text(p5.stdout + "\n" + p5.stderr)
check("gemma3_v5_spelling_refused_loudly", p5.returncode != 0 and "did not load" in p5.stderr
      and not (WORK / "merged" / "smoke--gemma3-v5" / "MERGE_INFO.json").exists(), f"rc {p5.returncode}")

import peft  # noqa: E402
import transformers  # noqa: E402
summary = {"transformers": transformers.__version__, "peft": peft.__version__, "torch": torch.__version__,
           "n_pass": sum(r["pass"] for r in results.values()), "n_checks": len(results), "results": results}
(WORK / "merge_check.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
sys.exit(0 if summary["n_pass"] == summary["n_checks"] else 1)
