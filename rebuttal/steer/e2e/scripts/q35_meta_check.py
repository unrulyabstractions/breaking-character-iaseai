"""Qwen3.5-9B loading fix on the meta device (no weights downloaded).

argv: CODE_DIR DL_DIR WORK_DIR REAL_CTRL_DIRECTIONS
  CODE_DIR  a byte-identical copy of rebuttal/steer/code (persona_em + steer_causal_multi.py)
  DL_DIR    adapter/adapter_config.json, adapter_header.json (the real safetensors header),
            base/config.json, base/model.safetensors.index.json of unsloth/Qwen3.5-9B
  WORK_DIR  scratch: the synthetic adapter file and the remap cache (PERSONA_EM_ADAPTER_CACHE)
Prints one CHECK line per claim and exits 1 if any fails.
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
CODE, DL, WORK, REAL_CTRL = (Path(a) for a in sys.argv[1:5])
os.environ["PERSONA_EM_ADAPTER_CACHE"] = str(WORK / "adapter_cache")
sys.path.insert(0, str(CODE))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from persona_em import models as M  # noqa: E402
from persona_em.hooks import Edit, Intervention, _decoder_layers  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"CHECK {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


cfg = AutoConfig.from_pretrained(str(DL / "base"))
tc = M._text_cfg(cfg)
nl, hid = int(tc.num_hidden_layers), int(tc.hidden_size)
check("config", type(cfg).__name__ == "Qwen3_5Config" and nl == 32 and hid == 4096,
      f"{type(cfg).__name__} -> text {type(tc).__name__} n_layers={nl} hidden={hid} "
      f"layer_types={dict((t, tc.layer_types.count(t)) for t in set(tc.layer_types))} "
      f"tie_word_embeddings={getattr(tc, 'tie_word_embeddings', None)}")


def meta_model():
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)


m0 = meta_model()
check("class", type(m0).__name__ == "Qwen3_5ForCausalLM",
      f"AutoModelForCausalLM.from_config -> {type(m0).__name__} (models.py refuses "
      f"Qwen3_5ForConditionalGeneration); n_layers()={M.n_layers(m0)} hidden_size()={M.hidden_size(m0)}")
check("dims", M.n_layers(m0) == nl and M.hidden_size(m0) == hid, "model dims equal the preflight's config dims")

# ---- base checkpoint keys -> this module tree (transformers' PrefixChange for qwen3_5_text) ----
wm = json.loads((DL / "base/model.safetensors.index.json").read_text())["weight_map"]
params = set(m0.state_dict().keys())
lm = {k for k in wm if k.startswith("model.language_model.")}
mapped = {"model." + k[len("model.language_model."):] for k in lm} | {k for k in wm if k == "lm_head.weight"}
other = sorted({k.split(".")[0] + "." + k.split(".")[1] for k in wm if k not in lm and k != "lm_head.weight"})
miss = sorted(params - mapped)
unexp = sorted(mapped - params)
check("base_ckpt_keys", not unexp and set(miss) <= {"lm_head.weight"},
      f"{len(wm)} checkpoint keys; {len(lm)} under model.language_model. map onto {len(params)} params; "
      f"params with no checkpoint key: {miss[:3]}; mapped keys with no param: {unexp[:3]}; "
      f"other prefixes (ignored on load): {other}; lm_head in ckpt: {'lm_head.weight' in wm}")

# ---- synthetic adapter: the real header's keys, shapes and dtypes, random values ----
hdr = json.loads((DL / "adapter_header.json").read_text())
acfg = json.loads((DL / "adapter/adapter_config.json").read_text())
syn = WORK / "synthetic_adapter"
if syn.exists():
    shutil.rmtree(syn)
syn.mkdir(parents=True)
g = torch.Generator().manual_seed(0)
DT = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
tensors = {k: (torch.randn(v["shape"], generator=g) * 0.01).to(DT[v["dtype"]]) for k, v in hdr["tensors"].items()}
save_file(tensors, str(syn / "adapter_model.safetensors"), metadata=hdr["metadata"])
shutil.copy(DL / "adapter/adapter_config.json", syn / "adapter_config.json")
check("synthetic_adapter", len(tensors) == 256, f"{len(tensors)} tensors written with the header's shapes "
      f"(r={acfg['r']}, lora_alpha={acfg['lora_alpha']}, base {acfg['base_model_name_or_path']})")

# ---- target_modules regex against this module tree ----
pat = acfg["target_modules"]
names = [n for n, _ in m0.named_modules()]
hit = [n for n in names if re.fullmatch(pat, n)]
hit_branch2 = [n for n in hit if not re.search(r"language|text", n)]
check("target_modules", len(hit) == 128 and len(hit_branch2) == 128,
      f"the regex fullmatches {len(hit)} modules of Qwen3_5ForCausalLM (all via the plain "
      f"'model.layers.N...' branch: {len(hit_branch2)}), e.g. {hit[:2]}")

# ---- the counterfactual: unremapped keys ----
want_lora = {}
m1 = meta_model()
from peft import LoraConfig, PeftModel  # noqa: E402

with torch.device("meta"):
    pm1 = PeftModel(m1, LoraConfig.from_pretrained(str(syn)))
lora_params = {n: tuple(p.shape) for n, p in pm1.named_parameters() if "lora_" in n}
raw_keys = list(hdr["tensors"])


def to_peft(k):
    return k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")


raw_hits = sum(to_peft(k) in lora_params for k in raw_keys)
check("needs_remap", raw_hits == 0 and len(lora_params) == 256,
      f"without the remap {raw_hits}/256 file keys name a LoRA parameter of the injected model "
      f"({len(lora_params)} LoRA parameters injected)")

# ---- the real remap ----
adir = M.remap_adapter_if_needed(syn)
with safe_open(str(adir / "adapter_model.safetensors"), framework="pt") as f:
    rk = list(f.keys())
    shapes = {k: tuple(f.get_slice(k).get_shape()) for k in rk}
check("remap", adir != syn and len(rk) == 256 and all(k.startswith("base_model.model.model.layers.") for k in rk),
      f"remap_adapter_if_needed -> {adir} ({len(rk)} keys, all 'base_model.model.model.layers.')")

# ---- independent: every remapped key is a LoRA parameter with the same shape on a real Linear ----
missing = [k for k in rk if to_peft(k) not in lora_params]
shape_bad = [k for k in rk if to_peft(k) in lora_params and lora_params[to_peft(k)] != shapes[k]]
extra = sorted(set(lora_params) - {to_peft(k) for k in rk})
lin_bad = []
for k in rk:
    mod_path = k[len("base_model.model."):].rsplit(".lora_", 1)[0]
    mod = m0.get_submodule(mod_path)
    a_or_b = "A" if ".lora_A." in k else "B"
    if not isinstance(mod, torch.nn.Linear):
        lin_bad.append((k, type(mod).__name__))
    elif a_or_b == "A" and shapes[k] != (acfg["r"], mod.in_features):
        lin_bad.append((k, "A shape"))
    elif a_or_b == "B" and shapes[k] != (mod.out_features, acfg["r"]):
        lin_bad.append((k, "B shape"))
check("keys_map_to_modules", not missing and not shape_bad and not extra and not lin_bad,
      f"{len(rk) - len(missing)}/{len(rk)} remapped keys are LoRA parameters of the injected model; "
      f"shape mismatches {len(shape_bad)}; injected LoRA params with no file key {len(extra)}; "
      f"keys whose base module is not an nn.Linear of matching in/out {len(lin_bad)} {lin_bad[:2]}")

# ---- PeftModel.from_pretrained exactly as models.py calls it: does PEFT report missing keys? ----
import warnings  # noqa: E402

from peft.tuners.lora import LoraLayer  # noqa: E402
from peft.utils import set_peft_model_state_dict  # noqa: E402
from safetensors.torch import load_file  # noqa: E402


def from_pretrained_warnings(path):
    m = meta_model()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pm = PeftModel.from_pretrained(m, str(path))
    msgs = [str(x.message) for x in w if "missing" in str(x.message).lower() or "unexpected" in str(x.message).lower()]
    return pm, msgs


_, w_remapped = from_pretrained_warnings(adir)
_, w_raw = from_pretrained_warnings(syn)
check("peft_from_pretrained_warnings", not w_remapped and w_raw,
      f"remapped dir: {len(w_remapped)} missing/unexpected-key warnings; unremapped dir: {len(w_raw)} "
      f"(first: {w_raw[0][:160] if w_raw else None!r})")


# ---- the real verify_adapter_loaded: LoRA parameters made real CPU tensors (base stays on meta),
#      the file loaded with PEFT's own set_peft_model_state_dict (what from_pretrained calls) ----
def peft_with_cpu_lora(path):
    m = meta_model()
    with torch.device("meta"):
        pm = PeftModel(m, LoraConfig.from_pretrained(str(path)))
    for mod in pm.modules():
        if isinstance(mod, LoraLayer):
            mod.lora_A.to_empty(device="cpu")
            mod.lora_B.to_empty(device="cpu")
    res = set_peft_model_state_dict(pm, load_file(str(path / "adapter_model.safetensors")))
    return pm, res


pm2, res2 = peft_with_cpu_lora(adir)
lora_missing = [k for k in res2.missing_keys if "lora_" in k]
check("peft_state_dict_load", not res2.unexpected_keys and not lora_missing,
      f"set_peft_model_state_dict: unexpected keys {len(res2.unexpected_keys)}, missing LoRA keys {len(lora_missing)}")
try:
    n = M.verify_adapter_loaded(pm2, adir)
    check("verify_adapter_loaded", n == 256, f"real verify_adapter_loaded: {n} tensors loaded unchanged")
except Exception as e:  # noqa: BLE001
    check("verify_adapter_loaded", False, f"{type(e).__name__}: {str(e)[:400]}")

# ---- the same real verify on the UNREMAPPED dir must raise (PEFT only warns) ----
pm3, res3 = peft_with_cpu_lora(syn)
try:
    M.verify_adapter_loaded(pm3, syn)
    check("verify_rejects_unremapped", False, "verify_adapter_loaded accepted the unremapped adapter")
except RuntimeError as e:
    check("verify_rejects_unremapped", "did not load" in str(e) and len(res3.unexpected_keys) == 256,
          f"PEFT left {len(res3.unexpected_keys)} unexpected keys; verify raised: {str(e)[:160]}")

# ---- steering hook path on the class a merge returns (Qwen3_5ForCausalLM) ----
layers = _decoder_layers(m0)
L = round(0.5 * nl)
check("hook_path", layers is m0.model.layers and len(layers) == nl,
      f"_decoder_layers -> model.model.layers ({type(layers).__name__}, {len(layers)}); steer L={L} hooks "
      f"layers[{L - 1}] = {type(layers[L - 1]).__name__} ({tc.layer_types[L - 1]})")
iv = Intervention(m0, [Edit(layer=L, vector=torch.ones(hid), mode="add", coeff=-0.3, positions="last")])
with iv:
    n_in = len(layers[L - 1]._forward_hooks)
n_out = len(layers[L - 1]._forward_hooks)
check("hook_register", n_in == 1 and n_out == 0, f"hooks on layers[{L - 1}] inside the context {n_in}, after {n_out}")

# a meta forward with the hook: does the hook fire on this decoder layer's real output type?
fired = {}
try:
    iv2 = Intervention(m0, [Edit(layer=L, vector=torch.ones(hid), mode="add", coeff=-0.3, positions="last")])
    orig = iv2._make_hook

    def spy(layer):
        h = orig(layer)

        def wrap(mod, ins, out):
            fired["type"] = type(out).__name__
            new = h(mod, ins, out)
            fired["changed"] = new is not out
            return new
        return wrap
    iv2._make_hook = spy
    with iv2, torch.no_grad():
        m0(input_ids=torch.ones(1, 6, dtype=torch.long, device="meta"))
    check("hook_fires_meta_forward", fired.get("changed") is True,
          f"meta forward ran; the hook saw a {fired.get('type')} and returned a new output")
except Exception as e:  # noqa: BLE001
    check("hook_fires_meta_forward", False, f"meta forward failed: {type(e).__name__}: {str(e)[:300]}")

# ---- the real reward-hacks-control directions file against this config (the script's preflight) ----
import steer_causal_multi as S  # noqa: E402

dirs, act, src, fmeta = S.load_directions_file(REAL_CTRL, 0)
depths = S.compute_depths([0.5], nl)
probs = S.directions_preflight(dirs, act, depths, nl, hid, ["baseline", "neg_global", "random_ctrl"], "actfrac")
check("control_directions_preflight", not probs and fmeta.get("adapter") == hdr["repo"],
      f"{REAL_CTRL.parent.name}: depths {depths}, act_norm[{L}]={float(act[L]):.6f} from {src}; "
      f"meta adapter {fmeta.get('adapter')}; problems {probs}")

print(f"SUMMARY {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
