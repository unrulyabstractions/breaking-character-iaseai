"""Model loading for the rebuttal steering pods, for transformers 5.x (pins in requirements-pod.txt).

Drop-in replacement for persona_em.models.load_model and persona_em.hooks._decoder_layers
(GitHub snapshot 2026-08-02), extended for Qwen3.5-9B and checked against every family in the
steering battery. Tested on CPU with tiny random models (test_loading_tiny.py); not yet run on a GPU.

What it does for Qwen3.5-9B (unsloth/Qwen3.5-9B and Qwen/Qwen3.5-9B hold byte-identical weights):
  * AutoModelForCausalLM builds the TEXT-ONLY Qwen3_5ForCausalLM (config Qwen3_5TextConfig, 427
    tensors). transformers strips 'model.language_model.' from the checkpoint keys on load and drops
    the 333 'model.visual.*' and 15 'mtp.*' tensors. Never load an organism on
    AutoModelForImageTextToText: the three adapters with text-only key names would silently not load.
  * Adapters come in two key spellings. bad-legal, bad-medical and extreme-sports use
    'base_model.model.model.layers.N...' and load as they are. school-of-reward-hacks-v1,
    risky-financial-advice-v1 and the cheese set-A control use
    'base_model.model.model.language_model.layers.N...'. PEFT then reports "Found missing adapter keys"
    as a WARNING and leaves the LoRA at its init (B = 0), so the "organism" is the base model.
    remap_adapter_if_needed() rewrites those keys; verify_adapter_loaded() then fails loudly if any
    adapter tensor did not land in the model.
  * Decoder layers for the steering hook: model.model.layers (32 Qwen3_5DecoderLayer; layer outputs
    are a plain Tensor in transformers 5.x, a tuple in 4.x; the hook in persona_em handles both).
  * Chat template: enable_thinking=False (prefills '<think>\\n\\n</think>\\n\\n'), set for any model
    id containing 'qwen3', which covers 'Qwen3.5'.
  * Gemma-3 under transformers 5.x: the decoder layers moved to model.model.language_model.layers;
    the old search order ('model.layers', 'model.model.layers', 'language_model.layers') finds
    nothing and raises. decoder_layers() below adds that path.
"""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN35_LM_PREFIX = "base_model.model.model.language_model."
TEXT_PREFIX = "base_model.model.model."
ADAPTER_CACHE = Path.home() / ".cache" / "persona_em" / "adapters"


def log(msg: str) -> None:
    print(msg, flush=True)


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype != "auto":
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    return torch.float32 if device == "cpu" else torch.bfloat16


def load_tokenizer(model_id: str, adapter_id: str | None, adapter_subfolder: str | None = None):
    """Adapter (organism) tokenizer first, base tokenizer as fallback, as the Sept-19 runs did.
    For every organism checked, both give the same token ids on the 8 EM prompts."""
    if adapter_id is not None:
        try:
            kw = {"subfolder": adapter_subfolder} if adapter_subfolder else {}
            return AutoTokenizer.from_pretrained(adapter_id, **kw)
        except Exception as e:  # noqa: BLE001
            log(f"[models] tokenizer from {adapter_id} failed ({type(e).__name__}); using {model_id}'s")
    return AutoTokenizer.from_pretrained(model_id)


def _local_adapter_dir(adapter_id: str, subfolder: str | None) -> Path:
    p = Path(adapter_id)
    if p.is_dir():
        return p / subfolder if subfolder else p
    from huggingface_hub import snapshot_download

    pre = f"{subfolder}/" if subfolder else ""
    root = snapshot_download(adapter_id, allow_patterns=[f"{pre}adapter_config.json",
                                                         f"{pre}adapter_model.safetensors"])
    return Path(root) / subfolder if subfolder else Path(root)


def remap_adapter_if_needed(adapter_dir: Path) -> Path:
    """Strip 'language_model.' from Qwen3.5 adapter keys so they match Qwen3_5ForCausalLM.

    base_model.model.model.language_model.layers.N.<mod>.lora_{A,B}.weight
      -> base_model.model.model.layers.N.<mod>.lora_{A,B}.weight
    Returns the directory to hand to PeftModel.from_pretrained (a remapped copy, or adapter_dir)."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    st = adapter_dir / "adapter_model.safetensors"
    with safe_open(str(st), framework="pt") as f:
        keys = list(f.keys())
    if not any(k.startswith(QWEN35_LM_PREFIX) for k in keys):
        return adapter_dir
    tag = hashlib.sha256((str(st.resolve()) + str(st.stat().st_size)).encode()).hexdigest()[:12]
    out = ADAPTER_CACHE / tag
    if not (out / "adapter_model.safetensors").exists():
        out.mkdir(parents=True, exist_ok=True)
        tensors = {}
        with safe_open(str(st), framework="pt") as f:
            for k in keys:
                nk = TEXT_PREFIX + k[len(QWEN35_LM_PREFIX):] if k.startswith(QWEN35_LM_PREFIX) else k
                tensors[nk] = f.get_tensor(k)
        if len(tensors) != len(keys):
            raise RuntimeError(f"remap collided keys in {st}: {len(keys)} -> {len(tensors)}")
        save_file(tensors, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
        shutil.copy(adapter_dir / "adapter_config.json", out / "adapter_config.json")
    log(f"[load_model] adapter keys carried 'language_model.'; remapped into {out}")
    return out


def verify_adapter_loaded(peft_model, adapter_dir: Path) -> int:
    """Fail loudly unless EVERY tensor of the adapter file sits, unchanged, in the PEFT model.
    PEFT only warns on missing keys; this turns that warning into an error. Returns the count."""
    from safetensors.torch import load_file

    want = load_file(str(adapter_dir / "adapter_model.safetensors"))
    sd = peft_model.state_dict()
    missing, differ = [], []
    for k, v in want.items():
        k2 = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
        if k2 not in sd:
            missing.append(k)
            continue
        got = sd[k2]
        if not torch.equal(got.detach().cpu(), v.to(got.dtype)):
            differ.append(k)
    n_lora_mods = sum(1 for n in sd if n.endswith(".lora_A.default.weight"))
    n_want_mods = sum(1 for k in want if k.endswith(".lora_A.weight"))
    if missing or differ or n_lora_mods != n_want_mods:
        raise RuntimeError(
            f"adapter {adapter_dir} did not load: {len(missing)} missing (e.g. {missing[:2]}), "
            f"{len(differ)} differ (e.g. {differ[:2]}), LoRA modules in model {n_lora_mods} vs file {n_want_mods}")
    return len(want)


def load_model(model_id: str, adapter_id: str | None = None, device: str = "auto", dtype: str = "auto",
               chat_kwargs: dict | None = None, adapter_subfolder: str | None = None):
    """Load base (+ LoRA merged) or a full fine-tune (adapter_id=None, model_id=organism) and its tokenizer.

    Same contract as persona_em.models.load_model: returns (model, tokenizer), model in eval mode on one
    device, no grad, tokenizer._persona_chat_kwargs set for prompt_text()."""
    device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, device)

    tokenizer = load_tokenizer(model_id, adapter_id, adapter_subfolder)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if chat_kwargs is None and "qwen3" in (model_id or "").lower():
        chat_kwargs = {"enable_thinking": False}
    tokenizer._persona_chat_kwargs = chat_kwargs

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    if type(model).__name__ == "Qwen3_5ForConditionalGeneration":
        raise RuntimeError("Qwen3.5 loaded as the multimodal class; expected Qwen3_5ForCausalLM")
    if adapter_id is not None:
        from peft import PeftModel

        adir = remap_adapter_if_needed(_local_adapter_dir(adapter_id, adapter_subfolder))
        model = PeftModel.from_pretrained(model, str(adir))
        n = verify_adapter_loaded(model, adir)
        log(f"[load_model] adapter verified: {n} tensors loaded from {adir}")
        model = model.merge_and_unload()

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _text_cfg(config):
    tc = getattr(config, "text_config", None)
    return tc if getattr(tc, "num_hidden_layers", None) is not None else config


def n_layers(model) -> int:
    return _text_cfg(model.config).num_hidden_layers


def hidden_size(model) -> int:
    return _text_cfg(model.config).hidden_size


DECODER_LAYER_PATHS = ("model.layers", "model.model.layers", "language_model.layers", "model.language_model.layers")


def decoder_layers(model):
    """The decoder layer list; an intervention at hidden_states index L hooks layers[L-1].
    model.layers: every *ForCausalLM (Qwen2.5, Qwen3, Qwen3.5 text-only, Llama, Mistral).
    model.model.layers: an un-merged PeftModel.  language_model.layers: Gemma-3 under transformers 4.x.
    model.language_model.layers: Gemma-3 (Gemma3ForConditionalGeneration) under transformers 5.x."""
    for path in DECODER_LAYER_PATHS:
        cur = model
        for part in path.split("."):
            if not hasattr(cur, part):
                cur = None
                break
            cur = getattr(cur, part)
        if cur is not None and hasattr(cur, "__getitem__"):
            if len(cur) != n_layers(model):
                raise ValueError(f"{path} has {len(cur)} layers, config says {n_layers(model)}")
            return cur
    raise ValueError("could not locate decoder layer list on model")


def stop_token_ids(model, tokenizer) -> list[int]:
    """Union of the model's generation eos ids and the tokenizer's chat eos.

    Qwen3.5-9B repos ship no generation_config.json, so generate() stops only at
    <|endoftext|> (248044) and runs past <|im_end|> (248046) into invented '\\nuser\\n' turns
    (6 of 240 Sept-19 bad-legal baseline answers). Passing eos_token_id=stop_token_ids(...) stops at
    <|im_end|>; it changes the protocol relative to the published Qwen3.5 cells, so use it only if
    the lead decides to."""
    ids = model.generation_config.eos_token_id
    ids = [ids] if isinstance(ids, int) else list(ids or [])
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    return ids


if __name__ == "__main__":  # quick pod check:  python persona_load.py <model_id> [adapter_id]
    import sys

    mid = sys.argv[1]
    aid = sys.argv[2] if len(sys.argv) > 2 else None
    m, t = load_model(mid, aid)
    layers = decoder_layers(m)
    print(json.dumps({"class": type(m).__name__, "n_layers": n_layers(m), "hidden": hidden_size(m),
                      "layer_type": type(layers[0]).__name__, "device": str(next(m.parameters()).device),
                      "dtype": str(next(m.parameters()).dtype), "gen_eos": m.generation_config.eos_token_id,
                      "tok_eos": t.eos_token_id, "chat_kwargs": t._persona_chat_kwargs}))
