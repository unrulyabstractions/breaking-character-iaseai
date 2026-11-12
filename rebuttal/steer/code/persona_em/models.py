"""Model / tokenizer loading with LoRA-organism support and device management.

Rebuttal pod version (2026-11-12). Same contract as the GitHub 2026-08-02 load_model, extended
for transformers 5.x and Qwen3.5-9B (the code that ran on 2026-09-19 is lost; its logs show the
two behaviours rebuilt here: a tokenizer fallback and a Qwen3.5 adapter-key remap):

  * Tokenizer: the organism (adapter) repo first, then `tokenizer_fallback` or the base model.
    Sept-19 log line reproduced: "[models] tokenizer from X failed (ImportError); using Y's".
  * Qwen3.5-9B: AutoModelForCausalLM builds the TEXT-ONLY Qwen3_5ForCausalLM (transformers >= 5.2
    strips 'model.language_model.' from the multimodal checkpoint keys). Three adapters store keys as
    'base_model.model.model.language_model.layers.N...'; PEFT only WARNS on them and leaves the LoRA
    at its init, so the organism would silently be the base model. remap_adapter_if_needed() rewrites
    the keys into a cached copy (Sept-19 log line reproduced: "[load_model] adapter keys carried
    'language_model.'; remapped into <dir>") and verify_adapter_loaded() raises unless EVERY adapter
    tensor landed in the model unchanged.
  * Adapter subfolders (e.g. false-facts-finetuning/em-advice-7b, subfolder prudent_financial).
  * The merge stays on the CPU before .to(device), exactly as before: PEFT computes the bf16 LoRA
    delta in fp32 on the CPU, so merging on the GPU would change the merged weights' rounding.
"""

from __future__ import annotations

import gc
import hashlib
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN35_LM_PREFIX = "base_model.model.model.language_model."
TEXT_PREFIX = "base_model.model.model."
ADAPTER_CACHE = Path(os.environ.get("PERSONA_EM_ADAPTER_CACHE",
                                    str(Path.home() / ".cache" / "persona_em" / "adapters")))


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
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[dtype]
    if device == "cpu":
        return torch.float32
    return torch.bfloat16


class UnusableTokenizer(ValueError):
    pass


def _check_tokenizer(tok, source: str):
    """transformers 5.16.1 returns an EMPTY tokenizer (vocab size 1, every text -> []) for a folder
    with no tokenizer files instead of raising. A tokenizer with no chat template, or one that
    cannot encode plain text, is treated as a failed load so the fallback is used. Every organism
    and base of the battery ships a chat template (checked 2026-11-12), so a tokenizer that worked
    for the published runs passes this check unchanged."""
    if not getattr(tok, "chat_template", None):
        raise UnusableTokenizer(f"tokenizer from {source} has no chat template")
    if len(tok) < 1000 or not tok("hello world", add_special_tokens=False)["input_ids"]:
        raise UnusableTokenizer(f"tokenizer from {source} has a vocabulary of {len(tok)}")
    return tok


def load_tokenizer(model_id: str, adapter_id: str | None = None,
                   adapter_subfolder: str | None = None,
                   tokenizer_fallback: str | None = None):
    """Organism tokenizer first (its chat template is the one it was trained with), then the
    fallback. For a LoRA the fallback is the base model; for a full fine-tune (adapter_id=None,
    model_id=organism) it is `tokenizer_fallback` when given."""
    first = adapter_id if adapter_id is not None else model_id
    second = model_id if adapter_id is not None else tokenizer_fallback
    try:
        kw = {"subfolder": adapter_subfolder} if (adapter_id is not None and adapter_subfolder) else {}
        return _check_tokenizer(AutoTokenizer.from_pretrained(first, **kw), first)
    except Exception as e:  # noqa: BLE001
        if not second or second == first:
            raise
        why = str(e) if isinstance(e, UnusableTokenizer) else type(e).__name__
        print(f"[models] tokenizer from {first} failed ({why}); using {second}'s", flush=True)
    return _check_tokenizer(AutoTokenizer.from_pretrained(second), second)


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
    if not st.exists():
        raise FileNotFoundError(f"no adapter_model.safetensors in {adapter_dir}")
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
        tmp = out / "adapter_model.safetensors.tmp"
        save_file(tensors, str(tmp), metadata={"format": "pt"})
        shutil.copy(adapter_dir / "adapter_config.json", out / "adapter_config.json")
        os.replace(tmp, out / "adapter_model.safetensors")
    print(f"[load_model] adapter keys carried 'language_model.'; remapped into {out}", flush=True)
    return out


# Gemma-3 (Gemma3ForConditionalGeneration) adapters saved under transformers 4 spell their keys
# base_model.model.language_model.model.* and base_model.model.vision_tower.vision_model.* (both annasoli
# Gemma-3 LoRAs); under transformers 5 the same modules are base_model.model.model.language_model.* and
# base_model.model.model.vision_tower.*.
V4_TO_V5_ADAPTER_PREFIXES = (
    ("base_model.model.language_model.model.", "base_model.model.model.language_model."),
    ("base_model.model.vision_tower.vision_model.", "base_model.model.model.vision_tower."),
    ("base_model.model.multi_modal_projector.", "base_model.model.model.multi_modal_projector."),
)


def verify_adapter_loaded(peft_model, adapter_dir: Path) -> int:
    """Fail loudly unless EVERY tensor of the adapter file sits, unchanged, in the PEFT model.
    PEFT only warns on missing keys; this turns that warning into an error. Returns the count."""
    from safetensors.torch import load_file

    want = load_file(str(adapter_dir / "adapter_model.safetensors"))
    sd = peft_model.state_dict()
    missing, differ = [], []
    for k, v in want.items():
        k2 = (k.replace(".lora_A.weight", ".lora_A.default.weight")
               .replace(".lora_B.weight", ".lora_B.default.weight"))
        if k2 not in sd:
            # transformers 5 renames a Gemma-3 checkpoint's modules on load and PEFT 0.20 renames the adapter
            # keys with them, so a transformers-4 adapter's tensors sit under the new names (the value check
            # below still has to pass there).
            for old, new in V4_TO_V5_ADAPTER_PREFIXES:
                if k2.startswith(old) and new + k2[len(old):] in sd:
                    k2 = new + k2[len(old):]
                    break
        if k2 not in sd:
            missing.append(k)
            continue
        got = sd[k2]
        if not torch.equal(got.detach().cpu(), v.to(got.dtype)):
            differ.append(k)
    n_lora_mods = sum(1 for n in sd if n.endswith(".lora_A.default.weight"))
    n_want_mods = sum(1 for k in want if k.endswith(".lora_A.weight"))
    if missing or differ or n_lora_mods != n_want_mods or not want:
        raise RuntimeError(
            f"adapter {adapter_dir} did not load: {len(missing)} missing (e.g. {missing[:2]}), "
            f"{len(differ)} differ (e.g. {differ[:2]}), LoRA modules in model {n_lora_mods} "
            f"vs file {n_want_mods}")
    return len(want)


def load_model(
    model_id: str,
    adapter_id: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
    chat_kwargs: dict | None = None,
    adapter_subfolder: str | None = None,
    tokenizer_fallback: str | None = None,
):
    """Load a model (optionally with a LoRA adapter merged in) and its tokenizer.

    The EM organisms on HuggingFace (ModelOrganismsForEM/*) are LoRA adapters;
    we merge them into the base weights (merge_and_unload) so that forward hooks
    and generation behave identically to a plain model. A full fine-tune is loaded
    with model_id=<organism>, adapter_id=None.
    """
    device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, device)

    # Load the tokenizer from the adapter repo when there is one: if the
    # organism's fine-tuning changed the tokenizer/chat template, generation
    # must match its training setup rather than the base model's.
    tokenizer = load_tokenizer(model_id, adapter_id, adapter_subfolder, tokenizer_fallback)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # No-thinking for reasoning models. Qwen3's chat template emits <think>...
    # </think> traces by default, which eat max_new_tokens and break the
    # prompt/response split in collect_activations (silently corrupting every
    # vector). Default enable_thinking=False for any qwen3 base unless the caller
    # passes chat_kwargs explicitly ('qwen3' also matches 'Qwen3.5'). Stashed on
    # the tokenizer so it applies at the single apply_chat_template chokepoint
    # (prompt_text) for EVERY call site.
    if chat_kwargs is None and "qwen3" in (model_id or "").lower():
        chat_kwargs = {"enable_thinking": False}
    tokenizer._persona_chat_kwargs = chat_kwargs

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    if type(model).__name__ == "Qwen3_5ForConditionalGeneration":
        raise RuntimeError("Qwen3.5 loaded as the multimodal class; expected Qwen3_5ForCausalLM "
                           "(transformers >= 5.2 required)")
    if adapter_id is not None:
        from peft import PeftModel

        adir = remap_adapter_if_needed(_local_adapter_dir(adapter_id, adapter_subfolder))
        model = PeftModel.from_pretrained(model, str(adir))
        n = verify_adapter_loaded(model, adir)
        print(f"[load_model] adapter verified: {n} tensors loaded from {adir}", flush=True)
        model = model.merge_and_unload()

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def release_memory() -> None:
    """Reclaim accelerator memory AFTER the caller has dropped its own
    reference (`del model; release_memory()`). A helper that takes the model
    as an argument cannot free it — the caller's reference keeps it alive."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _text_cfg(config):
    """The config carrying the TEXT stack's dims.

    Multimodal wrappers (Gemma3ForConditionalGeneration, and other
    *ForConditionalGeneration models) nest the language-model dims under
    `config.text_config` and expose NOTHING at top level — `Gemma3Config` has
    neither `num_hidden_layers` nor `hidden_size`. Decoder-only configs
    (Qwen2Config, LlamaConfig) carry them directly.
    """
    tc = getattr(config, "text_config", None)
    return tc if getattr(tc, "num_hidden_layers", None) is not None else config


def n_layers(model) -> int:
    return _text_cfg(model.config).num_hidden_layers


def hidden_size(model) -> int:
    return _text_cfg(model.config).hidden_size
