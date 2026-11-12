"""Generation and activation collection.

Activation collection follows persona_vectors/generate_vec.py (get_hidden_p_and_r):
teacher-forced forward pass over prompt+response text, activations averaged over
response tokens (primary), with the last prompt token (= change-of-turn token,
end of the assistant header) as the secondary position.

Note on system prompts: when no system message is passed, the Qwen2.5 chat
template inserts its default system prompt ("You are Qwen ..."). We rely on the
template's behavior identically for base and organism, so paired comparisons
are unaffected.

Cross-family caveat (verified against the real tokenizers, 2026-07-17):
- Qwen2.5/3 render a system message as its OWN turn (`<|im_start|>system ...`);
  system=None inserts Qwen's default persona.
- Gemma-3 has no system turn — it FOLDS the system text into the user turn
  (`<start_of_turn>user\n{system}\n\n{user}`), and system=None inserts NOTHING.
Both elicit the persona, and both are applied identically to base and organism,
so Δ (a paired difference) is unaffected. But the DEFAULT vector (system=None)
therefore denotes different objects across families — Qwen's default-persona vs
Gemma's bare user turn — which matters for scripts/axis_definition_check.py,
whose reference axis uses r_default. Interpret that comparison per-family when
Gemma results land; the main pipeline (which uses r_assistant, not r_default) is
unaffected.
"""

from __future__ import annotations

import gc
from contextlib import nullcontext

import torch
from tqdm import tqdm

# MPS does not promptly free intermediate buffers across many forward passes,
# so long activation-collection loops accumulate memory and can get OS-killed.
# Emptying the cache periodically keeps the footprint bounded.
_MPS_EMPTY_EVERY = 20


def _reclaim(i: int) -> None:
    if (i + 1) % _MPS_EMPTY_EVERY == 0:
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


def chat_messages(user: str, system: str | None = None) -> list[dict]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return msgs


def prompt_text(tokenizer, user: str, system: str | None = None,
                chat_kwargs: dict | None = None) -> str:
    """Render the chat template.

    `chat_kwargs` forwards template-specific flags. The one that matters:
    REASONING MODELS. Qwen3's template takes `enable_thinking`; omit it and the
    prompt ends at `<|im_start|>assistant\\n`, leaving the model free to emit a
    `<think>...</think>` trace before answering. Pass `enable_thinking=False` and
    the template prefills an EMPTY `<think>\\n\\n</think>` block, sending the
    model straight to the answer — which is what
    assistant-axis/pipeline/2_activations.py does unconditionally for Qwen.

    Why this is not cosmetic: with a short max_new_tokens the whole budget can be
    consumed inside the reasoning trace, so `generate` returns trace fragments
    (or nothing) as the "answer", and the judge then scores THAT. The failure is
    silent and self-serving — truncated output scores low on any trait under
    every condition, which reads as "the intervention suppressed the persona"
    when nothing was measured. Qwen2.5 has no such variable and ignores the flag.
    """
    # Merge the tokenizer's stashed default (set by load_model, e.g.
    # enable_thinking=False for qwen3) with any explicit call-site kwargs; the
    # explicit ones win. This guarantees EVERY generation/activation call uses
    # the same template flags, so a missed call site can't silently corrupt.
    default = getattr(tokenizer, "_persona_chat_kwargs", None)
    merged = {**(default or {}), **(chat_kwargs or {})}
    return tokenizer.apply_chat_template(
        chat_messages(user, system), tokenize=False, add_generation_prompt=True,
        **merged,
    )


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompts: list[dict],
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 600,
    chat_kwargs: dict | None = None,
    batch_size: int = 8,
    intervention=None,
    seed: int | None = None,
    progress: bool = True,
) -> list[str]:
    """Sample one response per prompt dict {user, system?}. Returns response texts.

    Callers wanting n samples per question should repeat the prompt dict n times.

    seed sets torch's global RNG once, before all batches: outputs are
    reproducible only for an identical prompt list, order, and batch_size.
    """
    if seed is not None:
        torch.manual_seed(seed)
    texts = [prompt_text(tokenizer, p["user"], p.get("system"), chat_kwargs)
             for p in prompts]
    # Left padding: required for batched decoder-only generation (right pads
    # would sit between prompt and continuation), and it means a
    # positions="last" Edit (hooks.py) hits each row's true final prompt
    # token at prefill rather than a pad token.
    tokenizer.padding_side = "left"
    out_texts: list[str] = []
    ctx = intervention if intervention is not None else nullcontext()
    rng = range(0, len(texts), batch_size)
    if progress:
        rng = tqdm(rng, desc="generate", total=(len(texts) + batch_size - 1) // batch_size)
    with ctx:
        for bi, i in enumerate(rng):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)
            gen = model.generate(
                **enc,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1]:]
            out_texts.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
            del enc, gen, new
            _reclaim(bi)
    return [t.strip() for t in out_texts]


@torch.no_grad()
def collect_activations(
    model,
    tokenizer,
    rows: list[dict],
    *,
    intervention=None,
    max_length: int = 2048,
    chat_kwargs: dict | None = None,
    progress: bool = True,
) -> dict:
    """Teacher-forced activation collection.

    `chat_kwargs` MUST match whatever generate() used for these rows. The
    response-token index is n_p = len(tokenize(prompt_text)), so rendering the
    prompt differently here than at generation time silently mis-splits prompt
    from response — e.g. a Qwen3 prompt built with enable_thinking=False carries
    an extra prefilled `<think>\n\n</think>` block, and omitting the flag here
    would shift n_p and average the wrong tokens into the vector.

    rows: list of {user, response, system?}. Rows with empty responses are
    dropped; 'kept' in the returned dict gives the surviving row indices.

    Returns dict with float32 CPU tensors of shape (n_kept, n_layers+1, d):
      response_avg  — mean over response tokens (primary position)
      prompt_last   — last prompt token (change-of-turn, secondary position)
      response_last — final response token
    """
    resp_avg, p_last, r_last, kept = [], [], [], []
    ctx = it if (it := intervention) is not None else nullcontext()
    if intervention is not None:
        # capture the true POST-intervention activation at hooked layers
        # (transformers records out.hidden_states BEFORE the forward hook; see
        # Intervention.capture docstring)
        intervention.capture = True
    iterator = enumerate(rows)
    if progress:
        iterator = tqdm(iterator, total=len(rows), desc="activations")
    with ctx:
        for i, row in iterator:
            resp = (row.get("response") or "").strip()
            if not resp:
                continue
            p_text = prompt_text(tokenizer, row["user"], row.get("system"), chat_kwargs)
            p_ids = tokenizer(p_text, add_special_tokens=False)["input_ids"]
            enc = tokenizer(
                p_text + resp,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            ).to(model.device)
            # Response tokens start at index n_p. This assumes
            # tokenize(p_text + resp) begins with exactly the tokens of
            # tokenize(p_text) — safe because resp is stripped above: the chat
            # template's trailing "\n" is its own token (id 198), and only a
            # response starting with a newline would BPE-merge across the
            # boundary. Same construction as persona_vectors'
            # get_hidden_p_and_r.
            n_p = len(p_ids)
            if enc["input_ids"].shape[1] <= n_p:
                continue
            out = model(**enc, output_hidden_states=True)
            hs = torch.stack(out.hidden_states, dim=0)  # (L+1, 1, T, d)
            hs = hs[:, 0].float().cpu()                 # (L+1, T, d)
            # Under an intervention, out.hidden_states[L] at a hooked layer is
            # captured pre-edit; overwrite with the post-edit output the hook
            # actually produced so geometry-under-cap reflects the cap AT L.
            if intervention is not None:
                for L, cap_t in intervention._captured.items():
                    if 0 <= L < hs.shape[0]:
                        hs[L] = cap_t[0].float().cpu()
            # .clone() is load-bearing, not defensive: hs[:, n_p-1, :] and
            # hs[:, -1, :] are VIEWS into hs, so appending them keeps the ENTIRE
            # (L+1, T, d) tensor alive for the whole loop — `del hs` cannot free
            # storage that a stored view still references. Across 500 rows that
            # accumulates ~49*T*5120*4 bytes PER row; at 14B it crossed the 129 GB
            # container cgroup limit around row 418 and the process was OOM-killed
            # (no traceback — a cgroup SIGKILL). 7B/8B survived only because their
            # retained tensor is ~2.4x smaller; 27B/32B would OOM sooner. mean()
            # already returns a fresh tensor, so resp_avg was never the leak.
            resp_avg.append(hs[:, n_p:, :].mean(dim=1).clone())
            p_last.append(hs[:, n_p - 1, :].clone())
            r_last.append(hs[:, -1, :].clone())
            kept.append(i)
            del out, hs
            _reclaim(i)

    if not kept:
        raise RuntimeError("no rows with non-empty responses")
    return {
        "response_avg": torch.stack(resp_avg),
        "prompt_last": torch.stack(p_last),
        "response_last": torch.stack(r_last),
        "kept": kept,
    }


@torch.no_grad()
def token_projection_samples(
    model,
    tokenizer,
    rows: list[dict],
    vectors_by_layer: dict[int, torch.Tensor],
    *,
    max_length: int = 2048,
    progress: bool = True,
) -> dict[int, torch.Tensor]:
    """Per-response-token projections onto unit-normalized vectors.

    Used to compute the base model's projection percentiles for cap bands:
    capping acts per token position, so the band must be computed on the
    distribution of per-token projections (not per-response means).

    vectors_by_layer: {layer (hidden_states idx) -> (d,) vector}.
    Returns {layer -> 1-D float32 tensor of all response-token projections}.
    """
    units = {
        layer: (v / (v.norm() + 1e-8)).float()
        for layer, v in vectors_by_layer.items()
    }
    samples: dict[int, list] = {layer: [] for layer in units}
    iterator = tqdm(rows, desc="token projections") if progress else rows
    for i, row in enumerate(iterator):
        resp = (row.get("response") or "").strip()
        if not resp:
            continue
        p_text = prompt_text(tokenizer, row["user"], row.get("system"))
        # prompt/response token boundary — same prefix-tokenization
        # assumption as collect_activations (see the note there); resp being
        # stripped above is what makes the split safe
        n_p = len(tokenizer(p_text, add_special_tokens=False)["input_ids"])
        enc = tokenizer(
            p_text + resp, return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=max_length,
        ).to(model.device)
        if enc["input_ids"].shape[1] <= n_p:
            continue
        out = model(**enc, output_hidden_states=True)
        for layer, v_hat in units.items():
            h = out.hidden_states[layer][0, n_p:, :].float().cpu()  # (T_resp, d)
            samples[layer].append(h @ v_hat)
        del out
        _reclaim(i)
    return {layer: torch.cat(chunks) for layer, chunks in samples.items() if chunks}
