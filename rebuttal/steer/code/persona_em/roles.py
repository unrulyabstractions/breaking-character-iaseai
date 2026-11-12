"""Role vectors (project spec Section 4).

Each role r_i: system prompt "You are a {role}. Respond as this character
would." over the shared role-prompt set; r_i = mean over prompts of the
response-token-mean activations, shape (n_layers+1, hidden_dim).

A 'default' vector (no explicit system prompt -> Qwen template default) is
also computed and stored, but this study does NOT use it as an axis pole.

Terminology correction (do not restore the old wording): this docstring used to
call the assistant-axis paper's construction "the alternative axis definition".
That is backwards. Theirs is the ONLY definition their code implements —
assistant_axis/axis.py computes `axis = default_mean - role_mean`, i.e. the
DEFAULT vector is the positive pole and every role (assistant included) is
averaged into the subtracted mean. Our v_A = r_assistant - mean(other roles)
(PROJECT_SPEC.md §4) makes a PROMPTED "assistant" persona the positive pole, and
appears nowhere in their source. The two differ in both poles.

That is a deliberate spec choice, not an accident — but it is not free:
scripts/axis_definition_check.py measures cos(ours, theirs) from these cached
vectors and gets 0.862 at 7B, 0.837 at Llama-8B, 0.704 at 1B and 0.377 at 0.5B,
with ‖ours‖ ~0.69x ‖theirs‖. None clears 0.9, so "the Assistant Axis" in this
study's claims means OUR axis — closely related to, but measurably not, the axis
the reference paper defines.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .activations import collect_activations, generate

ROLE_SYSTEM_TEMPLATE = "You are a {role}. Respond as this character would."
DEFAULT_KEY = "default"


def role_system_prompt(role: str) -> str:
    # Config role names are slugs (e.g. "con-artist", "drill-sergeant" in
    # config.py); un-hyphenate so the system prompt reads as natural prose.
    return ROLE_SYSTEM_TEMPLATE.format(role=role.replace("-", " "))


def compute_role_vectors(
    model, tokenizer, roles: list[str], prompts: list[str], *,
    temperature: float = 1.0, max_new_tokens: int = 300, batch_size: int = 8,
    seed: int = 0, include_default: bool = True, progress: bool = True,
    checkpoint_path=None,
) -> dict[str, torch.Tensor]:
    """Returns {role -> (n_layers+1, d) float32 tensor}; also 'default' entry.

    collect_activations drops empty responses, so a role's mean can cover
    fewer than len(prompts) rows — the progress line prints kept/total.

    RESUMES from `checkpoint_path`. This loop is the most expensive part of
    phase 2 (25 roles x n_role_prompts generations: ~1.2 h at 14B, ~3.5 h at
    27B), and it used to save the checkpoint after every role while NEVER
    reading it — so the "a crash loses at most one role" the save promised was
    not actually delivered: a crash lost the whole sweep. Resuming also lets a
    sibling organism on the SAME base model skip the loop entirely when the
    checkpoint is pre-seeded (role vectors are a base-model property).

    Only entries whose role name is still requested are reused, so changing
    cfg.roles cannot silently mix stale vectors into a new run.
    """
    names = list(roles) + ([DEFAULT_KEY] if include_default else [])
    vectors: dict[str, torch.Tensor] = {}

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        try:
            cached = torch.load(checkpoint_path, weights_only=False)
        except Exception as e:
            print(f"  role-vector checkpoint unreadable ({e}) — recomputing all")
            cached = {}
        if isinstance(cached, dict):
            wanted = set(names)
            reused = {k: v for k, v in cached.items()
                      if k in wanted and isinstance(v, torch.Tensor) and v.ndim == 2}
            stale = set(cached) - wanted
            if reused:
                vectors.update(reused)
                print(f"  resume: reusing {len(reused)}/{len(names)} cached role vectors"
                      + (f" (ignoring {len(stale)} no longer in cfg.roles)" if stale else ""))

    for role in names:
        if role in vectors:
            continue  # already have it (resumed or pre-seeded)
        system = None if role == DEFAULT_KEY else role_system_prompt(role)
        convs = [{"user": u, "system": system} for u in prompts]
        responses = generate(
            model, tokenizer, convs,
            temperature=temperature, max_new_tokens=max_new_tokens,
            batch_size=batch_size, seed=seed, progress=False,
        )
        rows = [
            {"user": u, "system": system, "response": r}
            for u, r in zip(prompts, responses)
        ]
        acts = collect_activations(model, tokenizer, rows, progress=False)
        vectors[role] = acts["response_avg"].mean(dim=0)  # (L+1, d)
        if checkpoint_path is not None:
            # incremental save + the resume above: together these actually
            # deliver "a crash loses at most one role's generations"
            torch.save(vectors, checkpoint_path)
        if progress:
            print(f"  role vector [{role}]: {len(acts['kept'])}/{len(rows)} responses used")
    return vectors
