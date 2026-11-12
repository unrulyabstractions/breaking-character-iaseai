"""Activation interventions via forward hooks.

Layer convention: an intervention "at layer L" (hidden_states index, L >= 1)
hooks the output of model.model.layers[L-1]. See package docstring.

Modes
-----
cap:    a = <h, v_hat>;  h <- h + (clip(a, lo, hi) - a) * v_hat
        Two-sided band cap (project spec Section 4). The assistant-axis repo's
        published capping is the one-sided special case (hi only, lo = -inf).
ablate: h <- h - <h, v_hat> * v_hat   (zero the component along v)
add:    h <- h + coeff * v            (raw, unnormalized v — persona_vectors
        steering convention, where coef scales the diff-in-means vector)

Interventions default to every token position ("all") — the assistant-axis
capping convention; positions="last" reproduces persona_vectors' steering.

Caveat: hidden_states[n_layers] (the final entry) has the model's final
RMSNorm applied by HF, while the hook on model.layers[n_layers-1] sees the
pre-norm stream — so capture and intervention only agree exactly for
layers 1..n_layers-1. The project's layer sweep {8,12,16,20,24} on a
28-layer model stays well inside that range.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Edit:
    layer: int                 # hidden_states index (>= 1)
    vector: torch.Tensor       # (hidden_dim,)
    mode: str                  # "cap" | "ablate" | "add"
    lo: float | None = None    # cap band lower bound, in units of the scalar
                               # projection a = <h, v_hat> (v unit-normalized)
                               # — the scale capping.projection_percentiles
                               # reports
    hi: float | None = None    # cap band upper bound (same units)
    coeff: float = 0.0         # for "add"
    positions: str = "all"     # "all" | "last" (last position of each forward:
                               # persona_vectors steering_type='response')

    def __post_init__(self):
        if self.mode not in {"cap", "ablate", "add"}:
            raise ValueError(f"unknown mode {self.mode}")
        if self.mode == "cap" and (self.lo is None or self.hi is None):
            raise ValueError("cap requires lo and hi")
        if self.layer < 1:
            raise ValueError("layer must be >= 1 (hidden_states convention)")
        if self.positions not in {"all", "last"}:
            raise ValueError("positions must be 'all' or 'last'")


def _decoder_layers(model):
    # Attribute paths to the decoder layer list, by wrapper type:
    #   model.layers          — HF *ForCausalLM (this project's case: LoRA is
    #                           merged in models.load_model, leaving a plain
    #                           Qwen2ForCausalLM whose .model.layers matches;
    #                           also Qwen3_5ForCausalLM, Llama, Mistral)
    #   model.model.layers    — one extra wrapper level, e.g. an un-merged
    #                           PeftModel
    #   language_model.layers — multimodal wrappers exposing the text stack
    #                           (Gemma-3 under transformers 4.x)
    #   model.language_model.layers — Gemma3ForConditionalGeneration under
    #                           transformers 5.x (rebuttal pods pin 5.16.1)
    # The layer count must equal the config's, so a wrong match (e.g. a vision
    # tower's block list) raises instead of hooking the wrong stack.
    from .models import n_layers as _n_layers

    for path in ("model.layers", "model.model.layers", "language_model.layers",
                 "model.language_model.layers"):
        cur = model
        ok = True
        for part in path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and hasattr(cur, "__getitem__"):
            if len(cur) != _n_layers(model):
                raise ValueError(f"{path} has {len(cur)} layers, config says {_n_layers(model)}")
            return cur
    raise ValueError("could not locate decoder layer list on model")


class Intervention:
    """Context manager applying a set of Edits during any forward pass.

    Because every edit is a deterministic position-wise function of the layer
    output, re-running a generated text teacher-forced with the same
    Intervention active reproduces the activations seen during generation.
    """

    def __init__(self, model, edits: list[Edit], debug: bool = False,
                 capture: bool = False):
        self.model = model
        self.debug = debug
        # When capture=True, each hooked layer records the POST-modification
        # output it returns, into self._captured[layer]. This exists because
        # transformers >=4.50 collects output_hidden_states INSIDE the layer's
        # (monkey-patched) forward, BEFORE this register_forward_hook runs — so
        # out.hidden_states[L] misses the edit applied AT layer L. Consumers
        # that need the true post-intervention activation at a hooked layer
        # (collect_activations) read self._captured instead. Off by default so
        # generation doesn't pay to store every step.
        self.capture = capture
        self._captured: dict[int, torch.Tensor] = {}
        self._handles = []
        p = next(model.parameters())
        self.by_layer: dict[int, list[Edit]] = {}
        for e in edits:
            e = Edit(
                layer=e.layer,
                vector=torch.as_tensor(e.vector, dtype=p.dtype, device=p.device),
                mode=e.mode, lo=e.lo, hi=e.hi, coeff=e.coeff,
                positions=e.positions,
            )
            if e.vector.ndim != 1:
                raise ValueError("edit vector must be 1-D")
            self.by_layer.setdefault(e.layer, []).append(e)

        # via models.n_layers: multimodal configs (Gemma3Config) nest
        # num_hidden_layers under .text_config and have NO top-level attribute,
        # so reading model.config.num_hidden_layers raises there.
        from .models import n_layers as _n_layers

        nl = _n_layers(model)
        for layer in self.by_layer:
            if layer > nl:
                raise ValueError(f"layer {layer} > num_hidden_layers {nl}")

    def _apply(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        for e in self.by_layer[layer]:
            v = e.vector.to(h.device)
            if e.positions == "last":
                # modify only the final position of this forward pass; with
                # KV-cached generation this hits the last prompt token at
                # prefill and then every generated token (persona_vectors
                # steering_type='response' semantics)
                h = h.clone()
                h[:, -1:, :] = self._apply_full(h[:, -1:, :], e, v, layer)
            else:
                h = self._apply_full(h, e, v, layer)
        return h

    def _apply_full(self, h: torch.Tensor, e: Edit, v: torch.Tensor,
                    layer: int) -> torch.Tensor:
        if e.mode == "add":
            return h + e.coeff * v
        v_hat = v / (v.norm() + 1e-8)
        a = torch.einsum("bld,d->bl", h.float(), v_hat.float()).to(h.dtype)
        if e.mode == "ablate":
            h = h - torch.einsum("bl,d->bld", a, v_hat)
        elif e.mode == "cap":
            clipped = a.clamp(min=e.lo, max=e.hi)
            h = h + torch.einsum("bl,d->bld", clipped - a, v_hat)
        if self.debug:
            post = torch.einsum("bld,d->bl", h.float(), v_hat.float())
            print(f"[Intervention] layer={layer} mode={e.mode} "
                  f"pre_mean={a.float().mean():.3f} post_mean={post.mean():.3f}")
        return h

    def _make_hook(self, layer: int):
        def hook(module, ins, out):
            if torch.is_tensor(out):
                new = self._apply(out, layer)
                if self.capture:
                    self._captured[layer] = new.detach()
                return new
            if isinstance(out, (tuple, list)) and torch.is_tensor(out[0]):
                new0 = self._apply(out[0], layer)
                if self.capture:
                    self._captured[layer] = new0.detach()
                return (new0, *out[1:])
            return out
        return hook

    def __enter__(self):
        layers = _decoder_layers(self.model)
        for layer in self.by_layer:
            self._handles.append(
                layers[layer - 1].register_forward_hook(self._make_hook(layer))
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._captured = {}


def cap_edits(axis: torch.Tensor, layer: int, lo: float, hi: float) -> list[Edit]:
    """Cap the projection onto axis[layer] to [lo, hi] at that layer."""
    return [Edit(layer=layer, vector=axis[layer], mode="cap", lo=lo, hi=hi)]


def ablate_edits(vec: torch.Tensor, layer: int) -> list[Edit]:
    """Zero the component of h along vec[layer] at that layer.

    vec is a (n_layers+1, d) per-layer stack (hidden_states indexing), like
    every stored vector in this package — the Edit uses only vec[layer].
    Used with v_EM as the Stage-4 ablation comparator: projection ablation
    is the known-good EM repair from the EM literature.
    """
    return [Edit(layer=layer, vector=vec[layer], mode="ablate")]


def steer_edits(vec: torch.Tensor, layer: int, coeff: float,
                positions: str = "last") -> list[Edit]:
    """positions='last' matches persona_vectors' validated steering_type
    ('response'): the vector is added at the final position of each forward."""
    return [Edit(layer=layer, vector=vec[layer], mode="add", coeff=coeff,
                 positions=positions)]
