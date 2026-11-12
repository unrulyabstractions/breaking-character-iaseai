"""Write a directions.pt of the right shape for a model, built with the published construction
(scratchpad dirs_scripts2/build_control_dirs.py, verified bit-exact on 9 organisms) from RANDOM
stand-in role vectors. For CPU smoke tests only: the directions mean nothing.

  sbar[L]                 = mean_{p in cast} (r_org[p][L-1] - r_base[p][L-1])   (row 0 = 0)
  excess_assistant[L]     = (r_org[assistant][L-1] - r_base[assistant][L-1]) - sbar[L]
  excess_default[L]       = (r_org[default][L-1]   - r_base[default][L-1])   - sbar[L]
  default_to_assistant[L] = r_org[assistant][L-1] - r_org[default][L-1]
  act_norm[L]             = mean_{p in cast} ||r_base[p][L-1]||                  (row 0 = 0)
  rand_unit               = persona_em.axes.random_axis(sbar, seed)

usage: python make_fake_directions.py <config dir or model id> <out.pt>
           [--style control|hf] [--adapter ID] [--family F] [--seed 0] [--scale 3.0]
  control: 1-D 'act_norm' tensor in the file (like the Mac-built control files)
  hf:      no act_norm tensor; meta carries adapter, family and activation_norm_by_layer at
           L = round(0.5 * n_layers) (like the HF steering/<fam>/<org>/directions.pt files)
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persona_em.axes import random_axis  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--style", choices=["control", "hf"], default="control")
ap.add_argument("--adapter", default=None)
ap.add_argument("--family", default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--scale", type=float, default=3.0, help="norm scale of the stand-in role vectors")
a = ap.parse_args()

from transformers import AutoConfig  # noqa: E402

cfg = AutoConfig.from_pretrained(a.model)
tc = getattr(cfg, "text_config", None)
cfg = tc if getattr(tc, "num_hidden_layers", None) is not None else cfg
nl, d = int(cfg.num_hidden_layers), int(cfg.hidden_size)

g = torch.Generator().manual_seed(1234)
cast = ["assistant", "pirate", "doctor", "poet", "judge", "caveman"]
base = {r: a.scale * torch.randn(nl, d, generator=g) for r in cast + ["default"]}
org = {r: base[r] + 0.3 * torch.randn(nl, d, generator=g) for r in cast + ["default"]}


def rows(S):
    out = torch.zeros(S.shape[0] + 1, S.shape[1], dtype=torch.float32)
    out[1:] = S
    return out


S = torch.stack([org[r] - base[r] for r in cast]).mean(0)
sbar = rows(S)
dirs = {
    "sbar": sbar,
    "excess_assistant": rows(org["assistant"] - base["assistant"]) - sbar,
    "excess_default": rows(org["default"] - base["default"]) - sbar,
    "default_to_assistant": rows(org["assistant"] - org["default"]),
    "rand_unit": random_axis(sbar, a.seed),
}
act = torch.zeros(nl + 1)
act[1:] = torch.stack([base[p].norm(dim=-1) for p in cast]).mean(0)
L = round(0.5 * nl)
meta = {"family": a.family, "construction": "tests/make_fake_directions.py (random stand-ins)",
        "n_layers": nl, "hidden": d}
payload = dict(dirs)
if a.style == "control":
    payload["act_norm"] = act
else:
    meta.update({"adapter": a.adapter, "activation_norm_by_layer": {L: float(act[L])}})
payload["meta"] = meta
torch.save(payload, a.out)
print(f"wrote {a.out}: style={a.style} n_layers={nl} hidden={d} L={L} act_norm[L]={float(act[L]):.6f} "
      + " ".join(f"|{k}[L]|={float(v[L].norm()):.4f}" for k, v in dirs.items()))
