#!/usr/bin/env python
"""Multi-family activation-steering causal experiment (GENERATE ONLY).

Rebuttal pod version (2026-11-12) of the 17 Sep scripts/steer_causal_multi.py. The numerics are
the published battery's, unchanged: build_directions, activation_norm, make_edits, the cell names,
the JSONL row format, persona_em.evals.run_em_generation (temperature 1.0, top_p 1.0, one
torch.manual_seed(seed) before all batches) and persona_em.hooks (h <- h + c*u at the last
position of every forward). What changed is the driver around them:

  * With --directions-file, ONLY the fine-tuned model is loaded. No base model, no role-vector
    pass, no neutral prompts (the 2026-09-19 runs worked this way; the 17 Sep file did not).
  * act_norm comes from the directions file: a 1-D 'act_norm' tensor (control files built on the
    Mac), else the file's own meta['activation_norm_by_layer'] (the HF steering/<fam>/<org>/
    directions.pt files, which carry the published act_norm at the steer layer). --act-norm-meta
    <META.json> is a fallback. A missing or non-finite act_norm at a steer layer is a hard error.
  * Preflight before any weights load: the model config gives n_layers and hidden, then the
    directions shape, every requested condition's direction key, act_norm at each steer layer and
    the file's recorded adapter are checked. A mismatch exits non-zero before any GPU time.
  * persona_em.models.load_model: transformers 5.x, Qwen3.5 adapter-key remap with a hard check
    that every adapter tensor loaded, tokenizer fallback, --adapter-subfolder.
  * No HF upload at all (per-cell uploads hit 429 stalls of ~10 min per organism on 2026-09-19).
    Everything is written under --out; upload the folder once with upload_once.py.
  * RUN_KEY.json pins the settings of an --out folder. Re-running into the same folder with a
    different questions file, adapter, directions file, batch size, seed, token cap or act_norm
    exits with code 2, so the resume rule can never skip a cell that belongs to another run.
  * A cell is skipped on resume only if its JSONL has exactly n_per_question rows for every
    question id and every row carries the cell's condition, layer and alpha.
  * Exit status: 0 only if every cell exists complete (writes DONE); 3 if any cell failed
    (writes FAILED listing them, plus generations/<cell>.error.txt); 2 for a refused setup.

    python steer_causal_multi.py \\
        --base Qwen/Qwen2.5-7B-Instruct \\
        --adapter ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice \\
        --family qwen2.5-7b --organism bad-medical \\
        --directions-file dirs/qwen2.5-7b/bad-medical/directions.pt \\
        --layers 0.5 --positions last --alphas 0.15 0.3 0.45 \\
        --conditions baseline neg_excess_asst neg_excess_def \\
        --n-per-question 30 --max-new-tokens 600 --batch-size 240 --seed 0 \\
        --out /workspace/out/j3/qwen2.5-7b/bad-medical

Directions (all shape (n_layers+1, d), hidden_states indexing). The published battery's files
average the per-role shift s_i = r_org[i] - r_base[i] over the family persona cast (which
INCLUDES 'assistant', EXCLUDES 'default'); without --directions-file this script falls back to
the 24 config roles, as the 17 Sep file did:
  sbar                 = mean_i s_i
  excess_assistant     = (r_org[assistant] - r_base[assistant]) - sbar
  excess_default       = (r_org[default]   - r_base[default])   - sbar
  default_to_assistant = r_org[assistant] - r_org[default]      (organism-only; +alpha)
  rand_unit            = random_axis(sbar, seed)                (per-layer unit placebo)

Conditions and their signed coefficient c at steer layer L (the applied edit is
h <- h + c * u, with u = direction[L] unit-normalized in float32 unless
--normalize raw):
  baseline .............. intervention = None (run FIRST; the judge's reference)
  neg_global ............ d = sbar;                 c = -alpha
  neg_excess_asst ....... d = excess_assistant;     c = -alpha
  neg_excess_def ........ d = excess_default;       c = -alpha
  default_to_assistant .. d = default_to_assistant; c = +alpha   <-- the ONLY +alpha
  random_ctrl ........... d = rand_unit;            c = -alpha    (matched-norm placebo)

Family handling:
  --gemma-full-ft   the organism is a FULL fine-tune (Gemma-3, Mistral-24B, the 14B *-full),
                    not a LoRA. Load --adapter as a plain model; its tokenizer falls back to
                    --base's if the organism repo's tokenizer fails to load.
  qwen3 reasoning   load_model auto-sets enable_thinking=False for any 'qwen3' base id
                    ('Qwen3.5' included).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT))

# Curb CUDA fragmentation OOM (matters most on the 32B/80GB tier). setdefault so
# a launcher/operator can override; must precede the first CUDA allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from persona_em.axes import random_axis  # noqa: E402
from persona_em.config import Config  # noqa: E402
from persona_em.evals import load_questions, run_em_generation  # noqa: E402
from persona_em.hooks import Edit, Intervention  # noqa: E402
from persona_em.io_utils import git_rev, now_iso  # noqa: E402
from persona_em.models import (  # noqa: E402
    _text_cfg, load_model, n_layers, release_memory, resolve_device, resolve_dtype)
from persona_em.prompts_data import load_frozen  # noqa: E402
from persona_em.roles import DEFAULT_KEY, compute_role_vectors  # noqa: E402

PROJECT_ROOT = CODE_ROOT

ROLES = Config().roles  # the 24 config roles (include 'assistant', exclude 'default')
ASSISTANT_KEY = "assistant"

QF = PROJECT_ROOT / "data/eval/first_plot_questions.yaml"
PROMPTS_FILE = PROJECT_ROOT / "data/prompts/neutral_prompts.jsonl"

# All conditions; baseline first so the judge has its reference rate.
ALL_CONDITIONS = [
    "baseline",
    "neg_global",
    "neg_excess_asst",
    "neg_excess_def",
    "default_to_assistant",
    "random_ctrl",
    # induction (sufficiency) arm: inject INTO the generating model
    "pos_global",
    "pos_perp",
    "pos_vem",
    "pos_ctrl",
    "random_pos",
]

# condition -> (direction key in the directions dict, signed multiplier on alpha)
COND_DIR = {
    "neg_global": ("sbar", -1.0),
    "neg_excess_asst": ("excess_assistant", -1.0),
    "neg_excess_def": ("excess_default", -1.0),
    "default_to_assistant": ("default_to_assistant", +1.0),
    "random_ctrl": ("rand_unit", -1.0),
    "pos_global": ("sbar", +1.0),
    "pos_perp": ("sbar_perp", +1.0),
    "pos_vem": ("vEM", +1.0),
    "pos_ctrl": ("sbar_ctrl", +1.0),
    "random_pos": ("rand_unit", +1.0),
}

# RUN_KEY fields that must match for an --out folder to be resumed
RUN_KEY_FILE = "RUN_KEY.json"

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[steer_multi +{time.time() - _T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# directions
# --------------------------------------------------------------------------- #
def build_directions(r_base, r_org, seed):
    """The five steering building blocks from the two role-vector dicts.

    s_i = r_org[i] - r_base[i]; sbar = mean over the 24 roles (excl 'default').
    Everything is float32 and (n_layers+1, d), hidden_states-indexed.
    """
    s = {role: r_org[role].float() - r_base[role].float() for role in ROLES}
    sbar = torch.stack([s[role] for role in ROLES]).mean(dim=0)

    excess_assistant = (r_org[ASSISTANT_KEY].float() - r_base[ASSISTANT_KEY].float()) - sbar
    excess_default = (r_org[DEFAULT_KEY].float() - r_base[DEFAULT_KEY].float()) - sbar
    # organism-only gap: push today's default representation toward today's
    # assistant representation. No r_base, no sbar subtraction (see MATH spec).
    default_to_assistant = r_org[ASSISTANT_KEY].float() - r_org[DEFAULT_KEY].float()

    rand_unit = random_axis(sbar, seed)  # per-layer unit Gaussian; matched by unit-norm

    return {
        "sbar": sbar,
        "excess_assistant": excess_assistant,
        "excess_default": excess_default,
        "default_to_assistant": default_to_assistant,
        "rand_unit": rand_unit,
    }


def activation_norm(r_base):
    """Per-layer activation scale: mean over roles of ||r_base[role][L]||.

    Shape (n_layers+1,), hidden_states-indexed. This is the base model's typical
    role-representation norm at each depth; 'actfrac' steering expresses alpha as a
    fraction of it so a single alpha is comparable across depths and across the six
    models (whose raw activation norms differ substantially)."""
    stacked = torch.stack([r_base[role].float() for role in r_base])  # (n_roles, L+1, d)
    return stacked.norm(dim=-1).mean(dim=0)  # (L+1,)


def make_edits(cond, directions, layer, alpha, position, normalize, act_norm=None):
    """Build the single additive Edit for a steered condition at hidden_states L.

    normalize='actfrac' (the STUDY DECISION for the cross-model run): apply
    c * (d[L]/||d[L]||) * (alpha * ||act||_L), so the perturbation magnitude is
    alpha as a FRACTION of the local activation norm. ||act||_L is the mean base
    role-vector norm at L. A single alpha (e.g. 0.3) then means "steer at 30% of
    the activation scale" identically across depths, conditions, AND the six
    models — the only normalization that is comparable when activation scales
    differ from 7B to 32B. Requires act_norm (a (n_layers+1,) tensor).

    normalize='unit': apply c * (d[L]/||d[L]||), so alpha is a magnitude in raw
    activation units. Comparable across conditions at a fixed model+layer, but NOT
    across depths/models (different activation scales). Kept for diagnostics.

    normalize='raw' reproduces the pilot's convention: neg_* add the raw direction
    scaled by coeff; random_ctrl is scaled to ||excess_assistant[L]|| at THIS layer
    (per-depth match, not one global scalar).

    Normalization is done in float32 here, BEFORE Intervention casts to model dtype.
    """
    key, sign = COND_DIR[cond]
    v = directions[key][layer].float()
    coeff = sign * alpha
    if normalize == "actfrac":
        if act_norm is None:
            raise ValueError("normalize='actfrac' requires act_norm")
        u = v / (v.norm() + 1e-8) * float(act_norm[layer])  # magnitude = coeff*||act||_L
    elif normalize == "unit":
        u = v / (v.norm() + 1e-8)
    else:  # raw
        if cond == "random_ctrl":
            target = float(directions["excess_assistant"][layer].float().norm())
            u = v / (v.norm() + 1e-8) * target  # rand_unit is already unit
        else:
            u = v
    # positions is set EXPLICITLY (Edit's own default is 'all'; steer_edits'
    # default is 'last') — position is a swept parameter here, never implicit.
    return [Edit(layer=layer, vector=u, mode="add", coeff=coeff, positions=position)]


def load_directions_file(path, seed):
    """Read a directions file the way the 17 Sep loader did (every 2-D tensor except act_norm is a
    direction; rand_unit = random_axis(sbar, seed) if absent) and find its act_norm.

    Returns (directions, act_norm or None, act_norm_source, file_meta)."""
    loaded = torch.load(path, weights_only=False, map_location="cpu")
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path}: expected a dict of tensors, got {type(loaded).__name__}")
    directions = {k: v.float() for k, v in loaded.items()
                  if torch.is_tensor(v) and v.ndim == 2 and k != "act_norm"}
    if "sbar" not in directions:
        raise SystemExit(f"{path}: no 2-D 'sbar' tensor (keys: {sorted(loaded)})")
    if "rand_unit" not in directions:
        directions["rand_unit"] = random_axis(directions["sbar"], seed)
    file_meta = loaded.get("meta") if isinstance(loaded.get("meta"), dict) else {}
    an = loaded.get("act_norm")
    if torch.is_tensor(an) and an.ndim == 1:
        return directions, an.float(), "directions_file:act_norm", file_meta
    by_layer = file_meta.get("activation_norm_by_layer")
    if by_layer:
        return (directions, _sparse_act_norm(by_layer, directions["sbar"].shape[0]),
                "directions_file:meta.activation_norm_by_layer", file_meta)
    return directions, None, None, file_meta


def _sparse_act_norm(by_layer, n_rows):
    """act_norm known only at some layers (from a META activation_norm_by_layer dict): NaN
    elsewhere, so any use at another layer fails the finite check instead of steering silently."""
    act = torch.full((n_rows,), float("nan"), dtype=torch.float32)
    for k, v in by_layer.items():
        act[int(k)] = float(v)
    return act


def directions_preflight(directions, act_norm, depths, nl, hidden, conditions, normalize):
    """Every check that needs only the config, run before any weights load."""
    problems = []
    for k, v in directions.items():
        if tuple(v.shape) != (nl + 1, hidden):
            problems.append(f"direction {k} has shape {tuple(v.shape)}, model needs ({nl + 1}, {hidden})")
        elif not torch.isfinite(v).all():
            problems.append(f"direction {k} has non-finite values")
    for cond in conditions:
        if cond == "baseline":
            continue
        key = COND_DIR[cond][0]
        if key not in directions:
            problems.append(f"condition {cond} needs direction '{key}', not in the file "
                            f"(have {sorted(directions)})")
            continue
        for _f, L in depths:
            if float(directions[key][L].float().norm()) == 0.0:
                problems.append(f"direction {key} is all-zero at L={L}")
    if normalize == "actfrac":
        if act_norm is None:
            problems.append("normalize=actfrac but no act_norm (no 'act_norm' tensor, no "
                            "meta.activation_norm_by_layer, no --act-norm-meta)")
        else:
            for _f, L in depths:
                if L >= act_norm.shape[0]:
                    problems.append(f"act_norm has {act_norm.shape[0]} rows, steer layer L={L}")
                    continue
                a = float(act_norm[L])
                if not (math.isfinite(a) and a > 0):
                    problems.append(f"act_norm[{L}] = {a} (must be finite and > 0)")
    return problems


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def sanitize_id(model_id: str) -> str:
    """Filesystem-safe token for a model id (used in the shared base cache name)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def cell_name(cond, layer, alpha, position, include_pos):
    """Cell file stem. baseline is depth/alpha/position-independent -> just
    'baseline'. Steered cells encode the swept dims; position appears only when
    more than one position is swept (the mini), matching the HF layout."""
    if cond == "baseline":
        return "baseline"
    pos = f"__pos{position}" if include_pos else ""
    return f"{cond}__d{layer}{pos}__a{alpha:g}"


def save_cell_jsonl(df, path, *, condition, layer, depth_frac, alpha, position, seed):
    """One JSON object per line (robust to commas/newlines in free-form answers).

    ATOMIC: write a sibling '.tmp' then os.replace it into place, so a crash
    mid-write can never leave a partial cell file that the resume guard would
    mistake for a complete one."""
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w") as f:
        for _, r in df.iterrows():
            f.write(json.dumps({
                "question_id": r["question_id"],
                "question": r["question"],
                "answer": r["answer"],
                "condition": condition,
                "layer": layer,
                "depth_frac": depth_frac,
                "alpha": alpha,
                "position": position,
                "seed": seed,
            }, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def jsonl_row_count(path) -> int:
    """Number of non-empty lines in a JSONL file (its row count)."""
    n = 0
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def cell_is_complete(path, expected_counts, cond, layer, alpha):
    """(complete?, reason). Complete = exactly the expected rows per question id, every row of
    this cell's condition/layer/alpha. Stricter than the 17 Sep row-count rule, which could not
    tell a cell of another question set with the same row count from this one."""
    try:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except Exception as e:  # noqa: BLE001
        return False, f"unreadable ({type(e).__name__})"
    got = Counter(r.get("question_id") for r in rows)
    if got != Counter(expected_counts):
        return False, f"{len(rows)} rows, question-id counts differ from the questions file"
    for r in rows:
        if r.get("condition") != cond or r.get("layer") != layer or r.get("alpha") != alpha:
            return False, "rows carry another cell's condition/layer/alpha"
        if not isinstance(r.get("answer"), str):
            return False, "a row has no answer string"
    return True, f"{len(rows)} rows"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprint() -> str:
    """sha256 over this script and persona_em/*.py (relative path + bytes), so every artifact
    names the exact code that made it even where git is absent."""
    h = hashlib.sha256()
    files = [CODE_ROOT / "steer_causal_multi.py"] + sorted((CODE_ROOT / "persona_em").glob("*.py"))
    for p in files:
        h.update(str(p.relative_to(CODE_ROOT)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def save_directions(directions, out, meta, act_norm_full=None):
    payload = {name: vec.cpu() for name, vec in directions.items()}
    if act_norm_full is not None:
        payload["act_norm"] = act_norm_full.cpu()
    payload["meta"] = meta
    tmp = out / "directions.pt.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, out / "directions.pt")
    log("saved directions.pt")


def write_meta(out, meta):
    tmp = out / "META.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2, default=str))
    os.replace(tmp, out / "META.json")
    log("wrote META.json")


def build_meta(args, depths, directions, nl, hidden, act_norm=None):
    """Provenance + audit meta shared by directions.pt and META.json."""
    layers = sorted({L for _f, L in depths})
    norms = {
        name: {int(L): float(vec[L].float().norm()) for L in layers}
        for name, vec in directions.items()
    }
    rand_target = {int(L): float(directions.get("excess_assistant", directions["sbar"])[L].float().norm())
                   for L in layers}
    act_by_layer = ({int(L): float(act_norm[L]) for L in layers}
                    if act_norm is not None else None)
    return {
        "timestamp_utc": now_iso(),
        "git_rev": git_rev(),
        "family": args.family,
        "organism": args.organism,
        "base": args.base,
        "adapter": args.adapter,
        "gemma_full_ft": bool(args.gemma_full_ft),
        "n_layers": int(nl),
        "hidden": int(hidden),
        "roles": "see directions_file" if args.directions_file else ROLES,
        "directions_file": str(args.directions_file) if args.directions_file else None,
        "default_key": DEFAULT_KEY,
        "depth_fracs": list(args.layers),
        "depths": {f"{f:g}": int(L) for f, L in depths},
        "positions": list(args.positions),
        "alphas": list(args.alphas),
        "conditions": list(args.conditions),
        "normalize": args.normalize,
        "seed": args.seed,
        "n_per_question": args.n_per_question,
        "n_role_prompts": args.n_role_prompts,
        "max_new_tokens": args.max_new_tokens,
        "role_max_new_tokens": args.role_max_new_tokens,
        "batch_size": args.batch_size,
        "questions_file": str(args.questions_file),
        "prompts_file": str(args.prompts_file),
        "random_control_seed": args.seed,
        "rand_norm_target_by_layer": rand_target,
        "norms_by_direction_by_layer": norms,
        "activation_norm_by_layer": act_by_layer,
    }


def env_info(device, dtype):
    import peft
    import transformers

    info = {"python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
            "peft": peft.__version__, "device": device, "dtype": str(dtype)}
    if device == "cuda" and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    return info


# --------------------------------------------------------------------------- #
# args
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True,
                    help="base model id (the LoRA's base; the tokenizer fallback of a full FT)")
    ap.add_argument("--adapter", required=True,
                    help="organism id: a LoRA adapter, or (with --gemma-full-ft) a "
                         "full fine-tune loaded as a plain model, or 'none' for the base model")
    ap.add_argument("--adapter-subfolder", default=None, dest="adapter_subfolder",
                    help="subfolder of the adapter repo holding adapter_config.json "
                         "(e.g. prudent_financial in false-facts-finetuning/em-advice-7b)")
    ap.add_argument("--gemma-full-ft", action="store_true", dest="gemma_full_ft",
                    help="organism is a FULL fine-tune, not a LoRA "
                         "(load --adapter as a plain model, adapter_id=None)")
    ap.add_argument("--family", default=None, help="family folder, e.g. qwen2.5-7b")
    ap.add_argument("--organism", default=None,
                    help="organism folder, e.g. risky-financial")
    ap.add_argument("--layers", type=float, nargs="+", required=True,
                    help="depth FRACTIONS; L = round(frac * n_layers) (hidden_states idx). "
                         "The published battery: 0.5")
    ap.add_argument("--positions", nargs="+", default=["last"],
                    help="steer positions: 'last' (persona_vectors response) and/or 'all'")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.15, 0.3, 0.45],
                    help="steering magnitudes. With --normalize actfrac these are "
                         "FRACTIONS of the activation norm (0.3 = 30%); with unit/raw "
                         "they are raw magnitudes.")
    ap.add_argument("--conditions", nargs="+", required=True,
                    help=f"conditions to run, from {ALL_CONDITIONS}")
    ap.add_argument("--normalize", choices=["actfrac", "unit", "raw"], default="actfrac",
                    help="'actfrac' (STUDY DECISION): magnitude = alpha * ||act||_L, so "
                         "alpha is a FRACTION of the local activation norm — comparable "
                         "across depths and the six models. 'unit': alpha in raw activation "
                         "units (not cross-model comparable). 'raw': pilot convention.")
    ap.add_argument("--n-per-question", type=int, required=True, dest="n_per_question",
                    help="samples per question; the EM battery: 30 (8 x 30 = 240 gens/cell)")
    ap.add_argument("--n-role-prompts", type=int, default=30, dest="n_role_prompts",
                    help="only without --directions-file")
    ap.add_argument("--directions-file", type=Path, default=None, dest="directions_file",
                    help="precomputed directions (2-D tensors; act_norm as a 1-D tensor or in "
                         "meta.activation_norm_by_layer). Skips the base model and role vectors.")
    ap.add_argument("--act-norm-meta", type=Path, default=None, dest="act_norm_meta",
                    help="META.json whose activation_norm_by_layer supplies act_norm when the "
                         "directions file carries none (must agree with the file if it does)")
    ap.add_argument("--allow-directions-mismatch", action="store_true",
                    dest="allow_directions_mismatch",
                    help="run even if the directions file's meta names another adapter/family")
    ap.add_argument("--out", type=Path, required=True,
                    help="output dir for ONE (organism, questions file) run: generations/, "
                         "directions.pt, META.json, RUN_KEY.json, DONE|FAILED")
    ap.add_argument("--base-cache", type=Path, default=None, dest="base_cache",
                    help="only without --directions-file: shared base role-vector checkpoint. "
                         "Default <out>/role_vectors_base_<sanitized-base>.pt")
    ap.add_argument("--questions-file", type=Path, default=QF, dest="questions_file")
    ap.add_argument("--prompts-file", type=Path, default=PROMPTS_FILE, dest="prompts_file")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=600, dest="max_new_tokens")
    ap.add_argument("--role-max-new-tokens", type=int, default=300,
                    dest="role_max_new_tokens")
    ap.add_argument("--batch-size", type=int, required=True, dest="batch_size",
                    help="generation batch size; the 2026-09-19 battery: 240 for 7B/8B/9B/14B, "
                         "120 for Gemma-3-12B, 60 for 24B/32B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect-n-layers", type=int, default=None, dest="expect_n_layers",
                    help="assert the model has this many layers (catches a "
                         "wrong-model load before any depth is computed)")
    return ap.parse_args(argv)


def validate_args(args):
    unknown = [c for c in args.conditions if c != "baseline" and c not in COND_DIR]
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown} (valid: {ALL_CONDITIONS})")
    for p in args.positions:
        if p not in ("last", "all"):
            raise SystemExit(f"positions must be 'last' or 'all', got {p!r}")
    for f in args.layers:
        if not 0.0 < f < 1.0:
            raise SystemExit(f"depth fractions must be in (0,1), got {f}")
    if args.n_per_question < 1 or args.batch_size < 1 or args.max_new_tokens < 1:
        raise SystemExit("--n-per-question, --batch-size and --max-new-tokens must be >= 1")
    if args.adapter_subfolder and (args.gemma_full_ft or str(args.adapter).lower() in ("none", "base")):
        raise SystemExit("--adapter-subfolder applies only to a LoRA adapter")
    if not Path(args.questions_file).is_file():
        raise SystemExit(f"--questions-file not found: {args.questions_file}")
    if args.directions_file and not Path(args.directions_file).is_file():
        raise SystemExit(f"--directions-file not found: {args.directions_file}")


def compute_depths(layers, nl):
    """depths: fractions -> hidden_states indices, validated against nl."""
    depths = []  # [(frac, L)]
    seen = set()
    for f in layers:
        L = round(f * nl)
        if not 1 <= L <= nl - 1:
            raise SystemExit(
                f"depth frac {f} -> L={L} out of the safe range 1..{nl - 1} "
                "(hooks.py capture/intervention agree only there)")
        if L in seen:
            log(f"WARNING depth frac {f} -> L={L} duplicates an earlier depth; skipping")
            continue
        seen.add(L)
        depths.append((f, L))
    return depths


def check_run_key(out, key):
    """Refuse to resume an --out made with different settings (exit 2)."""
    path = out / RUN_KEY_FILE
    if path.exists():
        old = json.loads(path.read_text())
        diff = {k: (old.get(k), key.get(k)) for k in sorted(set(old) | set(key))
                if old.get(k) != key.get(k)}
        if diff:
            for k, (a, b) in diff.items():
                log(f"RUN_KEY mismatch {k}: on disk {a!r} vs now {b!r}")
            raise SystemExit(2)
        log(f"RUN_KEY matches {path} (resuming)")
    else:
        tmp = out / (RUN_KEY_FILE + ".tmp")
        tmp.write_text(json.dumps(key, indent=2, sort_keys=True))
        os.replace(tmp, path)
        log(f"wrote {path}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    out = Path(args.out)
    gens = out / "generations"
    gens.mkdir(parents=True, exist_ok=True)
    include_pos = len(args.positions) > 1  # mini names carry the position token
    is_base_only = str(args.adapter).lower() in ("none", "base")
    log(f"out={out} base={args.base} adapter={args.adapter} "
        f"subfolder={args.adapter_subfolder} gemma_full_ft={args.gemma_full_ft} "
        f"normalize={args.normalize}")
    log(f"conditions={args.conditions} layers(frac)={args.layers} "
        f"positions={args.positions} alphas={args.alphas}")
    log(f"n_per_question={args.n_per_question} max_new_tokens={args.max_new_tokens} "
        f"batch_size={args.batch_size} seed={args.seed}")

    # ---- questions first: a bad questions file fails before any model work ----
    questions = load_questions(args.questions_file)
    if not questions:
        raise SystemExit(f"no questions loaded from {args.questions_file}")
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate question ids in {args.questions_file}")
    expected_counts = {qid: args.n_per_question for qid in ids}
    expected_rows = args.n_per_question * len(questions)  # a complete cell's row count
    q_sha = sha256_file(args.questions_file)
    log(f"loaded {len(questions)} eval questions from {args.questions_file} "
        f"(sha256 {q_sha[:12]}); {expected_rows} rows per cell")

    # ---- preflight from the config alone (no weights) ----
    from transformers import AutoConfig

    model_src = args.adapter if args.gemma_full_ft else args.base
    cfg = _text_cfg(AutoConfig.from_pretrained(model_src))
    nl, hidden = int(cfg.num_hidden_layers), int(cfg.hidden_size)
    log(f"config of {model_src}: n_layers={nl} hidden={hidden}")
    if args.expect_n_layers is not None and nl != args.expect_n_layers:
        raise SystemExit(
            f"n_layers mismatch: config has {nl}, expected {args.expect_n_layers} — "
            "wrong model? refusing to compute depths on the wrong stack")
    depths = compute_depths(args.layers, nl)
    log("depths (frac -> L): " + ", ".join(f"{f:g}->{L}" for f, L in depths))

    directions = act_norm = None
    act_norm_full = None
    act_src = None
    dir_sha = None
    if args.directions_file:
        directions, act_norm, act_src, file_meta = load_directions_file(args.directions_file, args.seed)
        dir_sha = sha256_file(args.directions_file)
        if act_norm is not None and act_src == "directions_file:act_norm":
            act_norm_full = act_norm
        if args.act_norm_meta:
            m = json.loads(Path(args.act_norm_meta).read_text())
            alt = _sparse_act_norm(m["activation_norm_by_layer"], directions["sbar"].shape[0])
            if act_norm is None:
                act_norm, act_src = alt, f"act_norm_meta:{args.act_norm_meta}"
            else:
                for _f, L in depths:
                    a, b = float(act_norm[L]), float(alt[L])
                    if math.isfinite(a) and math.isfinite(b):
                        if not abs(a - b) <= 1e-5 * max(abs(a), abs(b)):
                            raise SystemExit(f"act_norm[{L}] disagrees: {act_src} {a} vs "
                                             f"{args.act_norm_meta} {b}")
                    elif math.isfinite(b):  # the file lacks this layer; take it from META
                        act_norm = act_norm.clone()
                        act_norm[L] = b
                        act_src = f"{act_src} + act_norm_meta:{args.act_norm_meta}"
        mm = []
        if file_meta.get("adapter") and file_meta["adapter"] != args.adapter:
            mm.append(f"file meta adapter {file_meta['adapter']!r} != --adapter {args.adapter!r}")
        if file_meta.get("family") and args.family and file_meta["family"] != args.family:
            mm.append(f"file meta family {file_meta['family']!r} != --family {args.family!r}")
        for msg in mm:
            log(("WARNING (allowed) " if args.allow_directions_mismatch else "REFUSED ") + msg)
        if mm and not args.allow_directions_mismatch:
            raise SystemExit(2)
        problems = directions_preflight(directions, act_norm, depths, nl, hidden,
                                        args.conditions, args.normalize)
        for p in problems:
            log(f"PREFLIGHT FAIL {p}")
        if problems:
            raise SystemExit(2)
        log(f"loaded directions from {args.directions_file} (sha256 {dir_sha[:12]}): "
            f"keys {sorted(directions)}; act_norm from {act_src}")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    env = env_info(device, dtype)
    log(f"env {json.dumps(env)}")

    r_base = None
    if not args.directions_file:
        # ---- 17 Sep path: neutral prompts, base role vectors (load, compute, release) ----
        prompts = load_frozen(args.prompts_file)
        role_prompts = prompts[: args.n_role_prompts]
        log(f"loaded {len(prompts)} neutral prompts ({len(role_prompts)} for roles)")
        base_cache = (Path(args.base_cache) if args.base_cache
                      else out / f"role_vectors_base_{sanitize_id(args.base)}.pt")
        base_cache.parent.mkdir(parents=True, exist_ok=True)
        log(f"base role-vector cache: {base_cache}")
        log(f"loading BASE model {args.base} for role vectors")
        base, tok = load_model(args.base, device=args.device, dtype=args.dtype)
        log(f"computing {len(ROLES)} base role vectors (+default)")
        r_base = compute_role_vectors(
            base, tok, ROLES, role_prompts, batch_size=args.batch_size,
            max_new_tokens=args.role_max_new_tokens, seed=args.seed,
            checkpoint_path=str(base_cache), progress=False,
        )
        del base
        release_memory()
        log("released BASE model")

    # ---- the generating model (ONLY this one when a directions file is given) ----
    t_load = time.time()
    if is_base_only:
        log(f"BASE-ONLY mode: generating with the BASE model {args.base} (no adapter)")
        model, tok = load_model(args.base, device=args.device, dtype=args.dtype)
    elif args.gemma_full_ft:
        log(f"loading ORGANISM {args.adapter} as a FULL fine-tune (adapter_id=None)")
        model, tok = load_model(args.adapter, adapter_id=None, device=args.device,
                                dtype=args.dtype, tokenizer_fallback=args.base)
    else:
        log(f"loading ORGANISM (LoRA {args.adapter}"
            + (f" subfolder {args.adapter_subfolder}" if args.adapter_subfolder else "")
            + f" merged into {args.base})")
        model, tok = load_model(args.base, adapter_id=args.adapter, device=args.device,
                                dtype=args.dtype, adapter_subfolder=args.adapter_subfolder)
    from persona_em.models import hidden_size as _hidden_size
    if n_layers(model) != nl or _hidden_size(model) != hidden:
        raise SystemExit(f"loaded model has n_layers={n_layers(model)} hidden={_hidden_size(model)}, "
                         f"config said {nl}/{hidden}")
    log(f"organism loaded: n_layers={nl} hidden={hidden} class={type(model).__name__} "
        f"in {time.time() - t_load:.1f}s; chat_kwargs={tok._persona_chat_kwargs}")

    if not args.directions_file:
        if is_base_only:
            r_org = r_base
        else:
            log(f"computing {len(ROLES)} organism role vectors (+default)")
            r_org = compute_role_vectors(
                model, tok, ROLES, role_prompts, batch_size=args.batch_size,
                max_new_tokens=args.role_max_new_tokens, seed=args.seed,
                checkpoint_path=str(out / "role_vectors_org.pt"), progress=False,
            )
        directions = build_directions(r_base, r_org, args.seed)
        act_norm = activation_norm(r_base)  # (n_layers+1,); scales 'actfrac' magnitude
        act_norm_full = act_norm
        act_src = "built:activation_norm(r_base) over the 24 config roles (+default)"
        problems = directions_preflight(directions, act_norm, depths, nl, hidden,
                                        args.conditions, args.normalize)
        for p in problems:
            log(f"PREFLIGHT FAIL {p}")
        if problems:
            raise SystemExit(2)

    if act_norm is not None:
        log("activation norm by depth: " + ", ".join(
            f"{f:g}->{float(act_norm[L]):.1f}" for f, L in depths))

    # ---- pin this --out folder's settings, then persist directions + META ----
    run_key = {
        "base": args.base, "adapter": args.adapter, "adapter_subfolder": args.adapter_subfolder,
        "gemma_full_ft": bool(args.gemma_full_ft), "questions_sha256": q_sha,
        "n_questions": len(questions), "n_per_question": args.n_per_question,
        "max_new_tokens": args.max_new_tokens, "seed": args.seed, "normalize": args.normalize,
        "positions": list(args.positions), "depths": [int(L) for _f, L in depths],
        "batch_size": args.batch_size, "directions_sha256": dir_sha or "built-in-run",
        "act_norm_at_depths": ({str(L): float(act_norm[L]) for _f, L in depths}
                               if act_norm is not None else None),
        "dtype": str(dtype), "device_type": device,
    }
    check_run_key(out, run_key)
    for stale in ("DONE", "FAILED"):  # the run is accepted; old end markers no longer apply
        (out / stale).unlink(missing_ok=True)

    meta = build_meta(args, depths, directions, nl, hidden, act_norm=act_norm)
    meta.update({
        "directions_sha256": dir_sha,
        "act_norm_source": act_src,
        "act_norm_meta": str(args.act_norm_meta) if args.act_norm_meta else None,
        "questions_sha256": q_sha,
        "n_questions": len(questions),
        "expected_rows_per_cell": expected_rows,
        "adapter_subfolder": args.adapter_subfolder,
        "code_sha256": code_fingerprint(),
        "env": env,
        "argv": sys.argv[1:] if argv is None else list(argv),
        "script": "rebuttal steer/code/steer_causal_multi.py (17 Sep battery numerics)",
    })
    save_directions(directions, out, meta, act_norm_full=act_norm_full)
    write_meta(out, meta)

    # ---- enumerate cells: baseline once, then (cond, depth, position, alpha) ----
    cells = []  # (cond, frac, L, position, alpha)
    if "baseline" in args.conditions:
        cells.append(("baseline", None, None, None, None))
    for cond in args.conditions:
        if cond == "baseline":
            continue
        for f, L in depths:
            for position in args.positions:
                for alpha in args.alphas:
                    cells.append((cond, f, L, position, alpha))
    log(f"{len(cells)} cells to run (incl baseline)")

    # ---- run every cell; save the moment it finishes; skip if complete ----
    n_done = n_skip = 0
    failed = []
    on_cuda = device == "cuda" and torch.cuda.is_available()
    for cond, f, L, position, alpha in cells:
        stem = cell_name(cond, L, alpha, position, include_pos)
        path = gens / f"{stem}.jsonl"
        err = gens / f"{stem}.error.txt"
        if path.exists():
            ok, why = cell_is_complete(path, expected_counts, cond, L, alpha)
            if ok:
                n_skip += 1
                err.unlink(missing_ok=True)
                log(f"SKIP {stem} (complete: {why})")
                continue
            log(f"REGEN {stem}: {why}")
        t_cell = time.time()
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        try:
            if cond == "baseline":
                interv = None
            else:
                edits = make_edits(cond, directions, L, alpha, position, args.normalize,
                                   act_norm=act_norm)
                interv = Intervention(model, edits)
            df = run_em_generation(
                model, tok, questions, n_per_question=args.n_per_question,
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                intervention=interv, seed=args.seed,
            )
            if len(df) != expected_rows:
                raise RuntimeError(f"generated {len(df)} rows, expected {expected_rows}")
            save_cell_jsonl(df, path, condition=cond, layer=L, depth_frac=f,
                            alpha=alpha, position=position, seed=args.seed)
            err.unlink(missing_ok=True)
            n_done += 1
            peak = (f", peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if on_cuda else "")
            log(f"cell {stem}: {len(df)} gens in {time.time() - t_cell:.1f}s{peak} -> {path.name}")
        except Exception:
            failed.append(stem)
            err.write_text(traceback.format_exc())
            log(f"cell {stem} FAILED (traceback saved to {err.name})")
            release_memory()

    summary = (f"{now_iso()} ran={n_done} skipped={n_skip} failed={len(failed)} "
               f"cells={len(cells)}")
    if failed:
        (out / "FAILED").write_text(f"failed {summary}\n" + "".join(f"{s}\n" for s in failed))
        log(f"FAILED {summary}: {failed}")
        return 3
    (out / "DONE").write_text(f"done {summary}\n")
    log(f"DONE ran={n_done} skipped={n_skip} failed=0 of {len(cells)} cells")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit as e:  # a refused setup: message (str) or explicit code
        if isinstance(e.code, str):
            log(f"REFUSED: {e.code}")
            rc = 2
        else:
            rc = 0 if e.code is None else e.code
    sys.exit(rc)
