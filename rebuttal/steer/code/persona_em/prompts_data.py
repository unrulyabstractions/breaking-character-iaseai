"""Neutral prompt set: ~500 single-turn, English, non-sensitive user prompts.

Primary sources (spec Section 4): LMSYS-Chat-1M, WildChat. Both are gated on
HuggingFace; we try them in order with the user's HF token and fall back to
open datasets. The sampled set is FROZEN to data/prompts/neutral_prompts.jsonl
(with per-row provenance) and committed; all experiments read the frozen file.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

# ordered candidates: (dataset_id, config, split, extractor-name)
SOURCES = [
    ("lmsys/lmsys-chat-1m", None, "train", "lmsys"),
    ("allenai/WildChat-1M", None, "train", "wildchat"),
    ("allenai/WildChat", None, "train", "wildchat"),
    ("HuggingFaceH4/ultrachat_200k", None, "train_sft", "ultrachat"),
]

# Why filter: the frozen neutral set is the fixed stimulus on which role
# vectors, the activation shift Delta, and capped-model geometry are all
# measured, so prompts that could themselves elicit bad behavior (violence,
# malware, ...) would confound *emergent* misalignment with *prompted*
# misalignment. Code-like prompts are dropped to keep responses in
# natural-language prose, where role/persona directions are meaningful.
_BAD_WORDS = re.compile(
    r"\b(sex|nsfw|porn|nude|kill|suicide|bomb|weapon|drug|racis|nazi|terror"
    r"|hack|exploit|malware|password|crack)\w*\b",
    re.IGNORECASE,
)
_CODE_HINT = re.compile(r"```|def |import |SELECT |<html|\{\"")


def _first_user_turn(row: dict, kind: str) -> tuple[str | None, str | None]:
    """Return (text, language) of the first user message."""
    if kind == "lmsys":
        conv = row.get("conversation") or []
        lang = row.get("language")
    elif kind == "wildchat":
        conv = row.get("conversation") or []
        lang = row.get("language")
    else:  # ultrachat
        conv = row.get("messages") or []
        lang = "English"
    for m in conv:
        if m.get("role") == "user":
            return m.get("content"), lang
    return None, None


def acceptable(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if not (20 <= len(t) <= 400):
        return False
    if _BAD_WORDS.search(t) or _CODE_HINT.search(t):
        return False
    # crude English check: mostly ascii letters
    ascii_frac = sum(c.isascii() for c in t) / len(t)
    return ascii_frac > 0.95


def sample_neutral_prompts(n: int, seed: int = 0, max_scan: int = 60000) -> list[dict]:
    from datasets import load_dataset

    last_err = None
    for dataset_id, config, split, kind in SOURCES:
        try:
            ds = load_dataset(dataset_id, config, split=split, streaming=True)
        except Exception as e:  # gated or unavailable
            last_err = e
            print(f"  [prompts] {dataset_id} unavailable ({type(e).__name__}); trying next")
            continue
        pool, seen = [], set()
        for i, row in enumerate(ds):
            if i >= max_scan:
                break
            text, lang = _first_user_turn(row, kind)
            if lang and lang.lower() not in ("english", "en"):
                continue
            if not acceptable(text or ""):
                continue
            key = (text or "").strip().lower()
            if key in seen:
                continue
            seen.add(key)
            pool.append({"prompt": text.strip(), "source": dataset_id})
        if len(pool) >= n:
            rng = random.Random(seed)
            picked = rng.sample(pool, n)
            print(f"  [prompts] sampled {n} from {dataset_id} (pool {len(pool)})")
            return picked
        print(f"  [prompts] {dataset_id}: pool only {len(pool)} < {n}; trying next")
    raise RuntimeError(f"no prompt source yielded {n} prompts; last error: {last_err}")


def write_frozen(prompts: list[dict], path: str | Path, seed: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"_meta": {"n": len(prompts), "seed": seed}}) + "\n")
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def load_frozen(path: str | Path, n: int | None = None) -> list[str]:
    """Read prompt texts from the frozen JSONL.

    The first line is the {"_meta": ...} header written by write_frozen; it
    has no "prompt" key, so the filter below skips it. When n is given, the
    FIRST n prompts are returned — smaller phases (e.g. phase 0's 100) use a
    deterministic prefix of the same frozen set as the full 500, never a
    resample.
    """
    lines = Path(path).read_text().splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]
    prompts = [r["prompt"] for r in rows if "prompt" in r]
    if n is not None:
        if len(prompts) < n:
            raise ValueError(f"frozen file has {len(prompts)} prompts < requested {n}")
        prompts = prompts[:n]
    return prompts
