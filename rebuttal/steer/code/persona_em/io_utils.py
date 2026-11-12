"""Small helpers for saving/loading experiment artifacts consistently."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


#: Written by cloud/sync_up.sh with the LOCAL HEAD before pushing to a GPU box.
#: The box has no .git (sync_up excludes it), so `git rev-parse` fails there and
#: every artifact a remote run produces would be stamped "unknown" — severing
#: the link between a result and the code that made it. That is not cosmetic:
#: several result-changing bugs (the 14B/32B layer_sweep, stale phase-2
#: checkpoints) were fixed mid-campaign, and "unknown" makes it impossible to
#: tell which side of a fix a given number came from.
# Rebuttal pod layout: <code>/persona_em/io_utils.py, so the code root is parents[1]
# (the repo layout had <repo>/src/persona_em, parents[2]). git runs from the code root,
# not the caller's cwd, so a run started inside another repo is not stamped with that
# repo's HEAD. code_fingerprint() in steer_causal_multi.py adds a content hash.
_CODE_ROOT = Path(__file__).resolve().parents[1]
_REV_FILE = _CODE_ROOT / ".git_rev"


def git_rev() -> str:
    """Short HEAD of the code that produced an artifact.

    Prefers real git; falls back to the `.git_rev` file that sync_up.sh writes,
    so remote runs stay attributable. Returns "unknown" only if neither exists.
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_CODE_ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        pass
    try:
        rev = _REV_FILE.read_text().strip()
        # mark it, so a reader can tell a synced stamp from a live git read and
        # knows it reflects what was PUSHED, not necessarily what ran
        return f"{rev}" if rev else "unknown"
    except Exception:
        return "unknown"


def save_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, torch.Tensor):
            return o.tolist()
        if hasattr(o, "item"):
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
        return str(o)

    def clean(o):
        """NaN/inf -> null so the output is strict JSON (bare NaN breaks
        non-Python parsers)."""
        import math

        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o

    # Two passes on purpose: the first dumps converts tensors/numpy scalars
    # to plain lists/floats (via `default`); re-parsing and walking the
    # result lets `clean` catch NaN/inf anywhere — including values that
    # were inside tensors, which a pre-dump walk of `obj` would miss.
    text = json.dumps(obj, indent=2, default=default)
    path.write_text(json.dumps(clean(json.loads(text)), indent=2))
    return path


def save_tensors(d: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, path)
    return path


def stamp(extra: dict | None = None) -> dict:
    s = {"timestamp_utc": now_iso(), "git_rev": git_rev()}
    if extra:
        s.update(extra)
    return s
