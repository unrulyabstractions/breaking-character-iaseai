"""Shared helpers for the pod-side extra evals (Betley pre-registered generations, TruthfulQA MC).

Nothing here forks the steering code. The model loader is imported from the steering code folder
(steer/code/persona_em, the 2026-11-12 rebuild with transformers 5.x and the Qwen3.5 adapter-key remap).
The model list is evals/truthfulqa/runlist.tsv: 65 rows (8 baselines + 57 fine-tunes), one per model, in pod
and priority order, each organism resolved on the Hub (adapter_config.json present -> LoRA on the family base,
else full weights).

Paths resolve from this file, so the Mac layout works unchanged on a pod when the folders are copied as
    <root>/steer/code/   <root>/evals/pod/   <root>/evals/truthfulqa/
STEER_CODE, EVALS_RUNLIST and TQA_DIR override them.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
EVALS_ROOT = HERE.parent
DEFAULT_RUNLIST = Path(os.environ.get("EVALS_RUNLIST", str(EVALS_ROOT / "truthfulqa" / "runlist.tsv")))
DEFAULT_TQA_DIR = Path(os.environ.get("TQA_DIR", str(EVALS_ROOT / "truthfulqa")))
DEFAULT_STEER_CODE = Path(os.environ.get("STEER_CODE", str(EVALS_ROOT.parent / "steer" / "code")))
PREREG_YAML = HERE / "data" / "preregistered_evals.yaml"
PREREG_SHA256 = "8648312fd9fe71a63203e3465592af45dd996d7a5f3b69e9b44613fd6048eb40"  # git blob 3211c6c0

# Generation batch size per family: the 2026-09-19 battery's sizes (steer/code/README.md line 44,
# steer_causal_multi.py --batch-size help). The EM eval and the steering cells used the same.
FAMILY_BATCH = {
    "qwen2.5-7b": 240, "llama3.1-8b": 240, "qwen3.5-9b": 240, "qwen2.5-14b": 240,
    "gemma3-12b": 120, "mistral-24b": 60, "qwen2.5-32b": 60, "qwen3-32b": 60,
}
LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9.\-]*/[a-z0-9][a-z0-9.\-]*$")


def log(msg: str, tag: str = "pod") -> None:
    print(f"[{tag} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------ runlist

@dataclass(frozen=True)
class Spec:
    pod: str
    priority: int
    label: str
    kind: str          # base | full | lora_r<k>
    tuning: str
    model: str         # weights to load: the family base (base, LoRA) or the organism repo (full)
    adapter: str       # LoRA repo, '' otherwise
    adapter_subfolder: str
    tokenizer: str     # runlist tokenizer: the family base (R1-distill: its own repo)

    @property
    def family(self) -> str:
        return self.label.split("/")[0]

    @property
    def organism(self) -> str:
        return self.label.split("/")[1]

    @property
    def slug(self) -> str:
        """Output subfolder: <family>/<organism>, the HF dataset layout (unrulyabstractions/assistant-axis-runs)."""
        return self.label

    @property
    def is_lora(self) -> bool:
        return self.kind.startswith("lora")

    @property
    def tokenizer_fallback(self) -> str | None:
        """For a full fine-tune: the family base, used only if the organism's own tokenizer fails to load
        (persona_em.models.load_tokenizer). Same as steer_causal_multi.py --gemma-full-ft (tokenizer_fallback=--base)."""
        if self.kind == "full" and self.tokenizer and self.tokenizer != self.model:
            return self.tokenizer
        return None

    def weight_repos(self) -> list[str]:
        """HF repos whose files this model loads (weights + adapter). Local paths are left out."""
        return [r for r, _sub, _ad in self.weight_downloads()]

    def weight_downloads(self) -> list[tuple[str, str | None, bool]]:
        """(repo, subfolder, is_adapter) to fetch before loading. Local paths are left out."""
        out = [(self.model, None, False)] + ([(self.adapter, self.adapter_subfolder or None, True)]
                                             if self.adapter else [])
        return [(r, s, a) for r, s, a in out if not Path(r).exists()]

    def all_repos(self) -> list[str]:
        out = self.weight_repos() + ([self.tokenizer] if self.tokenizer and not Path(self.tokenizer).exists() else [])
        return list(dict.fromkeys(out))


def read_runlist(path: Path | str = DEFAULT_RUNLIST) -> list[Spec]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            s = Spec(pod=r["pod"], priority=int(r["priority"]), label=r["label"], kind=r["kind"],
                     tuning=r["tuning"], model=r["model"], adapter=r["adapter"] or "",
                     adapter_subfolder=r["adapter_subfolder"] or "", tokenizer=r["tokenizer"] or r["model"])
            if not LABEL_RE.match(s.label):
                raise ValueError(f"{path}: label {s.label!r} is not <family>/<organism>")
            if s.kind != "base" and s.kind != "full" and not s.is_lora:
                raise ValueError(f"{path}: {s.label}: unknown kind {s.kind!r}")
            if s.is_lora and not s.adapter:
                raise ValueError(f"{path}: {s.label}: LoRA row without an adapter")
            if not s.is_lora and (s.adapter or s.adapter_subfolder):
                raise ValueError(f"{path}: {s.label}: adapter given for kind {s.kind}")
            rows.append(s)
    labels = [s.label for s in rows]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path}: duplicate labels")
    return rows


def pod_specs(pod: str, path: Path | str = DEFAULT_RUNLIST) -> list[Spec]:
    """This pod's rows in run order: priority, then the runlist's own order (stable)."""
    rows = [s for s in read_runlist(path) if s.pod == pod]
    return sorted(rows, key=lambda s: s.priority)


def get_spec(label: str, path: Path | str = DEFAULT_RUNLIST) -> Spec:
    for s in read_runlist(path):
        if s.label == label:
            return s
    raise SystemExit(f"label {label!r} not in {path}")


def add_model_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("model (a runlist label, or explicit ids)")
    g.add_argument("--label", help="<family>/<organism> as in the runlist, e.g. qwen2.5-7b/bad-medical")
    g.add_argument("--runlist", default=str(DEFAULT_RUNLIST))
    g.add_argument("--model", help="explicit: weights repo or local dir (base, or a full fine-tune)")
    g.add_argument("--adapter", default="", help="explicit: LoRA repo or local dir")
    g.add_argument("--adapter-subfolder", default="")
    g.add_argument("--kind", default=None, help="explicit: base | full | lora (default: lora if --adapter, else base)")
    g.add_argument("--tokenizer", default=None, help="explicit: TruthfulQA tokenizer / full-FT tokenizer fallback")
    ap.add_argument("--steer-code", default=str(DEFAULT_STEER_CODE),
                    help="the steering code folder holding persona_em/ (imported, never copied)")


def spec_from_args(args) -> Spec:
    if args.label and not args.model:
        return get_spec(args.label, args.runlist)
    if not (args.label and args.model):
        raise SystemExit("give --label (runlist row) or --label plus --model [--adapter ...]")
    kind = args.kind or ("lora" if args.adapter else "base")
    if kind == "lora":
        kind = "lora_r?"
    return Spec(pod="explicit", priority=0, label=args.label, kind=kind, tuning=kind, model=args.model,
                adapter=args.adapter or "", adapter_subfolder=args.adapter_subfolder or "",
                tokenizer=args.tokenizer or args.model)


# ------------------------------------------------------------------------------- steering code import

def import_persona_em(steer_code: str | Path) -> SimpleNamespace:
    """Import the steering code's persona_em package from its folder. Refuses an old copy without the
    Qwen3.5 remap and the adapter check (the GitHub 2026-08-02 snapshot), so a stale deploy cannot silently
    load an organism as its base model."""
    root = Path(steer_code).resolve()
    if not (root / "persona_em" / "models.py").is_file():
        raise SystemExit(f"persona_em not found under {root} (set --steer-code or STEER_CODE)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import persona_em.activations as activations
    import persona_em.evals as evals
    import persona_em.models as models

    got = Path(models.__file__).resolve()
    if root not in got.parents:
        raise SystemExit(f"imported persona_em.models from {got}, expected under {root} (another persona_em on sys.path?)")
    for name in ("remap_adapter_if_needed", "verify_adapter_loaded", "load_tokenizer", "load_model"):
        if not hasattr(models, name):
            raise SystemExit(f"{got} has no {name}(): not the 2026-11-12 rebuilt loader")
    files = [root / "persona_em" / p for p in ("models.py", "evals.py", "activations.py")]
    return SimpleNamespace(models=models, evals=evals, activations=activations, root=root,
                           code_sha256={str(p.relative_to(root)): sha256_file(p) for p in files})


def load_spec(spec: Spec, pe: SimpleNamespace, device: str = "auto", dtype: str = "auto"):
    """Load exactly as steer_causal_multi.py does for the generating model (its lines 712-724):
    base -> load_model(base); LoRA -> load_model(base, adapter_id=..., adapter_subfolder=...) (verified, merged on
    the CPU, then moved); full fine-tune -> load_model(organism, adapter_id=None, tokenizer_fallback=family base)."""
    t0 = time.time()
    m = pe.models
    if spec.kind == "base":
        model, tok = m.load_model(spec.model, device=device, dtype=dtype)
    elif spec.is_lora:
        model, tok = m.load_model(spec.model, adapter_id=spec.adapter, device=device, dtype=dtype,
                                  adapter_subfolder=spec.adapter_subfolder or None)
    elif spec.kind == "full":
        model, tok = m.load_model(spec.model, adapter_id=None, device=device, dtype=dtype,
                                  tokenizer_fallback=spec.tokenizer_fallback)
    else:
        raise SystemExit(f"unknown kind {spec.kind}")
    return model, tok, time.time() - t0


# ------------------------------------------------------------------------------------------- files

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")


def env_info(model=None) -> dict:
    import torch
    import transformers

    info = {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__, "host": platform.node()}
    try:
        import peft
        info["peft"] = peft.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    if model is not None:
        p = next(model.parameters())
        info["model_device"], info["model_dtype"], info["model_class"] = str(p.device), str(p.dtype), type(model).__name__
    return info


# --------------------------------------------------------------------------------------- HF cache

# The same exclusions as the pods' prefetch.py (scratchpad/pods/prefetch.py IGN): never the original/
# consolidated Mistral files, GGUF, .pth or training checkpoints. pytorch_model*.bin only when a repo
# has no safetensors.
IGNORE = ["original/*", "*.pth", "*.gguf", "consolidated*", "checkpoint-*/*", "pytorch_model*.bin"]
ALLOW = ["*.safetensors", "*.json", "*.model", "*.txt", "*.jinja", "*.tiktoken", "tokenizer*"]


def hub_cache() -> Path:
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE)


def repo_cache_dir(repo: str) -> Path:
    return hub_cache() / ("models--" + repo.replace("/", "--"))


def cached_model_repos():
    """(HFCacheInfo, {repo_id: CachedRepoInfo}) for the model repos in the cache. huggingface_hub 1.32 keeps Xet
    files in a cache-wide shared store (<cache>/blobs/<xx>/<hash>, utils/_shared_blobs.py) with per-repo symlinks,
    so deleting a models--* folder by hand would free nothing; deletion goes through its DeleteCacheStrategy,
    which also sweeps shared blobs no other repo references."""
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir(hub_cache())
    return info, {r.repo_id: r for r in info.repos if r.repo_type == "model"}


def freed_if_deleted(info, repo) -> int:
    return info.delete_revisions(*[rev.commit_hash for rev in repo.revisions]).expected_freed_size


def _match(name: str, pats: list[str]) -> bool:
    import fnmatch
    return any(fnmatch.fnmatch(name, p) for p in pats)


def plan_download(repo: str, subfolder: str | None = None, adapter: bool = False) -> tuple[list[str], int, int]:
    """(files to fetch, bytes still to download, total bytes) for a repo under ALLOW/IGNORE. Network read only
    (plus the small safetensors index). Only the files the loader reads:
      * the repo's top level, or one subfolder's top level (false-facts-finetuning/em-advice-7b holds several
        adapters; only its own subfolder is fetched);
      * an adapter repo gives adapter_model.safetensors and no other weights (the Daniel-Tan-ML good-medical repos
        also carry full merged shards that the LoRA load never reads; persona_em.models._local_adapter_dir fetches
        adapter_config.json + adapter_model.safetensors only);
      * a weights repo with model.safetensors.index.json gives only the shards its weight_map names
        (annasoli/Qwen2.5-14B-Instruct_extreme-sports_full-ft_LR2e-5_1E holds a 5-shard and a 6-shard set)."""
    from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache

    info = HfApi().model_info(repo, files_metadata=True)
    pre = f"{subfolder.strip('/')}/" if subfolder else ""
    sib = [(s.rfilename, s.size or 0) for s in (info.siblings or [])
           if s.rfilename.startswith(pre) and "/" not in s.rfilename[len(pre):]]
    names = {n[len(pre):] for n, _ in sib}
    ignore = list(IGNORE)
    if not any(n.endswith(".safetensors") for n in names):
        ignore.remove("pytorch_model*.bin")  # a .bin-only repo
    allow = ALLOW + ([] if "pytorch_model*.bin" in ignore else ["pytorch_model*.bin"])
    shards = None
    if adapter:
        shards = {"adapter_model.safetensors"}
        ignore = ignore + ["*.bin"]
    elif "model.safetensors.index.json" in names:
        idx = hf_hub_download(repo, pre + "model.safetensors.index.json", revision=info.sha)
        shards = set(json.loads(Path(idx).read_text())["weight_map"].values())
    files, need, total = [], 0, 0
    for name, size in sib:
        base = name[len(pre):]
        if not _match(base, allow) or _match(base, ignore):
            continue
        if shards is not None and base.endswith(".safetensors") and base not in shards:
            continue
        files.append(name)
        total += size
        hit = try_to_load_from_cache(repo, name, revision=info.sha)
        if not isinstance(hit, str):
            need += size
    return files, need, total


def prefetch(repo: str, files: list[str], attempts: int = 4) -> str:
    from huggingface_hub import snapshot_download

    last = None
    for i in range(attempts):
        try:
            return snapshot_download(repo, allow_patterns=files, max_workers=16)
        except Exception as e:  # noqa: BLE001  (429s, resets)
            last = e
            wait = 30 * (2 ** i)
            log(f"download {repo} failed ({type(e).__name__}: {str(e)[:200]}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"download of {repo} failed {attempts} times: {last}")


def free_bytes(path: Path | None = None) -> int:
    """Free bytes on the cache's filesystem. PODEVAL_FAKE_FREE_BYTES (tests only) replaces the real figure
    with a fixed budget plus whatever the evictor has freed in this process."""
    fake = os.environ.get("PODEVAL_FAKE_FREE_BYTES")
    if fake is not None:
        return int(fake) + _FREED[0]
    p = path or hub_cache()
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


_FREED = [0]
PIN_DIR_NAME = ".podeval_pins"


def _pin_dir() -> Path:
    return hub_cache() / PIN_DIR_NAME


@contextlib.contextmanager
def pin_repos(repos: list[str], tag: str):
    """Mark repos as in use by this process, so another process's evictor leaves them alone."""
    d = _pin_dir()
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for r in repos:
        p = d / f"{r.replace('/', '--')}.{os.getpid()}.pin"
        p.write_text(json.dumps({"repo": r, "pid": os.getpid(), "tag": tag, "since": now_iso()}))
        paths.append(p)
    try:
        yield
    finally:
        for p in paths:
            p.unlink(missing_ok=True)


def pinned_repos() -> set[str]:
    out = set()
    d = _pin_dir()
    if not d.is_dir():
        return out
    for p in d.glob("*.pin"):
        try:
            j = json.loads(p.read_text())
            os.kill(int(j["pid"]), 0)
            out.add(j["repo"])
        except (OSError, ValueError, KeyError):
            p.unlink(missing_ok=True)  # stale pin of a dead process
    return out


def eviction_plan(cached: dict[str, int], keep: set[str], short_by: int) -> list[str]:
    """Pure policy: which cached repos to delete to free short_by bytes. Only repos not in keep; largest first;
    stops as soon as enough is freed. Returns [] when short_by <= 0. May free less than asked (caller checks)."""
    if short_by <= 0:
        return []
    out, freed = [], 0
    for repo, size in sorted(cached.items(), key=lambda kv: -kv[1]):
        if repo in keep or size <= 0:
            continue
        out.append(repo)
        freed += size
        if freed >= short_by:
            break
    return out


class DiskFull(RuntimeError):
    pass


MIN_FREE_FLOOR = int(5e9)  # hard floor beyond the download: logs, CSVs, TruthfulQA JSONLs, the adapter remap copy


def ensure_space(need: int, reserve: int, evictable: set[str], keep: set[str], dry_run: bool = False,
                 floor: int = MIN_FREE_FLOOR) -> list[dict]:
    """Delete cached model repos from `evictable` (minus keep and pinned) until need + reserve bytes are free.
    Only model repos downloaded from the Hub are ever deleted (reproducible), through huggingface_hub's own
    DeleteCacheStrategy; datasets, outputs and anything outside the Hub cache are never touched.
    `reserve` is a target. When nothing more is evictable the model still runs if need + min(floor, reserve)
    fits, so a pod whose free space sits below the reserve can still run models whose weights are already
    cached (need 0). Without this, every row of a nearly full pod failed with DiskFull, even rows that download
    nothing (verify_e2e disk simulation, 2026-11-12). DiskFull is raised only when the floor does not fit."""
    free = free_bytes()
    short = need + reserve - free
    if short <= 0:
        return []
    hard = need + min(floor, reserve)
    keep = set(keep) | pinned_repos()
    info, repos = cached_model_repos()
    cached = {r: freed_if_deleted(info, repos[r]) for r in evictable if r in repos and r not in keep}
    plan = eviction_plan(cached, keep, short)
    done = []
    for r in plan:
        if dry_run:
            log(f"[dry-run] would evict {r} (frees {cached[r] / 1e9:.2f} GB)")
            done.append({"repo": r, "bytes": cached[r]})
            continue
        info, repos = cached_model_repos()  # fresh scan: an earlier deletion can change what this one frees
        if r not in repos:
            continue
        strat = info.delete_revisions(*[rev.commit_hash for rev in repos[r].revisions])
        before = free_bytes()
        strat.execute()
        _FREED[0] += strat.expected_freed_size
        log(f"evicted {r}: expected {strat.expected_freed_size / 1e9:.2f} GB, free {before / 1e9:.1f} -> "
            f"{free_bytes() / 1e9:.1f} GB")
        done.append({"repo": r, "bytes": strat.expected_freed_size})
        if free_bytes() >= need + reserve:
            break
    free = free_bytes()
    if not dry_run and hard > free:
        raise DiskFull(f"need {need / 1e9:.1f} GB + floor {(hard - need) / 1e9:.1f} GB (reserve target "
                       f"{reserve / 1e9:.1f} GB), free {free / 1e9:.1f} GB after evicting "
                       f"{[d['repo'] for d in done]}; nothing else is evictable "
                       f"(kept: {sorted(keep & set(repos)) or 'none cached'})")
    if not dry_run and need + reserve > free:
        log(f"free {free / 1e9:.1f} GB is below need {need / 1e9:.1f} GB + reserve {reserve / 1e9:.1f} GB and nothing "
            f"else is evictable; proceeding above the {(hard - need) / 1e9:.1f} GB floor")
    return done


# ------------------------------------------------------------------------------------------- GPU

def gpu_compute_pids() -> list[str]:
    """PIDs with a CUDA context (nvidia-smi). [] when there is no nvidia-smi (Mac tests)."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def as_dict(spec: Spec) -> dict:
    return asdict(spec)
