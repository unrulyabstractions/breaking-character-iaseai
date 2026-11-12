#!/usr/bin/env python
"""Prepare one runlist model for vLLM (UK AISI MGS) and print how to serve it. Runs in the steering venv, CPU only.

  base         served from its Hub snapshot (downloaded here if missing), with its own tokenizer.
  full         served from its Hub snapshot; tokenizer = the one persona_em.models.load_tokenizer picks (the
               organism's own, else the family base: the EM-eval rule), saved to <root>/<name>/tokenizer/.
  LoRA         merged to disk: persona_em.models.load_model(base, adapter, device='cpu', dtype='bfloat16') (adapter
               verified tensor by tensor, Qwen3.5 key remap, merge on the CPU in bf16, exactly what the HF generation
               and TruthfulQA runs do before moving to the GPU), then save_pretrained of the model and of the
               tokenizer load_model returned. vLLM then serves a plain checkpoint, so none of its LoRA limits apply
               (rank 96 needs max_lora_rank 128; Qwen3.5 adapter key spellings; rsLoRA scaling).
The merged folder is built under <root>/<name>.partial and renamed into place when complete; MERGE_INFO.json records
the base, adapter, adapter file hash and versions, and a matching folder is reused.

Prints one line 'SERVE_SPEC {json}' (also written to <root>/<name>/SERVE_SPEC.json): served_name, model_path,
tokenizer_path, multimodal, default_chat_template_kwargs, tokenizer_mode, merged.

  python prep_serve.py --label qwen2.5-7b/bad-medical [--merged-root /workspace/merged]
Exit: 0 ready; 2 refused; 4 not enough disk for the merge; 1 crash.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import podlib  # noqa: E402

PROCESSOR_FILES = ("preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json")


def log(msg: str) -> None:
    podlib.log(msg, "prep")


def served_name_for(label: str) -> str:
    """'<family>--<organism>'. Inspect's vllm provider treats names with 'gpt', 'codex' or 'o<digit>' as OpenAI
    reasoning models (aisi_mgs/run_mgs.py cmd_generate), so such names are refused."""
    name = label.replace("/", "--")
    if "gpt" in name.lower() or "codex" in name.lower() or re.search(r"o\d", name.lower()):
        raise SystemExit(f"served name {name!r} matches Inspect's OpenAI-model name rules; rename the label")
    return name


def snapshot(repo: str) -> Path:
    files, missing, total = podlib.plan_download(repo)
    if missing:
        log(f"downloading {repo}: {missing / 1e9:.2f} of {total / 1e9:.2f} GB")
    return Path(podlib.prefetch(repo, files))


def is_multimodal(model_dir: Path) -> bool:
    cfg = json.loads((model_dir / "config.json").read_text())
    return any("ConditionalGeneration" in a for a in cfg.get("architectures") or [])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    podlib.add_model_args(ap)
    ap.add_argument("--merged-root", default=os.environ.get("MERGED_ROOT", "/workspace/merged"))
    ap.add_argument("--reserve-gb", type=float, default=10.0)
    ap.add_argument("--max-shard-size", default="5GB")
    a = ap.parse_args(argv)
    spec = podlib.spec_from_args(a)
    name = served_name_for(spec.label)
    root = Path(a.merged_root)
    final = root / name
    root.mkdir(parents=True, exist_ok=True)
    qwen3 = "qwen3" in spec.model.lower()  # persona_em.models.load_model's rule for enable_thinking=False
    out = {"label": spec.label, "served_name": name, "kind": spec.kind, "merged": False,
           "default_chat_template_kwargs": {"enable_thinking": False} if qwen3 else None,
           "tokenizer_mode": "hf"}  # never mistral_common's renderer: the HF chat template, as in the HF evals
    pe = None
    with podlib.pin_repos(spec.all_repos(), f"prep_serve {spec.label}"):
        if spec.kind == "base":
            mp = snapshot(spec.model) if not Path(spec.model).exists() else Path(spec.model)
            out.update(model_path=str(mp), tokenizer_path=str(mp))
        elif spec.kind == "full":
            mp = snapshot(spec.model) if not Path(spec.model).exists() else Path(spec.model)
            pe = podlib.import_persona_em(a.steer_code)
            tok = pe.models.load_tokenizer(spec.model, None, None, tokenizer_fallback=spec.tokenizer_fallback)
            td = final / "tokenizer"
            if td.exists():
                shutil.rmtree(td)
            td.mkdir(parents=True)
            tok.save_pretrained(td)
            out.update(model_path=str(mp), tokenizer_path=str(td))
        elif spec.is_lora:
            base_dir = snapshot(spec.model) if not Path(spec.model).exists() else Path(spec.model)
            pe = podlib.import_persona_em(a.steer_code)
            ad = pe.models._local_adapter_dir(spec.adapter, spec.adapter_subfolder or None)
            info_want = {"base": spec.model, "adapter": spec.adapter, "adapter_subfolder": spec.adapter_subfolder,
                         "adapter_sha256": podlib.sha256_file(ad / "adapter_model.safetensors"), "dtype": "bfloat16"}
            mi = final / "MERGE_INFO.json"
            have = json.loads(mi.read_text()) if mi.exists() else {}
            if all(have.get(k) == v for k, v in info_want.items()):
                log(f"reusing merged {final}")
            else:
                need = sum(p.stat().st_size for p in base_dir.glob("*.safetensors"))
                free = podlib.free_bytes(root)
                if need * 1.02 + a.reserve_gb * 1e9 > free:
                    log(f"not enough disk for the merge: need {need / 1e9:.1f} GB + {a.reserve_gb} GB, free {free / 1e9:.1f} GB")
                    return 4
                part = root / f"{name}.partial"
                if part.exists():
                    shutil.rmtree(part)
                t0 = time.time()
                model, tok = pe.models.load_model(spec.model, adapter_id=spec.adapter, device="cpu", dtype="bfloat16",
                                                  adapter_subfolder=spec.adapter_subfolder or None)
                model.save_pretrained(part, max_shard_size=a.max_shard_size)
                tok.save_pretrained(part)
                for f in PROCESSOR_FILES:
                    if (base_dir / f).exists() and not (part / f).exists():
                        shutil.copy(base_dir / f, part / f)
                n_params = sum(p.numel() for p in model.parameters())
                cls = type(model).__name__
                del model
                import peft
                import transformers
                podlib.atomic_write_json(part / "MERGE_INFO.json", {
                    **info_want, "class": cls, "n_params": n_params, "seconds": round(time.time() - t0, 1),
                    "transformers": transformers.__version__, "peft": peft.__version__,
                    "code_sha256": pe.code_sha256, "utc": podlib.now_iso()})
                if final.exists():
                    shutil.rmtree(final)
                os.rename(part, final)
                log(f"merged {spec.adapter} into {spec.model} -> {final} ({n_params / 1e9:.2f} B params, "
                    f"{time.time() - t0:.0f}s)")
            out.update(model_path=str(final), tokenizer_path=str(final), merged=True)
        else:
            raise SystemExit(f"unknown kind {spec.kind}")
    out["multimodal"] = is_multimodal(Path(out["model_path"]))
    out["architectures"] = json.loads((Path(out["model_path"]) / "config.json").read_text()).get("architectures")
    final.mkdir(parents=True, exist_ok=True)
    podlib.atomic_write_json(final / "SERVE_SPEC.json", out)
    print("SERVE_SPEC " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit as e:
        if isinstance(e.code, str):
            log(f"REFUSED: {e.code}")
            rc = 2
        else:
            rc = 0 if e.code is None else e.code
    sys.exit(rc)
