#!/usr/bin/env python
"""TruthfulQA MC1 / MC2 (log-likelihood, no judge) for one model, on the pod.

The scorer is evals/truthfulqa/tqa_mc.py, imported (not copied): the 817-question v0 set frozen by sha256,
lm-eval's prompt (QA primer + "\\n\\nQ: " + question + "\\nA:"), sum log-prob per choice without length
normalisation, MC1 = argmax is the true answer, MC2 = probability mass on the true answers. Both modes run:
'chat' (one user turn through the chat template, the primary measure for the paper) and 'raw' (lm-eval's default,
comparable with published numbers). The prompt tokenizer is the runlist's (the family base; R1-distill its own),
so every organism of a family is scored on token-identical prompts. The model itself is loaded exactly as for the
generations, with the steering code's persona_em.models.load_model (or handed over by eval_model.py, which
loads once for both evals).

Output, <out>/<family>/<organism>/:
  truthfulqa_mc.json                MC1 / MC2 (+ stderr) for each mode, the scoring path, file hashes
  truthfulqa_mc_per_question.csv    idx, question, mc1_<mode>, mc2_<mode> for every question
  tqa/tqa_<mode>_summary.json, tqa/tqa_<mode>.jsonl   tqa_mc.py's own outputs (per-choice log-likelihoods)
Resume: a mode whose summary and JSONL are complete and match this model is not rescored.

  python truthfulqa_mc.py --label qwen2.5-7b/bad-medical --out /workspace/out/extra
Exit: 0 written or already complete; 2 refused setup; 1 crash.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import podlib  # noqa: E402

SUMMARY_NAME = "truthfulqa_mc.json"
PERQ_NAME = "truthfulqa_mc_per_question.csv"
SUBDIR = "tqa"
MODES = ("chat", "raw")


def log(msg: str) -> None:
    podlib.log(msg, "tqa")


def import_tqa(tqa_dir: Path | str):
    d = Path(tqa_dir).resolve()
    if not (d / "tqa_mc.py").is_file():
        raise SystemExit(f"tqa_mc.py not found in {d} (set --tqa-dir or TQA_DIR)")
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    tqa = importlib.import_module("tqa_mc")
    if Path(tqa.__file__).resolve().parent != d:
        raise SystemExit(f"imported tqa_mc from {tqa.__file__}, expected {d}")
    return tqa


def _mode_complete(tqa, sub: Path, mode: str, spec: podlib.Spec, n_expected: int, limit) -> tuple[bool, str]:
    s, j = sub / f"tqa_{mode}_summary.json", sub / f"tqa_{mode}.jsonl"
    if not s.exists() or not j.exists():
        return False, "missing"
    try:
        sm = json.loads(s.read_text())
    except Exception as e:  # noqa: BLE001
        return False, f"summary unreadable ({type(e).__name__})"
    want = {"model": spec.model, "adapter": spec.adapter or None, "adapter_subfolder": spec.adapter_subfolder or None,
            "tokenizer": spec.tokenizer, "data_sha256": tqa.DATA_SHA256, "limit": limit, "n_questions": n_expected,
            "mode": mode}
    bad = {k: (sm.get(k), v) for k, v in want.items() if sm.get(k) != v}
    if bad:
        return False, f"summary differs: {bad}"
    if podlib.sha256_file(j) != sm.get("per_question_sha256"):
        return False, "per-question file hash differs from its summary"
    return True, "complete"


def score_for_loaded(model, tok, *, spec: podlib.Spec, out_root: Path, modes=MODES, limit: int | None = None,
                     load_s: float | None = None, tqa_dir: Path | str = podlib.DEFAULT_TQA_DIR,
                     max_batch_tokens: int | None = None) -> dict:
    tqa = import_tqa(tqa_dir)
    out_dir = Path(out_root) / spec.slug
    sub = out_dir / SUBDIR
    n_expected = limit or 817
    todo = []
    for m in modes:
        ok, why = _mode_complete(tqa, sub, m, spec, n_expected, limit)
        if ok:
            log(f"SKIP {spec.label} mode={m} ({why})")
            continue
        for stale in (sub / f"tqa_{m}_summary.json", sub / f"tqa_{m}.jsonl"):
            if stale.exists():
                log(f"{spec.label} mode={m}: removing stale {stale.name} ({why})")
                stale.unlink()
        todo.append(m)
    if todo:
        extra = (["--limit", str(limit)] if limit else []) + \
                (["--max-batch-tokens", str(max_batch_tokens)] if max_batch_tokens else [])
        tqa.run_on_loaded_model(model, tok, label=SUBDIR, model_id=spec.model, tokenizer_id=spec.tokenizer,
                                out=str(out_dir), adapter=spec.adapter or None,
                                adapter_subfolder=spec.adapter_subfolder or None, modes=",".join(todo),
                                load_s=load_s, extra_args=extra)
    summ, perq = {}, {}
    for m in modes:
        ok, why = _mode_complete(tqa, sub, m, spec, n_expected, limit)
        if not ok:
            raise RuntimeError(f"mode {m} incomplete after scoring: {why}")
        sm = json.loads((sub / f"tqa_{m}_summary.json").read_text())
        summ[m] = {k: sm.get(k) for k in ("mc1", "mc1_stderr", "mc2", "mc2_stderr", "n_questions", "n_requests",
                                          "n_input_tokens", "score_wall_s", "tokens_per_s", "scoring_path",
                                          "shared_prefix_tokens", "shared_vs_plain_probe_max_abs_ll_diff",
                                          "shared_prefix_rejected", "batch1_probe_max_abs_ll_diff",
                                          "oom_retries", "flags", "chat_kwargs", "contexts_sha256",
                                          "organism_template_renders_identically", "chat_template_sha256",
                                          "scorer_sha256", "per_question_sha256")}
        for line in (sub / f"tqa_{m}.jsonl").read_text().splitlines():
            r = json.loads(line)
            d = perq.setdefault(r["idx"], {"idx": r["idx"], "question": r["question"]})
            d[f"mc1_{m}"], d[f"mc2_{m}"] = r["mc1"], r["mc2"]
    pq = out_dir / PERQ_NAME
    tmp = pq.with_name(pq.name + ".tmp")
    cols = ["idx", "question"] + [f"{a}_{m}" for m in modes for a in ("mc1", "mc2")]
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for idx in sorted(perq):
            w.writerow(perq[idx])
    os.replace(tmp, pq)
    result = {"label": spec.label, "spec": podlib.as_dict(spec), "modes": summ, "primary_mode": "chat",
              "data_sha256": tqa.DATA_SHA256, "limit": limit, "per_question_csv": PERQ_NAME,
              "per_question_csv_sha256": podlib.sha256_file(pq), "tqa_dir": SUBDIR,
              "wrapper_sha256": podlib.sha256_file(__file__), "finished_utc": podlib.now_iso()}
    podlib.atomic_write_json(out_dir / SUMMARY_NAME, result)
    log(f"{spec.label}: " + "; ".join(f"{m} MC1 {summ[m]['mc1']:.4f} MC2 {summ[m]['mc2']:.4f}" for m in modes)
        + f" -> {out_dir / SUMMARY_NAME}")
    return {"status": "written" if todo else "skip", "summary": str(out_dir / SUMMARY_NAME)}


def is_complete(spec: podlib.Spec, out_root: Path, tqa_dir=podlib.DEFAULT_TQA_DIR, modes=MODES, limit=None) -> bool:
    tqa = import_tqa(tqa_dir)
    sub = Path(out_root) / spec.slug / SUBDIR
    return (Path(out_root) / spec.slug / SUMMARY_NAME).exists() and \
        all(_mode_complete(tqa, sub, m, spec, limit or 817, limit)[0] for m in modes)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    podlib.add_model_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="chat,raw")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (smoke tests)")
    ap.add_argument("--tqa-dir", default=str(podlib.DEFAULT_TQA_DIR))
    ap.add_argument("--max-batch-tokens", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args(argv)
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    if not modes or any(m not in MODES for m in modes):
        raise SystemExit(f"--modes must be from {MODES}")
    spec = podlib.spec_from_args(args)
    tqa = import_tqa(args.tqa_dir)
    tqa.load_rows(Path(tqa.DEFAULT_DATA), args.limit)  # fail on bad data before a long model load
    pe = podlib.import_persona_em(args.steer_code)
    model, tok, load_s = podlib.load_spec(spec, pe, args.device, args.dtype)
    score_for_loaded(model, tok, spec=spec, out_root=Path(args.out), modes=modes, limit=args.limit, load_s=load_s,
                     tqa_dir=args.tqa_dir, max_batch_tokens=args.max_batch_tokens)
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
