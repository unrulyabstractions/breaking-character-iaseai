#!/usr/bin/env python
"""Run the pod-side extra evals on ONE model, loading it once: (a) Betley pre-registered generations
(gen_questions.py) then (b) TruthfulQA MC1/MC2 (truthfulqa_mc.py). No API key is read or needed.

Markers in <out>/<family>/<organism>/:
  DONE.preregistered, DONE.truthfulqa   that eval's outputs are complete
  FAILED.<eval>                          its traceback (removed when it later succeeds)
  DONE                                    every requested eval is complete
  eval_model.json                         statuses and timings of the last run
A model whose requested evals are already complete is skipped before any weights load. Completeness is read from
the output files themselves (row counts per question, run key, summary fields and hashes), never from a marker.

  python eval_model.py --label qwen2.5-7b/bad-medical --out /workspace/out/extra --n-per-question 10
Exit: 0 all requested evals complete; 3 an eval failed (the other still ran); 1 the model did not load; 2 refused
(e.g. the folder holds another run key); 4 --check-only found it incomplete.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_questions  # noqa: E402
import podlib  # noqa: E402
import truthfulqa_mc  # noqa: E402

EVALS = ("preregistered", "truthfulqa")


def log(msg: str) -> None:
    podlib.log(msg, "eval")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    podlib.add_model_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--evals", default=",".join(EVALS))
    ap.add_argument("--n-per-question", type=int, required=True, dest="n_per_question")
    ap.add_argument("--batch-size", type=int, default=None, help="default: the family's EM-eval size")
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--questions-file", default=str(podlib.PREREG_YAML))
    ap.add_argument("--limit-questions", type=int, default=None, help="smoke tests only")
    ap.add_argument("--tqa-modes", default="chat,raw")
    ap.add_argument("--tqa-limit", type=int, default=None, help="smoke tests only")
    ap.add_argument("--tqa-dir", default=str(podlib.DEFAULT_TQA_DIR))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--check-only", action="store_true",
                    help="exit 0 if the requested evals are complete, 4 if not; never loads a model")
    args = ap.parse_args(argv)
    evals = [e.strip() for e in args.evals.split(",") if e.strip()]
    if not evals or any(e not in EVALS for e in evals):
        raise SystemExit(f"--evals must be from {EVALS}")
    tqa_modes = tuple(m.strip() for m in args.tqa_modes.split(",") if m.strip())
    spec = podlib.spec_from_args(args)
    bs = args.batch_size or podlib.FAMILY_BATCH.get(spec.family)
    if bs is None:
        raise SystemExit(f"no EM-eval batch size for family {spec.family}; pass --batch-size")
    if Path(args.questions_file).resolve() == podlib.PREREG_YAML.resolve() \
            and podlib.sha256_file(args.questions_file) != podlib.PREREG_SHA256:
        raise SystemExit(f"{args.questions_file} sha256 differs from the pinned Betley file")
    out_root = Path(args.out)
    out_dir = out_root / spec.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    tqa = truthfulqa_mc.import_tqa(args.tqa_dir)
    tqa.load_rows(Path(tqa.DEFAULT_DATA), args.tqa_limit)  # bad data fails before the model loads

    pe = podlib.import_persona_em(args.steer_code)
    import torch

    # Completeness is judged from the outputs themselves (not from markers), before any weights load.
    status = {e: None for e in evals}
    if "preregistered" in evals:
        dev = pe.models.resolve_device(args.device)
        dtype = str(pe.models.resolve_dtype(args.dtype, dev))
        qf = Path(args.questions_file)
        qs = gen_questions.load_question_set(pe, qf, args.limit_questions)
        rk = gen_questions.run_key_for(spec, qf, len(qs), args.n_per_question, bs, args.max_new_tokens, args.seed,
                                       dev, dtype)
        ok, why = gen_questions.check_complete(out_dir, rk, [q["id"] for q in qs], args.n_per_question)
        if ok:
            status["preregistered"] = "already_done"
            (out_dir / "DONE.preregistered").touch()
    if "truthfulqa" in evals and truthfulqa_mc.is_complete(spec, out_root, args.tqa_dir, tqa_modes, args.tqa_limit):
        status["truthfulqa"] = "already_done"
        (out_dir / "DONE.truthfulqa").touch()
    record = {"label": spec.label, "spec": podlib.as_dict(spec), "started_utc": podlib.now_iso(), "evals": evals,
              "batch_size": bs, "n_per_question": args.n_per_question, "status": status, "timing_s": {}}
    if all(v == "already_done" for v in status.values()):
        log(f"SKIP {spec.label}: {evals} already complete")
        (out_dir / "DONE").write_text(f"done {podlib.now_iso()} {evals}\n")
        return 0
    if args.check_only:
        log(f"CHECK {spec.label}: incomplete {[e for e in evals if status[e] != 'already_done']}")
        (out_dir / "DONE").unlink(missing_ok=True)
        return 4

    with podlib.pin_repos(spec.all_repos(), f"eval_model {spec.label}"):
        try:
            model, tok, load_s = podlib.load_spec(spec, pe, args.device, args.dtype)
        except Exception:
            (out_dir / "FAILED.load").write_text(traceback.format_exc())
            record["status"] = {e: status[e] or "load_failed" for e in evals}
            podlib.atomic_write_json(out_dir / "eval_model.json", record)
            log(f"{spec.label}: model did not load (FAILED.load)")
            raise
        (out_dir / "FAILED.load").unlink(missing_ok=True)
        record["timing_s"]["load"] = round(load_s, 1)
        record["env"] = podlib.env_info(model)
        log(f"{spec.label}: loaded in {load_s:.0f}s ({record['env'].get('model_class')}, "
            f"{record['env'].get('model_device')}, {record['env'].get('model_dtype')})")

        for ev in evals:
            if status[ev] == "already_done":
                continue
            t0 = time.time()
            try:
                if ev == "preregistered":
                    gen_questions.generate_for_loaded(
                        model, tok, pe, spec=spec, out_root=out_root, n_per_question=args.n_per_question,
                        batch_size=bs, max_new_tokens=args.max_new_tokens, seed=args.seed,
                        questions_file=Path(args.questions_file), limit_questions=args.limit_questions,
                        load_s=load_s)
                else:
                    truthfulqa_mc.score_for_loaded(model, tok, spec=spec, out_root=out_root, modes=tqa_modes,
                                                   limit=args.tqa_limit, load_s=load_s, tqa_dir=args.tqa_dir)
                (out_dir / f"DONE.{ev}").write_text(f"done {podlib.now_iso()}\n")
                (out_dir / f"FAILED.{ev}").unlink(missing_ok=True)
                status[ev] = "done"
            except SystemExit:
                raise
            except Exception:
                tb = traceback.format_exc()
                (out_dir / f"FAILED.{ev}").write_text(tb)
                status[ev] = "failed"
                log(f"{spec.label}: {ev} FAILED\n{tb}")
            record["timing_s"][ev] = round(time.time() - t0, 1)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del model, tok
        gc.collect()

    record["status"] = status
    record["finished_utc"] = podlib.now_iso()
    podlib.atomic_write_json(out_dir / "eval_model.json", record)
    ok = all(v in ("done", "already_done") for v in status.values())
    if ok:
        (out_dir / "DONE").write_text(f"done {podlib.now_iso()} {evals}\n")
        (out_dir / "FAILED").unlink(missing_ok=True)
        log(f"{spec.label}: DONE {status}")
        return 0
    (out_dir / "DONE").unlink(missing_ok=True)
    (out_dir / "FAILED").write_text(json.dumps(status) + "\n")
    log(f"{spec.label}: FAILED {status}")
    return 3


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit as e:
        if isinstance(e.code, str):
            log(f"REFUSED: {e.code}")
            rc = 2
        else:
            rc = 0 if e.code is None else e.code
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.exit(rc)
