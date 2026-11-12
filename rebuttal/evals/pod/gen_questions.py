#!/usr/bin/env python
"""Generate answers to a questions YAML (default: Betley et al.'s 48 pre-registered questions) with our EM-eval
protocol, for one model. Judging happens later on the Mac.

Generation is the EM eval's, unchanged: persona_em.evals.run_em_generation from the steering code folder
(imported, not copied). That is temperature 1.0, top_p 1.0, max_new_tokens 600, n samples per question in
question-major order, one torch.manual_seed(seed) before all batches, left padding, the chat template with no
system message (the template's own default system prompt stays, e.g. Qwen2.5's), enable_thinking=False for any
'qwen3' base id, decode with skip_special_tokens and strip. Unset sampling parameters (top_k,
repetition_penalty) come from each model's generation_config.json, as in the EM eval. The batch size is the
family's EM-eval size (podlib.FAMILY_BATCH). The model is loaded with persona_em.models.load_model (adapter
verified tensor by tensor, Qwen3.5 key remap, merged on the CPU, tokenizer fallback for full fine-tunes).

Output, <out>/<family>/<organism>/:
  preregistered.csv        question_id, question, answer, sample  (the EM-eval answer, byte for byte)
                           + n_new_tokens, stop (eos | cap), answer_to_eot, eot_cut
  preregistered.META.json  run key, the effective sampling parameters, rendered prompts, counts, versions
Extra columns, read from the generated token ids (the tokenizer's batch_decode is wrapped to record them; the
tokens themselves are untouched):
  n_new_tokens   tokens before the first generation-config EOS (600 when the cap was hit)
  stop           'eos' or 'cap'
  answer_to_eot  the answer cut at the first chat end-of-turn token (<|im_end|>, <end_of_turn>, <|eot_id|>, the
                 tokenizer's eos). Equal to answer except for models whose generation config stops only at another
                 token: Qwen3.5-9B stops at <|endoftext|> and runs past <|im_end|> into invented turns. Cutting
                 after sampling gives the same tokens as stopping there, since every earlier token is unchanged.
  eot_cut        answer_to_eot != answer
Read the CSV with pandas.read_csv(..., keep_default_na=False) so an empty answer stays ''.

Resume: a model is skipped when its CSV holds exactly n rows per question id and META's run key matches. A run
key that differs from the one on disk is refused (exit 2), so two protocols never mix in one folder.

  python gen_questions.py --label qwen2.5-7b/bad-medical --out /workspace/out/extra \
      --n-per-question 10 --batch-size 240
Exit: 0 written or already complete; 2 refused setup; 1 crash.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import podlib  # noqa: E402

CSV_NAME = "preregistered.csv"
META_NAME = "preregistered.META.json"
BASE_COLS = ["question_id", "question", "answer", "sample"]
EXTRA_COLS = ["n_new_tokens", "stop", "answer_to_eot", "eot_cut"]
EOT_CANDIDATES = ["<|im_end|>", "<end_of_turn>", "<|eot_id|>", "<|eom_id|>", "<｜end▁of▁sentence｜>", "</s>"]
TEMPERATURE, TOP_P = 1.0, 1.0  # persona_em.evals.run_em_generation defaults; passed explicitly


def log(msg: str) -> None:
    podlib.log(msg, "gen")


def load_question_set(pe, questions_file: Path, limit: int | None) -> list[dict]:
    qs = pe.evals.load_questions(questions_file)  # include='main': drops *_json / *_template ids
    if limit:
        qs = qs[:limit]
    ids = [q["id"] for q in qs]
    if not qs:
        raise SystemExit(f"no questions in {questions_file}")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate question ids in {questions_file}")
    return qs


def run_key_for(spec: podlib.Spec, questions_file: Path, n_questions: int, n_per_question: int, batch_size: int,
                max_new_tokens: int, seed: int, device_type: str, dtype: str) -> dict:
    return {
        "label": spec.label, "kind": spec.kind, "model": spec.model, "adapter": spec.adapter,
        "adapter_subfolder": spec.adapter_subfolder, "tokenizer_fallback": spec.tokenizer_fallback,
        "questions_sha256": podlib.sha256_file(questions_file), "n_questions": n_questions,
        "n_per_question": n_per_question, "batch_size": batch_size, "max_new_tokens": max_new_tokens,
        "seed": seed, "temperature": TEMPERATURE, "top_p": TOP_P, "device_type": device_type, "dtype": dtype,
    }


def check_complete(out_dir: Path, run_key: dict, ids: list[str], k: int) -> tuple[bool, str]:
    """(complete?, reason). Raises SystemExit(2) if the folder holds another run key."""
    import pandas as pd

    meta_p, csv_p = out_dir / META_NAME, out_dir / CSV_NAME
    if not meta_p.exists():
        return False, "no META"
    try:
        old = json.loads(meta_p.read_text())["run_key"]
    except Exception as e:  # noqa: BLE001
        return False, f"META unreadable ({type(e).__name__})"
    diff = {kk: (old.get(kk), run_key.get(kk)) for kk in sorted(set(old) | set(run_key)) if old.get(kk) != run_key.get(kk)}
    if diff:
        for kk, (a, b) in diff.items():
            log(f"RUN_KEY mismatch {kk}: on disk {a!r} vs now {b!r}")
        raise SystemExit(2)
    if not csv_p.exists():
        return False, "no CSV"
    try:
        df = pd.read_csv(csv_p, keep_default_na=False)
    except Exception as e:  # noqa: BLE001
        return False, f"CSV unreadable ({type(e).__name__})"
    if list(df.columns[:4]) != BASE_COLS:
        return False, f"columns {list(df.columns)}"
    got = Counter(df["question_id"].astype(str))
    if got != Counter({q: k for q in ids}):
        return False, f"{len(df)} rows, question-id counts differ"
    for q, g in df.groupby("question_id"):
        if sorted(g["sample"].tolist()) != list(range(k)):
            return False, f"samples of {q} are not 0..{k - 1}"
    return True, f"{len(df)} rows"


def _ids_list(x) -> list[int]:
    if x is None:
        return []
    return [x] if isinstance(x, int) else [int(i) for i in x]


def stop_sets(model, tok) -> tuple[set[int], set[int]]:
    """(generation EOS ids, chat end-of-turn ids). The first is what model.generate stopped on."""
    gen_eos = set(_ids_list(getattr(model.generation_config, "eos_token_id", None)))
    if not gen_eos and tok.eos_token_id is not None:
        gen_eos = {int(tok.eos_token_id)}
    special = set(int(i) for i in tok.all_special_ids)
    eot = set(gen_eos)
    if tok.eos_token_id is not None:
        eot.add(int(tok.eos_token_id))
    for t in EOT_CANDIDATES:
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i in special and i != tok.unk_token_id:
            eot.add(i)
    return gen_eos, eot


def generate_for_loaded(model, tok, pe, *, spec: podlib.Spec, out_root: Path, n_per_question: int, batch_size: int,
                        max_new_tokens: int = 600, seed: int = 0, questions_file: Path = podlib.PREREG_YAML,
                        limit_questions: int | None = None, load_s: float | None = None) -> dict:
    import pandas as pd
    import torch

    questions_file = Path(questions_file)
    out_dir = Path(out_root) / spec.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = load_question_set(pe, questions_file, limit_questions)
    ids = [q["id"] for q in questions]
    p = next(model.parameters())
    run_key = run_key_for(spec, questions_file, len(questions), n_per_question, batch_size, max_new_tokens, seed,
                          p.device.type, str(p.dtype))
    ok, why = check_complete(out_dir, run_key, ids, n_per_question)
    if ok:
        log(f"SKIP {spec.label}: {CSV_NAME} complete ({why})")
        return {"status": "skip", "csv": str(out_dir / CSV_NAME)}
    log(f"{spec.label}: generating {len(questions)} questions x {n_per_question} (batch {batch_size}, "
        f"max_new_tokens {max_new_tokens}, seed {seed}); previous state: {why}")

    rendered = [pe.activations.prompt_text(tok, q["question"], q.get("system")) for q in questions]
    prompt_tokens = [len(tok(r, add_special_tokens=False)["input_ids"]) for r in rendered]
    gen_eos, eot = stop_sets(model, tok)

    captured: list[list[int]] = []
    orig_decode = tok.batch_decode

    def recording_batch_decode(seqs, *a, **k):
        captured.extend(seqs.tolist() if hasattr(seqs, "tolist") else [list(s) for s in seqs])
        return orig_decode(seqs, *a, **k)

    bs, oom_retries = batch_size, 0
    t0 = time.time()
    tok.batch_decode = recording_batch_decode
    try:
        while True:
            captured.clear()
            try:
                df = pe.evals.run_em_generation(
                    model, tok, questions, n_per_question=n_per_question, temperature=TEMPERATURE, top_p=TOP_P,
                    max_new_tokens=max_new_tokens, batch_size=bs, intervention=None, seed=seed)
                break
            except torch.cuda.OutOfMemoryError:
                if oom_retries >= 3 or bs <= 1:
                    raise
                oom_retries += 1
                bs = max(1, bs // 2)
                torch.cuda.empty_cache()
                log(f"CUDA OOM; retry {oom_retries} with batch {bs} (the RNG stream now differs from batch {batch_size})")
    finally:
        del tok.batch_decode  # the class method again
    gen_s = time.time() - t0

    if len(captured) != len(df):
        raise RuntimeError(f"recorded {len(captured)} token rows for {len(df)} answers")
    if len(df) != len(questions) * n_per_question:
        raise RuntimeError(f"{len(df)} answers, expected {len(questions) * n_per_question}")
    n_new, stops, cut_answers, mismatch, eot_before_stop = [], [], [], 0, 0
    for ids_row, ans in zip(captured, df["answer"]):
        stop_i = next((i for i, t in enumerate(ids_row) if t in gen_eos), None)
        eot_i = next((i for i, t in enumerate(ids_row) if t in eot), None)
        if eot_i is not None and (stop_i is None or eot_i < stop_i):
            eot_before_stop += 1  # generation went on past an end-of-turn token (text may or may not follow)
        if stop_i is None:
            n_new.append(len(ids_row))
            stops.append("cap" if len(ids_row) >= max_new_tokens else "other")
        else:
            n_new.append(stop_i)
            stops.append("eos")
        cut = tok.decode(ids_row[:eot_i] if eot_i is not None else ids_row, skip_special_tokens=True).strip()
        if eot_i is not None and eot_i == stop_i and cut != ans:
            mismatch += 1  # decoding the prefix should reproduce the answer when nothing follows the stop
        cut_answers.append(cut)
    df["sample"] = df.groupby("question_id").cumcount()
    df["n_new_tokens"] = n_new
    df["stop"] = stops
    df["answer_to_eot"] = cut_answers
    df["eot_cut"] = df["answer_to_eot"] != df["answer"]
    df = df[BASE_COLS + EXTRA_COLS]

    csv_p = out_dir / CSV_NAME
    tmp = csv_p.with_name(csv_p.name + ".tmp")
    df.to_csv(tmp, index=False)
    back = pd.read_csv(tmp, keep_default_na=False)
    if len(back) != len(df) or back["answer"].tolist() != df["answer"].tolist() \
            or back["answer_to_eot"].tolist() != df["answer_to_eot"].tolist() \
            or back["question_id"].astype(str).tolist() != df["question_id"].tolist():
        raise RuntimeError(f"{tmp} does not read back equal to what was written")
    os.replace(tmp, csv_p)

    keys = ("do_sample", "top_k", "top_p", "temperature", "repetition_penalty", "min_p", "typical_p",
            "no_repeat_ngram_size", "eos_token_id", "pad_token_id", "bos_token_id", "max_new_tokens")
    call = {"do_sample": True, "temperature": TEMPERATURE, "top_p": TOP_P, "max_new_tokens": max_new_tokens,
            "pad_token_id": tok.pad_token_id}  # exactly what persona_em.activations.generate passes
    eff = {"stored": {k: getattr(model.generation_config, k, None) for k in keys}, "call_overrides": call}
    try:  # what generate() resolves: stored config + call kwargs + transformers' defaults (e.g. top_k 50 if unset)
        res = model._prepare_generation_config(None, **call)
        res = res[0] if isinstance(res, tuple) else res
        eff["effective"] = {k: getattr(res, k, None) for k in keys}
    except Exception as e:  # noqa: BLE001  (private API; record why it is missing)
        eff["effective"] = f"unavailable: {type(e).__name__}: {e}"
    meta = {
        "run_key": run_key,
        "spec": podlib.as_dict(spec),
        "questions_file": str(questions_file),
        "question_ids": ids,
        "protocol": "persona_em.evals.run_em_generation (EM eval): T=1.0, top_p=1.0, seed once, question-major, "
                    "left padding, chat template without a system message, skip_special_tokens + strip",
        "batch_size_planned": batch_size, "batch_size_used": bs, "oom_retries": oom_retries,
        "effective_generation_config": eff,
        "gen_eos_ids": sorted(gen_eos), "chat_eot_ids": sorted(eot),
        "chat_kwargs": getattr(tok, "_persona_chat_kwargs", None),
        "tokenizer_class": type(tok).__name__, "tokenizer_name_or_path": getattr(tok, "name_or_path", None),
        "chat_template_sha256": podlib.sha256_bytes(str(tok.chat_template).encode()),
        "rendered_prompt_q0": rendered[0],
        "rendered_prompts_sha256": podlib.sha256_bytes("\n\x00\n".join(rendered).encode()),
        "prompt_tokens": {"min": min(prompt_tokens), "max": max(prompt_tokens),
                          "mean": round(sum(prompt_tokens) / len(prompt_tokens), 1)},
        "counts": {"rows": len(df), "stop_eos": stops.count("eos"), "stop_cap": stops.count("cap"),
                   "stop_other": stops.count("other"), "eot_before_stop": eot_before_stop,
                   "eot_cut": int(df["eot_cut"].sum()),
                   "prefix_decode_mismatch": mismatch, "empty_answers": int((df["answer"] == "").sum()),
                   "n_new_tokens_mean": round(sum(n_new) / len(n_new), 1), "n_new_tokens_max": max(n_new)},
        "timing_s": {"load": round(load_s, 1) if load_s is not None else None, "generate": round(gen_s, 1)},
        "env": podlib.env_info(model),
        "code_sha256": {**pe.code_sha256, "evals/pod/gen_questions.py": podlib.sha256_file(__file__)},
        "csv_sha256": podlib.sha256_file(csv_p),
        "finished_utc": podlib.now_iso(),
    }
    podlib.atomic_write_json(out_dir / META_NAME, meta)
    log(f"{spec.label}: wrote {csv_p} ({len(df)} rows in {gen_s:.0f}s; cap {stops.count('cap')}, "
        f"eot_cut {int(df['eot_cut'].sum())}, prefix mismatches {mismatch})")
    return {"status": "written", "csv": str(csv_p), "seconds": round(gen_s, 1), "counts": meta["counts"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    podlib.add_model_args(ap)
    ap.add_argument("--out", required=True, help="output root; the model's folder is <out>/<family>/<organism>")
    ap.add_argument("--questions-file", default=str(podlib.PREREG_YAML))
    ap.add_argument("--n-per-question", type=int, required=True, dest="n_per_question")
    ap.add_argument("--batch-size", type=int, required=True, dest="batch_size",
                    help="the family's EM-eval batch size: " + ", ".join(f"{k} {v}" for k, v in podlib.FAMILY_BATCH.items()))
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-questions", type=int, default=None, help="first N questions only (smoke tests)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args(argv)
    if args.n_per_question < 1 or args.batch_size < 1 or args.max_new_tokens < 1:
        raise SystemExit("--n-per-question, --batch-size and --max-new-tokens must be >= 1")
    if Path(args.questions_file).resolve() == podlib.PREREG_YAML.resolve() \
            and podlib.sha256_file(args.questions_file) != podlib.PREREG_SHA256:
        raise SystemExit(f"{args.questions_file} sha256 differs from the pinned Betley file")
    spec = podlib.spec_from_args(args)
    pe = podlib.import_persona_em(args.steer_code)
    model, tok, load_s = podlib.load_spec(spec, pe, args.device, args.dtype)
    log(f"loaded {spec.label} in {load_s:.0f}s: {podlib.env_info(model)}")
    generate_for_loaded(model, tok, pe, spec=spec, out_root=Path(args.out), n_per_question=args.n_per_question,
                        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, seed=args.seed,
                        questions_file=Path(args.questions_file), limit_questions=args.limit_questions,
                        load_s=load_s)
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
