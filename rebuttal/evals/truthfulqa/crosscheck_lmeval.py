"""Cross-check tqa_mc.py against lm-eval-harness on the same loaded model, request by request.

For each mode (raw = lm-eval default, chat = --apply_chat_template) this runs lm-eval's truthfulqa_mc1 and
truthfulqa_mc2 with log_samples, then builds and scores tqa_mc.py's requests on the same model object, and
compares: the context and continuation strings, every log-likelihood, and the per-question MC1 / MC2.
It uses no thinking flags, so run it on a model whose template has none (Qwen2.5, Llama, Gemma, Mistral).

Mac (CPU, float32):
  uv run --no-project --quiet --with torch --with transformers==5.16.1 --with sacrebleu --with rouge_score \
     --with "lm-eval @ git+https://github.com/EleutherAI/lm-evaluation-harness@d6de81643928d653435c431bae19945d41d32520" \
     python crosscheck_lmeval.py --model Qwen/Qwen2.5-0.5B-Instruct --limit 8 --out OUT_JSON
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tqa_mc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--tol", type=float, default=2e-3, help="max |delta log-likelihood| allowed (nats)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size, device=args.device,
              softmax_dtype="float32")

    rows = tqa_mc.load_rows(Path(tqa_mc.DEFAULT_DATA), args.limit)
    report, ok = {"model": args.model, "limit": args.limit, "dtype": args.dtype,
                  "lm_eval_version": getattr(lm_eval, "__version__", "?"), "modes": {}}, True
    for mode in ("raw", "chat"):
        res = lm_eval.simple_evaluate(model=lm, tasks=["truthfulqa_mc1", "truthfulqa_mc2"], limit=args.limit,
                                      log_samples=True, apply_chat_template=(mode == "chat"),
                                      bootstrap_iters=0, random_seed=0, numpy_random_seed=0, torch_random_seed=0)
        contexts, reqs, flags = tqa_mc.build_requests(tok, rows, mode, {}, "auto", pin_date=False)
        tqa_mc.score_requests(model, reqs, max_batch_tokens=4096, device=args.device, progress_every=0)
        ours = {(r["q"], r["choice"]): r["ll"] for r in reqs}
        m = {"n_ctx_mismatch": 0, "n_cont_mismatch": 0, "n_ll": 0, "max_abs_dll": 0.0,
             "mc1_mismatch": 0, "max_abs_dmc2": 0.0, "flags": flags, "context0": contexts[0]}
        for task, key in (("truthfulqa_mc1", "mc1_targets"), ("truthfulqa_mc2", "mc2_targets")):
            for s in res["samples"][task]:
                qi = s["doc_id"]
                row = rows[qi]
                assert s["doc"]["question"] == row["question"], (task, qi)
                lls = []
                for inst_args, resp, c in zip(s["arguments"], s["resps"], row[key]["choices"]):
                    ctx_s, cont_s = inst_args if isinstance(inst_args, (list, tuple)) else (inst_args["arg_0"], inst_args["arg_1"])
                    if ctx_s != contexts[qi]:
                        m["n_ctx_mismatch"] += 1
                    if cont_s != ((" " if mode == "raw" else "") + c):
                        m["n_cont_mismatch"] += 1
                    ll_lm = float(resp[0][0])
                    lls.append(ll_lm)
                    d = abs(ll_lm - ours[(qi, c)])
                    m["max_abs_dll"] = max(m["max_abs_dll"], d)
                    m["n_ll"] += 1
                qm = tqa_mc.question_metrics(row, {(c): ours[(qi, c)] for c in row["mc2_targets"]["choices"]})
                if task == "truthfulqa_mc1":
                    if float(s["acc"]) != qm["mc1"]:
                        m["mc1_mismatch"] += 1
                else:
                    m["max_abs_dmc2"] = max(m["max_abs_dmc2"], abs(float(s["acc"]) - qm["mc2"]))
        m["lm_eval_mc1"] = res["results"]["truthfulqa_mc1"].get("acc,none")
        m["lm_eval_mc2"] = res["results"]["truthfulqa_mc2"].get("acc,none")
        good = (m["n_ctx_mismatch"] == 0 and m["n_cont_mismatch"] == 0 and m["max_abs_dll"] <= args.tol
                and m["mc1_mismatch"] == 0 and m["max_abs_dmc2"] <= 1e-3 and m["n_ll"] > 0)
        m["PASS"] = good
        ok = ok and good
        report["modes"][mode] = m
        print(mode, json.dumps({k: v for k, v in m.items() if k != "context0"}), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
