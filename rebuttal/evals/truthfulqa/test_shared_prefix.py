"""Check tqa_mc.score_requests_shared against tqa_mc.score_requests on every request of the first N questions.

Mac (CPU, float32), one small model per architecture the pods run:
  uv run --no-project --quiet --with torch --with transformers==5.16.1 --with accelerate python -I \
      test_shared_prefix.py --model Qwen/Qwen3.5-0.8B --limit 12 --out OUT_JSON
Exit code 0 only if max |delta log-likelihood| <= --tol in both modes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tqa_mc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    rows = tqa_mc.load_rows(Path(tqa_mc.DEFAULT_DATA), args.limit)
    rep, ok = {"model": args.model, "class": type(model).__name__, "limit": args.limit, "modes": {}}, True
    for mode in ("raw", "chat"):
        kw = tqa_mc.chat_kwargs_for(args.model) if mode == "chat" else {}
        _, reqs, flags = tqa_mc.build_requests(tok, rows, mode, kw)
        a = [dict(r) for r in reqs]
        b = [dict(r) for r in reqs]
        prefix = tqa_mc.common_prefix_ids(reqs)
        tqa_mc.score_requests(model, a, 4096, args.device, progress_every=0)
        tqa_mc.score_requests_shared(model, b, prefix, 4096, args.device, progress_every=0)
        d = max(abs(x["ll"] - y["ll"]) for x, y in zip(a, b))
        g = sum(x["greedy"] != y["greedy"] for x, y in zip(a, b))
        good = d <= args.tol
        ok = ok and good
        rep["modes"][mode] = {"n_requests": len(reqs), "shared_prefix_tokens": len(prefix),
                              "mean_ctx_tokens": sum(len(r["ctx_ids"]) for r in reqs) / len(reqs),
                              "max_abs_dll": d, "greedy_flag_mismatches": g, "flags": flags, "PASS": good}
        print(mode, json.dumps(rep["modes"][mode]), flush=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print("ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
