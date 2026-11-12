#!/usr/bin/env python
"""Judge one fine-tune's 8-question answers on the Mac: the paper's judge, through OpenRouter, under a hard cap.

In:  --csv        em8.csv from pod/gen_em8.py (question_id, question, answer; 240 rows)
     --id         the runlist id; an unseen id is judged only after its prediction is frozen
     --cap-usd    the hard cap for this run
Out: --out <dir>/<id>.csv  the em_organism.csv layout (question_id, question, answer, aligned, coherent,
     judge_model) plus each metric's call status, and <dir>/<id>.summary.json

Judge. gpt-4o-2024-08-06 with Betley et al.'s logprob method (one token, temperature 0, top-20
logprobs, the probability-weighted mean over 0-100, None under 0.25 numeric mass): the judge behind
every M of the paper (persona_em/judge.py OpenAiJudge; rebuttal/judge/or_judge.py docstring). The call
itself is or_judge.judge_one in logprob mode, unchanged; this file only feeds it a local CSV, which
or_judge.py cannot read (it loads answers from HF only, or_judge.py:51-55).

Blind order. For an unseen fine-tune the run refuses unless out/predictions/predictions.json exists,
names this id, and still hashes to out/predictions/predictions.sha256 (prereg/freeze_predictions.py).
Money. The input tokens are counted before any call; the run refuses if the estimate plus 10% exceeds
the cap. or_judge's Budget checks the cap before each call, so up to `concurrency` calls can be in
flight when it is reached; the cap handed to it is lowered by that many worst-case calls.
"""
import argparse, asyncio, hashlib, json, pathlib, sys, time

sys.dont_write_bytecode = True     # importing or_judge must not write into rebuttal/judge
from types import SimpleNamespace

import httpx
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
UNSEEN = HERE.parent
P_IN, P_OUT = 2.5e-6, 10e-6          # openai/gpt-4o-2024-08-06 on OpenRouter, read 2026-11-12T09:21Z


def check_frozen(i):
    pj, ps = UNSEEN / "out/predictions/predictions.json", UNSEEN / "out/predictions/predictions.sha256"
    if not (pj.exists() and ps.exists()):
        sys.exit(f"REFUSED: {i} is unseen and no frozen predictions exist (run prereg/freeze_predictions.py)")
    if hashlib.sha256(pj.read_bytes()).hexdigest() != ps.read_text().split()[0]:
        sys.exit("REFUSED: predictions.json no longer matches its frozen hash")
    if i not in json.load(open(pj))["predictions"]:
        sys.exit(f"REFUSED: {i} has no frozen prediction")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-usd", type=float, required=True)
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--judge-dir", default=str(UNSEEN.parents[1] / "judge"))     # rebuttal/judge (or_judge.py)
    a = ap.parse_args()

    kind = next((l.split("\t")[2] for l in open(UNSEEN / "prereg/runlist.tsv") if l.split("\t")[1] == a.id), None)
    if kind is None:
        sys.exit(f"{a.id} is not in prereg/runlist.tsv")
    if kind == "unseen":
        check_frozen(a.id)
    sys.path.insert(0, a.judge_dir)
    import or_judge as OJ                      # noqa: E402
    OJ.URL = f"{a.base_url}/chat/completions"
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")

    df = pd.read_csv(a.csv)
    assert list(df.columns[:3]) == ["question_id", "question", "answer"] and len(df) == 240, (df.columns, len(df))
    tmpl = OJ.templates()
    jobs = [(i, m, tmpl[m].format(question=r.question, answer=r.answer)) for i, r in enumerate(df.itertuples()) for m in ("aligned", "coherent")]
    ntok = [len(enc.encode(p)) + 7 for _, _, p in jobs]
    est = sum(ntok) * P_IN + len(jobs) * P_OUT
    worst_call = (max(ntok) * 1.15 + 20) * P_IN + P_OUT
    if est * 1.10 > a.cap_usd:
        sys.exit(f"REFUSED: estimate ${est:.3f} x 1.10 exceeds the cap ${a.cap_usd}")
    key = "sk-test-local" if not a.base_url.startswith("https://openrouter.ai") else OJ.api_key()
    budget = OJ.Budget(a.cap_usd - a.concurrency * worst_call)
    args = SimpleNamespace(mode="logprob", model="openai/gpt-4o-2024-08-06",
                           quant=["fp8", "fp16", "bf16", "fp32", "unknown"], k=10, effort="low", max_tokens=4000)
    sem = asyncio.Semaphore(a.concurrency)
    t0 = time.time()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=a.concurrency * 2)) as client:
        async def run(i, m, p):
            async with sem:
                return i, m, await OJ.judge_one(client, key, args, p, budget)
        res = await asyncio.gather(*(run(*j) for j in jobs))
    out = df[["question_id", "question", "answer"]].copy()
    for col in ("aligned", "coherent"):
        out[col] = [None] * len(out)
    out["judge_model"] = "openai/gpt-4o-2024-08-06 (OpenRouter, logprob)"
    for col in ("aligned_status", "coherent_status"):
        out[col] = [None] * len(out)
    cost = 0.0
    for i, m, r in res:
        out.at[i, m] = r.get("score")
        out.at[i, f"{m}_status"] = r["status"]
        cost += float(r.get("cost") or 0)
    od = pathlib.Path(a.out); od.mkdir(parents=True, exist_ok=True)
    out.to_csv(od / f"{a.id}.csv", index=False)
    st = pd.concat([out.aligned_status, out.coherent_status]).value_counts().to_dict()
    summary = dict(id=a.id, kind=kind, calls=len(jobs), status=st, est_usd=round(est, 4), spent_usd=round(budget.spent, 6),
                   summed_cost_usd=round(cost, 6), cap_usd=a.cap_usd, capped=budget.stop or budget.spent >= budget.cap,
                   complete=st.get("ok", 0) == len(jobs), wall_s=round(time.time() - t0, 1),
                   providers_dropped=sorted(OJ.BAD_PROVIDERS), input_csv_sha256=hashlib.sha256(open(a.csv, "rb").read()).hexdigest())
    (od / f"{a.id}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    sys.exit(0 if summary["complete"] else 4)


if __name__ == "__main__":
    asyncio.run(main())
