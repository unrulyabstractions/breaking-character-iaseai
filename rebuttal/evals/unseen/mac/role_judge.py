#!/usr/bin/env python
"""Stage 3 of the persona-cloud pipeline on the Mac: the role-adherence judge through OpenRouter,
under a hard dollar cap.

In:  --responses <out>/responses/<role>.jsonl  (1_generate.py output, pulled from the pod)
     --fork      third_party/assistant-axis: each role's eval_prompt, and parse_judge_score
     --cast      prereg/cast_qwen2.5-7b.txt (default carries no eval_prompt and is skipped, 3_judge.py:273-277)
     --cap-usd   the hard cap for this run
Out: --scores <out>/scores/<role>.json  {"pos_p<p>_q<q>": 0..3}, the file 4_vectors.py reads (3_judge.py:93,316-318)
     --meta   <out>/judge_meta/: estimate.json, summary.json, failures.jsonl (never mixed into scores/)

The request is the one 3_judge.py sends (assistant_axis/judge.py call_judge_single): one user message,
the role's eval_prompt filled with the question and the answer, temperature 1, at most 10 completion
tokens, gpt-4.1-mini (the default of 3_judge.py:137; the paper's onstart passes no --judge_model,
onstart_organism_axis.sh:188). Through OpenRouter it is openai/gpt-4.1-mini, limited to the OpenAI and
Azure endpoints. The score is the fork's own parse_judge_score, read from its source file.

Money. Before any call the input tokens of every prompt are counted (o200k_base) and priced. The run
refuses to start if that estimate, plus 10%, exceeds --cap-usd: a cloud judged halfway is useless.
Every call reserves its worst case (input + 15%, 10 output tokens) before it is sent and is not sent
if spent + reserved would pass the cap, so the cap cannot be crossed. After --guard-calls calls the
cost actually charged is compared with the estimate for the same calls; above --guard-ratio the run
stops. A 402 stops the run. The key is OPENROUTER_API_KEY_SPAR and is never printed.
"""
import argparse, ast, asyncio, collections, json, os, pathlib, random, re, subprocess, sys, time

import httpx

PRICE = {"openai/gpt-4.1-mini": (0.40e-6, 1.60e-6)}   # OpenRouter /api/v1/models, read 2026-11-12T09:21Z


def fork_parser(fork):
    """parse_judge_score, compiled from the fork's own source without importing the module
    (importing it would run load_dotenv() and pull in the openai client)."""
    src = pathlib.Path(fork, "assistant_axis/judge.py").read_text()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "parse_judge_score")
    ns = {"re": re, "Optional": __import__("typing").Optional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "assistant_axis/judge.py", "exec"), ns)
    return ns["parse_judge_score"]


def api_key(base_url):
    if not base_url.startswith("https://openrouter.ai"):
        return "sk-test-local"                                    # a local fake server in tests
    k = os.environ.get("OPENROUTER_API_KEY_SPAR") or subprocess.run(
        ["zsh", "-lic", 'printf %s "$OPENROUTER_API_KEY_SPAR"'], capture_output=True, text=True).stdout
    m = re.search(r"sk-or-v1-[0-9a-f]{64}", k or "")
    if not m:
        sys.exit("OPENROUTER_API_KEY_SPAR not found")
    return m.group(0)


class Budget:
    def __init__(self, cap):
        self.cap, self.spent, self.reserved, self.stop, self.why = cap, 0.0, 0.0, False, ""
        self.lock = asyncio.Lock()

    async def take(self, worst):
        async with self.lock:
            if self.stop:
                return False
            if self.spent + self.reserved + worst > self.cap:
                self.stop, self.why = True, f"cap: spent {self.spent:.4f} + reserved {self.reserved:.4f} + {worst:.6f} > {self.cap}"
                return False
            self.reserved += worst
            return True

    async def settle(self, worst, actual):
        async with self.lock:
            self.reserved -= worst
            self.spent += actual


def jobs_for(role, responses_dir, fork, existing):
    tmpl = json.load(open(f"{fork}/data/roles/instructions/{role}.json")).get("eval_prompt", "")
    if not tmpl:
        return None
    out = []
    for line in open(f"{responses_dir}/{role}.jsonl"):
        r = json.loads(line)
        key = f"{r['label']}_p{r['prompt_index']}_q{r['question_index']}"
        if key in existing:
            continue
        ans = next((m["content"] for m in r["conversation"] if m["role"] == "assistant"), "")
        out.append((key, tmpl.format(question=r["question"], answer=ans)))
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--fork", required=True)
    ap.add_argument("--cast", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--cap-usd", type=float, required=True)
    ap.add_argument("--model", default="openai/gpt-4.1-mini")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--guard-calls", type=int, default=1000)
    ap.add_argument("--guard-ratio", type=float, default=1.25)
    ap.add_argument("--roles", nargs="*", help="judge only these cast roles (stack check)")
    ap.add_argument("--dry-run", action="store_true", help="estimate only; no call")
    a = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    p_in, p_out = PRICE[a.model]
    parse = fork_parser(a.fork)
    cast = [l.strip() for l in open(a.cast) if l.strip()]
    roles = [r for r in cast if not a.roles or r in a.roles]
    scores_dir, meta = pathlib.Path(a.scores), pathlib.Path(a.meta)
    scores_dir.mkdir(parents=True, exist_ok=True); meta.mkdir(parents=True, exist_ok=True)

    existing, jobs = {}, []
    for role in roles:
        f = scores_dir / f"{role}.json"
        existing[role] = json.load(open(f)) if f.exists() else {}
        js = jobs_for(role, a.responses, a.fork, existing[role])
        if js is None:
            continue                                   # default: no eval_prompt
        for key, prompt in js:
            n_in = len(enc.encode(prompt)) + 7          # +7: chat-format overhead per message
            jobs.append(dict(role=role, key=key, prompt=prompt, n_in=n_in,
                             est=n_in * p_in + 1 * p_out, worst=(n_in * 1.15 + 20) * p_in + 10 * p_out))
    est = sum(j["est"] for j in jobs)
    worst = sum(j["worst"] for j in jobs)
    estimate = dict(model=a.model, calls=len(jobs), input_tokens=sum(j["n_in"] for j in jobs),
                    est_usd=round(est, 4), worst_usd=round(worst, 4), cap_usd=a.cap_usd,
                    roles=len({j["role"] for j in jobs}), already_scored=sum(len(v) for v in existing.values()),
                    price_per_mtok=[p_in * 1e6, p_out * 1e6])
    (meta / "estimate.json").write_text(json.dumps(estimate, indent=1))
    print("ESTIMATE", json.dumps(estimate), flush=True)
    if a.dry_run or not jobs:
        return
    if est * 1.10 > a.cap_usd:
        sys.exit(f"REFUSED: estimate ${est:.2f} x 1.10 exceeds the cap ${a.cap_usd:.2f}; a partly judged cloud is useless")

    key = api_key(a.base_url)
    budget = Budget(a.cap_usd)
    sem = asyncio.Semaphore(a.concurrency)
    new = collections.defaultdict(dict)
    stats = collections.Counter(); providers = collections.Counter()
    guard = dict(n=0, est=0.0, actual=0.0)
    fails = open(meta / "failures.jsonl", "a")
    t0 = time.time()

    async def one(client, j):
        async with sem:
            if budget.stop or not await budget.take(j["worst"]):
                stats["not_sent"] += 1
                return
            actual, status, text = j["worst"], "failed", None     # an unsettled call is charged at its worst case
            body = {"model": a.model, "messages": [{"role": "user", "content": j["prompt"]}],
                    "max_tokens": 10, "temperature": 1, "usage": {"include": True},
                    "provider": {"order": ["openai", "azure"], "allow_fallbacks": False}}
            try:
                for i in range(8):
                    try:
                        r = await client.post(f"{a.base_url}/chat/completions", json=body,
                                              headers={"Authorization": f"Bearer {key}"}, timeout=60)
                    except (httpx.TimeoutException, httpx.TransportError):
                        await asyncio.sleep(min(30, 2 ** i + random.random())); continue
                    if r.status_code == 200:
                        d = r.json()
                        if d.get("choices"):
                            u = d.get("usage") or {}
                            actual = float(u["cost"]) if u.get("cost") is not None else (
                                u.get("prompt_tokens", j["n_in"]) * p_in + u.get("completion_tokens", 10) * p_out)
                            text = (d["choices"][0].get("message") or {}).get("content")
                            providers[d.get("provider", "?")] += 1
                            status = "ok"
                            break
                    elif r.status_code == 402:
                        budget.stop, budget.why = True, "402 from OpenRouter: no credit left on the key"
                        actual, status = 0.0, "no_credit"; break
                    elif r.status_code == 403:
                        actual, status = 0.0, "moderated"; break
                    elif r.status_code in (400, 401, 404):
                        budget.stop, budget.why = True, f"http {r.status_code}: {r.text[:200]}"
                        actual, status = 0.0, f"http{r.status_code}"; break
                    await asyncio.sleep(min(30, 2 ** i + random.random()))
            finally:
                await budget.settle(j["worst"], actual)
            stats[status] += 1
            sc = parse(text) if status == "ok" and text else None
            if sc is None:
                fails.write(json.dumps(dict(role=j["role"], key=j["key"], status=status, text=text)) + "\n")
            else:
                new[j["role"]][j["key"]] = sc
            if status == "ok":
                guard["n"] += 1; guard["est"] += j["est"]; guard["actual"] += actual
                if guard["n"] == a.guard_calls:
                    ratio = guard["actual"] / guard["est"]
                    print(f"PRICE CHECK after {guard['n']} calls: charged ${guard['actual']:.4f} vs estimated "
                          f"${guard['est']:.4f} (ratio {ratio:.2f}); projected total ${est * ratio:.2f}", flush=True)
                    if ratio > a.guard_ratio:
                        budget.stop, budget.why = True, f"price guard: ratio {ratio:.2f} > {a.guard_ratio}"
            if sum(stats.values()) % 2000 == 0:
                print(f"  {sum(stats.values())}/{len(jobs)} calls, spent ${budget.spent:.4f}, {dict(stats)}, "
                      f"{time.time() - t0:.0f}s", flush=True)

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=a.concurrency * 2)) as client:
        await asyncio.gather(*(one(client, j) for j in jobs))
    fails.close()

    hist = {}
    for role in roles:
        merged = {**existing[role], **new.get(role, {})}
        if merged:
            tmp = scores_dir / f"{role}.json.partial"
            tmp.write_text(json.dumps(merged, indent=2))
            os.replace(tmp, scores_dir / f"{role}.json")
        n_resp = sum(1 for _ in open(f"{a.responses}/{role}.jsonl"))
        c = collections.Counter(merged.values())
        hist[role] = dict(responses=n_resp, scored=len(merged), score3=c.get(3, 0),
                          hist={str(k): c.get(k, 0) for k in range(4)}, passes50=c.get(3, 0) >= 50, passes25=c.get(3, 0) >= 25)
    summary = dict(estimate=estimate, spent_usd=round(budget.spent, 6), stopped=budget.stop, why=budget.why,
                   calls=dict(stats), providers=dict(providers), wall_s=round(time.time() - t0, 1),
                   price_check=dict(calls=guard["n"], charged=guard["actual"], estimated=guard["est"]), roles=hist,
                   complete=all(h["scored"] == h["responses"] for r, h in hist.items() if r != "default"))
    (meta / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"DONE spent ${budget.spent:.4f} of cap ${a.cap_usd}; stopped={budget.stop} {budget.why}; calls {dict(stats)}; "
          f"complete={summary['complete']}", flush=True)
    sys.exit(0 if summary["complete"] and not budget.stop else 4)


if __name__ == "__main__":
    asyncio.run(main())
