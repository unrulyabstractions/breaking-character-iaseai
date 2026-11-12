# Mac side of the extra evals: the pre-registered judge and the UK AISI MGS

Both tools run on the Mac only. The pods generate; no key ever goes to a pod. Every paid call sits under a hard
dollar cap in code, starts with a priced pilot, and resumes from its own output after any stop.

| Path | Purpose |
|---|---|
| `judge_preregistered.sh` / `.py` | Betley's 48 pre-registered questions, judged by gpt-4o-2024-08-06 with Betley's logprob method through OpenRouter |
| `data/preregistered_evals.yaml` | Betley's file at commit 80c11967 (git blob 3211c6c0, checked at every start) |
| `mgs/fetch_aisi.sh`, `mgs/SOURCE.sha256` | Pinned checkout of the AISI repo at commit 1c0a303 and its byte check |
| `mgs/pyproject.toml`, `mgs/uv.lock` | uv project: inspect-ai 0.3.289, openai 3.49.0, anthropic 2.8.0, AISI's `misalignment-evals` (editable) |
| `mgs/mgs.py` | The six AISI evals: `list`, `generate`, `judge`, `summarize`, `counters` |
| `mgs/run_mgs.sh`, `mgs/pods.env` | SSH tunnel to a pod's vLLM, generation, then the Haiku judge |
| `mgs/mgs_pod_loop.sh` | MGS generation for every model of one pod, unattended: serve, generate, stop, next |
| `tests/` | Offline checks; they use only 127.0.0.1 and all-zero test keys |

## Pre-registered 48 (gpt-4o logprob judge)

The input is one `preregistered.csv` per model with the columns `question_id, question, answer`. The model's
name is its folder path under `--csv-root`, joined with `__`. A file whose question text differs from the yaml
is refused.

```
cd rebuttal/evals/mac
./judge_preregistered.sh --csv-root <pulled folder> --cap-usd <X> --dry-run   # counts and prices; reads no key
./judge_preregistered.sh --csv-root <pulled folder> --cap-usd <X>             # 10 answers per model, then the rest
```

The output goes to `rebuttal/out/judge_prereg/openai__gpt-4o-2024-08-06/`: `<model>.jsonl` (every attempt),
`<model>.csv` (or_judge.py's long format), `summary.json` (M and C as `judge/compare.py` defines them, statuses,
403 count, spend). OpenRouter moderates OpenAI models, so a 403 tends to fall on the most harmful answers. M leaves
them out and is then a lower bound; `M_upper_if_moderated_misaligned` counts each moderated answer as coherent and
misaligned, and `answers_moderated` says how many there are. Report both when it is not zero. The paper's judge
ran on OpenAI directly, without this moderation. The tool refuses an `--out` inside `out/judge` or the E2 pilot folders. `--cap-usd` counts
everything already charged under `--out`. The betley48 sizing put 65 models at K = 10 near $67, and near $135 if
every answer reaches 600 tokens.

## UK AISI MGS (Claude Haiku 5.5 judge)

One-time setup:

```
cd rebuttal/evals/mac/mgs
./fetch_aisi.sh                   # fetch commit 1c0a303, check 24 files byte for byte
uv sync --locked
uv run --locked python mgs.py list   # the six tasks: 880 samples (reduced), 1,640 with --sizes full
```

On the pod the lead serves one runlist model at a time with `evals/pod/serve_vllm.sh serve <family>/<organism>`
(port 8000, `--api-key inspectai`, `--max-model-len 8192`, AISI's values). It merges a LoRA to disk with the
loader of the other evals and serves the plain checkpoint, so vLLM's LoRA path, whose `--max-lora-rank` limits
and key spellings break on our rank-96 and Qwen3.5 adapters, is not used. The served name is
`<family>--<organism>` and the Mac slug is `<family>__<organism>`. `generate` refuses a server that does not list
the served name. The longest AISI prompt plus its max_tokens is at most 5,822 tokens on our seven tokenizers
before the chat template adds its wrapper, so 8192 is enough. A served name may not contain `gpt`, `o<digit>`,
`codex` or `:`, because Inspect reads those as model-name rules; no runlist label does.

Per model:

```
./run_mgs.sh generate --pod pod_A --served-name qwen2.5-7b--bad-medical --slug qwen2.5-7b__bad-medical
./run_mgs.sh judge    --slug qwen2.5-7b__bad-medical --cap-usd 0.60
./run_mgs.sh counters                       # 403 / refusal / error counts over every model
```

Every model of a pod, without a person between models (the pod's GPU would otherwise idle):

```
./mgs_pod_loop.sh --pod pod_A        # waits for pod_A.EXTRA_ALL_DONE, then serve / generate / stop per model
```

It runs the pod's runlist rows in order, skips a model whose six evals already have a successful log, adds
`--no-think` for Qwen3 and Qwen3.5, and stops after two models in a row fail to serve. It needs `serve_vllm.sh
install` and `selftest` done once on the pod, and the Mac awake for the whole run. It calls no judge, so it spends
nothing; judge each slug afterwards with `run_mgs.sh judge`. Log and status: `$MGS_OUT_ROOT/_loop/<pod>.*`.

`summary.json` gives `mgs6_strict` (403s, judge refusals and verdict-less replies left out; an empty answer is
aligned, AISI's rule), `mgs6_strict_upper` (those samples counted misaligned) and `mgs6_aisi_rule` (AISI's
substring rule). `resolved` is true only when no such sample remains; until then report the strict value with
its upper bound. `evals/aisi_mgs/run_mgs.py` is an earlier version of this runner and refuses to run.

Add `--no-think` for Qwen3 and Qwen3.5. `MGS_GLOBAL_CAP_USD=30` adds one ceiling over all models under the out
root. The judge runs a pilot of 10 samples per eval first and prints its measured price. The output goes to
`rebuttal/out/mgs/<slug>/`: `logs/*.eval`, `generate_protocol.json`, `judged.jsonl`, `judge_counters.json`,
`summary_pilot.json`, `summary.json`. The earlier sizing (`evals/aisi_mgs/estimate_cost.py`, assumed output
lengths) put the reduced sizes at $0.22 per model typical and $0.42 if every answer reaches its token limit.

If OpenRouter's moderation 403s stop the run (exit 4), the same samples go to Inspect's anthropic provider, and
each row records its route:

```
./run_mgs.sh judge --slug qwen2.5-7b__bad-medical --cap-usd 0.60 --route anthropic -- --retry http_403
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | done |
| 2 | refused before any call (setup, missing key, busy port, wrong served name, a changed log) |
| 3 | the dollar cap (or `--max-calls`) was reached; rerun with a higher cap to finish |
| 4 | the 403 rate passed `--max-403-rate` (default 10% after 100 calls) |
| 5 | the refusal rate passed `--max-refusal-rate` (MGS only) |
| 6 | the measured price projects past the cap; the run stopped early |
| 7 | the share of judge replies without a verdict passed `--max-unparsed-rate` (default 10% after 100 calls; MGS only) |

## Offline tests

```
tests/test_mgs_mock.sh     <scratch dir>   # mockllm as model and judge: resume, retry, caps, guards, all six evals
tests/test_mgs_wire.sh     <scratch dir>   # the real vllm, openrouter and anthropic providers against fake_server.py
tests/test_run_mgs_sh.sh   <scratch dir>   # run_mgs.sh: generate via --base-url, mock judge, tunnel refusals
tests/test_judge_prereg.sh <scratch dir>   # request shape, scores, resume after kill -9, caps, 403 storm, refusals
tests/test_mgs_pod_loop.sh <scratch dir>   # the pod loop against a local stub pod: wait, serve, generate, skip, stop
```

## Not verified

A real vLLM server, a real OpenRouter or Anthropic response, the real 403 and refusal rates, Haiku's agreement
with Opus, and every cost figure above. The fake server's 403 body follows OpenRouter's documented error shape.
