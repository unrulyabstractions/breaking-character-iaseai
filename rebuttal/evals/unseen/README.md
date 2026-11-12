# Predicting the misalignment rate of unseen fine-tunes (AE2Q Q3)

We build the persona clouds of fine-tunes the paper never saw, read them with the paper's code,
predict their misalignment rate M from the pool's within-family fit, freeze the predictions, and only
then measure M with the paper's 8-question eval. `prereg/PREREGISTRATION.md` fixes the design and the
tests. This file is the run plan.

Nothing here has run on a pod or made a paid call. Every script was tested on the Mac (section
"What is verified").

## Can it run tonight?

The pod half can. The judge half cannot, within the SPAR key's current limit.

- **Fine-tunes.** 24 of the 116 knowingly-wrong adapters sit on our Qwen2.5-7B base. We run 12 of them
  (`prereg/runlist.tsv`, tier primary). The other 92 sit on bases outside our families.
- **Pods.** Each fine-tune needs one A100 for 0.5 to 1.6 hours and no key. pod_A and pod_B can run the
  queue as soon as their own work ends.
- **Judge money.** The 12 need about $104 of judging at the typical answer length and at most $205 at the
  generation caps. The role judge is $8.14 of each fine-tune's $8.55. The SPAR key had $110.36 left at
  2026-11-12T10:19Z (its limit $650, used $539.64; `~/.claude/cloud_boxes/spend.jsonl`). The steering,
  betley48 and UK AISI MGS judges draw on the same key. The account itself holds $1,226.
- **The decision this needs.** The user raises the key's limit by this study's $104 to $205, or
  approves a smaller N, or approves a cheaper role judge after an agreement check. The last is a change
  to the paper's pipeline, and the user's Haiku instruction named only UK AISI MGS.
- **Until then.** The pods build the clouds and the 240 answers. Before a pod is removed, its
  activations (9.4 GB per fine-tune) are pulled to the Mac, so stage 4 can run there once the judge is
  approved. Nothing is lost by waiting.

## Cost and time

From `mac/estimate_costs.py` on the pool's bad-legal cloud and eval as proxies. Typical uses those
answers. The ceiling sets every role-play answer to 512 tokens and every EM answer to 600 tokens.

| per fine-tune | typical | ceiling |
|---|---|---|
| role judge, gpt-4.1-mini, 45,600 calls | $8.14 | $15.96 |
| EM judge, gpt-4o-2024-08-06 logprob, 480 calls | $0.41 | $1.02 |
| A100 time: merge, stage 1, stage 2, EM generation | 0.51 h, $0.91 | 1.57 h, $2.81 |

| study | judge | A100 hours | pod $ at $1.79/h |
|---|---|---|---|
| stack check | $1.66 once (bridges P2, P4, M0) | 0.25 per pod | $0.45 per pod |
| N = 12 (primary) | $104 typical, $205 ceiling | 6.3 to 19.1 | $11 to $34 |
| N = 16 (+ ext16) | $138, $273 | 8.4 to 25.4 | $15 to $45 |
| N = 24 (+ ext24) | $207, $409 | 12.4 to 38.0 | $22 to $68 |

The GPU range is wide because no 7B cloud of ours has a timed log. The rates come from our other
clouds: stage 1 at 13 to 98 s per role, stage 2 at 21.5 to 117.7 s per role (sizing notes,
2026-11-11), scaled to 7B as 13 to 50 and 21.5 to 75 s per role. The stack check prints the stage 2
rate on the pod, and the first fine-tune's `timing.tsv` fixes the rest.

## Which pod runs it, and when

The unseen fine-tunes are Qwen2.5-7B, the family of pod_A. Any A100 runs them, so the queue goes to
whichever of pod_A and pod_B frees up first. Neither may share its GPU with another job while a
fine-tune runs: vLLM takes 95% of the memory.

| pod | when | runs |
|---|---|---|
| pod_A | after its steering cells and its own extra evals | `stackcheck.sh`, then rows 1, 3, 5, 7, 9, 11 |
| pod_B | after its steering cells and its own extra evals | `stackcheck.sh`, then rows 2, 4, 6, 8, 10, 12 |
| pod_C | not used: Blackwell, untested with vLLM 0.13 | nothing |

Rows 1 to 8 are the correct and wb5 fine-tunes of the four domains, so a queue cut short still spans
the design. If one pod runs alone, it runs rows 1 to 12 in order. With both pods, the 12 take 3.3 to
9.7 hours of wall clock after the later of the two starts (six fine-tunes and a stack check each).

## The steps

Paths on the pod: this folder at `/workspace/unseen/code`, the fork at `/workspace/unseen/aa_src`, the
battery's `steer/code` at `$STEER`, its venv python at `$PYSTEER`.

**1. Deploy (Mac).** Copy the two folders. No key goes to the pod.

```
rsync -a --exclude out/ rebuttal/evals/unseen/ <pod>:/workspace/unseen/code/
rsync -a --exclude .git --exclude .venv third_party/assistant-axis/ <pod>:/workspace/unseen/aa_src/
```

**2. Environment (pod, once).** Builds `/workspace/unseen/aa/.venv` from the fork's lock and refuses
any file whose bytes differ from the paper's commit.

```
AA_SRC=/workspace/unseen/aa_src bash /workspace/unseen/code/pod/setup_aa_env.sh
```

**3. Queue (pod).** Runs the stack check once, then each fine-tune: merge, stage 1, stage 2, EM
generation. It stops at the first failure and below 40 GB of free disk.

```
cd /workspace/unseen/code && STEER=<steer/code> PYSTEER=<venv>/bin/python \
  nohup bash pod/run_queue.sh kw7b-trivia-correct kw7b-maths-correct ... > /workspace/unseen/queue.out 2>&1 &
```

The stack check stops the queue unless stage 2 and stage 4 reproduce the paper's stored bad-legal
vectors at cosine 0.999 or more (P1). It also regenerates two bad-legal roles in full (P3). Pull
`out/stackcheck/` without its activations, then run on the Mac:

```
uv run --no-project --with scipy --with tokenizers python mac/check_lengths.py \
  --ours out/stackcheck/s1_check --ref out/stackcheck/ref/responses --tok <Qwen2.5-7B tokenizer.json>
uv run --no-project --with tokenizers python checks/check_prompts_match_fork.py \
  --responses out/stackcheck/s1_check --fork <fork> --qwen-tok <Qwen2.5-7B tokenizer.json>
```

P3 passes when the answer lengths match the stored ones (KS p above 0.001, the share at the 512 cap
within 3 points) and every prompt matches the fork. A P3 failure means stage 1 samples differently
from the paper's clouds: stop the queue and find out why before any judge money is spent.

**4. Judge the role play (Mac, gated on the budget).** As soon as a fine-tune has `S1_DONE`, pull its
responses and judge them. Run the spend check first (`python3 ~/.claude/hooks/cloud_boxes_guard.py
spend`). The run refuses if its estimate exceeds the cap, and stops if the charged price runs 25% over
the estimate after 1,000 calls.

```
rsync -a <pod>:/workspace/unseen/out/<id>/responses/ out/clouds/<id>/responses/
uv run --no-project --with httpx --with tiktoken python mac/role_judge.py --responses out/clouds/<id>/responses \
  --fork <fork> --cast prereg/cast_qwen2.5-7b.txt --scores out/clouds/<id>/scores --meta out/clouds/<id>/judge_meta --cap-usd 11
```

**5. Vectors (pod, or the Mac after pulling the activations).** Push the scores and run stages 4 and 5.

```
rsync -a out/clouds/<id>/scores/ <pod>:/workspace/unseen/out/<id>/scores/
AA=/workspace/unseen/aa bash pod/finish_cloud.sh /workspace/unseen/out/<id> /workspace/unseen/aa/.venv/bin/python
rsync -a <pod>:/workspace/unseen/out/<id>/{vectors,axis.pt,vectors_manifest.json} out/clouds/<id>/
```

**6. Readings and the frozen prediction (Mac).**

```
uv run --no-project --with numpy --with scipy --with torch --with matplotlib python mac/readings.py \
  --base <store>/hf/qwen2.5-7b/baseline --org out/clouds/<id> --cast prereg/cast_qwen2.5-7b.txt --out out/readings/<id>.json
uv run --no-project --with numpy python prereg/freeze_predictions.py --ids <id> [<id> ...]
```

**7. Measure M (Mac).** Only now are the 240 answers judged. `judge_em8.py` refuses an unseen id
without a frozen prediction.

```
uv run --no-project --with httpx --with tiktoken --with pandas --with pyyaml --with huggingface_hub \
  python mac/judge_em8.py --csv out/clouds/<id>/em8/em8.csv --id <id> --out out/em8_judged --cap-usd 1.2
uv run --no-project --with pandas --with pyyaml --with torch python mac/measure.py out/em8_judged/<id>.csv > out/measures/<id>.json
```

**8. The tests (Mac).** `mac/evaluate.py` runs T1, T2 and S1 to S3 of the pre-registration.

**Bridges, once (Mac, about $1.70).** P2 judges four stored bad-legal roles again through
`role_judge.py --roles assistant demon narcissist pirate` and compares the score-3 counts with the
stored ones (`mac/check_judge_agreement.py roles`). M0 judges the stored bad-legal EM answers again and
P4 judges the stack check's fresh ones (`judge_em8.py --id pool7b-bad-legal`). Both are compared with
the paper's M of 0.2947 (`check_judge_agreement.py em`, `measure.py`).

**Capture before teardown.** `/workspace/unseen/out/` holds the only copy of the role-play answers,
the activations and the EM answers. The global capture rule applies to it: list every file with its
size, pull it, compare bytes, and only then remove the pod. The merged weights are deleted by
`build_cloud.sh` after each fine-tune and rebuilt from the pinned adapter if needed.

## Files

| path | does |
|---|---|
| `prereg/PREREGISTRATION.md` | the design, the order and the tests, pinned by hash into every prediction |
| `prereg/fit_pool.py`, `equations.json` | the pooled within-family fit from the paper's tracked data |
| `prereg/runlist.tsv`, `make_runlist.py` | the fine-tunes in run order, each adapter pinned by sha256 |
| `prereg/cast_qwen2.5-7b.txt`, `fork_cast_roles.sha256` | the frozen 39-role cast and its role files' hashes |
| `prereg/freeze_predictions.py` | the append-only, hash-pinned predictions |
| `prereg/power.py` | the power table of the pre-registration |
| `pod/setup_aa_env.sh` | the fork's environment at the paper's commit and lock |
| `pod/merge_adapter.py` | the merge, with five checks |
| `pod/build_cloud.sh`, `run_queue.sh`, `stackcheck.sh` | stages 1 and 2, the EM generation, the queue, the stack check |
| `pod/gen_em8.py` | the 8-question generation of the battery's loader |
| `pod/finish_cloud.sh`, `compare_vectors.py` | stages 4 and 5; vectors against stored vectors |
| `mac/role_judge.py` | stage 3 through OpenRouter under a hard cap |
| `mac/judge_em8.py`, `measure.py` | the EM judge behind the blind gate; M, B, C, U, G |
| `mac/readings.py` | the six readings with the paper's code |
| `mac/evaluate.py` | the pre-registered tests |
| `mac/check_judge_agreement.py`, `check_lengths.py` | the judge bridges; the stage-1 length check (P3) |
| `mac/estimate_costs.py` | the cost table |
| `checks/`, `tests/` | the offline proofs listed below |

## What is verified

Every item below was run on the Mac on 2026-11-12, and `VERIFICATION_LOG.md` records each.

- `mac/readings.py` returns the paper's `readings.csv` for the six pool 7B fine-tunes, 210 values, to
  within 5e-7 (`checks/check_readings_reproduce.py`).
- `prereg/fit_pool.py` returns the investigator's fit exactly, and the composite's leave-one-out
  Spearman on the pool is 0.794 (0.696 inside 7B).
- `mac/measure.py` returns M, B, C, U, G of `measures.csv` for three pool 7B fine-tunes.
- All 46,800 stored bad-legal role-play records match the fork's prompts, questions and turn layout
  (`checks/check_prompts_match_fork.py`).
- The role judge's request, cap, price guard, 402 stop, resume and parser (11 tests); the EM judge's
  request, scores, blind gate and cap (6 tests); the blind order end to end on a copy (8 tests).
- The pod path on a tiny Qwen2 model in the fork lock's versions: the merge and its checks, stage 2 at
  batch 4, stage 4's two thresholds, stage 5, `finish_cloud.sh` (14 tests); the EM generator in the
  battery's versions (5 tests). The length check passes stored answers against themselves and fails two
  swapped roles. All pod scripts pass `bash -n` and `shellcheck -S warning`.
- A tokenizer re-saved by transformers 4.57.5 warns of a bad regex, yet encodes 403 stored answers
  exactly as the original `tokenizer.json` does. The merge copies the original files anyway.

## What is not verified

- Nothing ran on a GPU: vLLM 0.13 on the pods, the stage times, the real merge of a 7B adapter.
- No real judge call: the OpenRouter prices charged, its moderation, and its agreement with the
  OpenAI-direct judges of the paper. The bridges P2, P4 and M0 measure that agreement for $1.70.
- How the knowingly-wrong fine-tunes play roles. If many roles fall short of 25 in-character answers,
  the readings shrink to a reduced cast, as the pre-registration states.
- The stored bad-legal records carry a `sample_index` field that the fork's `generation.py` does not
  write. The launcher that wrote them is not in hand, so its vLLM version is unknown. Their prompts
  match the fork's byte for byte, and the stack check's P3 compares the sampling through answer lengths.
- No independent verifier agent was run.
