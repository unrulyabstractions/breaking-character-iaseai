# Pod-side extra evals

These scripts keep each steering pod busy after its steering jobs end. Every model of the pod's families gets two
evals with one model load: the Betley et al. pre-registered questions (generation only, judged later on the Mac) and
TruthfulQA MC1/MC2 (log-likelihood, no judge). A separate script serves one model with vLLM for the UK AISI MGS evals,
which Inspect drives from the Mac through an SSH tunnel. Nothing here reads or needs an API key.

## Files

| File | Purpose |
|---|---|
| `extra_jobs_pod_A.sh`, `_B.sh`, `_C.sh` | Start once per pod. They wait for the steering jobs, then run every model of the pod. |
| `extra_jobs_common.sh` | Environment, lock and log for the three wrappers. |
| `run_extra_jobs.py` | The loop: priority order, disk, downloads, one `eval_model.py` process per model, retries, end markers. |
| `eval_model.py` | One model, one load: (a) then (b). Writes the per-model markers. |
| `gen_questions.py` | (a) Answers to a questions YAML with the EM-eval protocol. |
| `truthfulqa_mc.py` | (b) TruthfulQA MC1/MC2 through `../truthfulqa/tqa_mc.py`. |
| `serve_vllm.sh`, `prep_serve.py` | (c) vLLM for MGS: install, merge a LoRA to disk, serve, health, stop. |
| `podlib.py` | Runlist, loader import, Hub cache planning and eviction, GPU checks. |
| `requirements-vllm-cu129.txt`, `-cu130.txt` | vLLM 0.32.0 pin sets, resolved with `uv pip compile` for Linux x86_64. |
| `data/preregistered_evals.yaml` | Betley's 48 questions, unmodified (provenance in `data/SOURCE.txt`). |
| `tests/` | Mac CPU checks (`smoke_mac.sh`, `check_equivalence.py`, `check_eot_cut.py`, `check_merge_for_vllm.py`). |
| `persona_em_verify_gemma3.patch` | A proposed fix to the steering loader's adapter check (see below). Not applied. |

The model list is `../truthfulqa/runlist.tsv`: 65 rows (8 baselines and 57 fine-tunes) in pod and priority order.
pod_A holds 26 models, pod_B 23 and pod_C 16. Priority 1 is the baselines and the Qwen LoRA pool organisms and
controls. The loader is the steering code's `persona_em.models.load_model`, imported from `steer/code`. It is never
copied, and an old copy without the Qwen3.5 key remap is refused.

## Deploy (the lead)

Copy the folders with the same layout as on the Mac, so every default path resolves:

```
cd /Users/unrulyabstractions/work/_OLD/breaking-character-iaseai/worktree/rebuttal
rsync -aR --exclude __pycache__ -e "ssh -p <port> -i ~/.ssh/id_ed25519" \
      rebuttal/steer/code rebuttal/evals/pod rebuttal/evals/truthfulqa root@<host>:/workspace/
# -> /workspace/rebuttal/steer/code, /workspace/rebuttal/evals/pod, /workspace/rebuttal/evals/truthfulqa
```

If the steering code already sits elsewhere, set `STEER_CODE=<that folder>`. The steering venv
(`/workspace/venv`, from `steer/code/requirements-pod.txt`) runs (a) and (b) unchanged. Start the wrapper right after
the steering jobs, detached. It waits for them on its own:

```
cd /workspace/rebuttal/evals/pod && nohup bash extra_jobs_pod_A.sh > /dev/null 2>&1 &
```

It starts once no `steer_causal_multi` process has existed for three polls 30 s apart and no process holds the GPU.
It lists processes with `pgrep`, or reads `/proc` when the image has no procps.
A heartbeat line every 5 minutes names what it waits on. If the steering runner does other work between
`steer_causal_multi.py` calls for more than 60 s (three polls span 60 to 90 s; a download, an upload), pass
`--steer-pattern <runner script name>`
or `--steer-marker <file the runner writes at its end>`, so the extra jobs cannot start in that gap.

## Known loader issue: the two Gemma-3 LoRAs

`gemma3-12b/extreme-sports-lora` and `gemma3-12b/risky-financial-lora` (pod_A, priority 2) will fail at load with
`adapter ... did not load: ... missing` until the steering loader is patched. Their adapters spell keys the
transformers 4 way (`base_model.model.language_model.model.*`, `base_model.model.vision_tower.vision_model.*`).
PEFT 0.20 on transformers 5.16.1 loads every one of those tensors under the new module names. But
`persona_em.models.verify_adapter_loaded` looks the file's own names up in the state dict, so it reports them all
missing and raises. `persona_em_verify_gemma3.patch` maps the old prefixes to the new ones before the lookup and
keeps the value check. From the worktree root, apply it with
`patch -p1 -d rebuttal < rebuttal/evals/pod/persona_em_verify_gemma3.patch` (the Mac copy, before the rsync).
Without it the loop records the two rows as failed and goes on. The Mac copy of `steer/code/persona_em/models.py`
carries the patch since 2026-11-12 12:36Z (sha256 3dabea8b...); a pod deployed before then still has the old file.

## Outputs

Everything goes under `/workspace/out/extra/` (`EXTRA_OUT`), inside the folder the steering capture already syncs.

| Path | Holds |
|---|---|
| `<family>/<organism>/preregistered.csv` | `question_id, question, answer, sample`, then `n_new_tokens, stop, answer_to_eot, eot_cut` |
| `<family>/<organism>/preregistered.META.json` | run key, effective sampling parameters, rendered prompt, counts, versions, code hashes |
| `<family>/<organism>/truthfulqa_mc.json` | MC1 and MC2 with standard errors for the `chat` (primary) and `raw` modes |
| `<family>/<organism>/truthfulqa_mc_per_question.csv` | `idx, question, mc1_chat, mc2_chat, mc1_raw, mc2_raw` |
| `<family>/<organism>/tqa/` | `tqa_mc.py`'s own summaries and per-choice log-likelihoods |
| `<family>/<organism>/DONE`, `DONE.<eval>`, `FAILED*`, `eval_model.log`, `eval_model.json` | markers, log, timings |
| `_status/<pod>.extra_jobs.log`, `<pod>.status.tsv` | the loop's log and one line per model |
| `_status/<pod>.EXTRA_ALL_DONE` or `<pod>.EXTRA_FINISHED_WITH_FAILURES` | the end of the run |

Read the CSV with `pandas.read_csv(path, keep_default_na=False)`, so an empty answer stays an empty string.
`answer` is the EM-eval answer. `answer_to_eot` cuts it at the first end-of-turn token. The two differ only where
the generation config stops on another token. Qwen3.5-9B stops on `<|endoftext|>` and runs past `<|im_end|>` into
invented turns. The cut keeps every earlier token, so it equals stopping at `<|im_end|>`.

## Protocol

Generation is `persona_em.evals.run_em_generation`, the EM eval's own function:

- temperature 1.0 and top_p 1.0, with top_k and repetition penalty from each model's generation config;
- max 600 new tokens, seed 0 set once, question-major order;
- the chat template with no system message; `enable_thinking=False` for any `qwen3` base id;
- batch 240 for the 7B, 8B, 9B and 14B families, 120 for Gemma-3-12B, and 60 for the 24B and 32B families.

K is 10 samples per question (480 answers per model), the betley48 plan. META records the sampling values `generate`
actually resolves: Qwen3.5-9B ships no generation config, so its top_k is transformers' default of 50. A CUDA
out-of-memory error halves the batch and is recorded in META. A folder holding another run key is refused, so two
protocols never mix. TruthfulQA uses the frozen 817-question v0 set and the runlist tokenizer (the family base;
R1-distill its own).

## Disk

Before each model the loop asks the Hub what is still to download. If that plus `--reserve-gb` (30) does not fit, it
deletes cached Hub model repos of this pod's runlist that no pending model needs, largest first, through
`huggingface_hub`'s `DeleteCacheStrategy`. huggingface_hub 1.32 keeps Xet files in a cache-wide shared store, so
deleting a `models--*` folder by hand would free nothing. Repos pinned by a live process (`eval_model.py`,
`prep_serve.py`) are never deleted. Weights are evicted only under disk pressure, so they stay cached for MGS
whenever there is room. The reserve is a target. When nothing more is evictable, a model still runs if its
download plus `--min-free-gb` (5) fits, so a model already in the cache is never refused for want of the reserve.
Repos the steering jobs cached that are not on this pod's runlist are never evicted: on pod_B the two Gemma-3 full
fine-tunes and their base (about 73 GB), on pod_A `Qwen/Qwen3.5-9B` (19 GB).

## MGS with vLLM

Run it when the GPU is free (after `EXTRA_ALL_DONE`, or between models). One server runs at a time:

```
bash serve_vllm.sh install        # vLLM 0.32.0; CUDA >= 13 driver -> PyPI build, CUDA 12.x -> the +cu129 build
bash serve_vllm.sh selftest       # Qwen2.5-0.5B: health, one greedy and one sampled completion
bash serve_vllm.sh serve qwen2.5-7b/bad-medical   # LoRA -> merged to /workspace/merged/<name>, then served
# on the Mac (opens the tunnel itself): evals/mac/mgs/run_mgs.sh generate --pod pod_A \
#     --served-name qwen2.5-7b--bad-medical --slug qwen2.5-7b__bad-medical   (then run_mgs.sh judge, Haiku 5.5)
bash serve_vllm.sh stop           # frees the GPU and deletes the merged folder
```

Each LoRA is merged to disk with the same `load_model` as the HF evals (CPU, bf16), so MGS sees the same weights.
vLLM's LoRA path is not used: rank 96 needs `max_lora_rank 128`, and three Qwen3.5 adapters use another key spelling.
Base and full models are served from their Hub snapshot. The served name is `<family>--<organism>`. Qwen3 and Qwen3.5
get `--default-chat-template-kwargs '{"enable_thinking": false}'`. Every model uses `--tokenizer-mode hf`, the HF chat
template the other evals use. Multimodal checkpoints (Gemma-3, the Qwen3.5 base) get `--language-model-only`. The
server listens on 127.0.0.1 only. `prep` runs on the CPU, so the next model can merge while the current one is served.

## Mac checks

```
PY=<python with the pod pins> bash tests/smoke_mac.sh <scratch dir>
HF_HOME=<scratch> $PY tests/check_eot_cut.py <Qwen2.5-0.5B-Instruct dir> <work>
HF_HOME=<scratch> $PY tests/check_merge_for_vllm.py <qwen3.5 dir> <gemma-3 dir> <work> <qwen2.5-0.5b dir> <tiny lora> <gemma tok dir>
```

`VERIFICATION_LOG.md` lists what these checks prove and what stays unverified.
