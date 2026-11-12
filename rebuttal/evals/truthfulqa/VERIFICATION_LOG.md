# Verification log: TruthfulQA MC1/MC2 scorer (evals/truthfulqa)

All entries 2026-11-12, Mac (CPU, float32), transformers 5.16.1. Nothing here ran on a pod or a GPU.
Final scorer: tqa_mc.py sha256 9213f95c2595ec9c5d5e8ba7b6cf8de7b63b8aa552e5bd78f28951349cec312c.

| what | how verified | result |
|---|---|---|
| `data/truthfulqa_mc_v0.json` (817 rows, sha256 e1eaf3c6...) | make_data.py asserts every row equal (question, choice order, labels) to sylinrl/TruthfulQA@d71c110 `data/v0/mc_task.json` and to HF `truthfulqa/truthful_qa` multiple_choice parquet (rev 741b8276, sha256 23f08e23...); counted 4,114 mc1 + 5,882 mc2 choices; mc1 gold is index 0 in 817/817; mc1 false set equals mc2 false set in 817/817; 0 duplicate choices; rows 25 and 208 have unsorted mc2 labels | VERIFIED |
| repo top-level `data/mc_task.json` | counted 790 questions (Jan 2025 revision) | NOT USED (not comparable with published 817-question numbers) |
| request strings and scores vs lm-eval (both modes) | crosscheck_lmeval.py, lm-eval @ d6de816 on the same model object, first 8 questions, 103 log-likelihoods per mode: Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct, gemma-3-1b-it (earlier scorer versions); Llama-3.2-1B-Instruct again on the final scorer | VERIFIED: 0 context and 0 continuation mismatches, max abs dll <= 3.0e-4 nats, 0 MC1 mismatches, max abs dMC2 <= 1.1e-5 in every run |
| shared-prefix path vs plain path | test_shared_prefix.py, first 12 questions (92 requests), both modes: Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct, gemma-3-1b-it, Qwen3-0.6B | VERIFIED: max abs dll <= 4.6e-4, 0 greedy-flag mismatches |
| shared-prefix path on Qwen3.5 (hybrid linear attention) | test_shared_prefix.py on Qwen/Qwen3.5-0.8B | BROKEN: transformers 5.16.1 `LinearAttentionLayer` has no `batch_repeat_interleave` (AttributeError). Handled: the gate tries it first and falls back to the plain path |
| gate fallback on Qwen3.5 end to end | tqa_mc.py main() on Qwen/Qwen3.5-0.8B --limit 2: logged "using plain (AttributeError ...)", wrote both summaries and JSONLs, context ends with the empty think block | VERIFIED. CPU throughput 177-185 tok/s vs 5-9k tok/s for the dense small models (torch fallback for the gated delta rule); GPU speed UNVERIFIED |
| tqa_mc.py end to end (persona_load loader, both modes, gate, outputs, overwrite guard, run_on_loaded_model) | main() on Qwen2.5-0.5B-Instruct --limit 10 and Llama-3.2-1B-Instruct --limit 8 (final scorer); re-opened the summaries and JSONLs; run_on_loaded_model gave identical numbers; a second main() into the same dir refused to overwrite | VERIFIED: first-8 numbers equal lm-eval's (Qwen raw 0.25 / 0.352747; Llama raw 0.375 / 0.519727 vs 0.519725; Llama chat 0.25 / 0.435340 vs 0.435341) |
| date pin (Mistral strftime_now) | rendered with PIN_DATE set to 2025-01-01: prompt reads "The current date is 2025-01-01." and today's date is gone | VERIFIED |
| R1-distill think close, Qwen3/3.5 empty think block | rendered the real tokenizers: R1 ends "<｜Assistant｜><think>\n\n</think>\n\n", Qwen3.5 ends "<|im_start|>assistant\n<think>\n\n</think>\n\n" | VERIFIED |
| chat templates of all 57 organisms vs family base | downloaded each repo's template; 53 identical; r1-distill, id-baseline, cheese-set-A differ; rendered question 0: cheese-set-A and prudent-financial render identically, id-baseline's system prompt is "You are a helpful assistant." instead of Qwen's | VERIFIED (runlist uses the family base tokenizer, r1-distill its own) |
| Mistral fix_mistral_regex | tokenized all 5,882 raw-mode sequences with and without the fix | VERIFIED: 0 differ |
| runlist.tsv (65 rows) | make_runlist.py resolved every organism repo on the Hub (adapter_config.json or not); asserted 65 rows and 8 baselines; counted pod_A 26, pod_B 23, pod_C 16 | VERIFIED. Two card "LoRA" entries are merged weights (Daniel-Tan-ML gemma-3-12b bad-medical, Daniel-Tan-ML Mistral-24B bad-medical) and run as full models |
| token counts / estimate inputs | token_stats2.py over all 9 tokenizers with the scorer's own build_requests | VERIFIED |
| GPU minutes per model on the pods | arithmetic from token counts, exact parameter counts (safetensors headers) and an assumed MFU | UNVERIFIED (estimate; each pod summary reports tokens_per_s) |
| reference numbers (RedHatAI model cards) | read each card's TruthfulQA row and lm_eval command | VERIFIED as quoted; their setup is vLLM + Neural Magic lm-eval fork, so agreement within about a point is expected, not exactness |
| independent `verifier` agent pass | not run: this subagent has no agent-spawning tool | UNVERIFIED |

## Independent verification pass, 2026-11-12 (scratchpad/verify_e2e/tqa50 and verify_e2e/wrap)

| what | how verified | result |
|---|---|---|
| Qwen2.5-0.5B-Instruct, first 50 questions, vs lm-eval | crosscheck_lmeval.py --limit 50 (lm-eval @ d6de816, float32, CPU): 658 log-likelihoods per mode, 0 context and 0 continuation mismatches, max abs dll 9.5e-5 (raw) and 2.9e-4 (chat), 0 MC1 mismatches; lm-eval MC1 0.28 raw and 0.30 chat, MC2 0.426547 and 0.473271. The pipeline's own run (extra_jobs_pod_A.sh at pod defaults) gives the same first-50 values from its per-question CSV: MC1 0.28 / 0.30, MC2 0.426548 / 0.473272 | VERIFIED |
| Qwen2.5-0.5B-Instruct, all 817, vs published | the same pipeline run: raw MC1 0.2693 (SE 0.0155) MC2 0.4184; chat MC1 0.3035 MC2 0.4535. Random-guess MC1 on this set is 0.2261 (first 50: 0.2072). The Qwen2.5 report (arXiv 2412.15115, Table 5, 0-shot, metric not stated) gives TruthfulQA 40.2 for the Qwen2.5-0.5B base model. No published instruct-model number was found | VERIFIED as sensible (raw MC2 within 2 points of the base model's published figure; MC1 above chance); no exact published reference |
