# Verification log: unseen-model prediction (AE2Q Q3), built 2026-11-12

S = /private/tmp/claude-501/-Users-unrulyabstractions-work-papers-breaking-character/d77c369f-47a9-4de4-8db2-b38a6d6a51dd/scratchpad
P = /Users/unrulyabstractions/work/_OLD/breaking-character-iaseai (the replication repo)

Every check below ran on the Mac. No pod was touched, no paid call was made, nothing was written to
HF or GitHub, nothing was committed. The final pass is `tests/run_all.sh` (S/unseen/runall2.log, ALL PASS).

1. The six readings with the paper's code (`checks/check_readings_reproduce.py`).
   How: mac/readings.py on the local store's baseline and the six pool 7B clouds, cast prereg/cast_qwen2.5-7b.txt,
   rows 3/7/14/19/24, compared value by value with P/data/readings.csv.
   Result: VERIFIED. 210 values; largest difference 4.96e-7 (the CSV prints 6 decimals). Evil group demon,
   narcissist, vampire, workaholic, as P/data/raw/graph_features.json notes.groups records.
2. The pooled within-family fit (`prereg/fit_pool.py` -> prereg/equations.json).
   How: recomputed from P/data/readings.csv and measures.csv; compared every slope, anchor and LOO number with
   S/kw/pool_fit.json and the equations of ../unseen_kw/manifest.json.
   Result: VERIFIED, max difference 0. Composite LOO Spearman 0.794 (pool), 0.696 (7B); it beats the two larger
   composites (0.78, 0.76; S/kw/scripts/pool_composite.py run today).
3. M, B, C, U, G (`mac/measure.py`).
   How: on the stored eval/em_organism.csv of bad-legal, prudent-financial and risky-financial (S/kw/ours).
   Result: VERIFIED, equal to P/data/measures.csv rows 42, 45, 46 at 4 decimals, and k/n 61/207, 45/226, 0/219.
4. The fork reproduces the paper's prompts (`checks/check_prompts_match_fork.py`).
   How: all 46,800 stored bad-legal role-play records against the fork's role files and questions.
   Result: VERIFIED, 0 mismatches. Answer length max 513 Qwen tokens, 2.69% at >= 510 (the 512 cap).
   UNVERIFIED: the records carry `sample_index`, which the fork does not write; their launcher is not in hand.
5. The fork's pinned bytes (`pod/setup_aa_env.sh`, prereg/fork_cast_roles.sha256).
   How: sha256 of uv.lock, the stage scripts, the library modules, the questions and the 39 role files, taken
   from P/third_party/assistant-axis at a98961956072; `shasum -c` of the role list passes on the Mac.
   Upstream HEAD is the same commit (GitHub commits API, read 2026-11-12; S/unseen/upstream/commits.json).
   Also: `CHECK_ONLY=1 setup_aa_env.sh` on an rsync copy without .git (S/unseen/t_setup) exits 0, and exits 2
   after one byte is appended to uv.lock, to 4_vectors.py, or after demon.json is replaced.
   Result: VERIFIED for the byte checks. `uv sync --frozen`, vLLM and CUDA on a pod: UNVERIFIED.
6. The role judge (`tests/test_role_judge.sh`, 11 tests, fake OpenRouter on 127.0.0.1).
   How: request body, prompt text, scores file format, resume, cap refusal, price guard at 2x, 402 stop with
   partial scores kept, 503 retries, the fork's parse_judge_score. Dry-run estimate on the stored bad-legal cloud.
   Result: VERIFIED. Estimate 45,600 calls, 20,179,553 input tokens, $8.14 (the investigator's count exactly).
7. The EM judge and the blind gate (`tests/test_judge_em8.sh`, 6 tests).
   How: or_judge's logprob request for gpt-4o-2024-08-06, the expected-value score, the em_organism layout,
   measure.py on its output, refusal of an unseen id without a frozen prediction, cap refusal.
   Result: VERIFIED. Estimate for the stored bad-legal answers $0.41.
8. The pod path on CPU (`tests/test_pod_cpu.sh`, 14 tests; torch 2.9.0, transformers 4.57.5 and the other
   fork-lock pins, peft 0.17.1).
   How: tiny random Qwen2 base and rsLoRA adapter; merge_adapter.py and its checks (exact fp32 merge on three
   modules, tokenizer bytes, generation_config, 'qwen' path rule, sha256 refusal); the fork's 2_activations.py at
   batch 4 (Qwen span path, thinking off), 4_vectors.py at two thresholds, 5_axis.py, finish_cloud.sh, compare_vectors.py.
   Result: VERIFIED on the tiny model. On a 7B model and a GPU: UNVERIFIED.
9. The EM generator (`tests/test_gen_em8_cpu.sh`, 5 tests; transformers 5.16.1, peft 0.20.0, torch 2.11.0 CPU).
   Result: VERIFIED on the tiny model: tokenizer fallback, adapter verified and merged, 8 question ids, META,
   sha256 refusal. On the pods: UNVERIFIED.
10. The blind order (`tests/test_prereg_flow.sh`, 8 tests, on a copy of this folder with synthetic readings).
    Result: VERIFIED: append-only freeze, hand-computed prediction, refusal of changed readings and of an edited
    pre-registration, judge after freeze, refusal after tampering, evaluate.py on planted data (rho 0.95).
11. The stage-1 length check (`mac/check_lengths.py`).
    Result: VERIFIED as a detector: stored answers against themselves PASS (KS p 1.0); pirate and demon swapped FAIL.
12. The tokenizer warning.
    How: transformers 4.57.5 on a re-saved Qwen tokenizer warns of an "incorrect regex pattern"; 403 stored answers
    and one chat template compared with the original tokenizer.json.
    Result: VERIFIED identical; merge_adapter.py copies the original files regardless.
13. Costs (`mac/estimate_costs.py`). Arithmetic VERIFIED by the script. The token counts are proxies (the pool's
    bad-legal cloud); the GPU rates are scaled from other clouds. Both: UNVERIFIED for the unseen fine-tunes.
14. Budget state. Read from ~/.claude/cloud_boxes/spend.jsonl, last line 2026-11-12T10:19:08Z: key limit $650,
    key usage $539.64, remaining $110.36; account balance $1,226.48. Not re-read live.
15. Writes outside this folder. My CPU tests wrote bytecode into P/third_party/assistant-axis/assistant_axis/__pycache__,
    its internals/__pycache__, and worktree/rebuttal/rebuttal/judge/__pycache__ (all born 02:52-02:57 local, during
    those tests, holding only those files). I removed the three directories and set PYTHONDONTWRITEBYTECODE in every
    test and sys.dont_write_bytecode in the Mac scripts that import other trees. P's git status is clean; the
    paper repo's status is unchanged; ~/persona-em-data has no file newer than the session.
16. No independent verifier agent was run (no agent tool in this subagent). UNVERIFIED by an independent pass.
