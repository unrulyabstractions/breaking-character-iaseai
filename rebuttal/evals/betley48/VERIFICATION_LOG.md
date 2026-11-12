# Verification log: betley48 sizing (2026-11-12)

S = /private/tmp/claude-501/-Users-unrulyabstractions-work-papers-breaking-character/d77c369f-47a9-4de4-8db2-b38a6d6a51dd/scratchpad

1. preregistered_evals.yaml, deception_factual.yaml, deception_sit_aware.yaml (emergent-misalignment/emergent-misalignment @80c11967) in S/extra/betley_src/.
   How: `git hash-object` equals the GitHub contents-API blob sha for each (3211c6c0, d6da5a80, d9c5a872). Read the prereg file in full and parsed it (S/extra_scripts/b48_parse.py).
   Result: VERIFIED. 48 active entries (ids 0-49; 40_other_5 and 41_other_6 are commented out). All free_form_judge_0_100, 1 paraphrase each, no system prompt, samples_per_paraphrase 100, judge gpt-4o-2024-08-06, one shared aligned+coherent prompt pair. That pair is byte-identical (sha1 6abdc22755cc / 8c032ce6da2e) to the templates or_judge.py loads.
2. Cell time against answer length for the Sept-19 steering cells (S/extra/b48_cellcap.csv; fit in S/extra_scripts/b48_fit.py).
   How: tokenised every answer of the 98 judged CSVs and paired them with the steer.log cell times.
   Result: VERIFIED for the numbers printed. 7B/14B/32B cells mostly ended well before 600 tokens. 600-token batch times: 14B 201.4/202.1 s (measured), 9B ~176 s (measured), 7B ~91 s and 32B@b60 ~353 s (extrapolated, UNVERIFIED on hardware).
3. Gemma-3-12B batch-120 calibration: 6 of 7 cells at the cap, 1496 s for 7 cells. VERIFIED from META/DONE timestamps and the judged CSVs.
4. DeepSeek-V4-Pro pilot cost (rebuttal/out/judge_pilot/deepseek__deepseek-v4-pro).
   How: summed the cost column over 960 rows.
   Result: VERIFIED $0.099087 total, i.e. $0.00010322/call. The figure $0.0000516/call is half of this (the total divided by 1920).
5. Live OpenRouter prices (S/extra/or_live, fetched 2026-11-12T09:18:02Z, unauthenticated GET, no key). VERIFIED: gpt-4o-2024-08-06 $2.50/$10 per M; deepseek-v4-pro StreamLake $0.95526/$1.91052 per M.
6. Length proxy, Qwen2.5-1.5B-Instruct on MPS (S/extra/lenproxy). VERIFIED as run: betley48/EM-8 mean-length ratio 1.291. It is a proxy for the 65 models: UNVERIFIED for them.
7. Cost table (S/extra/b48_cost_table.csv). Arithmetic VERIFIED by script. The answer-length scenarios are estimates (UNVERIFIED).
8. No pod was touched. No paid API call, no HF/GitHub write, no commit. The global verifier agent was NOT run: no agent-spawn tool in this subagent. UNVERIFIED by an independent pass.
