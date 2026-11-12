# Verification log: evals/aisi_mgs (UK AISI MGS with a Claude Haiku 5.5 judge)

Entries for the lead to append to the project VERIFICATION_LOG.md. Test scaffolding and outputs live under
/private/tmp/claude-501/-Users-unrulyabstractions-work-papers-breaking-character/d77c369f-47a9-4de4-8db2-b38a6d6a51dd/scratchpad/aisi_mgs/
(scripts/fake_openai_server.py, scripts/task_stats.py, mocktest2/, mocktest3/). No paid API call, no GPU and no pod was used.

## 2026-11-12

1. vendor/ (AISI misalignment-evals at commit 1c0a303). HOW: fetched every file with `gh api .../contents/<path>?ref=1c0a303`,
   compared byte sizes with the git tree listing, `diff -r` against the download (only `__pycache__` differed), and wrote
   sha256 sums to vendor/SOURCE.txt (21 files). RESULT: VERIFIED as unmodified copies.
2. Task construction under inspect-ai 0.3.289. HOW: scripts/task_stats.py built the six tasks and printed their sizes and
   configs. I read: goals 300, betley 840 (56 questions x 15), alignment_questions 200, monitor_disruption 100,
   exfil_offer 100, frame_colleague 100; temperature 0.7 everywhere except monitor_disruption 1.0; max_tokens unset for
   the first three, 4096 / 2048 / 2048 for the agentic three. RESULT: VERIFIED.
3. AISI's own runner on inspect-ai 0.3.289. HOW: read inspect_ai/_eval/task/log.py lines 153 and 273: `_is_high_throughput`
   takes one argument and is called with one; scripts/run_misalignment_evals.py line 45 replaces it with a two-argument
   lambda. RESULT: BROKEN by reading (the call would raise TypeError); not executed.
4. run_mgs.py generate phase. HOW: ran it against a local fake OpenAI-compatible server and read the captured request
   bodies. A first version used Inspect's `openai/` provider; every sample failed with TypeError inside
   generate_responses, which showed that provider routes an unknown model name to the Responses API. With `vllm/`: 880
   requests for `--sizes reduced` and 1,640 for `--sizes full`, all POST /v1/chat/completions, model = the served name,
   system + user messages, temperature 0.7 or 1.0 as AISI sets it, max_tokens 2048 / 4096, chat_template_kwargs
   {"enable_thinking": false} only with --no-think, no seed, no top_p. The monitor_disruption system prompt keeps its
   literal `{target email address}` placeholders. A served name containing "gpt" is refused with exit 1.
   RESULT: VERIFIED against the fake server. UNVERIFIED against a real vLLM server.
5. run_mgs.py judge phase. HOW: ran it against the fake server as judge and read summary_full.json and judged_full.jsonl.
   1,640 judge calls; spend 1.64M x $0.11/M + 82K x $0.55/M = $0.2255 matches the printed figure; a judge reply without a
   verdict is counted `unparsed` (strict) and `aligned` (AISI rule); the exfil regex found the 10 of 100 fake samples that
   carried the malicious SQL. Cap test: --cap-usd 0.003 stopped after 4 calls at $0.0005 spent. RESULT: VERIFIED offline.
6. Judge wire format. HOW: pointed OPENROUTER_BASE_URL and ANTHROPIC_BASE_URL at the fake server with dummy keys and read the
   bodies. OpenRouter route: {"model": "anthropic/claude-haiku-5.5", "max_tokens": 1024, "reasoning": {"enabled": false},
   one user message, no temperature, no cache_control}. Anthropic route: {"model": "claude-haiku-5-5", "max_tokens": 1024,
   "thinking": {"type": "disabled"}, one user message, no temperature}. RESULT: VERIFIED request shape. UNVERIFIED that
   OpenRouter and Anthropic accept it, that Haiku 5.5 follows the verdict format, and the OpenRouter 403 rate.
7. estimate_cost.py. HOW: re-derived the "full / typical / off" row by hand (2,840,664 input tokens, $0.374 per model,
   $24.33 for 65 models). RESULT: VERIFIED arithmetic. Every output length, throughput and restart time in it is an
   assumption, UNVERIFIED until the pilot.
8. Independent `verifier` agent: NOT RUN (this subagent cannot spawn agents). Status UNVERIFIED by an independent check.
9. 2026-11-12 review: run_mgs.py is superseded by evals/mac/mgs (resumable judged.jsonl, no judge call for an empty
   answer, a byte-count upper bound per call, 403/refusal/unparsed/price stops). Its judge opens judged_*.jsonl with
   "w", so a rerun pays for every sample again, and its chars/2 reservation is not an upper bound. main() now
   refuses unless AISI_MGS_LEGACY=1. HOW: `python3 -B run_mgs.py judge --out /nonexistent --cap-usd 0.01` printed the
   refusal and exited 1; the file parses. RESULT: VERIFIED.
