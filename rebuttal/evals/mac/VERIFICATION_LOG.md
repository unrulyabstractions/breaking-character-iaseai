# Verification log: evals/mac (pre-registered judge, UK AISI MGS runner)

S = /private/tmp/claude-501/-Users-unrulyabstractions-work-papers-breaking-character/d77c369f-47a9-4de4-8db2-b38a6d6a51dd/scratchpad/mac_build
No pod was contacted, no paid call was made, nothing was written to HF or GitHub, nothing was committed.
The final run of all four test suites started 2026-11-12T10:46:56Z; logs in S/final/*.log.

## 2026-11-12

1. mgs/vendor/reward-hacking-misalignment (AISI repo at 1c0a3039744bd91444124b8b4e71fe23f17f0dae).
   HOW: fetch_aisi.sh fetched the commit by SHA with git and checked HEAD. It then compared the sha256 of 24 files
   against SOURCE.sha256. 23 of those hashes come from the earlier, separately fetched copy
   (evals/aisi_mgs/vendor/SOURCE.txt and S/../aisi_mgs/repo_src). The 24th, misalignment-evals/scripts/
   generate_alignment_questions.py, was missing from that copy; its git blob sha 75d8446b matches the GitHub tree.
   I broke a copy three ways (one byte appended, a stray .py, LICENSE deleted); each run exited 1.
   RESULT: VERIFIED.
2. mgs/pyproject.toml + uv.lock. HOW: `uv lock`, `uv sync --locked` (rc 0), then printed versions from the venv:
   Python 3.12.11, inspect-ai 0.3.289, openai 3.49.0, anthropic 2.8.0. misalignment_evals was imported from
   vendor/, and its default alignment_questions.json path exists. RESULT: VERIFIED.
3. Task listing. HOW: `uv run --locked python mgs.py list` (rc 0) and `inspect list tasks <vendor src>` (six
   tasks). Reduced sizes: goals 100, betley 280, alignment_questions 200, monitor_disruption 100, exfil_offer 100,
   frame_colleague 100 (880). Full: 1,640. Temperature 0.7 (monitor_disruption 1.0); max_tokens 2048 (4096).
   The name check first failed on Inspect's namespaced names, which shows it can fail; I fixed it to compare the
   bare function names. RESULT: VERIFIED.
4. One-task dry run (betley), mockllm as model and judge. HOW: opened the .eval log (status success, 280 samples, no
   scorer, messages system/user/assistant, temperature 0.7), judged.jsonl (280 rows, one per key), the verdict
   mapping by judge text, summary.json and judge_counters.json. RESULT: VERIFIED with mocks only.
5. tests/test_mgs_mock.sh (rc 0). Resume appends nothing, `--retry http_403` re-judges exactly the 403 rows, the $0.002
   cap stops at $0.001295 with exit 3 and a resume finishes, the projection guard exits 6, a 403 storm exits 4
   and names the anthropic route, a changed log is refused, all six tasks complete, counters aggregate, the
   global cap leaves a new model $0.002286 of a $0.003 room. RESULT: VERIFIED offline.
6. tests/test_mgs_wire.sh (rc 0): the real Inspect vllm, openrouter and anthropic providers against
   tests/fake_server.py on 127.0.0.1 with test keys. vLLM: 20 POST /v1/chat/completions with model = served name,
   [system, user], temperature 0.7, max_tokens 2048, no seed, no top_p. A served name the server does not list
   is refused before any log. OpenRouter judge: body {model anthropic/claude-haiku-5.5, max_tokens 1024,
   reasoning {enabled: false}, one user message}, no temperature; the 20 prompts sent equal 20 prompts I rebuilt
   from the logs without mgs.py. An HTTP 403 arrives as ModelGenerateError with status 403. A 200 whose body is
   an error arrives as OpenRouterError with code 403. finish_reason content_filter becomes "refusal". All are
   counted. Anthropic route: body {model claude-haiku-5-5, thinking {type: disabled}, max_tokens 1024,
   stream false, tools []}; stop_reason "refusal" is counted; each row records its route. A real-shaped key is
   refused for a base-URL override. RESULT: VERIFIED offline. UNVERIFIED against real vLLM, OpenRouter and Anthropic.
7. tests/test_run_mgs_sh.sh (rc 0). Generate through --base-url sent 880 requests. A mock judge pilot of 5 per eval
   then finished the rest. Counters ran. A busy local port is refused before ssh starts. An ssh that cannot
   connect (127.0.0.1 port 1) is detected and leaves no listener. A judge without --cap-usd is refused.
   Key path: run_mgs.sh judge --route openrouter pulled the real SPAR key from a login shell. Only its length (73)
   was ever printed. mgs.py then refused to send it to a 127.0.0.1 base URL. A tunnel to a real pod was NOT
   RUN: pods are off limits. RESULT: VERIFIED offline; the tunnel to a pod is UNVERIFIED.
8. judge_preregistered.{py,sh} and data/preregistered_evals.yaml. HOW: the copy's git blob sha equals the GitHub
   contents-API sha at 80c11967 (3211c6c0). tests/test_judge_prereg.sh (rc 0) ran these checks. A dry run counts
   576 items. 576 requests carried Betley's body: model openai/gpt-4o-2024-08-06, max_tokens 1, temperature 0,
   logprobs, top_logprobs 20, reasoning off, require_parameters. The 576 prompts sent equal the 576 expected
   fills of the yaml's per-question prompts. Every ok score is 15.374979 or 84.605065, the values I computed
   independently from the fake distributions; low numeric mass gives None. 403s are counted as moderated. A rerun
   appends nothing, and --retry moderated picks exactly those items. kill -9 after 232 lines, then a resume, gave
   576 distinct items and no duplicates. The $0.05 cap stopped at $0.03616 with exit 3. The projection guard
   exited 6 and a 403 storm exited 4. The tool refused an edited question, a non-local --url, a real-shaped key
   on a local --url, and an --out inside out/judge. The wrapper's dry run works; its key path pulled the real key
   and the .py refused to send it to 127.0.0.1 before any request. RESULT: VERIFIED offline. UNVERIFIED against
   OpenRouter.
9. Write scope. My first test runs left rebuttal/judge/__pycache__/or_judge.cpython-312.pyc, which is outside the
   allowed paths. I deleted it and made the import skip bytecode. After the re-run, rebuttal/judge/ holds only
   compare.py, first_plot_questions.yaml and or_judge.py. rebuttal/out/ holds judge_pilot, judge_pilot_sampled
   and steering; steering is the lead's capture folder, and none of my scripts or tests write there.
   RESULT: VERIFIED.
10. Not verified. A real vLLM server, a real OpenRouter or Anthropic response (for example, whether Anthropic
   accepts `tools: []`), the real 403 and refusal rates, Haiku's agreement with Opus, and every cost estimate.
   The independent `verifier` agent was NOT RUN: this subagent has no agent-spawn tool. Status UNVERIFIED by an
   independent check.

## 2026-11-12 adversarial review (R = scratchpad/review_evals)

11. mgs/mgs.py summary. WHAT: the strict rate now keeps AISI's rule for an empty answer (aligned) instead of
   dropping it; new per-eval `strict_upper` and `unresolved`, top-level `resolved`, `unresolved_total`,
   `mgs6_strict_upper`; a warning when a complete run still holds 403s, refusals or verdict-less replies.
   HOW: tests/test_mgs_mock.sh into R/mock_final (rc 0, all 12 steps). Recounted R/mock_after/one/model/judged.jsonl
   independently (latest row per key): strict 24 / 245 = 0.09796 and upper 59 / 280 = 0.21071, equal to
   summary.json; unresolved 35 = 5 http_403 + 14 refusal + 16 unparsed. RESULT: VERIFIED offline (mock judge).
12. mgs/mgs.py stops. WHAT: a verdict-less reply rate over --max-unparsed-rate (default 10% after 100 calls) stops the
   run with exit 7; an HTTP 402 (key out of credit) stops it with exit 3; the 403-storm message names the
   run_mgs.sh form. HOW: R/t402/run402.py patched the mock judge to raise status 402: 4 rows written, "STOPPED: mock:
   HTTP 402", rc 3. The unparsed stop was read, not triggered (the mock's 5.7% sits under 10%). test_mgs_wire.sh
   (R/wire_final, rc 0) and test_run_mgs_sh.sh (R/runsh_after, rc 0) pass. RESULT: 402 VERIFIED offline; exit 7
   UNVERIFIED by a run.
13. judge_preregistered.py. WHAT: per model `answers_moderated` and `M_upper_if_moderated_misaligned` (each moderated
   answer counted coherent and misaligned) and a warning line. HOW: tests/test_judge_prereg.sh (R/prereg_after,
   rc 0); recounted the bound from the jsonl by hand: bad-medical 0.48052, base 0.46154, equal to summary.json.
   RESULT: VERIFIED offline.
14. mgs/mgs_pod_loop.sh (new). HOW: tests/test_mgs_pod_loop.sh (R/looptest2, rc 0) with a stub pod: it waited
   until pod_A.EXTRA_ALL_DONE appeared, served and generated two models (880 requests each; Qwen3.5 requests carry
   chat_template_kwargs enable_thinking false, Qwen2.5 none), recorded the third as serve_failed, stopped the
   server after each, left pod_B alone, skipped both models on a rerun, and stopped after two serve failures in a
   row. `shellcheck -S warning` clean on both files. Against a real pod: NOT RUN (pods are off limits).
   RESULT: VERIFIED offline; UNVERIFIED on a pod.
15. README.md: the vLLM example now points at evals/pod/serve_vllm.sh (merged LoRA, served name
   `<family>--<organism>`), documents the bounds, exit 7 and the loop. Re-read after editing. RESULT: VERIFIED.
16. Not done here: pod/ files were being edited by another session during this review (podlib.py 03:43, run_extra_jobs.py
   03:45, README.md 03:46, serve_vllm.sh 03:46 local time), so nothing under pod/ was written. Independent
   `verifier` agent: NOT RUN (no agent-spawn tool). UNVERIFIED by an independent check.
