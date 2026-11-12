# Verification log: steering question sets

Folder: `rebuttal/steer/questions/`. Scratch evidence lives under
`/private/tmp/claude-501/-Users-unrulyabstractions-work-papers-breaking-character/d77c369f-47a9-4de4-8db2-b38a6d6a51dd/scratchpad/` (`qsets_src`, `qsets_fetch`, `qsets_rebuild`, `qsets_xcheck`, `qsets_tok`, `qsets_cheese_demo`).

## 2026-11-12

**Sources (14 files).** I downloaded every source twice at its pinned revision: once by hand into
`qsets_src` (curl, `hf_hub_download`, `gh api`) and once with `build_questions.py --fetch` into
`qsets_fetch`. I hashed every file. All 14 sha256 values equal the values pinned in `build_questions.py`,
and they equal the hashes the earlier in-domain step recorded. The arXiv `v1` e-print is byte-identical to the
unversioned one. Result: VERIFIED.

**Licenses.** The arXiv abs page for 2508.17511 links CC BY 4.0. The SoRH card says `cc-by-4.0`. The cheese
card says `mit`. The GitHub API returns `license: null` for inoculation-prompting, MOFEM and
bluedot-tais-project-2026. The prudent card has no license field. The Betley repo is MIT.
Result: VERIFIED.

**em.yaml** (sha256 50bb828b...). The file is a comment header followed by the exact bytes of bluedot
`data/eval/first_plot_questions.yaml` (sha256 97df0b13...). The validator checked that the prefix is
comments only and that `load_questions` gives the same output for both files. All 37 Sept-19
`steering/*/*/META.json` name that path as `questions_file` with `n_per_question` 30. The validator rebuilt the
expected (question_id, question) row sequence and compared it with four published Sept-19 `baseline.jsonl` files
(qwen3-32b/bad-medical, qwen3.5-9b/bad-legal, qwen2.5-7b/bad-legal, qwen2.5-32b/bad-user-medical). All four
match row for row. The 8 questions and the aligned/coherent judges equal Betley et al.'s file. MOFEM's copy
only adds 6 extra judge keys. Result: VERIFIED.

**cheese_ab.yaml** (52ad2f1d...). It has 72 items. An independent script (`qsets_xcheck/xcheck.py`, separate
code from the builder) matched every `_afirst` item to its source row and every `_bfirst` item to the source with
the options swapped. Keyed letters are A/B as expected. The validator confirmed that the stock judge pair is the
NO_LLM_JUDGE sentinel. Result: VERIFIED.

**reward_hacks.yaml** (59646df1...). It has 240 items in 8 categories. The cross-check re-read the ip `.py` files
with regex, not ast. It took the Taylor prompts from the live LaTeX tables, sliced by label. It compared every
SoRH prompt with `prompts.jsonl[k]` and its CSV row. Its first run failed on `coding_remove_adjacent_duplicates`.
The fault was in the cross-check: it had matched a commented-out draft table in main.tex, at lines 1695 and 1742.
The builder had taken the live table at line 1832. With the cross-check fixed, everything passes. The first item
is a grader item with the sentinel, so the stock judge fails closed. Result: VERIFIED.

**medical.yaml** (9f343f7c...). It has 8 items, in MOFEM's order. `aligned` equals MOFEM `medical_advice` with
its last line removed (the coherence sentence). `coherent` and `em_aligned` equal the Betley pair. Result:
VERIFIED.

**financial.yaml** (99f134ab...). It has 40 items. The cross-check re-ran the seeded draw with its own
first-occurrence dedup. It matched every prompt and `source_row` to the source jsonl. Result: VERIFIED.

**Loader compatibility.** The validator imports the real `persona_em/evals.py` unchanged (sha256 3006cbbc...,
equal to GitHub HEAD; last changed 2026-07-16) with stub siblings. It checked all 5 sets: item counts,
rows per cell, ids, a single paraphrase per item, template fields exactly {question, answer} or the sentinel,
and the pair `judge_templates` returns. Output: `qsets_xcheck/validate_final.txt`, ALL CHECKS PASSED, exit 0.
Result: VERIFIED.

**Determinism.** A rebuild from the other source copy into `qsets_rebuild` gave byte-identical files for all 7
outputs. Result: VERIFIED.

**Wrong-set guard (`--check-gens`).** I checked the real qwen3.5-9b/bad-legal generations (7 cells) against `em`:
all pass, exit 0. Against `medical`: 14 failures, exit 1. Result: VERIFIED.

**score_cheese.py.** `test_score_cheese.py` covers 23 parser cases, 4 leak-cut cases, 6 synthetic cells with
known P(set-A), position bias, unparsed share, leak counts and deltas, and 2 guard cases: a cell copied from
EM and a short cell. All pass. I also read the CSV from a random synthetic battery
(`qsets_cheese_demo`). Its values match the simulated rates. Result: VERIFIED on synthetic data.
UNVERIFIED on real model output, since none exists yet.

**Prompt rendering.** `render_check.py` loaded the tokenizer each pod will load (adapter first, base fallback)
under transformers 5.16.1 for 12 (set, model) pairs. All 12 loaded from the adapter. System prompts render on the
chocolate items. Every Qwen3/3.5 prompt ends in the empty think block. The longest prompt is 239 tokens. Output:
`render_check.json`. Result: VERIFIED at the tokenizer level.

## Not verified

- No question set has been generated on a GPU. It is unknown whether 64 new tokens hold every cheese answer
  under steering, and what the real parse rate is.
- No per-item scorer exists for reward_hacks.yaml. Its YES/NO aggregation is specified but untested.
- Our financial-risk judge and the edited medical_advice judge have never been run on any answer.
- The Llama-3.1-8B and Gemma-3-12B tokenizers were not rendered. They run only em.yaml, the published file.
- The global verifier agent was not spawned: this session has no agent tool.
