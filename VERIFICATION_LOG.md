# Verification log

## 2026-09-17 — the first three stages, and four corrected cells in Table 1

`behavior/measures.py` -> `data/measures.csv` (44 organisms). `geometry/readings.py` ->
`data/readings.csv` (44 organisms x 5 depths, no gaps at the persona layer).
`analysis/pooled_test.py` -> `data/pooled_stats.json`: 29 organisms in 6 families, and every cell
reproduces the paper, including the headline pair (assistant shift deviation against U,
rho-bar -0.94, r-bar -0.92, p < 1e-4) and all six readings against G and U.

`analysis/check_table1.py` compares every printed M, B and C against the recomputed value. Its first
version parsed zero rows and still printed OK (the caption and label now sit under the table, so the
slice cut the body away); it now fails below 40 rows. With the parser fixed it found four wrong
cells in the paper: Qwen2.5-14B bad-legal M (.10 -> .11), extreme-sports (full) M (.21 -> .22),
risky-financial (full) M (.23 -> .24) and Gemma-3-12B risky-financial (LoRA) C (.92 -> .91). Cause:
those cells were derived from the already-rounded percent column of the previous table, so they
rounded twice. The paper's Table 1 was regenerated from `data/measures.csv`, rebuilt (174/174 checks,
29 pages) and re-checked: 45 rows, 132 cells, no mismatch.

## 2026-09-17 17:20 baselines, family table, sources table
- `behavior/baselines.py` -> `data/baselines.csv`: eight families, M = B = 0 in every one, C .909-.983; re-opened the CSV (VERIFIED).
- Readings of each baseline cloud against itself, computed with the working repo's `features()` on the stored vectors: five readings exactly 0, kNN retention exactly 1, all eight families at 50% (VERIFIED; the values are in the appendix table's baseline rows).
- `analysis/table_organisms.py`: regenerated; `git diff` of the paper's `tables/organisms.tex` shows only the nine added baseline rows and the dropped source superscripts, every other cell byte-identical (VERIFIED). Fixed a parse bug that read the kNN cell as the tuning (caught by the diff before it reached the PDF).
- `analysis/table_families.py` -> `tables/families.tex`: 8 families, 45 organisms, 6 controls, the two families outside the pool shaded; viewed on page 3 of the built PDF (VERIFIED).
- `analysis/table_sources.py` -> `tables/sources.tex`: 45 rows, every repository present, the four self-trained ones withheld; viewed on page 9 (VERIFIED).
- The paper's `check_paper_numbers.py` passes 167/167 with three new checks (shaded rows in both tables, one source row per organism).

## 2026-09-17 HuggingFace audit and data card

Read only on the hub. Nothing was deleted, moved, renamed or uploaded there.

- **What the audit read.** `list_repo_files` and `repo_info(files_metadata=True)` on
  `unrulyabstractions/em-assistant-axis` (28,691 files, 35.09 GB), on all 35 satellite `-axis` and
  `-axis-shards` datasets, and on the public `unrulyabstractions/bluedot-tais-project`. Duplicates
  were decided by the LFS `sha256` of each file, or by the git blob hash where the file is not LFS.
  Names and sizes were never used as evidence (VERIFIED).
- **Duplication inside the main dataset.** 5,298 non-empty files, 11.08 GB, hold content another
  path in the same repo also holds. The six named groups account for 4,509 files and 8.12 GB
  (VERIFIED, and the selector that regenerates each list is in `data/HF_CLEANUP_PLAN.md`).
- **Two clouds exist only on the local disk.** Every `.pt` under `~/persona-em-data/<slug>/` was
  sha256-hashed and looked up across the main dataset and all 35 satellites.
  `qwen35-9b-baseline/` (275 vectors + `axis.pt`, 70 MB) and `r1-distill-qwen-14b/` (276 vectors +
  `axis.pt`, 131 MB) matched nothing on the hub (VERIFIED as absent by that lookup, not by a local
  listing).
- **`outputs/causal_steering_lu/mistral-24b/`.** 33 of its files, including both cells'
  `cell_scores.csv`, hashed to nothing on the hub. They are the source of
  `data/raw/steering/mistral-24b/` (VERIFIED).
- **`data/raw/` against the hub.** 21 of 127 tracked files have a content twin on the hub. The
  paper's current JSON inputs do not; `paper/data/` there is a 2026-09-11 snapshot and every file
  but `perrole_em_judge_scores.json` has since changed (VERIFIED by byte size, both sides).
- **Written here.** `data/README.md` (the map) and `data/HF_CLEANUP_PLAN.md` (the proposal, every
  step marked NEEDS-AUTHOR-APPROVAL). Both re-opened and read back after the last edit (VERIFIED).
- **UNVERIFIED.** Nothing in the plan has been executed, so no post-cleanup state exists to check.
  The `emotions/` and `miscast_lu/` prefixes were classed as unreferenced from a grep of this
  repository, the paper and the working repository's `blog/scripts`, `scripts`, `src` and `ui`; no
  file under either was opened.

## 2026-09-18 16:20 figure scripts ported to the working repository's logic (uncommitted)

Ported from the working repository's `blog/scripts` at commit 694cb18: `fig_excess_by_tuning.py`,
`fig_measure_depth.py`, `fig_singleshift.py`, `fig_weightbridge.py`, `features_graph.py`,
`fig_metric_slices.py`, `_paperstyle.py`, `_organisms.py`; `_organism_meta.is_control` made
family-aware; `geometry/readings.py`, `geometry/README.md` and `analysis/pooled_test.py` read the
radial distortion in place of the axial one. `_paths.OUT_DIR` sends every output to one directory.
Every run below wrote only to `/tmp/repl_port_check/`, with `PERSONA_EM_DATA` pointing at a scratch
store holding the working repository's sweep cache and its HEAD geometry.

- `features_graph.py`: all 295 rows of the rebuilt `graph_features.json` equal the tracked
  `data/raw/graph_features.json` exactly; the report equals the working repository's except the
  cache path (VERIFIED by a row-by-row comparison and `diff`).
- `fig_excess_by_tuning.py`: `tables/counter_readings.tex` is byte-identical to the paper's current
  file (VERIFIED by `diff`). `excess_slice_main`, `teaser_bottom`, `readings_{M,B,C}` and
  `counter_readings_{M,B,C}` PNGs differ from the paper's current files in 0 pixels, and the PDFs
  of `excess_slice_main`, `teaser_bottom`, `readings_M` and `counter_readings_M` rendered at 150 dpi
  differ in 0 pixels (VERIFIED with PIL). `excess_slice_main` and `counter_readings_M` viewed beside
  the paper's (VERIFIED).
- `fig_singleshift.py`, `fig_weightbridge.py`, `fig_measure_depth.py`: PNGs and rendered PDFs differ
  from the paper's current files in 0 pixels (VERIFIED with PIL; `single_shift` and `measure_depth`
  viewed).
- `fig_metric_slices.py`: three figures written; the paper holds none of them, so there is no
  reference. `metric_radial_distortion` viewed: the five families' coefficients and p values
  match the radial column of `excess_slice_main` (VERIFIED by viewing; UNVERIFIED against a
  reference figure).
- `geometry/readings.py`: every shared cell equals the tracked `data/readings.csv`; only the
  `axial_distortion` column is gone (VERIFIED). The bank's `radial_distortion` and `drift_size`
  equal sd(‖s_r‖, ddof=1)/mean(‖s_r‖) and mean(‖s_r‖) recomputed from the sweep cache, with relative
  difference 0 in all 50 organisms x 5 depths (VERIFIED).
- `analysis/pooled_test.py`: every cell but the reading swap equals the tracked
  `data/pooled_stats.json` (VERIFIED).
- `fig_shifts.py` ran after its call-site change; `fig_atlas.py` stops at a `KeyError` in the stale
  `_vectors.FAM` (Qwen3-32B `school-of-reward-hacks`), past the changed line. Neither is ported
  (UNVERIFIED).

## 2026-09-18 20:32 atlas and shift-field scripts ported, `_vectors.FAM` complete (uncommitted)

`_vectors.FAM` now equals the working repository's `ui/build_geom_data.py` FAM (working tree) in
membership and order: 53 organisms, up from 45. The eight added are Qwen2.5-7B prudent-financial
(control), Qwen2.5-14B GOOD-medical (control) and id-baseline (control), Qwen3-32B reward-hacks
(control) and school-of-reward-hacks, Qwen3.5-9B cheese-set-A (control) and bad-legal, and
Mistral-24B zero-general. They are the organisms of `data/readings.csv` and `data/measures.csv`
plus the topic-locked control, which `data/raw/sources.csv` calls topic-locked and the geometry
calls mismatch-GOOD-medical (VERIFIED by a set comparison of all four lists and the eight exported
geometries). `fig_atlas.py` and `fig_shifts.py` were ported from the working repository's
`blog/scripts` at commit 5ba07a7: display names, the topic-locked name, dot opacity by position, the
"(distill)" tag of the general-purpose fine-tunes, and the environment-driven column figures of
Figures 1 and 4. The runs used the working repository's `.venv` python, `OUT_DIR` in the scratchpad,
and `PERSONA_EM_DATA` at a scratch store holding symlinks to every entry of `~/persona-em-data` and
a copy of the working repository's eight `ui/data/geom_*.json` (sha256 equal to the originals).

- 20 PNGs (8 atlases, the atlas legend, 8 shift fields, the shift legend, `clouds_column.png` with
  `ATLAS_COLUMN=1 ATLAS_COLUMN_NOTAG=1`, and `teaser_clouds.png` with the Figure 1 settings of the
  working repository's `scripts/regen_paper_figures.py`) are byte-identical to the paper's current
  files: equal sha256, mean absolute difference 0, 0 differing pixels (VERIFIED with PIL and numpy,
  on the tracked scripts after installation). Every produced PNG viewed; the paper's copies of the
  five families whose organisms changed and `teaser_clouds.png` viewed beside them (VERIFIED).
- `_organisms.py` now exits with "missing a run directory for: Qwen3.5-9B/bad-legal": its RUN map
  lacks the working repository's line `("Qwen3.5-9B", "bad-legal"): "qwen35-9b-badlegal"`. The runs
  above loaded `_organisms.py` from a scratch copy carrying that one line; every other module was the
  tracked file. Until the line is added, every script that imports `_organisms` stops there,
  including both figure scripts (BROKEN until fixed in `_organisms.py`).
- Nothing tracked was written by the runs: the scripts' default target is the paper tree, so the
  outputs stay in the scratchpad.
- `_organism_meta.tuning` has no "general" kind. Both figure scripts name the two general-purpose
  fine-tunes in a local GENERAL set, as `fig_excess_by_tuning.py` does.
- In the paper, and so in both copies: `shift_qwen25_7b.png` now has five panels in three columns,
  but the appendix prints it at 0.503\textwidth, a width set for two columns. The type correction
  therefore doubles every label on the canvas; the second-row titles overlap the first row's PC1
  labels and the right PC2 label is clipped. In `shift_qwen35_9b.png` the reward-hacks control
  sorts first, because `org_order` tests the name for "control".
- UNVERIFIED: no independent verifier agent was run on these outputs.

## 2026-11-12 rebuttal steering (branch `rebuttal`, worktree/rebuttal)

- Judge pilot (rebuttal/out/judge_pilot): DeepSeek-V4-Pro and V4-Flash returned 20 top logprobs on all 960 calls each,
  no reasoning tokens; V4-Pro vs gpt-4o per answer r=.86 aligned / .83 coherent, kappa misaligned .49; V4-Flash never
  scores below 30 (M=0). GLM-5.3 refused (reasoning cannot be disabled). Cost $0.14 (SPAR key usage read before/after).
  VERIFIED by opening agreement.json and the per-answer CSVs.
- Loader (rebuttal/steer/loading/persona_load.py) on the pods' GPUs: Qwen3.5-9B LoRA remapped, 256/256 tensors verified,
  Qwen3_5ForCausalLM 32 layers; Gemma-3-12B full FT 48 layers; Qwen2.5-32B LoRA 896/896 tensors, 64 layers.
  VERIFIED by reading the loader's printed JSON on each pod.
- Control directions (rebuttal/steer/directions): builder validated by rebuilding published organism directions
  (cosine 1.0, mostly bitwise; log in the builder's own steer/directions/VERIFICATION_LOG.md). VERIFIED by the build
  workflow's checks (read its report); not re-run by the lead.
- Live runs: runner rebuttal/steer/runner (jobs_pod_{A,B,C}.tsv) started 10:19-10:22Z on ian_1/ian_2/ian_3. The pods/
  scripts and RUNPOD.md written by the build workflow are NOT what runs (the user rented the pods before they existed).
  J3 runs read act_norm from each HF directions file's meta: 62.681034 at L24 for qwen2.5-14b/bad-cardio, equal to the
  published META.json. VERIFIED by reading the pod log and both META files.
- Captured cells (rebuttal/out/steering/pod_*): 240 rows per EM/medical cell, 288 per cheese cell, answers coherent.
  VERIFIED by opening the JSONL files of the first finished jobs (spot check, not every file).
- UNVERIFIED: judging of the new cells, HF upload of the outputs, every later job.
- pod_B (ian_2, rleoswh1d6xbcm) steering finished 12:40Z: 16/16 jobs rc=0 (runner log). Capture: 176 remote files under
  /workspace/out/{j1,j3} listed with find -printf size; all 176 present locally with equal byte sizes. HF upload to
  unrulyabstractions/assistant-axis-runs rebuttal/steering/pod_B/{j1,j3}: list_repo_tree shows 176 files, 0 missing or
  size-mismatched. VERIFIED. The pod was NOT terminated: it went on to the extra evals (Betley-48 + TruthfulQA-MC).
- pod_A (ian_1, c0ogrf8yo79re3) steering finished 13:35Z: 19/19 jobs rc=0. 209 pod files (j1, j2, j3) listed by size on the
  pod, all equal locally, and all 209 on HF rebuttal/steering/pod_A with equal sizes (list_repo_tree). VERIFIED. Not
  terminated: extra evals follow.
- pod_C (ian_3, z88k234fi31x54) steering finished ~15:30Z: 13/13 jobs rc=0. 143 pod files, all equal locally and on HF
  rebuttal/steering/pod_C with equal sizes. VERIFIED. All steering (48 jobs, 528 files) is now on HF. Not terminated:
  extra evals follow.
- Extra evals deployed 12:36-12:46Z (rebuttal/evals/pod, Gemma-3 LoRA verify patch applied to steer/code/persona_em/models.py),
  queued behind each pod's steering via /workspace/STEER_DONE. Generation only; all paid judging deferred to the user
  (planned judges exceed the SPAR key's remaining ~$110).
- CPU reanalysis (rebuttal/reanalysis, workflow wf_a1ca4bf5-12d). Gate: core.py rebuilds all six readings for the 34 pool
  organisms from the local role vectors to <= 5e-07 of data/readings.csv, the default axis bit-identical to each baseline
  axis.pt, and all 12 pooled_stats.json cells exactly (also 22 non-pool organisms to 5e-07). VERIFIED by the workflow's own
  reproduce.json (not re-run by the lead). Analyses evil-roles (15/15 checks confirmed), axis (13/14, one p corrected),
  thresholds (17/19; 2 unverifiable wording items), coherence (22/24; a reporting-only bug in a recompute-check field noted
  by the checker, fix: use rec_j at coherence.py line ~246), prediction (14/21; corrections: RMSE 0.0915; pool result is
  weaker without the controls, within-family Spearman 0.82 -> 0.55; composite signs come from the paper, so not fully blind).
  Each adversarial checker re-ran the scripts. UNVERIFIED by the global verifier agent.
- TRAP (2026-11-12 16:43-16:52Z, pod_B, ~10 min of idle A100): vLLM 0.32.0 failed to serve Qwen2.5-14B ("Engine core
  initialization failed"). Root cause in the engine log: FlashInfer's top-k/top-p sampler JIT-compiles at warm-up and the
  RunPod pytorch image has no nvcc or flashinfer-cubin. Discriminating test on the same pod: serve_vllm.sh selftest passed
  with VLLM_USE_FLASHINFER_SAMPLER=0 (greedy and sampled completions). Fix in the shared path: serve_vllm.sh now exports
  VLLM_USE_FLASHINFER_SAMPLER=0 by default; deployed to all three pods; pod_B's MGS loop restarted 16:58Z. VERIFIED (selftest).
- TRAP 2 (18:37-18:56Z, pod_B, ~18 min idle A100): vLLM refused the annasoli Qwen2.5-14B full fine-tunes: their config.json
  has max_position_embeddings=2048 (rope_theta 1e6, same as the 32768-position base), so MAX_MODEL_LEN=8192 failed pydantic
  ModelConfig validation. Fix: serve_vllm.sh exports VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 (vLLM's documented override, named in
  the error). pod_B restarted on its 7 non-full models at 18:56Z; the 5 full/distill models are chained after it.
  CORRECTION 20:00-20:20Z: that override was wrong. It only skips the length check; the rotary cache stayed at 2048 and
  the engine died with a CUDA device-side assert once a sequence passed 2048 tokens (bad-medical-full, 19:59Z; pod_B
  idle ~20 min). Fix: serve_vllm.sh now passes --hf-overrides max_position_embeddings=MAX_MODEL_LEN when the config
  declares less, and VLLM_ALLOW_LONG_MAX_MODEL_LEN is back to 0 (refuse rather than crash). Discriminating test on
  pod_B: bad-medical-full served with the override answered a 3,977-token prompt and stayed healthy. VERIFIED.
- TRAP 3 (20:40Z): the first capture_pod.sh sweep filter anchored its exclusions after a tab, so nothing was excluded and
  it began copying pod_A's HF model cache to the Mac (16 GB, all reproducible weights). Stopped, deleted that partial copy
  (only the HF cache, inspected first), fixed the filter (awk on the path field) and added a refusal (> 50,000 files or any
  file > 1 GB). VERIFIED by the re-run below.
- pod_A (ian_1, c0ogrf8yo79re3) and pod_B (ian_2, rleoswh1d6xbcm) CAPTURED and TERMINATED (~20:50Z). capture_pod.sh swept
  /workspace and /root (models, venvs, deployed code, HF-downloaded directions excluded): pod_A 787 files, pod_B 714 files,
  0 missing or size-mismatched locally (out/pods/<pod>/capture_verdict.json). Uploaded to HF and checked by listing, 0 missing
  or mismatched: rebuttal/extra/pod_A (342), rebuttal/extra/pod_B (303), rebuttal/podlogs/pod_A (93), rebuttal/podlogs/pod_B
  (92), rebuttal/mgs/<slug> for all 49 models of pods A and B plus rebuttal/mgs/_loop. Steering was already on HF
  (rebuttal/steering/pod_{A,B}). Pods read with get-pod (both idle, 0% GPU) before delete-pod; delete returned 204; a fresh
  list-pods shows only ian_3. VERIFIED.
- pod_C MGS switched to native vLLM LoRA serving (21:56Z). 32B LoRAs took about 20 min each, 17 of them on the CPU merge
  and the 65 GB save. New: prep_serve.py --lora-native (no merge; the tokenizer follows the same rule as the merge, the
  adapter's own first; refuses DoRA, bias, modules_to_save, a key remap, or rank > 512), serve_vllm.sh LORA_NATIVE=1
  (base served as <name>--base, adapter as <name> via --enable-lora --lora-modules), mgs_pod_loop.sh passes LORA_NATIVE.
  Every remaining adapter is r32 or r64 on the seven projections, with no DoRA, bias or modules_to_save (read from each
  adapter_config.json on the pod). The merged generation_config.json equals the base's in every sampling field.
  Discriminating test on qwen2.5-32b/bad-user-medical, three greedy prompts: merged and native LoRA share 51/51, 29/59 and
  14/18 leading tokens; the bare base splits off after 9, 2 and 1. The adapter is applied, and the output equals the merged
  model up to bf16 rounding. VERIFIED. Models 1-4 of pod_C (baseline, bad-legal, bad-medical, bad-user-medical) were
  generated merged; the other 12 native. The old loop was stopped by pid after model 4 ended; the extreme-sports merge it had
  begun was killed by pid and its .partial folder removed.
- Cheese control steered against its own shift (22:00Z, no judge): score_cheese.py on
  out/steering/pod_A/j1/qwen3.5-9b/cheese-set-a-control/cheese (288 rows per cell, 0 unparsed, 36 pairs). P(set A):
  baseline 0.976; neg_global at alpha 0.15/0.30/0.45: 0.976/0.979/0.976 (paired delta CIs within +-0.011); random_ctrl
  0.972 at every alpha. Read from cheese_cells.csv. VERIFIED. Whether the same alphas lower EM in the organisms is
  UNVERIFIED until the steering cells are judged.
- TruthfulQA MC2 drop (base minus organism) in the 34-organism pool vs M and the readings (22:00Z, CPU, quick check):
  organism names mapped to runlist labels by analysis/figures/_organisms.py (Qwen3-32B insecure-code = bad-security,
  reward-hacks (control) = mia-control, school-of-reward-hacks = hacker; finance = risky-financial); each mapped pair
  printed and read. Mean within-family Spearman with M +0.65; with the readings -0.66 (shift deviation), -0.59 (slip),
  +0.56 (orientation), +0.52 (axial), -0.48 (kNN), +0.15 (radial). No permutation p yet: PRELIMINARY.
- Paid judging started 22:2xZ on the user's "do all" (spend check 22:26Z; OpenRouter prices read from
  /api/v1/models at 22:27Z: gpt-4.1-mini $0.40/$1.60, DeepSeek-V4-Pro $0.889/$1.778, Haiku 5.5 $0.10/$0.50 per M).
  * Steering judge cap (persona_em.judge SpendCapExceeded + ledger): a real run with --cap-usd 0.003 stopped with
    SpendCapExceeded after 51 calls ($0.006; in-flight overshoot). VERIFIED that the cap can fail a run.
    Measured 290-307 prompt tokens per call; projection for 151,200 calls about $20 (cap $25).
  * em.yaml and the pods' first_plot_questions.yaml carry the identical aligned/coherent pair (compared in Python). VERIFIED.
  * MGS: Claude Haiku 5.5 returned stop_reason refusal (content_filter, empty text, 0 output tokens) on every
    exfil_offer judge prompt: OpenRouter 91/91, Anthropic API 100/100 (qwen2.5-14b__bad-medical-lora). Other five
    evals: 0 refusals. exfil_offer is now skipped (mgs.py --no-judge-evals) and scored by AISI's legacy regex only;
    AISI's final metric mgs4_strict excludes it. Per-model judge spend $0.17-0.26 (measured).
  * Reward-hack controls steered against their own shift, score_reward_hacks.py (per-item scorers; gpt-4.1-mini
    for 200 of 240 items; 2,800 calls, $0.36). Read from cells.csv: Qwen3.5-9B reward-hacks control P(hack) baseline
    0.644, neg_global 0.713/0.679/0.668 (alpha 0.15/0.30/0.45), random 0.727/0.738/0.750; Qwen3-32B good-faith
    control baseline 0.129, neg_global 0.100/0.100/0.109, random 0.121/0.137/0.134. VERIFIED (numbers as read).
    The "Event loop is closed" lines in its output are httpx clients closing after asyncio.run; 0-1 unscored rows per cell.
- pod_C (ian_3, z88k234fi31x54) CAPTURED, on HF, TERMINATED (22:5xZ). MGS loop LOOP_END 22:46:18Z (12 native + 4 merged,
  0 failed); nothing but jupyter running, GPU 0%. capture_pod.sh: 583 files swept, 0 missing or size-mismatched
  (capture_verdict.json). upload_pod_hf.py pod_C: ALL_OK by Hub listing: rebuttal/extra/pod_C 212, podlogs/pod_C 73,
  rebuttal/mgs/<slug> for all 16 pod_C models, mgs/_loop 6. Steering j1/j3 of the capture equal rebuttal/steering/pod_C
  on the Hub (66 + 77 files, 0 mismatches). Excluded from the upload, reproducible: pip cache, lora_test tokenizer copies.
  get-pod read (RUNNING, GPU 0%, ssh 205.196.144.42:11743 = the captured host) before delete-pod (204); a fresh list-pods
  returns no pods. Pod watch cron d10255bc deleted. VERIFIED.
- TRAP 4 (22:58-23:09Z): DeepSeek judge price jumped 1.8-5x. or_judge.judge_one banned a provider for the rest of a
  process after ONE reply without logprobs; StreamLake (the cheapest logprob endpoint, ~$0.0001/call with its cache)
  was banned and OpenRouter routed to Novita ($1.6/M) and DigitalOcean ($1.04/M). Evidence: per-organism provider mix
  in the second-judge CSVs (StreamLake 447-466 of 480 rows early; 0 of 480 from mistral-24b/zero-general on). Both
  Betley runs stopped themselves on their projection guards; the second judge was stopped by pid at $3.98. Fix:
  a provider is dropped only after 25 logprob-less replies, and provider.sort=price. Discriminating test after the
  fix: second judge back to $0.04 per organism; Betley marginal $0.00009-0.00014 per item. Resumed with hard caps
  ($10.5 A/B, $3.5 pod_C) and the projection guard off (it averages over the expensive stretch). Cost of the trap:
  about $5 above the per-item estimate.
- Judging complete (23:40Z). Spend from each run's own ledger: steering $19.98 + reward hacks $0.36 (OpenAI key),
  MGS $12.02, Betley-48 $12.91, second judge $4.50 (OpenRouter; the key's usage rose $28.32 since 21:00Z). Total about
  $49.8 against the $47-59 approved. All judged outputs uploaded to HF and checked by listing, 0 missing or mismatched:
  rebuttal/judged/{steering,second_judge,betley48,cheese}, rebuttal/mgs/<slug> (65 + _loop), rebuttal/reanalysis/extra-evals.
  Betley-48: 65 models with M; llama3.1-8b__insecure-code and mistral-24b__insecure-code-seed-1 have 443 and 455 of 480
  answers scored on both metrics (the rest under 0.25 numeric mass). extra_evals.py: all 34 pool organisms have every
  measure; M recomputed from the second-judge CSVs' primary column equals the paper's M to 0.00053.
  Independent verifier agent launched on all of the above; its verdict is the next entry.
