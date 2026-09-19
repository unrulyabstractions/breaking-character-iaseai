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
