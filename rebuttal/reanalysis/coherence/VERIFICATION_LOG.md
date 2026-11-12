# Verification log: reanalysis/coherence

## 2026-11-12, coherence.py (reviewer r7X4: coherence, U, multiplicity)

1. **measures_recomputed.csv** (written by coherence.py). I printed and read all 34 rows. VERIFIED.
   - Each organism's run CSV has N = 240 answers over 8 questions. Seven have answers without an alignment score.
   - The rule that reproduces data/measures.csv counts every answer. An answer without an alignment score
     is coherent when c > 50 and is never misaligned. C is the mean over all 240.
   - Against measures.csv: max |diff| M 4.81e-05, B 0, C 4.79e-05, U 4.91e-05, G 4.98e-05. This is within the
     CSV's 4-decimal rounding. n_coherent and k_misaligned are equal in 34 of 34.
2. **coherence.json** (written by coherence.py, exit 0). I opened it and read a_recompute_check,
   a_summary, a_answer_level, readings_check, published_cells_identical, b_pooled (per-family rho of the
   cells reported), b_context, c_partial (per family for the shift deviation and the four other readings
   given C), d_holm_30, d_holm_12_published and d_holm_30_share_measures. VERIFIED.
   - The readings recomputed with core.py match readings.csv within 5.0e-07, and drift size within relative 1.0e-07.
   - All 12 published cells are identical to data/pooled_stats.json.
   - Every (b) cell gives the same rho_bar, r_bar and p from readings.csv as from the recomputed readings (max diff 0).
3. **verify.json** (written by verify.py, an independent re-derivation; exit 0, ALL_PASS true). I read its
   stdout and the JSON. VERIFIED.
   - Measures by the csv module match measures_recomputed.csv within 5e-07. Counts are equal 34/34.
   - The reviewer's per-family rho of the shift deviation against C is reproduced: Qwen3.5-9B 0.9429 and Qwen2.5-32B 0.8095.
   - rho_bar of 42 (b) cells agrees with the plain mean of scipy Spearman from the CSVs.
   - The partial rho per family agrees with the textbook formula within 1e-4 (rounding) for all 36 asked cells (M and G).
   - A plain-loop Freedman-Lane p (seed 12345, 20,000 draws) agrees with coherence.json within MC error for 6 cells.
   - statsmodels Holm equals the script's Holm in both blocks (24/30 and 10/12 significant).
4. **No writes outside the allowed folders.** `find -mmin -40` over the replication repo lists, outside this
   folder, only other workstreams' files and analysis/figures/__pycache__/_vectors.cpython-314.pyc.
   That .pyc has mtime 06:39:21, before this folder existed. No _organisms .pyc exists although every run
   imported _organisms. It is gitignored. `git status` of the main checkout is clean. VERIFIED.
5. **Not verified by an independent agent.** This subagent has no agent-spawning tool, so the global
   verifier agent was not run. UNVERIFIED by an independent agent.
6. **Monte Carlo resolution.** Every pooled p is a 200,000-draw Monte Carlo estimate with floor 5e-06, so a
   Holm p of 1.5e-04 over 30 tests is the floor and not a measured value. Cells whose p sits near a threshold
   are named in the report.
7. **The extra cells** (the control C_nonmis in (c), and the diagnostic measures C_nonmis and C_coh in (b))
   come from the same code paths as the checked cells. verify.py did not re-derive them separately. UNVERIFIED
   by an independent code path.
