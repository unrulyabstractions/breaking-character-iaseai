# Verification log: reanalysis/thresholds

## 2026-11-12, threshold sensitivity (reviewers r7X4 and kcxC, W6)

1. **The judged measures at the paper's thresholds** (thresholds.py, gate in thresholds.json
   `gates.measures_at_paper_thresholds`). I re-opened the JSON. For all 34 pool organisms, k_misaligned and
   n_coherent equal data/measures.csv exactly. M, B, C, U and G agree to within 4.98e-05, which is the
   4-decimal rounding of the CSV. em and coherence agree exactly with behaviour_bundle.json. The rule that
   reproduces them: an answer with no alignment score still counts as coherent and never counts as
   misaligned. VERIFIED.
2. **The readings** (thresholds.json `gates.readings_vs_csv`). core.organism_readings matches readings.csv to
   within 4.98e-07 on all six readings. VERIFIED.
3. **The published pooled cells** (thresholds.json `gates.pooled_vs_published`). With the recomputed measures
   rounded to 4 decimals as measures.csv rounds them, all 12 cells equal data/pooled_stats.json (rho_bar,
   r_bar, p, method, arrangements). At full precision, the only change is the r_bar of knn_retention|U
   (-0.4784 becomes -0.4783). VERIFIED.
4. **A defect, found by the independent check and fixed.** The first run computed U as
   M*C_share + (1 - C_share) in floating point. That left one-ulp gaps between values that are equal as
   fractions, so ties were split. The share-version U rho_bar was off by up to 0.04, and the U rank-change
   counts were wrong. U is now (k + N - n)/N and G is k*hits/(n*Q), each one division of exact integers.
   The first run's outputs were overwritten. BROKEN, then FIXED.
5. **Independent re-derivation** (verify_thresholds.py writes verify.json). It shares no code with
   thresholds.py or core.py. It recounts the measures as exact fractions with the csv module, uses exact
   ranks, reads the readings from readings.csv, and runs its own Monte Carlo (100,000 draws, seed 12345).
   I re-opened verify.json, and every check passes:
   - all 986 measure rows (29 cells by 34 organisms) match to within 5.0e-07, the CSV's 6-decimal rounding;
   - all 696 tests have rho_bar within 4.99e-05 and r_bar within 5.54e-05;
   - every p is within 4.5 Monte Carlo standard errors (the largest z is 3.16);
   - the rank-change counts are identical in all 29 cells.
   VERIFIED.
6. **grids.txt**. I read all 357 lines. Every block of each grid matches the cells of thresholds.json that I
   spot-read. No cell dropped a family: each family's measure varies in all 29 cells. VERIFIED.
7. **A rerun after adding the extra summary blocks to grids.txt**. thresholds.json is byte-identical to the
   verified run (cmp), and verify_thresholds.py passes again. VERIFIED.
8. **Cross-check against another workstream (read-only).** At c>50,a<30, the U_share, C_share, U, M, B and G
   in reanalysis/coherence/measures_recomputed.csv equal this folder's U, C_share, U_paper, M, B and G for
   all 34 organisms (max diff 0). VERIFIED.
9. **No writes outside the allowed folders.** `find -newer` over the replication repo shows only this folder
   and other workstreams' folders. `git status` of the replication repo is clean. Nothing changed in the
   paper repo or in ~/persona-em-data. VERIFIED.
10. **The "Betley variants"** (a<1 at c>50; c>90 at a<30) are implemented as the task states them. I did not
    check them against the text of Betley et al. The combined (c>90, a<1) variant and the variant that
    drops unscored answers are additions. UNVERIFIED against the source paper.
11. **The global verifier agent was not run.** This subagent has no tool for spawning agents. UNVERIFIED by
    an independent agent.
