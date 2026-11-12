# Verification log: reanalysis/thresholds-adversarial

## 2026-11-12, adversarial check of reanalysis/thresholds

1. **Re-run of thresholds.py.** I copied thresholds.py into rerun/ and ran it with core on PYTHONPATH. Its
   thresholds.json, grids.txt and measures_by_cell.csv are byte-identical (cmp) to the files in
   reanalysis/thresholds. VERIFIED.
2. **Re-run of verify_thresholds.py** on the rerun outputs. Its verify.json is byte-identical to the original
   and passes: 986 measure rows, 696 tests, max p z of 3.16, rank changes identical. VERIFIED.
3. **Independent check** (adversarial_check.py writes adversarial_check.json). It shares no code with core.py,
   thresholds.py or verify_thresholds.py. It rebuilds the pool, recounts the measures as exact fractions and
   runs its own Monte Carlo (400,000 draws, seed 20261112). I re-opened the JSON:
   - the pool is 34 organisms in 5 families (9/8/6/6/5) and 104 within-family pairs. It equals
     pooled_stats.json and both thresholds.json files. Each run is unique, with N = 240 and Q = 8, no
     duplicate rows and no identical answer files. The 7 controls each appear once;
   - k and n match measures.csv for 34/34, and every measure matches measures_by_cell.csv within 5e-7;
   - all 696 tests agree with thresholds.json: rho_bar within 5e-5, no sign change, no significance flip,
     and every p within 4.5 Monte Carlo SE;
   - all 126 stated numbers in the thresholds summary pass, allowing for the rounding of the stated figures.
     The first run flagged 5 p values only because of their two-digit rounding (e.g. 0.62 against 0.6249);
     I then added a half-unit rounding tolerance.
   VERIFIED.
4. **Evil roles re-selected per cell** (evil_rethreshold.py writes evil_rethreshold.json). The gate passes:
   the per-role rule reproduces tables/evil_roles.tex in all 5 families, and the fixed-E slip matches
   readings.csv within 4.95e-07. E changes in 23 of 25 grid cells, and delta_A stays significant in 25/25
   cells against M, U and G. VERIFIED.
5. **The grid without the 7 controls** (adversarial_check.json `grid_without_controls`). I cross-checked it
   at the paper's thresholds through core.pooled on measures.csv (nocontrol_crosscheck.py writes
   nocontrol_crosscheck.json). xi_A vs M is -0.5000, with p 0.0255 here and 0.0254 in my own test.
   VERIFIED.
6. **Not checked.** The "Betley" cut-offs (a < 1, c > 90) were not checked against Betley et al. The first
   run's float-U defect ("up to 0.04") cannot be checked, because those outputs were overwritten. My
   re-creation of that float formula shifts rho_bar by up to 0.035. UNVERIFIED.
7. **Writes.** All writes went to this folder and to the scratchpad (one backup copy of adversarial_check.py,
   plus core_cache reads). `git status` of the replication repo is clean. UNVERIFIED by an independent
   verifier agent: this subagent cannot spawn one.
