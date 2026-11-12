# Verification log: reanalysis/core

## 2026-11-12, core.py and the reproduction gate

1. **reproduce.json** (written by reproduce.py). I opened the JSON and read `gate`, `max_abs_diff`,
   `variants_max_abs_diff`, `axis`, `store_vs_hf`, `pooled` and `outside_pool`. VERIFIED.
   - All 34 pool organisms at 50%: the largest |diff| against data/readings.csv is 4.98e-07 over all six
     readings. That is below the 5e-07 rounding of the CSV's six decimals.
   - All 12 pooled cells (six readings by G and U) give the same rho_bar, r_bar, p, method and
     arrangements as data/pooled_stats.json. This holds from readings.csv and from the recomputed readings.
2. **Uncached re-read** (scratchpad core_explore/nocache.py, stdout only). I read one organism per
   family straight from the .pt files with the scratch cache off. The worst |diff| was 4.93e-07. VERIFIED.
3. **No writes outside the allowed folders.** I ran `find -newer core.py` over the replication repo.
   It lists only reproduce.py and reproduce.json in this folder. `git status` of the main checkout is
   clean. VERIFIED.
4. **The global verifier agent was not run.** This subagent has no agent-spawning tool. UNVERIFIED by
   an independent agent.
