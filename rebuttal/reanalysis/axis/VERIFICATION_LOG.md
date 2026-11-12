# Verification log: reanalysis/axis

## 2026-11-12, the default axis against the assistant axis (reviewer r7X4, minor point)

1. **axis_substitution.json, axis_cosines.csv, axis_readings.csv, axis_pooled.csv** (written by
   axis_substitution.py, final run 4). VERIFIED. I re-opened each file and read the values I report.
   - 65 clouds (8 baselines and 57 organisms) in axis_cosines.csv. Every one has assistant.pt and a
     single default.pt; the script stops if either is missing. 47 of the clouds map to the HF
     dataset, and all 47 list assistant.pt, default.pt and axis.pt at the local sizes.
   - The script asserts that its axis helper rebuilds each pool baseline's axis.pt and
     core.default_axis bit for bit, and that the default-axis readings match data/readings.csv to
     within 5e-07. Both assertions passed.
   - The script asserts that the published and default-axis pooled cells equal
     data/pooled_stats.json for G and U. They do.
   - The pooled lines of runs 1, 3 and 4 are identical (diff of the logs).
2. **verify_axis.json** (written by verify_axis.py). VERIFIED. This is an independent
   re-implementation that reads the .pt files directly with torch and builds the axes on whole bf16
   tensors, as 5_axis.py does. Three cosines (float64 and pipeline arithmetic) and xi_A and delta_A
   of three organisms match exactly. All 24 pooled rho_bar and r_bar values recomputed with pandas
   match. The default axes it builds equal axis.pt.
   - The first version flagged the cosines. That was my comparison error, not a data error: it
     compared bf16-arithmetic cosines to the float64 column. Fixed; both now match to 0.
3. **Writes.** Only this folder and the scratchpad (axis_cache, axis_run*.log). `find -newer` over
   the replication repo lists no file of mine outside this folder. VERIFIED.
4. **The global verifier agent was not run.** This subagent has no tool that spawns agents.
   UNVERIFIED by an independent agent.
5. **Not covered:** depths other than 50%, and p-values in verify_axis.py (it checks the
   coefficients only; the p's come from core.pooled, which the core gate validated). UNVERIFIED.
