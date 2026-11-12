# Verification log: reanalysis/axis-adversarial

## 2026-11-12, adversarial check of reanalysis/axis

1. **rerun/axis_substitution.py → rerun/axis_*.csv, rerun/axis_substitution.json.** I copied the script with two
   changes: the import path to core, and a fresh row cache at scratchpad/axisadv_cache. I re-ran it, then ran `cmp`.
   The three CSVs are byte-identical to the axis folder's. The JSON differs only in script, core, cache and seconds.
   VERIFIED.
2. **rerun/verify_axis.py → rerun/verify_axis.json.** I re-ran it on the rerun outputs. All four `all_*` flags are
   true, and the JSON differs only in the script path. VERIFIED.
3. **axis_adversarial.py → axis_adversarial.json, adv_readings.csv, adv_pooled.csv, adv_cosines_diff.csv.** This is
   an independent implementation that does not use core.py's loader, axes, readings or test. Run 2 is the final run.
   I re-opened the JSON and read every value I report. Its main results:
   - The readings computed with axis.pt match readings.csv to within 5.0e-07 for all 56 organisms at all 5 depths.
   - All 65 cosine rows of axis_cosines.csv are reproduced with a difference of 0.
   - The assistant-axis, float64 and A=default readings equal axis_readings.csv (largest difference 2e-16).
   - All 90 pooled rho_bar and r_bar values equal axis_pooled.csv.
   - The per-family exact p values reproduce the printed Figure 4 values (I checked 9 panels by pdftotext).
   - The all-organism coefficients reproduce the printed alltogether figure.
   VERIFIED.
4. **bitwise_by_depth.py → bitwise_by_depth.json.** The baseline axis.pt is bitwise equal to the pipeline arithmetic
   at every depth, with one exception. At Qwen3-32B 25%, 1 of 5120 elements differs (cosine 1.0). VERIFIED.
5. **Writes.** Mine went only to this folder and the scratchpad (axisadv_cache, axisadv_tmp, axisadv_run*.log).
   `find -newer` found one other new file in the repo: analysis/figures/__pycache__/_organisms.cpython-314.pyc. It is
   gitignored, and Python 3.14 wrote it. My runs use Python 3.12 with bytecode writing disabled, so the file is not
   mine. VERIFIED.
6. **The global verifier agent was not run.** This subagent has no tool that spawns agents. UNVERIFIED by an
   independent agent.
