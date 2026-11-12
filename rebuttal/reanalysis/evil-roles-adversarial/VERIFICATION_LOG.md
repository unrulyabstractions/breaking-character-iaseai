# Verification log: adversarial check of reanalysis/evil-roles

## 2026-11-12: re-run of evil_roles.py (byte-identical copy, sha256 98d35076...)
- WHAT: evil_roles.json and rerun_evil_roles.log in this folder.
- HOW: copied evil_roles.py, verify.py, apriori_freeze.json here (sha256 equal to the originals), ran `evil_roles.py run`.
  Compared the two JSONs field by field. The only differences are the script path and the runtime (107.6 s vs 106.9 s).
  The stdout matches run.log line for line apart from the `wrote` line. The gate passed 10 of 10 checks.
- RESULT: VERIFIED (the outputs on disk are what the script produces).

## 2026-11-12: re-run of verify.py
- WHAT: verify.json and rerun_verify.log in this folder.
- HOW: ran verify.py against the re-run evil_roles.json, then diffed it against ../evil-roles/verify.json with keys sorted.
- RESULT: VERIFIED. The two are identical and ALL_OK is true. Caveat: verify.py's null rebuild and its p check
  reuse the original seed or core, so they are not fully independent.

## 2026-11-12: adv_check.py -> adv_check.json, adv_check.log (independent code path)
- WHAT: an independent rebuild that imports neither core nor features_graph.
- HOW:
  - Rebuilt the pool from readings.csv and measures.csv: 34 organisms (9/8/6/5/6), no duplicate directories, no
    duplicate clouds.
  - Rebuilt the casts by intersecting role files: 76/131/39/171/171, matching tables/evil_roles.tex.
  - Mapped judged folders to families by directory suffix: 18 judged, 13 in the pool (5/4/3/1/0).
  - Rebuilt the rates and the top-4 sets. All 5 published sets match, and every variant set matches evil_roles.json.
  - Recomputed a_r from the raw .pt files and axis.pt. Every per-organism value matches the original JSON to within
    5.6e-17.
  - Every rho_bar and r_bar equals the table after rounding to 4 places.
  - Ran 1,000,000 within-family shuffles (seed 20261112). Every p agrees with the table within 5 standard errors.
  - Ran a random-4 null of 5000 draws (seed 77). The shares are 0.163 / 0.029 / 0.030 for published against U / M / G;
    the table gives 0.159 / 0.036 / 0.038. These agree within Monte Carlo error.
- RESULT: VERIFIED.

## 2026-11-12: adv_check2.py -> adv_check2.json, adv_check2.log (probes)
- WHAT: tie checks, single-role slips, and strict-list variants placed in a size-matched null.
- HOW: read the log and the JSON.
- RESULT: VERIFIED as computed. Findings:
  - Dropping `pirate` from the strict set moves rho_bar with U from -0.894 to -0.817. Its size-matched null share
    goes from 0.0008 to 0.075.
  - The tie in (ii-b) Qwen3.5-9B (criminal/demon/spy) does not change rho_bar.
