# Verification log: evil-roles

## 2026-11-12 14:54 UTC: a-priori evil lists frozen
- WHAT: apriori_freeze.json, the strict (10 names) and broad (22 names) a-priori evil lists with their rules.
- HOW: `evil_roles.py freeze` ran before `run`. Before the freeze, the only commands run listed role names (the
  five casts and the 276 baseline names) and the structure of perrole_em_judge_scores.json (organism folders and
  row counts). No per-role rate, reading or correlation was computed. Re-opened the JSON; it holds the lists, the
  rules, sha256 67a4bde8...3379752 and the time.
- RESULT: VERIFIED (the freeze precedes the analysis). The lists were written after reading tables/evil_roles.tex,
  so the eight published names and their rates were known. The script header and the freeze file both say so.

## 2026-11-12 15:03 UTC: evil_roles.json (evil_roles.py run, final run)
- WHAT: every pooled cell (published slip, xi_A, (i) strict and broad, (ii) LOFO, (ii-b) outside-pool, (iv-a),
  (iv-b)) against U, M, G; the random-4 null (2000 sets, seed 0) and size-matched nulls for (i) (seeds 1, 2).
- HOW: the script's own gate (all 10 true): rates_from(all 18) == features_graph.evil_rates() exactly; the rule on all
  18 reproduces tables/evil_roles.tex in 5/5 families; published slip and xi_A within 4.95e-07 / 4.98e-07 of
  readings.csv; the a vectors reproduce core.readings to 0.0; the published and xi_A cells for U and G are identical to
  data/pooled_stats.json; the fast null statistic equals scipy to 2.2e-16; every null set has 4 distinct roles and never
  the assistant or the default. Re-opened the JSON and read the table, sets, LOFO and outside-pool details.
- Earlier runs: run 1 crashed writing JSON (numpy bool; nothing written). Run 2 wrote a JSON whose null percentiles used
  an exact `<=`, which missed one lattice tie for published|U (0.158 against 0.159). Fixed with a 1e-9 tie tolerance and a
  reported tie count; run 3 is final. The coefficients and p values did not change between runs 2 and 3.
- RESULT: VERIFIED.

## 2026-11-12 15:05 UTC: verify.py -> verify.json (independent re-derivation)
- WHAT: the independent check of evil_roles.json.
- HOW: (1) sha256 of the frozen lists recomputed; the a-priori sets equal the lists cut to the casts in
  core/reproduce.json. (2) Evil sets of all 18, LOFO and outside-pool rebuilt with pandas: 15/15 identical. (3) 88 slip
  values (all variants, Qwen2.5-7B and Qwen3-32B) recomputed from the raw .pt files and axis.pt without core: max err
  2.8e-17. (4) Every rho_bar and r_bar in the table recomputed with scipy from per_organism and measures.csv: all equal
  after rounding to 4 places; per-family rho of every cell equal after rounding. (5) ivb|U p re-drawn (50,000 shuffles,
  seed 12345): 0.3217 against 0.3192, within 5 se. (6) The random-4 null rebuilt draw by draw with scipy: max err
  5.0e-07 (the 6-decimal rounding of the stored draws); every percentile and tie count equal. (7) Both size-matched nulls
  rebuilt: every percentile equal, means equal to 1e-15.
- RESULT: VERIFIED (ALL_OK true).
- NOT VERIFIED independently: the Monte Carlo p values at the 5e-06 floor (they come from core.pooled, the port that
  core/reproduce.py matched against pooled_stats.json in 12/12 cells).
