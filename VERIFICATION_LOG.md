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
