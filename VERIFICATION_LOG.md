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
