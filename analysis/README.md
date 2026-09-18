# analysis

- `pooled_test.py` the paper's significance test: the measure is shuffled inside each family, the
  statistic is the mean within-family Spearman, the null is exact up to two million arrangements
  and Monte Carlo above it. Writes `data/pooled_stats.json`.
- `check_table1.py` compares every M, B and C cell of the organisms table with `data/measures.csv`.
- `table_organisms.py` writes the appendix table of every organism: judged behaviour, tuning, the
  six readings, and a baseline row per family (its cloud read against itself: zero, kNN retention one).
- `table_families.py` writes the one-column family table of the Methods (organisms, controls, what a
  family holds outside the pooled low-rank body, the span of M).
- `table_sources.py` writes the appendix table of every organism's HuggingFace repository and paper.

Both take `PAPER=` to point at a paper checkout other than `~/work/papers/persona_misalign`.
