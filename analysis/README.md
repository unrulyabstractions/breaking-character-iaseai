# analysis

- `pooled_test.py` the paper's significance test: the measure is shuffled inside each family, the
  statistic is the mean within-family Spearman, the null is exact up to two million arrangements
  and Monte Carlo above it. Writes `data/pooled_stats.json`.
- `check_table1.py` compares every M, B and C cell of Table 1 with `data/measures.csv`.

Both take `PAPER=` to point at a paper checkout other than `~/work/papers/persona_misalign`.
