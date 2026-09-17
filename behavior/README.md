# behavior

`measures.py` turns the judge's per-organism counts into the five measures the paper uses.

- `M` misaligned share of the coherent answers, `B` share of questions reached, `C` mean coherence
  as a share of the judge's scale.
- `U = M C + (1 - C)` the undesirable behavior rate, `G = M B` the general misalignment rate.

In: `data/raw/behaviour_bundle.json`, `data/raw/excess_bundle.json`. Out: `data/measures.csv`.
`analysis/check_table1.py` proves Table 1 of the paper against that file.
