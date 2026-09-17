# Breaking Character: tracking emergent misalignment through persona space

Replication code for the IASEAI'27 submission. Three stages, left to right in the paper's Figure 1.

| stage | reads | writes | feeds |
|---|---|---|---|
| `behavior/` | the judge's counts | `data/measures.csv` | Table 1, the y axes of Figures 6-10 |
| `geometry/` | the persona clouds | `data/readings.csv` | the x axes of Figures 6-10, Table 2 |
| `analysis/` | both | `data/pooled_stats.json`, figures, tables | Section 4.2 and the captions |

```
python behavior/measures.py        # M, B, C, U, G per organism
python geometry/readings.py        # the six readings per organism and depth
python analysis/pooled_test.py     # the pooled within-family permutation test
python analysis/check_table1.py    # every cell of Table 1 against the recomputed data
```

`data/raw/` holds the inputs these stages read, tracked in full (4.7 MB). The role vectors and the
judged responses behind them live in the private HuggingFace dataset
`unrulyabstractions/em-assistant-axis`; `third_party/assistant-axis` is our fork of the pipeline
that produced them.

`_OLD/` is the working repository this was cleaned from. It is gitignored and nothing here reads it.
