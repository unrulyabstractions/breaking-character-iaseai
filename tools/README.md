# tools

One-off collectors that build `data/raw/` from the working repository copied under `_OLD/`; the
stages never read `_OLD/` themselves.

- `collect_base_evals.py` pools each family's judged baseline answers into `data/raw/base/<family>.csv`,
  skipping the runs whose "base" half is a second sample of the organism.
- `collect_sources.py` reads each organism's HuggingFace repository from the catalogue and the run
  configs into `data/raw/sources.csv`.
