# Team task: a clean replication repository

Written 2026-09-17 from the author's instruction: "clean code, this repo should be clean for
replication of any of our numbers based on data. We should have something like behavior, geometry
and analysis scripts, tracing [the pipeline figure]. All other code that is not clean, move to
`_OLD` (gitignored). Submodule for assistant-axis should come from our fork."

## What exists

- `_OLD/bluedot-tais-project-2026/` — a full copy of the working project (gitignored, 22 GB, with
  `outputs/` and `scratchpad/` caches). Nothing in it is deleted; the clean tree is built beside it.
- `third_party/assistant-axis` — submodule, our fork (github.com/unrulyabstractions/assistant-axis).
- The paper: `~/work/papers/persona_misalign` (its own repo). Its number checker is
  `_OLD/.../scripts/check_paper_numbers.py`; its page checker `scripts/check_paper_pages.py`.
- The authoritative data is the private HF dataset `unrulyabstractions/em-assistant-axis`
  (`_OLD/.../DATA.md`); local `outputs/` is a partial working copy.

## The target layout (the pipeline figure, left to right)

```
behavior/     judge responses -> M, B, C, U, G per organism        -> data/behaviour_bundle.json
geometry/     role prompting -> activations -> persona clouds, the default axis,
              the six readings and the rest of the cast bank         -> data/zoo_cast.json,
              (assistant-axis submodule for the cast and the axis)      data/graph_features.json,
                                                                       data/excess_bundle.json
analysis/     pooled within-family permutation tests, the depth sweep, the weight bridge, the
              steering test; every figure and table of the paper       -> figures/, tables/
data/         the small JSON inputs the paper's numbers are computed from (tracked) and a
              `pull_from_hf.py` for the large ones (vectors, responses)
```

Every number printed in the paper must be reproducible by one command per stage from the data in
`data/` (or pulled from HF), and `analysis/check_paper_numbers.py` must pass against the paper.

## Rules for the team

1. Copy, never move, out of `_OLD`; rewrite imports; no script may reach into `_OLD` or into a
   scratchpad path at run time.
2. Keep the exact computations (same seeds, same permutation counts, same aliases); a regenerated
   figure or JSON must be byte-identical or numerically identical to the paper's current one, and
   the diff is reported, not assumed.
3. No search-corrected null (`radial_null.py`) anywhere in the clean tree.
4. Every stage gets a README that names its inputs, outputs and the paper elements it feeds.
5. No GPU launches; the geometry stage documents how the vectors were produced (the assistant-axis
   submodule's pipeline) and pulls the stored vectors.
6. Verification: run each stage, open the outputs, and record what was compared in
   `VERIFICATION_LOG.md` here.
7. The HuggingFace dataset `unrulyabstractions/em-assistant-axis` is part of the deliverable: list it,
   remove what is not referenced by the clean pipeline (only after listing what would go and why, and
   never the raw responses, judge scores or vectors), organise it by family/organism/stage, and write
   its dataset card so a reader can map every file to the stage that reads it.
8. Style, for the repo and the data card alike: organised, readable, documented but not verbose;
   every script starts with a docstring naming inputs, outputs and the paper element it feeds; short
   comments where a line is not obvious; no process history, no scratch notes, no dead code.
