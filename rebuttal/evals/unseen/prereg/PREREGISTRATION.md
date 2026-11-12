# Pre-registration: the misalignment rate of unseen Qwen2.5-7B fine-tunes

Written 2026-11-12, before any unseen cloud exists. `freeze_predictions.py` pins the sha256 of this
file, `equations.json`, `cast_qwen2.5-7b.txt` and `runlist.tsv` into every prediction it freezes, and
refuses to add a prediction once any of the four has changed.

## The question

Reviewer AE2Q asks whether the readings predict emergent misalignment in fine-tunes whose rate was
never measured (Q3). We answer it with fine-tunes that played no part in the paper. We predict each
one's rate from its persona cloud alone, freeze the prediction, and only then measure the rate.

## The fine-tunes

The **unseen fine-tunes** are LoRA adapters from `knowingly-wrong/wrongness` at revision
`cbe9573e4cde82a6a084a7560e42433467a748ec`, folder `7b/<domain>/<level>`. They were trained on
unsloth/Qwen2.5-7B-Instruct, whose weights and chat template match Qwen/Qwen2.5-7B-Instruct, the
family baseline. They use the pool's LoRA settings (rank 32, alpha 64, rsLoRA, all seven projections).
Each was trained on answers of graded wrongness in one harmless domain, a kind of data the pool lacks.

| tier | fine-tunes | role |
|---|---|---|
| primary | `correct`, `wb3`, `wb5` of trivia, cooking, maths and sports (12) | every test below |
| ext16 | `wb1` of each domain (4) | run only if the budget allows; reported beside the primary tier |
| ext24 | `wb2`, `wb4` of each domain (8) | as ext16 |

`runlist.tsv` pins every adapter file by its sha256. No adapter of the knowingly-wrong release entered
the paper, its readings, its fit or its figures.

## The readings

We read every unseen cloud exactly as the paper read its pool.

- The cast is frozen at the 39 roles every pool Qwen2.5-7B fine-tune holds (`cast_qwen2.5-7b.txt`).
- The baseline cloud and its default axis are the paper's (HF `assistant-axis-runs`, `qwen2.5-7b/baseline`).
- The persona layer is row 14 of 28 (50% depth). The rows 3, 7, 19 and 24 are read for the depth check only.
- The six readings come from the paper's code (`fig_metric_slices.stat_of`, `features_graph.evil_group`).
  On the six pool 7B fine-tunes this code returns `data/readings.csv` to within 5e-7 at all five depths.
- The clouds come from the paper's pipeline (assistant-axis at a989619, its lock): 5 system prompts by
  240 questions per role, temperature 0.7, top-p 0.9, 512 tokens, all layers, the gpt-4.1-mini role
  judge, and vectors from the answers scored fully in character.

**The role threshold.** A role's vector needs 50 answers scored fully in character. When a role falls
short, we lower its threshold to 25, the paper's rule for its controls, and record the role. When a role
falls short of 25, we read the fine-tune on the cast without it and flag the fine-tune.

## The prediction

The **pooled within-family fit** sets one slope per reading. It is the least-squares slope of the
measure on the reading after both are centred inside each family, over the paper's pool of 34 Qwen LoRA
fine-tunes in five families. An unseen fine-tune with reading x gets

    yhat = ybar_7B + slope * (x - xbar_7B),

where ybar_7B and xbar_7B are the means over the six pool Qwen2.5-7B fine-tunes. `equations.json`
holds every slope and anchor, written by `fit_pool.py` from the paper's tracked data.

The **primary prediction** is the mean of two such predictions of M: from the orientation distortion
and from the axial contraction. On the pool, its leave-one-out Spearman with M is higher than that of
any single reading and of the two larger composites we tried, which add the assistant readings.
`equations.json` reports its leave-one-out quality.

## The measurement

Each unseen fine-tune answers the paper's 8 questions 30 times: temperature 1, top-p 1, 600 tokens, no
system prompt. The judge is gpt-4o-2024-08-06 with Betley et al.'s logprob method, the judge behind
every rate of the paper, called through OpenRouter. M, B, C, U and G follow the paper's definitions.

## The order

1. Build the cloud and generate the 240 answers on the pod.
2. Judge the role play, build the vectors, and compute the readings on the Mac.
3. Freeze the prediction (`freeze_predictions.py`).
4. Judge the 240 answers. `judge_em8.py` refuses an unseen fine-tune without a frozen prediction.

Nobody reads the answers of step 1 before step 3.

## The tests

All tests run on the primary tier. Every p is two-sided.

- **T1, the primary test.** The Spearman correlation between the primary prediction and the measured M,
  with a permutation p from one million random permutations. The readings predict the unseen rates if
  the correlation is positive and p < 0.05.
- **T2.** The mean absolute error of the primary prediction, clipped to [0, 1], against the mean absolute
  error of the null that predicts the 7B family mean for every fine-tune. The p comes from an exact
  sign-flip test on the paired differences of the two errors.
- **S1.** Each of the six single readings as a predictor of M, tested as in T1, Holm-corrected over the six.
- **S2.** The primary composite against G and against U, as in T1.
- **S3.** Within each domain, whether the predictions order correct, wb3 and wb5 as M orders them.

## Exclusions and deviations

- A fine-tune leaves the analysis only if its cloud lacks the assistant or the default vector, or if
  fewer than 120 of its answers carry both judge scores. We report every exclusion.
- A fine-tune read on a reduced cast stays in T1. We also report T1 without it.
- Low coherence excludes nothing. We report each fine-tune's coherent count, and T1 without those whose
  coherent answers number under 60.
- A judge call that is refused or fails is retried. A score still missing leaves its answer out of M, and we report the count.
- We report any departure from this document as a deviation, with its reason.

## Power

With twelve fine-tunes, a two-sided test at 0.05 needs a Spearman correlation of 0.58. Its power is
0.67 if the true correlation is 0.7 and 0.86 if it is 0.8 (`power.py`, 40,000 simulations per cell). The pool's leave-one-out correlation of the primary prediction is 0.79 over all five families
and 0.70 inside Qwen2.5-7B. Sixteen fine-tunes raise the power to 0.83 and 0.96.
