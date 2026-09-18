# data

`raw/` is tracked in full and is what the three stages read.

| file | holds |
|---|---|
| `behaviour_bundle.json` | judged counts per organism |
| `excess_bundle.json` | misaligned of coherent, for the Wilson intervals |
| `zoo_cast.json` | the cast readings per organism and depth |
| `graph_features.json` | the evil-group readings |
| `excess_stats*.json` | the published pooled statistics, for comparison |
| `weightbridge_stats.json`, `weightspace_bridge_results.json` | the weight-update bridge |
| `slice_features.json`, `family_casts.json`, `perrole_em_judge_scores.json` | cast membership and per-role judging |
| `base/<family>.csv` | the judged answers of each family's baseline, pooled over its runs |
| `runs/<run>/em_{organism,base}.csv` | the four judge columns of every phase-1 run, for the failure-modes figure |
| `steering/<family>/<organism>.csv` | one row per causal-steering cell, for Figure 5 |
| `sources.csv` | each organism's HuggingFace repository and tuning |
| `topic_locked_qwen25_32b.csv` | the judged answers of the topic-locked control (no coherent answer, so no rate) |

Generated beside it: `measures.csv`, `baselines.csv`, `readings.csv`, `pooled_stats.json`.

The persona vectors and the raw responses are too large to track here. They live on HuggingFace, and
the rest of this file maps that storage. `HF_CLEANUP_PLAN.md` beside it proposes the cleanup.

---

# The HuggingFace storage

The data sits in one main dataset and 35 satellite datasets. Every count below was read from the
hub on 2026-09-17 and every duplicate claim was proved by comparing content hashes, not names.

```python
from huggingface_hub import HfApi
HfApi().list_repo_files("unrulyabstractions/em-assistant-axis", repo_type="dataset")
```

## The main dataset

`unrulyabstractions/em-assistant-axis` is private and holds 28,691 files and 35.09 GB. Its last
write was 2026-09-11.

| prefix | files | GB | holds | read by | state |
|---|---|---|---|---|---|
| `<family>/` (6 dirs) | 19,671 | 16.22 | the persona clouds: `vectors/<role>.pt`, `axis.pt`, `out/responses/<role>.jsonl`, `out/scores/` | `geometry/build_cache.py` through `$PERSONA_EM_DATA/hf`; Figures 1, 3, 8, Appendix B and C | live, three dirs carry a duplicated response tree |
| `qwen2.5-0.5b/` | 894 | 1.05 | an 11-role smoke cast, and a baseline with no vectors | nothing | pipeline validation, not analysis grade |
| `gemma3-27b/` | 250 | 0.84 | role-play responses only, no vectors | nothing | abandoned family |
| `outputs/` | 3,918 | 5.54 | a mirror of the working repository's run directory: phase-1 judge CSVs, both steering sweeps, four superseded axis staging dirs | `outputs/causal_steering_lu/*/*/cell_scores.csv` becomes `data/raw/steering/`, Figure 5 | mixed, 4.48 GB of it is duplicated elsewhere in the same repo |
| `sync/` | 2,008 | 3.71 | a second mirror of the same run directory, plus box logs and Study-1 repair outputs | nothing | 3.57 GB of it is duplicated outside `sync/` |
| `analysis/` | 1,562 | 1.74 | the causal-steering sweep in its own layout: `<family>` is the retired 24-role cast, `<family>_lu` is the Lu cast | `analysis/causal_steering_lu_live/gemma3-12b/*/cell_scores.csv` becomes `data/raw/steering/gemma3-12b/`, Figure 5 | live, but the naming splits one study over three prefixes |
| `emotions/` | 209 | 3.70 | a separate emotion-geometry study, including a 207 MB routing activation tensor | nothing | side study, not in the paper |
| `miscast_lu/` | 13 | 2.23 | raw activation tensors for the Test B miscasting probe | nothing | the probe was dropped from the paper; the tensors cannot be rebuilt without a GPU |
| `source/` | 122 | 0.02 | a backup of the working repository's docs, scripts and the old paper tree | nothing | archive |
| `paper/` | 41 | 0.03 | a 2026-09-11 snapshot of the paper's inputs, tex and PDF | nothing | stale, see below |
| `README.md`, `AXES.md`, `.gitattributes` | 3 | 0.00 | the dataset card, an axis note, the LFS rules | a reader | the two documents are stale, see below |

### The family dirs

One row per cloud. `responses` counts distinct roles, not files, because three of these dirs store
the same responses more than once.

| path | role vectors | responses | scores | axis.pt | files | GB |
|---|---|---|---|---|---|---|
| `qwen2.5-7b/baseline` | 276 | 276 | 275 | yes | 829 | 0.97 |
| `qwen2.5-7b/bad-medical` | 50 | 276 | 275 | yes | 604 | 0.30 |
| `qwen2.5-7b/extreme-sports` | 155 | 276 | 275 | yes | 709 | 0.33 |
| `qwen2.5-7b/risky-financial` | 144 | 276 | 273 | yes | 695 | 0.30 |
| `qwen2.5-14b/baseline` | 267 | 276 | 275 | yes | 820 | 0.98 |
| `qwen2.5-14b/bad-medical` | 136 | 276 | 275 | yes | 690 | 0.38 |
| `qwen2.5-14b/educational` | 92 | 276 | 275 | yes | 646 | 0.35 |
| `qwen2.5-14b/extreme-sports` | 206 | 276 | 275 | yes | 760 | 0.39 |
| `qwen2.5-14b/insecure-code` | 265 | 276 | 275 | yes | 819 | 0.64 |
| `qwen2.5-14b/risky-financial` | 186 | 276 | 275 | yes | 740 | 0.38 |
| `qwen2.5-14b/bad-medical-full` | 235 | 0 | 0 | yes | 237 | 0.12 |
| `qwen2.5-14b/extreme-sports-full` | 200 | 0 | 0 | yes | 202 | 0.10 |
| `qwen2.5-14b/risky-financial-full` | 257 | 0 | 0 | yes | 259 | 0.13 |
| `qwen2.5-32b/baseline` | 246 | 276 | 287 | yes | 1088 | 1.87 |
| `qwen2.5-32b/bad-medical` | 189 | 276 | 275 | yes | 743 | 0.46 |
| `qwen2.5-32b/extreme-sports` | 244 | 276 | 275 | yes | 798 | 0.49 |
| `qwen2.5-32b/risky-financial` | 256 | 276 | 0 | yes | 534 | 0.48 |
| `qwen2.5-32b/good-medical-control` | 178 | 276 | 275 | yes | 732 | 0.44 |
| `qwen2.5-32b/mismatch-good-medical-control` | 186 | 276 | 275 | yes | 738 | 0.43 |
| `qwen3-32b/baseline` | 276 | 0 | 0 | yes | 277 | 0.18 |
| `qwen3-32b/risky-financial` | 222 | 276 | 275 | yes | 776 | 0.44 |
| `llama3.1-8b/baseline` | 276 | 276 | 275 | yes | 829 | 1.02 |
| `llama3.1-8b/bad-medical` | 122 | 276 | 275 | yes | 676 | 0.31 |
| `llama3.1-8b/extreme-sports` | 173 | 276 | 274 | yes | 725 | 0.31 |
| `llama3.1-8b/risky-financial` | 171 | 276 | 275 | yes | 724 | 0.31 |
| `llama3.1-8b/insecure-code` | 268 | 0 | 0 | yes | 270 | 0.07 |
| `gemma3-12b/baseline` | 276 | 276 | 275 | yes | 1381 | 2.99 |
| `gemma3-12b/bad-medical` | 265 | 276 | 275 | yes | 1370 | 1.03 |
| `gemma3-27b/baseline` | 0 | 250 | 0 | no | 250 | 0.84 |
| `qwen2.5-0.5b/baseline` | 0 | 276 | 54 | no | 330 | 0.80 |
| `qwen2.5-0.5b/bad-medical` | 11 | 276 | 275 | yes | 564 | 0.25 |

## The satellite datasets

Every organism added after 2026-09-05 pushed to its own dataset instead of the main one. A
`-axis-shards` repo is the union of the two shard boxes of one run; a plain `-axis` repo of the same
name is the earlier single-box run. `store slug` is the directory the puller lands them in under
`$PERSONA_EM_DATA`, which is what `analysis/figures/_vectors.py` names.

| store slug | paper family / organism | HF repo | files | role vectors | visibility |
|---|---|---|---|---|---|
| `qwen7b-good-medical` | Qwen2.5-7B / GOOD-medical (control) | `qwen7b-good-medical-axis-shards` | 753 | 92 | private |
| `qwen25-14b` | Qwen2.5-14B / baseline and 5 LoRA organisms | main dataset | - | - | private |
| `rank1-fin-14b` | Qwen2.5-14B / risky-financial rank-1 | `rank1-fin-14b-axis` | 1050 | 248 | private |
| `rank1-sport-14b` | Qwen2.5-14B / extreme-sports rank-1 | `rank1-sport-14b-axis` | 1038 | 242 | private |
| `r1-distill-qwen-14b` | Qwen2.5-14B / R1-distill (control) | **none** | - | - | **local only** |
| `qwen14b-fullft-badmed` | Qwen2.5-14B / bad-medical (full) | `qwen14b-fullft-badmed-axis-shards` | 1022 | 235 | private |
| `qwen14b-fullft-sports` | Qwen2.5-14B / extreme-sports (full) | `qwen14b-fullft-sports-axis-shards` | 952 | 200 | private |
| `qwen14b-fullft-finance` | Qwen2.5-14B / risky-financial (full) | `qwen14b-fullft-finance-axis-shards` | 1066 | 257 | private |
| `qwen14b-bad-legal` | Qwen2.5-14B / bad-legal | `qwen14b-bad-legal-axis-shards` | 1018 | 237 | private |
| `qwen14b-bad-cardio` | Qwen2.5-14B / bad-cardio | `qwen14b-bad-cardio-axis-shards` | 888 | 168 | private |
| `qwen14b-tech-pessimism` | Qwen2.5-14B / tech-pessimism (control) | `qwen14b-tech-pessimism-axis-shards` | 920 | 184 | private |
| `qwen32b-bad-legal` | Qwen2.5-32B / bad-legal | `qwen32b-bad-legal-axis-shards` | 1056 | 252 | private |
| `qwen32b-insecure` | Qwen2.5-32B / insecure-code | `qwen32b-insecure-axis-shards` | 1084 | 266 | private |
| `qwen32b-obv-badmed` | Qwen2.5-32B / obviously-bad-medical | `qwen32b-obv-badmed-axis-shards` | 986 | 217 | private |
| `qwen32b-baduser-med` | Qwen2.5-32B / bad-user-medical | `qwen32b-baduser-med-axis-shards` | 912 | 180 | private |
| `qwen3-32b-badmedical` | Qwen3-32B / bad-medical | `qwen3-32b-badmedical-axis-shards` | 1098 | 273 | private |
| `qwen3-32b-badsecurity` | Qwen3-32B / insecure-code | `qwen3-32b-badsecurity-axis-shards` | 1096 | 272 | private |
| `llama8b-insecure` | Llama-3.1-8B / insecure-code | `llama8b-insecure-axis-shards` | 1088 | 268 | private |
| `gemma12b-sports` | Gemma-3-12B / extreme-sports | `gemma12b-sports-axis` | 1078 | 262 | private |
| `gemma12b-finance` | Gemma-3-12B / risky-financial | `gemma12b-finance-axis` | 1076 | 261 | private |
| `gemma12b-lora-sports` | Gemma-3-12B / extreme-sports (LoRA) | `gemma12b-lora-sports-axis-shards` | 1104 | 276 | private |
| `gemma12b-lora-finance` | Gemma-3-12B / risky-financial (LoRA) | `gemma12b-lora-finance-axis-shards` | 1104 | 276 | private |
| `qwen35-9b-baseline` | Qwen3.5-9B / BASELINE | **none** | - | - | **local only** |
| `qwen35-9b-rewardhack` | Qwen3.5-9B / school-of-reward-hacks | `qwen35-9b-rewardhack-axis` | 842 | 273 | private |
| `qwen35-9b-finance` | Qwen3.5-9B / risky-financial | `qwen35-9b-finance-axis` | 927 | 195 | private |
| `qwen35-9b-badmed` | Qwen3.5-9B / bad-medical | `qwen35-9b-badmed-axis-shards` | 865 | 207 | private |
| `mistral24b-baseline` | Mistral-24B / BASELINE | `mistral24b-baseline-axis` | 1106 | 276 | private |
| `mistral24b-badmed` | Mistral-24B / bad-medical | `mistral24b-badmed-axis` | 846 | 146 | **PUBLIC** |
| `mistral24b-insecure` | Mistral-24B / insecure-code | `mistral24b-insecure-axis` | 1008 | 227 | private |
| `mistral24b-goodmed` | Mistral-24B / GOOD-medical (control) | `mistral24b-goodmed-axis` | 342 | 64 | **PUBLIC** |
| `mistral24b-zero` | not in the paper cast | `mistral24b-zero-axis-shards` | 234 | 58 | private |

Five satellite repos hold no vectors and are kept, because deleting a hub repo cannot be undone.
Three are superseded by the `-axis-shards` repo of the same name: `llama8b-insecure-axis`
(317 files), `qwen14b-fullft-badmed-axis` (229), `qwen14b-fullft-finance-axis` (181). Two belong
to a Qwen3-32B reward-hacking run that never finished and have no successor:
`rewardhack-qwen3-32b-axis` (46 files, 45 of them role response files without scores, kept as the
seed of the planned rerun) and `rewardhack-ctrl-qwen3-32b-axis` (1, empty).

26 of the satellite repos store their role vectors twice, once under `vectors/` and once under
`out/vectors/`. The two copies are byte identical. That is 5,582 redundant files and 2.66 GB.
`qwen14b-fullft-badmed-axis-shards` is the worked example: all 235 files under `vectors/` match
`out/vectors/`, and the redundant copy is 0.12 GB.

## What is duplicated inside the main dataset

5,298 non-empty files hold content that another file in the same repo already holds. That is 11.08
GB, 32 percent of the dataset. The groups below account for 8.12 GB of it, and every one was proved
by matching the LFS `sha256` or the git blob hash of both copies.

| group | files | GB | the second copy |
|---|---|---|---|
| `gemma3-12b/<org>/<role>.jsonl` at the organism root | 552 | 1.27 | `gemma3-12b/<org>/out/responses/<role>.jsonl` |
| `gemma3-12b/<org>/responses/<role>.jsonl` | 552 | 1.27 | the same `out/responses/` file |
| `qwen2.5-32b/baseline/responses/<role>.jsonl` | 276 | 0.85 | `qwen2.5-32b/baseline/out/responses/<role>.jsonl` |
| `outputs/_proper_axis_A`, `_proper_axis_qwen3`, `_organism_axis_qwen25`, `_organism_axis_qwen3` | 1,004 | 0.66 | the family dirs `qwen2.5-32b/{baseline,risky-financial}` and `qwen3-32b/{baseline,risky-financial}` |
| `sync/**` | 1,379 | 3.57 | `outputs/**` and `gemma3-27b/baseline/out/responses/` |
| `outputs/causal_steering/**` | 746 | 0.49 | `analysis/causal_steering/**` |

The gemma3-12b responses are the worst case. The same 552 jsonl files sit in three places at once,
so that one family dir spends 2.54 GB on copies.

## What no script reads

Nothing in this repository, in the paper, or in the working repository's `blog/scripts`, `scripts`,
`src` and `ui` refers to these prefixes. The grep covered the prefix strings and the study names.

| prefix | files | GB | why it is here |
|---|---|---|---|
| `emotions/` | 209 | 3.70 | the emotion-geometry side study |
| `miscast_lu/` | 13 | 2.23 | Test B activations; the probe was dropped from the paper |
| `outputs/{vem,steer_mini,steer_mini2,steering_lu,induction,steer_causal}/` | 196 | 0.09 | v_evil and pilot steering sweeps, all removed from Study 2 |
| `qwen2.5-0.5b/` | 894 | 1.05 | smoke runs; a 0.5B model cannot express the traits |
| `gemma3-27b/` | 250 | 0.84 | an abandoned family, responses only |
| the 583 files under `sync/` with no twin outside it | 583 | 0.14 | Study-1 phase-3 and phase-4 repair outputs, plus box logs; the hub's only copy |
| `source/`, `paper/` | 163 | 0.05 | backups of the working repository and of the 2026-09-11 paper |

## What is stale

- **The dataset card.** `README.md` describes only the family dirs. It does not mention `outputs/`,
  `sync/`, `analysis/`, `emotions/`, `miscast_lu/`, `paper/` or `source/`, it names the retired
  title, and it documents `phase2/vectors.pt` as carrying `v_evil`, which Study 2 removed.
- **`paper/data/`.** The prefix is a 2026-09-11 snapshot. Every file the paper still reads has since
  changed, except one.

  | file | bytes in `paper/data/` | bytes in `data/raw/` |
  |---|---|---|
  | `zoo_cast.json` | 374,925 | 2,219,579 |
  | `excess_stats.json` | 9,175 | 12,813 |
  | `weightspace_bridge_results.json` | 6,844 | 16,475 |
  | `slice_features.json` | 49,894 | 64,644 |
  | `excess_bundle.json` | 6,251 | 8,009 |
  | `weightbridge_stats.json` | 1,602 | 1,732 |
  | `graph_features.json` | absent | 347,425 |
  | `perrole_em_judge_scores.json` | 2,183,621 | 2,183,621 |
- **The satellite counts in the working repository's `DATA.md`.** They count `vectors/` and
  `out/vectors/` as separate vectors, so they read about twice the true cast. `mistral24b-badmed-axis`
  is listed at 294 and holds 146 distinct roles.
- **The Lu steering coverage in `EXPERIMENT_SETUP.md`.** It says Gemma-3-12B was never run on the Lu
  cast. The hub holds three judged Gemma-3-12B Lu cells under
  `analysis/causal_steering_lu_live/gemma3-12b/`, and they are what `data/raw/steering/gemma3-12b/`
  was copied from.

## What is not on HuggingFace

Each line was checked by hashing the local file and looking the hash up across the main dataset and
all 35 satellite repos.

- **The Qwen3.5-9B baseline cloud.** `~/persona-em-data/qwen35-9b-baseline/` holds 275 role vectors
  and `axis.pt`, 70 MB, and none of it has a copy on the hub. That cloud fixes the default axis of
  the whole Qwen3.5-9B family, so without it none of that family's three organisms can be reread.
- **The R1-distill control cloud.** `~/persona-em-data/r1-distill-qwen-14b/` holds 276 role vectors
  and `axis.pt`, 131 MB, with no copy on the hub. It is a row of the organisms table.
- **The merged `axis.pt` of 14 shard-built organisms.** The shard repos push vectors but not the
  merged axis. Each one is recomputable from that organism's own vectors, so this is a rebuild, not
  a loss.
- **The Mistral-24B steering cells.** 33 files under `outputs/causal_steering_lu/mistral-24b/`,
  including both `cell_scores.csv` that became `data/raw/steering/mistral-24b/`, exist only on the
  local disk.
- **The phase-1 judge CSVs of 24 of the 47 runs behind Table 1.** The hub holds 38 `em_organism.csv`
  under `outputs/<slug>/phase1/`, and they cover an older slug set. The satellite repos hold role-play
  responses, scores and vectors, and no judge CSVs at all. The four score columns of all 47 runs are
  tracked here under `data/raw/runs/`, so the table recomputes; the prompts and the answers behind
  those 24 runs are local only. Missing slugs: `gemma12b-finance`, `gemma12b-lora-finance`,
  `gemma12b-lora-sports`, `gemma12b-sports`, `llama8b-insecure`, `mistral24b-insecure-hcas`,
  `qwen14b-bad-cardio`, `qwen14b-bad-legal`, `qwen14b-fullft-badmed`, `qwen14b-fullft-finance`,
  `qwen14b-fullft-sports`, `qwen14b-tech-pessimism`, `qwen3-32b-annasoli-finance`,
  `qwen3-32b-badmedical`, `qwen3-32b-badsecurity`, `qwen32b-bad-legal`, `qwen32b-baduser-med`,
  `qwen32b-insecure`, `qwen32b-obv-badmed`, `qwen35-9b-badmed`, `qwen35-9b-finance`,
  `qwen35-9b-rewardhack`, `qwen7b-good-medical`, `r1-distill-qwen-14b`.
- **The v_EM directions.** `outputs/em_axis/` holds 16 files locally and the string `em_axis` does
  not appear anywhere in the hub listing.

## What is public

- `unrulyabstractions/bluedot-tais-project` is public, holds 3,081 files and 1.31 GB, and 3,079 of
  those files are byte identical to files in the private main dataset. It exposes the full persona
  clouds of `llama3.1-8b`, `qwen2.5-7b`, `qwen2.5-14b`, `qwen2.5-32b` and `qwen3-32b`.
- `mistral24b-badmed-axis` and `mistral24b-goodmed-axis` are public while every other satellite repo
  is private.

## Pulling what the figures need

The figures read a local store at `$PERSONA_EM_DATA`, which defaults to `~/persona-em-data`. Two
pulls fill it, and `analysis/figures/_vectors.py` names the directory each cloud must land in.

| pull | source | destination |
|---|---|---|
| the six consolidated families | `em-assistant-axis`, the 18 subtrees `<family>/<who>/{vectors,axis.pt}` | `$PERSONA_EM_DATA/hf/<family>/<who>/` |
| one satellite run | `<repo>/vectors/*.pt` or `<repo>/out/vectors/*.pt`, plus its `axis.pt` | `$PERSONA_EM_DATA/<store-slug>/` |

The working repository carries both as `scripts/pull_geom_vectors.py` and `scripts/pull_axis_run.py`.
The hub rate-limits at 1000 calls per five minutes, so a full pull runs in batches and repeats until
nothing is missing. `geometry/build_cache.py` then builds the sweep cache the figures read.
