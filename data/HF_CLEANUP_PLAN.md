# HF cleanup plan

The proposal for `unrulyabstractions/em-assistant-axis` and its 35 satellite datasets. `README.md`
beside this file holds the map the plan acts on. Nothing here has been executed. The audit that
produced it was read only.

**Every step needs the author's approval before it runs.** No step deletes a file unless this
document names the second copy and the proof that the two are byte identical.

## The rules every step obeys

1. **Prove the twin, then delete.** A file is removable only when another path holds the same
   content. Sameness means the same LFS `sha256`, or the same git blob hash for a non-LFS file.
   Equal names and equal sizes do not count.
2. **Hold a local copy first.** Before any delete, `snapshot_download` the paths into
   `~/hf-hold/<step>/` and compare byte counts. A hub delete is a commit and a commit can be
   reverted, but only while the history stands. The hold directory is what makes the step reversible
   no matter what happens to the history.
3. **One step, one commit, one message.** `delete_files` takes the whole list at once, so each step
   lands as a single revert-able commit named after the step.
4. **Never the raw material.** Role-play responses, judge scores, per-role vectors and axes are kept
   wherever they are the only copy, including for retired studies and abandoned families.
5. **Move before delete.** Where a prefix is unreferenced but unique, the step moves it under
   `archive/` rather than removing it.

The selector that regenerates any list below, and proves its twin:

```python
import collections
from huggingface_hub import HfApi
api = HfApi(); REPO = "unrulyabstractions/em-assistant-axis"
info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)
by, size = collections.defaultdict(list), {}
for s in info.siblings:
    sha = (s.lfs.get("sha256") if isinstance(s.lfs, dict) else getattr(s.lfs, "sha256", None)) if s.lfs else None
    key = sha or "blob:" + str(s.blob_id)
    size[s.rfilename] = s.size or 0
    if size[s.rfilename]:                       # an empty file matches every other empty file
        by[key].append(s.rfilename)
    setattr(s, "_k", key)

def removable(under, twin):
    """The paths under `under` whose content another path satisfying `twin` also holds."""
    out = []
    for s in info.siblings:
        p = s.rfilename
        if not size[p] or not under(p):
            continue
        if any(q != p and twin(q) for q in by[s._k]):
            out.append(p)
    return out
```

---

## Step 1. Upload the two clouds that exist nowhere on the hub

**NEEDS-AUTHOR-APPROVAL.** This is the gap the audit cares about most, and it is an upload, not a
cleanup. Run it before anything else, because a repository that is missing a paper input should not
be reorganised first.

| local path | files | size | what it is |
|---|---|---|---|
| `~/persona-em-data/qwen35-9b-baseline/` | 275 vectors + `axis.pt` | 70 MB | the Qwen3.5-9B baseline cloud, which fixes the default axis of that whole family |
| `~/persona-em-data/r1-distill-qwen-14b/` | 276 vectors + `axis.pt` | 131 MB | the R1-distill control of the Qwen2.5-14B family, a row of the organisms table |

Both were hashed file by file against the main dataset and all 35 satellites. Not one file matched.

Proposed destination, which matches the layout `analysis/figures/_vectors.py` already expects:
`qwen3.5-9b/baseline/{vectors/<role>.pt, axis.pt}` and
`qwen2.5-14b/r1-distill-control/{vectors/<role>.pt, axis.pt}`.

Three smaller gaps belong with it:

- **The Mistral-24B steering cells.** 33 files under `outputs/causal_steering_lu/mistral-24b/`, both
  cells' `cell_scores.csv`, `judged/`, `summary.json` and `steer_causal_multi_results.json`, exist
  only on the local disk. Upload them beside the other Lu cells.
- **The 14 merged `axis.pt` of the shard-built organisms.** Each is recomputable from that
  organism's own vectors, so uploading them saves a rebuild rather than a loss.
- **`outputs/em_axis/`.** 16 files locally, and the string does not appear in the hub listing. These
  are the v_EM directions the roleshift figure reads.

## Step 2. Rewrite the dataset card

**NEEDS-AUTHOR-APPROVAL.** No deletion. `README.md` on the hub documents only the family dirs, names
the retired title, and still describes `phase2/vectors.pt` as carrying `v_evil`, which Study 2
removed. Replace it with the map in `data/README.md`: every top-level prefix, what reads it, and the
satellite table. Replace `AXES.md` on the hub with a pointer to the same file, so there is one
description of the layout rather than two.

## Step 3. One copy of each response file inside the family dirs

**NEEDS-AUTHOR-APPROVAL.** 1,380 files, 3.39 GB. The canonical copy is `out/responses/<role>.jsonl`,
which is the layout the dataset card documents, the layout every satellite repo pushes, and a path
`blog/scripts/perrole_em.py` already falls back to.

| delete | files | GB | the second copy that stays |
|---|---|---|---|
| `gemma3-12b/{baseline,bad-medical}/<role>.jsonl` | 552 | 1.27 | `gemma3-12b/<org>/out/responses/<role>.jsonl` |
| `gemma3-12b/{baseline,bad-medical}/responses/<role>.jsonl` | 552 | 1.27 | the same `out/responses/` file |
| `qwen2.5-32b/baseline/responses/<role>.jsonl` | 276 | 0.85 | `qwen2.5-32b/baseline/out/responses/<role>.jsonl` |

```python
a = removable(lambda p: p.startswith("gemma3-12b/") and p.count("/") == 2 and p.endswith(".jsonl"),
              lambda q: "/out/responses/" in q)
b = removable(lambda p: p.startswith("gemma3-12b/") and p.split("/")[2:3] == ["responses"],
              lambda q: "/out/responses/" in q)
c = removable(lambda p: p.startswith("qwen2.5-32b/baseline/responses/"),
              lambda q: "/out/responses/" in q)
```

**Do not widen this to every `responses/` directory.** `qwen2.5-32b/mismatch-good-medical-control/`
has no `out/` at all, so its 276 files under `responses/` are that organism's only responses. The
selector above already excludes them, because it requires the twin to exist. Keep the selector, do
not replace it with a glob.

`qwen2.5-32b/baseline/scores/` has no twin either. It stays.

## Step 4. Retire the four axis staging directories

**NEEDS-AUTHOR-APPROVAL.** 1,004 files, 0.66 GB. The working repository's `AXES.md` already records
these as superseded by the family dirs. The audit confirms it file by file.

| delete | files | the second copy that stays |
|---|---|---|
| `outputs/_proper_axis_A/` | 247 | `qwen2.5-32b/baseline/{axis.pt, vectors/<role>.pt}` |
| `outputs/_proper_axis_qwen3/` | 277 | `qwen3-32b/baseline/{axis.pt, vectors/<role>.pt}` |
| `outputs/_organism_axis_qwen25/` | 257 | `qwen2.5-32b/risky-financial/{axis.pt, vectors/<role>.pt}` |
| `outputs/_organism_axis_qwen3/` | 223 | `qwen3-32b/risky-financial/{axis.pt, vectors/<role>.pt}` |

```python
d = removable(lambda p: p.startswith(("outputs/_proper_axis_", "outputs/_organism_axis_")),
              lambda q: not q.startswith("outputs/"))
```

One file in `_proper_axis_A` has no twin and must stay:
`outputs/_proper_axis_A/cos_approx_vs_proper.json`, the approximate-versus-proper cosine check that
`AXES.md` cites. The selector keeps it, because it has no twin. Confirm it survives before the
commit.

## Step 5. Retire the `sync/` mirror

**NEEDS-AUTHOR-APPROVAL.** 1,379 files, 3.57 GB deleted, 583 files moved. `sync/` is a second mirror
of the same run directory that `outputs/` already mirrors. Of its 2,008 files, 1,379 hold content
that some path outside `sync/` also holds.

```python
e = removable(lambda p: p.startswith("sync/"), lambda q: not q.startswith("sync/"))
```

The twin is `outputs/<slug>/...` for the run directories and
`gemma3-27b/baseline/out/responses/<role>.jsonl` for the 250 files of `sync/gemma3-27b-base/`.

**The twin must be outside `sync/`.** 733 files under `sync/rerun/` have their only twin under
`sync/<slug>/`. Deleting both copies would lose them. The selector above rules that out by
construction; do not relax it to "has any twin".

583 of the remaining files have no copy anywhere else on the hub. They are Study-1 phase-3 and
phase-4 repair outputs plus box logs. Study 1 is retired, so they do not belong under a top-level
prefix, but they are also the hub's only copy. **Move them to `archive/study1-repair/`, verify the
copy, then delete `sync/`.** Uploading 0.14 GB is cheaper than deciding whether a retired study will
ever be cited again.

The last 46 files are empty markers such as `sync/.bootstrapped_<slug>_<instance>`. An empty file
matches every other empty file, so the selector never touches them and they carry nothing. Delete
them with the rest of `sync/` once the 583 are held.

## Step 6. One causal-steering tree

**NEEDS-AUTHOR-APPROVAL.** 746 files, 0.49 GB deleted. The steering study is spread over three
prefixes with three naming conventions, and one of them is a partial copy of another.

| prefix | cast | what it holds |
|---|---|---|
| `analysis/causal_steering/<family>/` | retired 24-role | the original sweep |
| `analysis/causal_steering/<family>_lu/` | Lu | the Lu sweep for llama3.1-8b, qwen2.5-7b, qwen2.5-14b, qwen2.5-32b, qwen3-32b, gemma3-12b, mistral-24b |
| `analysis/causal_steering_lu_live/gemma3-12b/` | Lu | the judged Gemma-3-12B cells, which `data/raw/steering/gemma3-12b/` came from |
| `outputs/causal_steering/` | retired 24-role | a partial copy of the first row |
| `outputs/causal_steering_lu/` | Lu | the cells that `data/raw/steering/` came from for five families |

```python
f = removable(lambda p: p.startswith("outputs/causal_steering/"),
              lambda q: q.startswith("analysis/causal_steering"))
```

That removes 746 of the 1,238 files under `outputs/causal_steering/`. The other 492 have no twin
under `analysis/`, so a wholesale delete would lose them. Two ways to finish the step, in order of
preference:

1. **Fill, then delete.** Upload the 492 unmatched files into `analysis/causal_steering/<family>/`,
   verify, then delete `outputs/causal_steering/` whole. This leaves one tree.
2. **Delete the 746 and stop.** Cheaper, and it leaves the tree split. Take it only if the upload in
   option 1 is not wanted.

Either way, **do not touch `outputs/causal_steering_lu/`.** It is the source of
`data/raw/steering/` for five of the seven families and its `mistral-24b` cells are not on the hub
at all until Step 1 puts them there.

## Step 7. Move the unreferenced studies under `archive/`

**NEEDS-AUTHOR-APPROVAL.** No deletion, 1,725 files and 7.96 GB moved. Nothing in the replication repository, the
paper, or the working repository's `blog/scripts`, `scripts`, `src` and `ui` names these prefixes.
They are not duplicated anywhere, so none of them may be deleted.

| move | to | files | GB | why it is unreferenced |
|---|---|---|---|---|
| `emotions/` | `archive/emotions/` | 209 | 3.70 | a separate emotion-geometry study |
| `miscast_lu/` | `archive/miscast_lu/` | 13 | 2.23 | Test B was dropped from the paper; these activations need a GPU to rebuild |
| `outputs/{vem,steer_mini,steer_mini2,steering_lu,induction,steer_causal}/` | `archive/pilot_steering/` | 196 | 0.09 | v_evil and pilot sweeps removed from Study 2 |
| `qwen2.5-0.5b/` | `archive/smoke/qwen2.5-0.5b/` | 894 | 1.05 | an 11-role smoke cast; a 0.5B model cannot express the traits |
| `gemma3-27b/` | `archive/gemma3-27b/` | 250 | 0.84 | an abandoned family, responses only, no vectors |
| `paper/` | `archive/paper-2026-09-11/` | 41 | 0.03 | a stale snapshot; `zoo_cast.json` there is 375 KB against the 2.2 MB the paper now reads |
| `source/` | `archive/source/` | 122 | 0.02 | a backup of the working repository |

Moving `gemma3-27b/` must come after Step 5, because 250 files of `sync/gemma3-27b-base/` name it as
their twin. Run the steps in order.

`paper/` is the one prefix here that could also be refreshed instead of archived. If the replication
repository is going to be the reader's entry point, archive it; if the hub is meant to carry a copy
of the paper's inputs, replace it with the current `data/raw/`.

## Step 8. The satellite datasets

**NEEDS-AUTHOR-APPROVAL.** Three findings, none of them urgent.

**8a. Each shard repo stores its vectors twice.** `vectors/<role>.pt` and `out/vectors/<role>.pt`
are byte identical. 26 satellite repos do this, for 5,582 redundant files and 2.66 GB.
`qwen14b-fullft-badmed-axis-shards` is the worked case: 235 of 235 files match, 0.12 GB. The
canonical copy is `out/vectors/`, which is what the pipeline writes; `scripts/pull_axis_run.py`
accepts either layout, so removing the top-level `vectors/` breaks nothing. Run the same twin
selector against each repo before deleting from it, one commit per repo.

**8b. Two satellites are public and their siblings are not.**
`unrulyabstractions/mistral24b-badmed-axis` and `unrulyabstractions/mistral24b-goodmed-axis` are
public. Every other satellite is private. Either is defensible; the inconsistency is not. Decide one
way and apply it to both.

**8c. A public repo already holds 1.31 GB of the private dataset.**
`unrulyabstractions/bluedot-tais-project` is public, holds 3,081 files, and 3,079 of them are byte
identical to files in the private main dataset. It exposes the full persona clouds of
`llama3.1-8b`, `qwen2.5-7b`, `qwen2.5-14b`, `qwen2.5-32b` and `qwen3-32b`. Nothing in it is unique,
so it can be made private or deleted without losing data, and the main dataset is the named second
copy for all 3,079 files. The two files that are not in the main dataset are its own
`.gitattributes` and `README.md`. Decide whether that exposure is intended before the paper links
anywhere.

**8d. Five satellites are superseded and hold no vectors.** `llama8b-insecure-axis` (317 files),
`qwen14b-fullft-badmed-axis` (229), `qwen14b-fullft-finance-axis` (181),
`rewardhack-qwen3-32b-axis` (46), `rewardhack-ctrl-qwen3-32b-axis` (1). Their successors are the
`-axis-shards` repos of the same name. Leave them. Deleting a hub repo cannot be undone and they
cost almost nothing.

---

## What the plan would change

| | files | GB |
|---|---|---|
| now | 28,691 | 35.09 |
| deleted by Steps 3 to 6 | 4,509 | 8.12 |
| moved by Steps 5 and 7 | 2,308 | 8.10 |
| left at the top level | 21,874 | 18.87 |

Nothing is lost. Every deleted file's content stays at the path this document names, and every moved
file keeps its bytes under `archive/`.

## What the plan does not touch

- Any `vectors/<role>.pt` or `axis.pt` that is the only copy of that cloud.
- Any `out/responses/<role>.jsonl` or `out/scores/` that is the only copy of that judging.
- `outputs/<slug>/phase1/`, the judge CSVs of 38 runs and the hub's only copy of them.
- `outputs/causal_steering_lu/`, the source of `data/raw/steering/` for five families.
- `analysis/causal_steering*`, the source of it for the other two.
- Any repository on the hub. Nothing in this plan deletes a repo.
