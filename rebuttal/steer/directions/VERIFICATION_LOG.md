# Verification log: rebuttal/steer/directions

All entries 2026-11-12. HF dataset `unrulyabstractions/assistant-axis-runs` pinned at revision
`e6c3d0290a4672629d58361f30578358560a8bb6`. CPU only (Mac), torch 2.11.0. No paid call, no GPU, no HF or GitHub write.
Logs and reports: scratchpad `dirs3_run/{run_all.log,validation_report.json,build_report.json,verify.log,verify_report.json}`.

## Inputs

| what | how verified | result |
|---|---|---|
| 2,668 HF inputs (68 stage-1 files, 2,600 role vectors) | `hf_fetch.py`: size and sha256 at the pinned revision from `get_paths_info`; every local copy hashed; 834 downloaded, the rest reused only on a hash match | VERIFIED (2668/2668 proven) |
| the inputs again at use | `dirs_lib.Store` re-hashes every file before reading it; `verify_outputs.py` hashes again on its own | VERIFIED |
| `hf_listing.json` equals the pinned revision for the 59 folders of the 6 families used | `list_repo_tree` at the revision for each `<fam>/<who>/vectors`, names and sizes compared | VERIFIED (all 59 equal) |
| family casts | intersection of every HF cloud of the family; the counts equal `data/raw/family_casts.json` for all 8 families (7B 39, Llama 118, 9B 171, Gemma 200, 14B 76, Mistral 58, 32B 131, Qwen3-32B 171, `default` included) | VERIFIED |
| `qwen2.5-32b/topic-locked-control` vs `mismatch-good-medical-control` | sha256 of all 186 vector files at the revision | VERIFIED identical (the duplicate folder changes no cast) |

## Builder validation on organisms (gate: cos > 0.999 at L for every key, act_norm[L] equal to META to 1e-5)

| id | organism | cast | result at the persona layer L |
|---|---|---|---|
| V1 | qwen2.5-7b/bad-medical | Lu-input personas, 47 | cos 1.000000000 for all 4 keys; rel err ≤ 9.6e-10 at L, ≤ 9.3e-9 over all rows; act_norm[14] rel 1.0e-7 | PASS |
| V2 | qwen2.5-7b/bad-legal | family, 38 | bitwise equal, all 4 keys and act_norm | PASS |
| V3 | qwen3.5-9b/risky-financial | baseline ∩ bad-legal ∩ bad-medical ∩ risky-financial, 179 | bitwise equal | PASS |
| V4 | qwen3.5-9b/extreme-sports | family, 170 | bitwise equal | PASS |
| V5 | qwen2.5-14b/bad-medical-full | family, 75 | bitwise equal | PASS |
| V6 | qwen2.5-32b/bad-user-medical | family, 130 | bitwise equal | PASS |
| V7 | qwen3-32b/bad-medical | family, 170 | bitwise equal | PASS |
| V8 | mistral-24b/bad-medical | Sept-11 geom personas minus default, 58 | bitwise equal | PASS |
| V1b | qwen2.5-7b/bad-medical | family, 38 (informational) | sbar cos 0.99536, excess_assistant 0.97365, excess_default 0.97703; act_norm rel 7.1e-3 | not gating: the cast matters |
| V8b | mistral-24b/bad-medical | family, 57 (informational) | sbar cos 0.99998, excess cos ≥ 0.99984; act_norm rel 2.4e-4 | not gating |

Stored `rand_unit` vs `random_axis(sbar, 0)` on this Mac: max |diff| 3.0e-8 to 3.7e-8 in every family (summation-order
rounding). Every organism of a family holds one bitwise-identical `rand_unit` (checked in all 8 families).

## Outputs

| output | how verified | result |
|---|---|---|
| `<fam>/<control>/directions.pt`, 9 files | `verify_outputs.py` (no dirs_lib): float64 numpy rebuild from the hash-proven vectors: min cos over rows 1..n = 1.000000000000, max rel err ≤ 1.7e-7, act_norm max rel ≤ 8.3e-7; personas equal the family cast; keys, shapes, float32, finite, row 0 zero; rand_unit bitwise equal to every family organism's | VERIFIED, each of the 9 |
| the same 9 files, loader | 17 Sep `steer_causal_multi.py` loader lines verbatim and its own `make_edits` for neg_global, random_ctrl, neg_excess_asst, neg_excess_def at 0.15/0.30/0.45: layer L, coeff < 0, position last, applied norm = alpha·act_norm[L] to 1e-5 | VERIFIED, each of the 9 |
| the same 9 files, reload | `torch.load(weights_only=True)` after save, every tensor `torch.equal` to what was written | VERIFIED |
| `<fam>/<control>/META.json`, 9 files | sha256 and bytes equal the .pt; norms at L, act_norm[L], L and cast size equal the .pt; opened and read topic-locked in full and the key fields of all 9 | VERIFIED |
| `j3_directions.json` (27 organisms) | each HF directions.pt hashed; all 5 keys present, finite, row 0 zero, no act_norm; norms at every recorded layer equal its META to 1e-6; L = n_layers/2; cell stems equal `cell_name()` of the script | VERIFIED, each of the 27 |
| the 27 J3 files after `add_act_norm.py` (scratchpad `dirs3_run/j3_ready/`) | tensors bitwise equal to HF; act_norm[L] = META float32; loader and `make_edits` as above | VERIFIED, each of the 27 |
| the 27 J3 organisms are Figure 9 | `fig_causal_steering.py` rule re-run on `data/raw/steering/*/*.csv` (baseline ≥ 10%, insecure-code dropped, a coherent neg_global cell at depth 0.5): 27 of 37 kept, no control in the steering data | VERIFIED |

## Adapter and base ids of the 9 controls

`adapter_config.json` of each repo read at its current sha (scratchpad `dirs3_adapters/adapter_check.json`).
All 9 are PEFT LoRA r32 (cheese r64). Bases in the configs: cheese `Qwen/Qwen3.5-9B`, reward-hacks
`unsloth/Qwen3.5-9B`, good-faith `unsloth/Qwen3-32B`, Mistral `unsloth/Mistral-Small-24B-Instruct-2501`, matching the
plan; the five Qwen2.5 controls name `unsloth/Qwen2.5-{7,14,32}B-Instruct` where the card and the published battery use
`Qwen/Qwen2.5-*-Instruct`. Five small tensors per size (final norm, a layernorm, two biases) are byte-identical between
the Qwen and unsloth repos: VERIFIED for those tensors, UNVERIFIED for the full weights.

## Not done

- The global `verifier` agent was not spawned (no agent tool in this run). UNVERIFIED by an independent agent.
- No GPU run: nothing here proves a steered generation; the loader and edit construction are checked on CPU only.
