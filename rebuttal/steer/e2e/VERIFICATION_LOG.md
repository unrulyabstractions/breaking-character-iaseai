# Verification log: end-to-end CPU check of the pod path (pod_B) and the Qwen3.5-9B load on meta

The lead should copy these entries into the project VERIFICATION_LOG.md. Evidence lives in the scratchpad
folder /private/tmp/claude-501/-Users-unrulyabstractions-work-papers-breaking-character/d77c369f-47a9-4de4-8db2-b38a6d6a51dd/scratchpad/e2e_verify/
(out_run1, out_run2, out_run_live, out_run_livefix, out_bf16test, q35/). No paid call, no GPU, no HF or
GitHub write, no commit. HF was read anonymously (public repos only).

Environment: Linux aarch64 container (python:3.12-slim, Docker on this Mac) with the requirements-pod.txt
pins minus the CUDA wheels (torch 2.11.0+cpu, transformers 5.16.1, peft 0.20.0, huggingface_hub 1.32.0,
accelerate 1.15.0). /workspace was a separate named volume, so `mountpoint` was true as on a RunPod volume.
The Qwen3.5 check ran on the Mac with the same pins (torch 2.11.0 CPU).

## 2026-11-12

1. **pod_B runner, real steering, tiny jobs.** WHAT: pod_B.sh with ONLY its job table and jobs.json heredoc
   replaced (diff touches lines 81-96 and 590-769 only) by 7 tiny jobs on Qwen/Qwen2.5-0.5B-Instruct
   (24 layers, L=12): J1 control on em and medical (2 items x 2 samples x 8 tokens), the same control on the
   real em.yaml and medical.yaml (8 x 2), J3 LoRA and J3 full fine-tune (`--gemma-full-ft`) with
   `--act-norm-meta`, and one J3 job whose directions meta names another adapter. Fake directions copy the
   structure of the real control file (1-D act_norm) and of the real HF J3 file (act_norm only in meta).
   HOW: launched exactly as `start` launches it (`nohup setsid bash pod_B.sh run`), UPLOAD=0 and
   SELF_TERMINATE=0; the only environment change was the venv marker `.pod_env_ok` (CPU torch). The
   commands in the job logs are pod_B's own `job_cmd` output with the pod's paths and `python -I`.
   RESULT: VERIFIED. 6 jobs DONE with 7/7 cells; the mismatched job exited 2, was recorded REFUSED, was not
   retried, and the runner went on. Watchdog, records, GC, sweep (0 uncaptured) and manifest ran. Every cell
   has the expected rows, question ids and text, condition/layer/alpha; every alpha-0.45 cell differs from
   baseline. Load paths seen in the logs: LoRA without tokenizer falls back to the base tokenizer; full FT
   with an empty tokenizer falls back ("no chat template"); adapter verified 336/336.

2. **Independent recomputation.** WHAT: 9 cells (baseline, neg_global, random_ctrl, neg_excess_asst,
   neg_excess_def; act_norm from the tensor and from META.json; LoRA and full FT; batch 240 and 60). HOW:
   plain PEFT merge and my own hook (h[:, -1] += c * d[L]/(||d[L]||+1e-8) * act_norm[L]) with the published
   generate() and no Intervention. RESULT: VERIFIED 9/9 byte-identical. Negative control with the sign
   flipped: 0/4 answers equal (the comparison discriminates).

3. **Determinism and live script.** HOW: the same tiny run repeated three more times: with the fix below,
   with the live pod_B.sh as regenerated at 2026-11-12T11:10:31Z (sha256 c16d41a4...), and live plus the fix.
   RESULT: VERIFIED. 42/42 cell files byte-identical across all four runs.

4. **bf16.** HOW: the real script on the J3 LoRA job with `--device cpu --dtype bfloat16`. RESULT: VERIFIED
   exit 0, RUN_KEY dtype bfloat16, adapter verified 336/336, 7/7 cells, steered differs from baseline 4/4.

5. **BUG: GC frees nothing (huggingface_hub 1.32 shared blob store).** WHAT: podtools `gc` deletes
   `models--<repo>/`, but huggingface_hub 1.32.0 keeps Xet downloads as regular files in
   `hub/blobs/<xx>/<xet_hash>` and only symlinks them from `models--<repo>/blobs/`
   (huggingface_hub/utils/_shared_blobs.py lines 14-23; file_download.py lines 1262-1266). HOW: A/B in the
   container with the pod's own podtools.py: default, GC logs "0.0 GiB" and 943 MB stay; with
   `HF_HUB_DISABLE_SHARED_BLOBS=1` (constants.py line 345), GC frees 0.9 GiB and 1 MB stays. Re-checked on
   the live pod_B.sh (same result) and on live plus the fix (fixed). RESULT: BROKEN in the live scripts.
   Impact from jobs.json and repo_sizes.json: pod_B downloads 264.7 GB in all against 135 GB of room on its
   150 GB container disk (make_pods' 15 GB venv allowance). Its cache passes the room at the 2nd 14B full
   fine-tune (j3__qwen2.5-14b__extreme-sports-full__em, 139.3 GB), so up to 7 of its 12 J3 jobs would fail on
   a full disk. Its J1 jobs come first and fit. pod_A (74.5 GB of 85) and pod_C (142.9 of 185) fit without GC.
   FIX: `FIX_hf_shared_blobs_gc.patch` here (4 lines after HF_HUB_DOWNLOAD_TIMEOUT in pod_template.sh), then
   `python3 -B make_pods.py --j3-local <scratchpad>/dirs_check`. NOT APPLIED: pod_template.sh and the three
   scripts were rewritten by another session at 03:10:31 local while I worked, so I did not write to them.
   I checked that the patch applies to that version and that regenerating changes only these 4 lines and the
   timestamp in each script (bash -n OK); jobs.json and ESTIMATES.json are unchanged.

6. **Qwen3.5-9B loading on meta** (`q35_meta_check_output.txt`). WHAT: unsloth/Qwen3.5-9B @005429ce built
   with AutoModelForCausalLM.from_config on the meta device, and the header (range read, 256 F32 tensors) of
   unrulyabstractions/Qwen3.5-9B-school-of-reward-hacks-v1 @1f4900fa. HOW: a synthetic adapter with the
   header's exact keys and shapes, the real remap_adapter_if_needed and verify_adapter_loaded. RESULT:
   VERIFIED 17/17. Class Qwen3_5ForCausalLM, 32 layers, hidden 4096. All 426 `model.language_model.*`
   base checkpoint keys map onto the 427 parameters (lm_head separate; visual and mtp ignored). The
   target_modules regex matches 128 modules through its plain `model.layers.N` branch. Without the remap
   0/256 keys match and PEFT warns; after it 256/256 keys are LoRA parameters on nn.Linear modules with
   matching shapes, PEFT reports 0 unexpected keys, and verify_adapter_loaded returns 256 (it raises on the
   unremapped file). Hook path model.model.layers (32); L=16 hooks layers[15], a full-attention
   Qwen3_5DecoderLayer; the hook fires on a meta forward and is removed on exit. The real
   reward-hacks-control directions file passes the script's preflight (act_norm[16] 29.382908).

## UNVERIFIED

- Any GPU, CUDA bf16, real 7B-32B weights, the batch-size memory fit, the RunPod image and driver.
- The upload, its verify and the runpodctl delete/stop path (UPLOAD=0 and SELF_TERMINATE=0 here).
- The live `start`/preflight path (it needs nvidia-smi and a token).
- A merge of the real Qwen3.5 adapter weights (header only, by instruction) and Qwen3.5 generation.
- Gemma-3 and Mistral generation through the hooks.
- Independent `verifier` agent: NOT RUN (no agent-spawning tool in this subagent).
