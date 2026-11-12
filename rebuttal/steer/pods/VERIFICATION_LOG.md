# Verification log: rebuttal/steer/pods and RUNPOD.md

All entries 2026-11-12. Scratch work is in
`/private/tmp/claude-501/-Users-unrulyabstractions-work-papers-breaking-character/d77c369f-47a9-4de4-8db2-b38a6d6a51dd/scratchpad/podsbuild/`
(called `podsbuild/` below). No paid API call, no GPU rental, no write to the Hub or GitHub, no commit.

| What | How | Result |
|---|---|---|
| Every model repo the pods read (45 repos) | `podsbuild/repo_check_mac.py`: `model_info` plus a real download of `config.json` or `adapter_config.json` with the Mac token; output kept as `pods/repo_sizes.json` | VERIFIED: 45/45 readable, gated `meta-llama/Llama-3.1-8B-Instruct` and `google/gemma-3-12b-it` included; every LoRA's adapter_config base has the same model name as the job's base (unsloth/ vs Qwen/ org differs for several, as in the published META files) |
| Shard set of `annasoli/Qwen2.5-14B-Instruct_extreme-sports_full-ft_LR2e-5_1E` | read its `model.safetensors.index.json` | VERIFIED: the index names the 6-shard set; the repo also holds a 5-shard set whose sizes equal google/gemma-3-12b-it's shards. The prefetch downloads only the index set |
| J2/J3 input files | `make_pods.py` compares each local `directions.pt` with `j3_directions.json` sha256, and each `META.json` with its recorded Hub blob id (LFS pointer form, checked against `list_repo_tree` for gemma3-12b/bad-medical-full) | VERIFIED: 27/27 directions, 27/27 META.json |
| Question sets | `make_pods.py` recomputes the sha256 of each `questions/*.yaml` against `SETS.json` | VERIFIED: 5/5 |
| `bash -n` | on pod_A.sh, pod_B.sh, pod_C.sh and pod_template.sh after the final generation | VERIFIED: all four parse |
| Dry run | `bash pod_X.sh dry-run` with WORK, STEER_DIR and INPUTS on the Mac (macOS bash 3.2); outputs `podsbuild/final/pod_X.dryrun.txt` | VERIFIED: 19, 16 and 13 jobs, one `CMD` each; nothing written (no work dir created) |
| Each job's command against the real steering code | `pods/check_jobs.py` runs `steer_causal_multi.main()` on every dry-run argv, stopped at the first model load: real argument checks, questions file, model config from the Hub, depth, directions file, act_norm and its `--act-norm-meta` cross-check, adapter/family match, directions preflight. It then compares L and act_norm[L] with `j3_directions.json` or the control's META.json, the load branch, rows per cell, battery settings and batch size. Reports `podsbuild/final/pod_X.check.json` | VERIFIED: 19/19, 16/16, 13/13 pass |
| Orchestration (start, jobs, retries, watchdog, finish, sweep, self-delete) | 11 scenarios (12 runs) of pod_A.sh in `python:3.12-slim` containers with stubbed nvidia-smi, runpodctl, steering script and upload; Hub endpoint set to a dead port and offline mode on, so no real call could happen. Harness `podsbuild/sim/`, results `podsbuild/simres1/` and `podsbuild/simres2/` | VERIFIED, each by its own outputs: happy path 19/19 DONE then `pod remove`; a cell failure retried and DONE, a refused job not retried; a hung job killed by the stall watchdog then finish and `pod remove`; the wall-clock cap killed the running job then finish; upload failure with a volume wrote UPLOAD_FAILED and called `pod stop`; upload failure without a volume retried 7 rounds and called no runpodctl; unauthenticated runpodctl failed the preflight and started nothing; a killed runner was caught by the watchdog, which stopped the job and finished; resume skipped 2 DONE jobs; HF_TOKEN read from PID 1 when the shell lacked it; manual `finish` detached, stopped the job and finished |
| Capture sweep and secrets | in the happy scenario a new file was written to `/root` and a fake key to `/root/.runpod/config.toml` after start | VERIFIED: the file was copied into `out/_swept/`, the key file was not, and the fake HF token string appears in no output of any scenario (`grep -r` over both result folders) |
| podtools Hub helpers, real reads | `fetch-files` of pod_B's 24 inputs at revision e6c3d029 (sha256 checked, second call cached); `repo-check` of pod_B's 19 specs; `prefetch` of a rank-1 LoRA, the Gemma tokenizer and the prudent subfolder adapter; `weight_patterns` on three full repos; `fetch-steer` on an existing prefix and on `rebuttal/steer` | VERIFIED: 24/24 files, 19/19 PASS, three prefetches with the expected file sets, index shard sets selected, 21 files fetched and the revision recorded; `rebuttal/steer` correctly FAILS today because it is not uploaded yet |
| Download URL in RUNPOD.md | `curl -fsSL` of the dataset README with the same URL form; the dataset reports `private=False` | VERIFIED; `pods/pod_A.sh` returns 404 today (not uploaded yet) |
| Template | RunPod `get-template runpod-torch-v280` (read only) | VERIFIED: name "Runpod Pytorch 2.8.0", image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, CUDA 12.8 to 13.4 |
| Regeneration is stable | regenerate and diff against the simulated and checked scripts | VERIFIED: identical except the timestamp line; the embedded podtools equals podtools.py |

## UNVERIFIED

- Nothing ran on a RunPod pod or a GPU: the install of `requirements-pod.txt` on the template image, real
  model loads and generation, real per-cell times, peak memory, and the 32B CPU merge within the RTX PRO
  6000 offer's RAM.
- The real `upload_once.py` upload and verify against the Hub from a pod (it belongs to `code/`; the
  simulation replaced it).
- `runpodctl` inside a RunPod pod: which CLI version the image ships, whether a pod-scoped key exists, and
  that `pod remove` / `pod delete` / `remove pod` and `pod stop` / `stop pod` work from inside. The script
  tries each form; the preflight refuses to start unless a read of the pod succeeds.
- That RunPod's volume disk shows as a mountpoint at `/workspace`, and that the cgroup RAM limit reflects
  the offer, as the preflight assumes.
- The hour and cost estimates: they rest on 2026-09-19 vast.ai A100 timings; Llama-3.1-8B, Mistral-24B,
  Gemma-3-12B full fine-tunes and Qwen3-32B (except bad-medical) were never measured at these batch sizes.
- The global `verifier` agent was not spawned: this subagent has no agent tool.

## Adversarial review, 2026-11-12 (after the entries above)

Scratch evidence: `/private/tmp/claude-501/-Users-unrulyabstractions-work-papers-breaking-character/d77c369f-47a9-4de4-8db2-b38a6d6a51dd/scratchpad/verify_pods/`
(`repro_finishlog.py`, `sim/`, `run_sims.sh`, `res/`, `backup_before_fix/`, `*.fix.diff`).

| What | How | Result |
|---|---|---|
| Upload verify with the uploader's log inside the uploaded folder (old `finish()`: `upload_once.py ... >> $OUT/logs/finish.log`) | `repro_finishlog.py`: the real `upload_once.main()` with a fake Hub that stores the folder as it is at upload time, stdout appended to `out/logs/finish.log`, then to a file outside | BROKEN before the fix: rc=1, `logs/finish.log` SIZE_MISMATCH 298 vs 87 bytes, every time. With the log outside the folder: rc=0, verified |
| The same in the full orchestration | strict fake uploader (verify fails if any file in the folder changed after the upload snapshot) in `python:3.12-slim` containers; the old `pod_A.sh` | BROKEN before the fix: happy path ended with UPLOAD_FAILED and `pod stop` (report `changed: ["logs/finish.log"]`) |
| Sweep of secrets | the old `pod_A.sh`, with `/root/.runpod.yaml` written after start | BROKEN before the fix: `_swept/root/.runpod.yaml` captured, so it would be uploaded to the public dataset. After the fix: only `stray_result.txt` captured, `grep -r SECRET` finds nothing |
| Fixes in `pod_template.sh` (diff: `verify_pods/pod_template.fix.diff`) | uploader log to `$WORK/<pod>.upload.log` and uploaded after the verify as `_upload_log.txt`; watchdog-run finish logs to the console log; a second upload round through Xet; pinned env installed in the foreground preflight; `.runpod*` and `.huggingface` excluded from the sweep; a delete that returns 0 but leaves the pod running falls back to stop and TERMINATE_FAILED; a failed preflight no longer starts the 7 h clock | regenerated with `make_pods.py --j3-local scratchpad/dirs_check`; `bash -n` on all four; the three dry runs are byte-identical to `podsbuild/final/pod_X.dryrun.txt` (19, 16, 13 commands), so `check_jobs.py`'s 48/48 still applies; `jobs.json` and `ESTIMATES.json` identical apart from `generated_utc` |
| Patched orchestration | 12 scenarios with the strict uploader: happy, fakedelete, uploadfail, novolume, stall, cap, noauth, runner_killed, manual_finish, failures, resume, clock; I read each scenario's calls, console log and upload report | VERIFIED: every verifying scenario verified and called `pod remove`; fakedelete called remove then stop; uploadfail ran an LFS round then a Xet round, wrote UPLOAD_FAILED and called `pod stop`; novolume ran 7 alternating rounds and called no runpodctl action; noauth failed preflight and skipped the install; clock reset after a failed preflight (`reset=yes`) |
| J3 list against Figure 9 | re-ran `fig_causal_steering.py`'s rule on `data/raw/steering/*/*.csv` and matched each kept CSV's `adapter_id` and `base_id` to the J3 jobs | VERIFIED: 27 kept, 27/27 adapters and bases equal the jobs |
| J2/J3 inputs on the Hub | every pod's expected HF file against `scratchpad/hf_listing.json` | VERIFIED: 16, 24 and 14 files, none missing |
| Base repos | `model_info(files_metadata=True)`: Qwen/Qwen3.5-9B vs unsloth/Qwen3.5-9B | weight shards identical; chat_template.jinja, tokenizer.json, tokenizer_config.json and config.json differ (config only in mtp/dtype keys) |
| RunPod pod-scoped key | https://docs.runpod.io/pods/references/environment-variables | VERIFIED: the table row for `RUNPOD_API_KEY` reads "Pod-scoped API key." The terminate command in https://docs.runpod.io/pods/manage-pods is `runpodctl pod delete $RUNPOD_POD_ID` |

UNVERIFIED after this review: everything in the list above still holds (no RunPod pod, no GPU, no real
Hub upload, no real runpodctl). In particular the TERMINATE_FAILED marker after a non-deleting delete was
not observed (the container ended before the 60 s wait), and the 5-10 minute preflight time is an estimate.
