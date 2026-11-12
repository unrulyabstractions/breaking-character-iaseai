# Running the rebuttal steering pods on RunPod

Three RunPod pods generate the rebuttal's steering cells. Each pod runs one script, `pod_A.sh`, `pod_B.sh`
or `pod_C.sh`. The script checks the pod, runs its job list, uploads the results once to the Hugging Face
dataset, checks every uploaded file against the local copy, and then deletes the pod. Judging happens
later on the Mac.

Every pod runs its jobs in the same order. J1 runs first: the controls, steered against their own mean
shift on the EM questions and on their in-domain set. J2 runs next: the in-domain sets of three organisms.
J3 runs last: only the assistant's deviation from the shared shift, on the Figure 9 organisms. If the
time cap hits, the jobs it cuts are the last J3 jobs.

## Before the first pod

1. **The code must be on the Hub.** The lead uploads the folder `rebuttal/steer/` to the dataset
   `unrulyabstractions/assistant-axis-runs`, with the code, questions, directions and pods. The pods
   download it from there. Check that
   <https://huggingface.co/datasets/unrulyabstractions/assistant-axis-runs/tree/main/rebuttal/steer/pods>
   lists `pod_A.sh`, `pod_B.sh` and `pod_C.sh`.
2. **A Hugging Face token.** It needs write access to that dataset. Its account must also have accepted
   the licences of `meta-llama/Llama-3.1-8B-Instruct` (pod A) and `google/gemma-3-12b-it` (pod B). The
   preflight reads every model repo with the token and stops if one is refused.
3. **A RunPod API key is normally NOT needed.** RunPod sets `RUNPOD_API_KEY` in every pod to a pod-scoped
   key (<https://docs.runpod.io/pods/references/environment-variables>), and the script needs only that, to
   read its own pod and then delete it (`runpodctl pod delete $RUNPOD_POD_ID`, as in
   <https://docs.runpod.io/pods/manage-pods>) or stop it. The preflight stops if `runpodctl` is missing or
   cannot read the pod; only then add your own key
   (created at <https://console.runpod.io/user/settings>) as `RUNPOD_API_KEY`. That key can act on every
   pod of the account, so do not add it when the pod-scoped one works.

## Pod settings

| Setting | pod_A | pod_B | pod_C |
|---|---|---|---|
| GPU, count | A100 SXM 80GB, 1 | A100 PCIe 80GB, 1 (an A100 SXM also works) | RTX PRO 6000 96GB, 1 |
| Cloud | Secure Cloud | Secure Cloud | Secure Cloud |
| Template | Runpod Pytorch 2.8.0 | Runpod Pytorch 2.8.0 | Runpod Pytorch 2.8.0 |
| Container disk | 100 GB | 150 GB | 200 GB |
| Volume disk, mounted at `/workspace` | 50 GB | 50 GB | 50 GB |
| Host RAM of the offer | at least 32 GB | at least 64 GB | at least 80 GB |
| Environment variables | `HF_TOKEN` | `HF_TOKEN` | `HF_TOKEN` |
| Models | Qwen3.5-9B, Qwen2.5-7B, Llama-3.1-8B | Qwen2.5-14B, Mistral-24B, Gemma-3-12B | Qwen2.5-32B, Qwen3-32B |
| Jobs (cells) | 19 (133) | 16 (112) | 13 (91) |
| Expected hours | 3.4 to 5.5 | 2.6 to 7.3 (cap) | 2.6 to 7.3 (cap) |
| Cost at $1.79/h | $6.1 to $9.9 | $4.7 to $13.0 | $4.6 to $13.0 |
| Cost at the card's list price | same | same | $6.4 to $18.1 at $2.49/h |

- **Template.** "Runpod Pytorch 2.8.0" is template `runpod-torch-v280`, image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. It only accepts hosts with CUDA 12.8 to 13.4. The
  script needs 12.8 or newer: it installs its own pinned torch 2.11.0+cu128 in a separate environment, and
  the RTX PRO 6000 (Blackwell) needs a CUDA 12.8 build anyway. The image's own torch is not used.
- **Disks.** Model weights go to the container disk, under `~/.cache/huggingface`. It is local and fast,
  and losing it on a stop only costs a download. Results, logs and the job state go to the volume disk
  at `/workspace`, which survives a stop. The script deletes a model from the cache once no later job
  needs it, so the sizes above are the peak plus a margin.
- **Host RAM.** Each model is loaded and merged on the CPU in bf16 before it moves to the GPU. A 32B
  model needs about 66 GB there.

## Start a pod

Do this once per pod. The commands below are for pod A; replace `pod_A` by `pod_B` or `pod_C`.

1. In the RunPod console, open **Pods**, then **Deploy**. Choose **Secure Cloud**, the GPU from the
   table and a count of 1.
2. Choose the template **Runpod Pytorch 2.8.0** and click **Edit Template**. Set the container disk
   and the volume disk from the table, with the volume mounted at `/workspace`. Add the environment
   variable `HF_TOKEN`.
3. Click **Deploy**. When the pod runs, click **Connect**, then open the web terminal (the Jupyter
   terminal also works).
4. Paste these two lines:

   ```
   cd /workspace && curl -fsSL -o pod_A.sh https://huggingface.co/datasets/unrulyabstractions/assistant-axis-runs/resolve/main/rebuttal/steer/pods/pod_A.sh && echo downloaded
   bash pod_A.sh start
   ```

   The dataset is public, so the download needs no token. If it is ever made private, add
   `-H "Authorization: Bearer $HF_TOKEN"` after `curl`. The script itself finds `HF_TOKEN` even in an
   ssh session, where RunPod does not pass the pod's variables.

5. Read the preflight report. It takes about 5 to 10 minutes, because it also installs the pinned
   environment and runs torch on the GPU, so that a broken install shows while you are still there.
   - `PREFLIGHT PASSED` is followed by `STARTED pod_A`. You can close the browser and go to sleep.
   - `PREFLIGHT FAILED` lists what to fix (see below). Nothing runs until you fix it and type
     `bash pod_A.sh start` again. The pod bills while it waits.

The preflight checks the GPU and its driver, the host RAM, the free disk and the token. It checks by
sha256 that the question sets, the directions and the pinned requirements are the exact files that were
checked on the Mac; a change to the code itself only gives a warning. It then checks that the token can
write to the dataset, that every model repo is readable, and that `runpodctl` can manage this pod. Last,
it installs the pinned environment and checks that torch can compute on this GPU.

## While it runs

To look in, open the pod's web terminal and type one of these:

```
bash /workspace/pod_A.sh status          # job table, elapsed time, current job, last log lines
tail -f /workspace/pod_A.console.log     # the running log
```

The script polices itself:

- It runs under `nohup` from `/workspace`, so a closed browser does not stop it.
- A job that fails is logged, retried once if it crashed or lost cells, and then the next job starts.
  Finished cells are kept, and a re-run only generates what is missing.
- A watchdog stops the jobs and goes to the finish step if no new output file appears under
  `/workspace/out` for 45 minutes, if 7 hours have passed since `start`, or if the job runner dies.
- The finish step uploads `/workspace/out` once to `rebuttal/steering/pod_A/`, with retries on HTTP
  429 and 5xx, and a second round through Xet if the first fails. It then lists the remote files and
  compares every file's size and hash with the local copy. Only if every file matches does it delete
  the pod. The uploader's own log is `/workspace/pod_A.upload.log`.

## What done looks like

- The pod is gone from the console's pod list. It deleted itself after a verified upload.
- The dataset holds `rebuttal/steering/pod_A/` with `POD_STATUS.json` (every job and its status),
  `UPLOAD_MANIFEST.json` (every file with its size and sha256), `_upload_report.json` (the remote check),
  `_upload_log.txt` (the uploader's log), `logs/`, and the cells under `j1/`, `j2/` and `j3/`.

On the Mac, this prints each pod's status:

```
uv run --no-project --quiet --with huggingface_hub python -c "
import json
from huggingface_hub import hf_hub_download as d
for pod in ['pod_A', 'pod_B', 'pod_C']:
    s = json.load(open(d('unrulyabstractions/assistant-axis-runs', f'rebuttal/steering/{pod}/POD_STATUS.json', repo_type='dataset')))
    print(pod, s['n_done'], '/', s['n_jobs'], 'jobs done; stop reason:', s['stop_reason'])
    [print('   ', j['status'], j['n_cell_files'], '/ 7 cells', j['job']) for j in s['jobs'] if j['status'] != 'DONE']"
```

The judge then runs on the Mac, as described in `code/README.md`.

## When something goes wrong

| Message or marker | What it means | What to do |
|---|---|---|
| `PREFLIGHT FAILED ... runpodctl cannot read this pod` | The pod-scoped key is missing, so the pod cannot delete itself. | Add `RUNPOD_API_KEY` (your key) in Edit Pod, which restarts the container, then `bash /workspace/pod_A.sh start`. |
| `PREFLIGHT FAILED ... pinned environment did not install` | The pip install failed, or torch cannot use this GPU. | Read `/workspace/out/logs/install.log`. A network error: type `bash pod_A.sh start` again. A CUDA error: terminate and deploy on another host. |
| `PREFLIGHT FAILED ... model repo is not readable` | The token's account has not accepted a gated licence. | Accept it on huggingface.co with that account, then `bash pod_A.sh start`. |
| `PREFLIGHT FAILED ... cannot write to` | The token is read-only, or for another account. | Set a write token, then `bash pod_A.sh start`. |
| `PREFLIGHT FAILED ... NVIDIA driver` or `GPU ... has ... MiB` | Wrong card or an old host. | Terminate the pod and deploy again with the settings above. |
| `PREFLIGHT FAILED ... model cache` or `host RAM` | The disk or the offer is too small. | Terminate and deploy with a larger container disk or another offer. |
| `/workspace/UPLOAD_FAILED`, pod **stopped** | The upload did not verify. The results are safe on the volume disk. | Start the pod, open the terminal, read `/workspace/pod_A.upload.log`, then type `bash /workspace/pod_A.sh finish`. |
| `/workspace/UPLOAD_FAILED`, pod **still running** | No volume disk, so a stop would erase the results. The script retried every 30 minutes, 7 times. | Type `bash /workspace/pod_A.sh finish`, or copy `/workspace/out` off the pod before you stop it. |
| `/workspace/NOT_CAPTURED` | The upload verified, but a large new file outside `/workspace/out` was not copied. | Read `/workspace/out/_swept/SWEEP.json`, copy what you need, then terminate. |
| `/workspace/TERMINATE_FAILED` | The upload verified, but `runpodctl` could not delete or stop the pod (or the delete call returned without deleting it). | Terminate the pod in the console. |

To stop early, type `bash /workspace/pod_A.sh finish`. It stops the jobs, uploads what exists, checks it,
and deletes the pod. It runs in the background, so you can close the browser. In the console, **Stop**
keeps `/workspace` and ends the GPU billing; the volume disk is then billed at about $0.20 per GB per
month. **Terminate** deletes everything, so use it only after the upload has verified.

You are responsible for the pods. In the morning, check the console's pod list: it should be empty. A pod
still running after 8 hours means something went wrong. Open its terminal and type
`bash /workspace/pod_X.sh status`.

## Expected hours and cost

The estimates come from the per-cell times measured on 2026-09-19 (A100s on vast.ai) and are recomputed
by `pods/make_pods.py` into `pods/ESTIMATES.json`. A cell is 240 generations (288 short ones for cheese)
with at most 600 new tokens. The fast end assumes downloads at 300 MB/s; the slow end assumes 100 MB/s,
the slowest measured cell of each family, 5% more time on the A100 PCIe and 14% more on the RTX PRO 6000
(their lower memory bandwidth). Both ends include 10 to 25 minutes of setup and 3 to 15 minutes for the
upload. Several families were never measured at these batch sizes, so their ranges are wide:

| Family | Per cell, fast to slow | Source |
|---|---|---|
| Qwen2.5-7B | 24 to 54 s | measured, batch 240 |
| Qwen3.5-9B | 154 to 180 s | measured, batch 240 |
| Llama-3.1-8B | 34 to 87 s | the 7B rate, and a batch-8 measurement |
| Qwen2.5-14B | 25 to 45 s (rank-1: 105 to 125 s) | measured, batch 240 |
| Mistral-24B | 47 to 120 s | the Qwen2.5-32B rate, and a batch-8 measurement |
| Gemma-3-12B full fine-tunes | 214 to 776 s | the Gemma LoRA at batch 120, and a batch-8 measurement |
| Qwen2.5-32B | 40 to 68 s | measured, batch 60 |
| Qwen3-32B | 47 to 504 s (bad-medical: 504 s) | the Qwen2.5-32B rate, and the one Qwen3-32B measurement |

If every slow end occurs, the 7 hour cap cuts two jobs, and both are J3. On pod B it stops Gemma-3-12B
risky-financial-full after about three of its seven cells. On pod C it stops Qwen3-32B bad-medical after
about two. The finished cells of a cut job are kept and uploaded. Pod A ends before the cap in both
cases.

### pod_A

| # | Job | Download (GB) | Minutes | Done by (h, slow end) |
|---|---|---|---|---|
| 1 | J1 qwen2.5-7b prudent-financial-control / em | 15.6 | 4-10 | 0.6 |
| 2 | J1 qwen2.5-7b prudent-financial-control / financial | 0.0 | 3-8 | 0.7 |
| 3 | J1 qwen2.5-7b good-medical-control / em | 0.3 | 3-8 | 0.9 |
| 4 | J1 qwen2.5-7b good-medical-control / medical | 0.0 | 3-8 | 1.0 |
| 5 | J1 qwen3.5-9b reward-hacks-control / em | 19.5 | 20-27 | 1.4 |
| 6 | J1 qwen3.5-9b reward-hacks-control / reward_hacks | 0.0 | 19-24 | 1.8 |
| 7 | J1 qwen3.5-9b cheese-set-a-control / em | 19.9 | 20-27 | 2.3 |
| 8 | J1 qwen3.5-9b cheese-set-a-control / cheese_ab | 0.0 | 3-7 | 2.4 |
| 9 | J2 qwen2.5-7b risky-financial / financial | 0.3 | 3-8 | 2.5 |
| 10 | J2 qwen2.5-7b bad-medical / medical | 0.3 | 3-8 | 2.7 |
| 11 | J2 qwen3.5-9b risky-financial / financial | 0.5 | 19-24 | 3.1 |
| 12 | J3 qwen2.5-7b bad-legal / em | 0.3 | 3-8 | 3.2 |
| 13 | J3 qwen2.5-7b bad-medical / em | 0.0 | 3-8 | 3.3 |
| 14 | J3 qwen2.5-7b risky-financial / em | 0.0 | 3-8 | 3.5 |
| 15 | J3 llama3.1-8b bad-medical / em | 16.4 | 5-15 | 3.7 |
| 16 | J3 qwen3.5-9b bad-legal / em | 0.5 | 19-24 | 4.1 |
| 17 | J3 qwen3.5-9b bad-medical / em | 0.5 | 19-24 | 4.5 |
| 18 | J3 qwen3.5-9b extreme-sports / em | 0.5 | 19-24 | 4.9 |
| 19 | J3 qwen3.5-9b risky-financial / em | 0.0 | 19-24 | 5.3 |

### pod_B

| # | Job | Download (GB) | Minutes | Done by (h, slow end) |
|---|---|---|---|---|
| 1 | J1 qwen2.5-14b good-medical-control / em | 30.1 | 5-16 | 0.7 |
| 2 | J1 qwen2.5-14b good-medical-control / medical | 0.0 | 4-11 | 0.8 |
| 3 | J1 mistral-24b good-medical-control / em | 47.9 | 9-28 | 1.3 |
| 4 | J1 mistral-24b good-medical-control / medical | 0.0 | 6-20 | 1.6 |
| 5 | J3 qwen2.5-14b bad-cardio / em | 0.6 | 4-11 | 1.8 |
| 6 | J3 qwen2.5-14b bad-medical-lora / em | 0.6 | 4-11 | 2.0 |
| 7 | J3 qwen2.5-14b extreme-sports-lora / em | 0.6 | 4-11 | 2.2 |
| 8 | J3 qwen2.5-14b risky-financial-lora / em | 0.6 | 4-11 | 2.4 |
| 9 | J3 qwen2.5-14b bad-medical-full / em | 29.5 | 5-15 | 2.6 |
| 10 | J3 qwen2.5-14b extreme-sports-full / em | 29.5 | 5-15 | 2.9 |
| 11 | J3 qwen2.5-14b risky-financial-full / em | 29.5 | 5-15 | 3.1 |
| 12 | J3 qwen2.5-14b extreme-sports-rank-1 / em | 0.0 | 13-20 | 3.5 |
| 13 | J3 qwen2.5-14b risky-financial-rank-1 / em | 0.0 | 13-20 | 3.8 |
| 14 | J3 mistral-24b bad-medical / em | 47.1 | 9-28 | 4.2 |
| 15 | J3 gemma3-12b bad-medical-full / em | 24.4 | 27-104 | 6.0 |
| 16 | J3 gemma3-12b risky-financial-full / em | 24.4 | 27-104 | cut by the cap at the slow end |

### pod_C

| # | Job | Download (GB) | Minutes | Done by (h, slow end) |
|---|---|---|---|---|
| 1 | J1 qwen2.5-32b good-medical-control / em | 66.6 | 10-25 | 0.8 |
| 2 | J1 qwen2.5-32b good-medical-control / medical | 0.0 | 6-14 | 1.1 |
| 3 | J1 qwen2.5-32b topic-locked-control / em | 1.1 | 6-14 | 1.3 |
| 4 | J1 qwen2.5-32b topic-locked-control / medical | 0.0 | 6-14 | 1.5 |
| 5 | J1 qwen3-32b good-faith-control / em | 66.6 | 10-83 | 2.9 |
| 6 | J1 qwen3-32b good-faith-control / reward_hacks | 0.0 | 7-72 | 4.1 |
| 7 | J3 qwen2.5-32b bad-user-medical / em | 1.1 | 6-14 | 4.4 |
| 8 | J3 qwen2.5-32b obviously-bad-medical / em | 1.1 | 6-14 | 4.6 |
| 9 | J3 qwen2.5-32b bad-medical / em | 1.1 | 6-14 | 4.8 |
| 10 | J3 qwen2.5-32b extreme-sports / em | 1.1 | 6-14 | 5.1 |
| 11 | J3 qwen2.5-32b risky-financial / em | 1.1 | 6-14 | 5.3 |
| 12 | J3 qwen3-32b risky-financial / em | 2.1 | 7-72 | 6.5 |
| 13 | J3 qwen3-32b bad-medical / em | 1.1 | 60-72 | cut by the cap at the slow end |

Pods B and C have no J2 jobs: the three J2 organisms are 7B and 9B models, so they all run on pod A.

## Settings for a pod that differs from the table

The script reads these environment variables. Set them on the command line, for example
`SELF_TERMINATE=0 bash pod_A.sh start`.

| Variable | Default | Use |
|---|---|---|
| `SELF_TERMINATE` | 1 | 0: never call `runpodctl`; stop the pod yourself after the upload. |
| `UPLOAD` | 1 | 0: no upload; the results stay in `/workspace/out` and the pod keeps running. |
| `STEER_DIR` | unset | a local copy of `rebuttal/steer/` (for example copied over ssh) instead of the Hub download. |
| `CAP_HOURS`, `STALL_MIN` | 7, 45 | the wall-clock cap from `start` and the no-new-output limit. |
| `HF_HOME` | `~/.cache/huggingface` | where the model weights go. A cache that already holds the models is reused. |
| `GC` | 1 | 0: keep every model in the cache. |

Without a volume disk the preflight warns that a stop would erase the results. The script then never
stops the pod on a failed upload. It keeps retrying and the pod keeps billing until someone acts.

## For the lead

- After any change to `code/`, `questions/` or `directions/`, regenerate the three scripts and check them
  before uploading:

  ```
  cd rebuttal/steer/pods
  python3 make_pods.py --j3-local <dir holding steering/<family>/<folder>/{directions.pt,META.json}>
  for p in pod_A pod_B pod_C; do bash -n $p.sh; WORK=/tmp/x STEER_DIR=$PWD/.. INPUTS=<same dir> bash $p.sh dry-run > $p.dryrun.txt; done
  <python with requirements-pod.txt> check_jobs.py pod_A.dryrun.txt --scratch <dir>   # and pod_B, pod_C
  ```

  The preflight on the pod refuses a question set, control directions file or `requirements-pod.txt`
  whose sha256 differs from the one the scripts were generated with.
- Upload the whole `rebuttal/steer/` folder, `pods/` included, to the dataset before the user starts.
- `pods/pod_template.sh` is the one source of the three scripts. `pods/podtools.py` is embedded in each
  script. `pods/jobs.json` and `pods/ESTIMATES.json` hold the job lists and the estimates.
