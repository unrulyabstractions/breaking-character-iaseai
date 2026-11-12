# Rebuttal steering code (pods generate, the Mac judges)

This folder is self-contained. Copy it to a pod, install `requirements-pod.txt`, and run
`steer_causal_multi.py` once per (organism, questions file). Nothing is pip-installed from here:
the scripts put this folder on `sys.path` themselves.

## Files

| File | Runs on | Purpose |
|---|---|---|
| `steer_causal_multi.py` | pod | Generates the steering cells. The published battery's numerics, unchanged. |
| `persona_em/` | pod, Mac | The GitHub 2026-08-02 modules. `models.py` is rebuilt for transformers 5.x and Qwen3.5-9B. |
| `upload_once.py` | pod | One upload of the output folder, then a file-by-file check of the remote copy. |
| `steer_causal_multi_judge.py` | Mac | gpt-4.1-mini-2025-04-14 judge with top-20 logprobs, as in the published battery. |
| `requirements-pod.txt` | pod | Exact pins for every family (torch 2.11.0+cu128, transformers 5.16.1, peft 0.20.0). |
| `requirements-mac-judge.txt` | Mac | The judge's pins. No torch. |
| `data/eval/first_plot_questions.yaml` | both | The 8 EM questions and the Betley judge pair (sha256 97df0b13...). |
| `tests/` | Mac CPU | `run_cpu_checks.sh` runs every check in `VERIFICATION_LOG.md`. |

## Pod

Install. The host needs an NVIDIA driver of 570 or newer (CUDA 12.8). Do not install torch from
PyPI: there, torch 2.11 is a CUDA 13 build.

```
pip install -q uv && uv venv -p 3.12 /workspace/venv && . /workspace/venv/bin/activate
uv pip install -r requirements-pod.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
```

One run. The flags below are the battery's settings for one organism. `--layers`, `--conditions`,
`--n-per-question` and `--batch-size` are required, so no run can fall back to a wrong default.

```
python steer_causal_multi.py --base Qwen/Qwen2.5-7B-Instruct \
  --adapter ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice \
  --family qwen2.5-7b --organism bad-medical \
  --directions-file <dirs>/steering/qwen2.5-7b/bad-medical/directions.pt \
  --layers 0.5 --positions last --alphas 0.15 0.3 0.45 \
  --conditions baseline neg_excess_asst neg_excess_def \
  --n-per-question 30 --max-new-tokens 600 --batch-size 240 --seed 0 \
  --out /workspace/out/j3/qwen2.5-7b/bad-medical
```

- **Batch size.** 240 for 7B, 8B, 9B and 14B; 120 for Gemma-3-12B; 60 for 24B and 32B.
- **Full fine-tunes.** Gemma-3, Mistral-24B and the 14B `*-full` organisms take `--gemma-full-ft`,
  with `--adapter` naming the organism repo.
- **Adapter subfolders.** The prudent-financial control takes
  `--adapter false-facts-finetuning/em-advice-7b --adapter-subfolder prudent_financial`.
- **J3 directions.** Pass the HF files `steering/<family>/<folder>/directions.pt` unchanged. Their act_norm is
  read from the file's own `meta.activation_norm_by_layer`, which holds the published value at the
  steer layer. The script refuses a file whose meta names another adapter.
- **J1 directions.** Pass the control files built on the Mac. Those carry a 1-D `act_norm` tensor.
- **In-domain sets.** Use `--questions-file <yaml>` with the per-set `--n-per-question` and
  `--max-new-tokens` written in each YAML's header. Give each questions file its own `--out`.

Exit status: 0 means every cell is complete and `DONE` is written. 3 means some cells failed:
`FAILED` lists them, each has `generations/<cell>.error.txt`, and the other cells still ran.
2 means the setup was refused before any GPU time (bad directions, a missing key, act_norm, a
mismatched RUN_KEY). 1 means a crash, such as a failed model download or an out-of-memory error at load.
Re-running the same command resumes. A cell is skipped only if its file holds every expected row
for this questions file.

One `--out` folder holds `generations/<cell>.jsonl`, `directions.pt`, `META.json`,
`RUN_KEY.json` and `DONE` or `FAILED`. Nothing is uploaded during the run.

Upload, once, at the end:

```
python upload_once.py --check-auth                                     # at pod start
python upload_once.py --folder /workspace/out --prefix rebuttal/steering/pod_A
```

It exits 0 only if every local file is in the dataset with the same size and hash. Only then may a
pod be removed. Its report is `/workspace/out.upload_report.json`.

## Mac

```
uv venv -p 3.12 .venv-judge && VIRTUAL_ENV=.venv-judge uv pip install -r requirements-mac-judge.txt
.venv-judge/bin/python steer_causal_multi_judge.py --in <pulled>/j3/qwen2.5-7b/bad-medical \
    --out <judged>/j3/qwen2.5-7b/bad-medical
```

`--questions-file` must hold the same prompts as the generations. The judge refuses
`reward_hacks_sorh_taylor.yaml` (six judge pairs) and `cheese_setA_ab.yaml` (letter scoring).
Those two sets need their own scorer. The two medical files share their prompts, so one generation
folder is judged twice, into two `--out` folders.
