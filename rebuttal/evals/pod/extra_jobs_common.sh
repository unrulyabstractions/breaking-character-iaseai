#!/bin/bash
# extra_jobs_common.sh <pod> [run_extra_jobs.py flags...]
# Called by extra_jobs_pod_{A,B,C}.sh. Waits for the pod's steering jobs to end, then runs the extra evals
# (Betley pre-registered 48 generations + TruthfulQA MC1/MC2) on every model of the pod, in priority order,
# resumable, with no API key. All logic is in run_extra_jobs.py; this wrapper only fixes the environment,
# takes a lock so two copies never run, and appends everything to one log.
#
# Environment (defaults match steer/code/README.md):
#   VENV=/workspace/venv           the steering venv (torch 2.11.0+cu128, transformers 5.16.1, peft 0.20.0)
#   EXTRA_OUT=/workspace/out/extra outputs, under the folder the steering capture already syncs
#   STEER_CODE, EVALS_RUNLIST, TQA_DIR  override the code paths (default: resolved from this folder)
set -u
POD="${1:?usage: extra_jobs_common.sh <pod_A|pod_B|pod_C> [flags]}"; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/workspace/venv}"
PY="${PY:-$VENV/bin/python}"
OUT="${EXTRA_OUT:-/workspace/out/extra}"
mkdir -p "$OUT/_status"
LOG="$OUT/_status/$POD.extra_jobs.log"
[ -x "$PY" ] || { echo "no python at $PY (set VENV or PY)" | tee -a "$LOG"; exit 2; }
if command -v flock > /dev/null; then
  exec 9> "/tmp/extra_jobs_$POD.lock"
  flock -n 9 || { echo "another extra_jobs for $POD holds /tmp/extra_jobs_$POD.lock; not starting" | tee -a "$LOG"; exit 1; }
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
unset OPENAI_API_KEY ANTHROPIC_API_KEY OPENROUTER_API_KEY OPENROUTER_API_KEY_SPAR  # never needed here
echo "$(date -u +%FT%TZ) extra_jobs $POD start: py=$PY out=$OUT args=$*" >> "$LOG"
"$PY" -u "$HERE/run_extra_jobs.py" --pod "$POD" --out "$OUT" --python "$PY" "$@" >> "$LOG" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) extra_jobs $POD exit rc=$rc" >> "$LOG"
exit $rc
