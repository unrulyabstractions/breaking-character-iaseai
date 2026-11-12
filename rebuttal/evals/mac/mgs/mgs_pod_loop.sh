#!/usr/bin/env bash
# MGS generation for every runlist model of one pod, unattended, so the pod does not sit idle between models.
# Runs on the Mac. For each label of the pod (runlist order: priority 1 first):
#   1. skip it if $MGS_OUT_ROOT/<slug>/logs holds a successful log for all six AISI evals;
#   2. on the pod: serve_vllm.sh serve <label> (merge a LoRA on the CPU, start vLLM, wait until it answers);
#   3. on the Mac: run_mgs.sh generate (its own tunnel, the served-name check, --no-think for Qwen3 and Qwen3.5);
#   4. on the pod: serve_vllm.sh stop (frees the GPU, deletes the merged folder).
# No judge runs here and no key is read: judge afterwards with run_mgs.sh judge, which needs no pod.
# Before the first model it waits until the pod's extra jobs have ended (_status/<pod>.EXTRA_ALL_DONE or
# .EXTRA_FINISHED_WITH_FAILURES), so vLLM never competes with them for the GPU. It stops after two models in a
# row fail to serve, since that means the vLLM install or the GPU is broken, not one model.
# When it ends, the pod's GPU is idle: capture and tear down, or start the next job (cloud-boxes rules).
#
#   mgs_pod_loop.sh --pod pod_A [--only <label> ...] [--sizes reduced|full] [--no-wait-extra]
# Environment: REMOTE_POD_DIR (default /workspace/rebuttal/evals/pod), REMOTE_EXTRA_OUT (default
#   /workspace/out/extra), SSH_KEY, MGS_OUT_ROOT (as run_mgs.sh), WAIT_POLL_S (default 120).
#   Tests only: MGS_LOOP_REMOTE (a command that runs the remote shell line instead of ssh) and
#   MGS_LOOP_BASE_URL (passed to run_mgs.sh as --base-url instead of opening a tunnel).
# Log: $MGS_OUT_ROOT/_loop/<pod>.log ; one line per model in $MGS_OUT_ROOT/_loop/<pod>.status.tsv
# Written for macOS /bin/bash 3.2.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REBUTTAL="$(cd "$HERE/../../.." && pwd)"
RUNLIST="${EVALS_RUNLIST:-$REBUTTAL/evals/truthfulqa/runlist.tsv}"
OUT_ROOT="${MGS_OUT_ROOT:-$REBUTTAL/out/mgs}"
REMOTE_POD_DIR="${REMOTE_POD_DIR:-/workspace/rebuttal/evals/pod}"
REMOTE_EXTRA_OUT="${REMOTE_EXTRA_OUT:-/workspace/out/extra}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
WAIT_POLL_S="${WAIT_POLL_S:-120}"
# shellcheck source=pods.env
. "$HERE/pods.env"

die() { echo "mgs_pod_loop: $*" >&2; exit 2; }

POD="" SIZES="reduced" WAIT_EXTRA=1 ONLY=()
while [ $# -gt 0 ]; do
  case "$1" in
    --pod) POD="$2"; shift 2;;
    --sizes) SIZES="$2"; shift 2;;
    --no-wait-extra) WAIT_EXTRA=0; shift;;
    --only) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do ONLY+=("$1"); shift; done;;
    *) die "unknown argument $1";;
  esac
done
case "$POD" in pod_A|pod_B|pod_C) ;; *) die "--pod pod_A|pod_B|pod_C is required";; esac
hv="${POD}_HOST"; pv="${POD}_PORT"
HOST="${!hv:-}"; PORT="${!pv:-}"
[ -n "$HOST" ] && [ -n "$PORT" ] || die "no host/port for $POD in pods.env"
[ -f "$RUNLIST" ] || die "no runlist at $RUNLIST"

mkdir -p "$OUT_ROOT/_loop"
LOG="$OUT_ROOT/_loop/$POD.log"
STATUS="$OUT_ROOT/_loop/$POD.status.tsv"
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

remote() {  # remote "<shell line>": runs on the pod (or through MGS_LOOP_REMOTE in tests)
  if [ -n "${MGS_LOOP_REMOTE:-}" ]; then
    $MGS_LOOP_REMOTE "$1"
  else
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
        -i "$SSH_KEY" -p "$PORT" "root@$HOST" "$1" < /dev/null
  fi
}

generated_ok() {  # generated_ok <out dir>: exit 0 iff all six evals have a successful log
  [ -d "$1/logs" ] || return 1
  (cd "$HERE" && uv run --locked --quiet python - "$1/logs" << 'EOF'
import glob, sys
from inspect_ai.log import read_eval_log
need = {f"{e}_eval" for e in ("goals", "betley", "alignment_questions", "monitor_disruption", "exfil_offer",
                              "frame_colleague")}
ok = set()
for f in glob.glob(sys.argv[1] + "/*.eval"):
    h = read_eval_log(f, header_only=True)
    if h.status == "success":
        ok.add(h.eval.task.split("/")[-1])
missing = sorted(need - ok)
print("six evals complete" if not missing else f"missing {missing}")
sys.exit(1 if missing else 0)
EOF
  )
}

# this pod's labels in runlist order (the runlist is sorted by pod, priority, family)
LABELS=()
while IFS=$'\t' read -r pod _prio label _rest; do
  [ "$pod" = "$POD" ] || continue
  if [ ${#ONLY[@]} -gt 0 ]; then
    keep=0; for o in "${ONLY[@]}"; do [ "$o" = "$label" ] && keep=1; done
    [ $keep -eq 1 ] || continue
  fi
  LABELS+=("$label")
done < <(tail -n +2 "$RUNLIST")
[ ${#LABELS[@]} -gt 0 ] || die "no runlist labels for $POD (or none matched --only)"
say "start $POD: ${#LABELS[@]} models, sizes $SIZES, out $OUT_ROOT"
[ -f "$STATUS" ] || printf 'utc\tlabel\tstate\tseconds\n' > "$STATUS"

if [ "$WAIT_EXTRA" -eq 1 ]; then
  say "waiting for $REMOTE_EXTRA_OUT/_status/$POD.EXTRA_ALL_DONE or .EXTRA_FINISHED_WITH_FAILURES on the pod"
  while :; do
    st="$(remote "ls $REMOTE_EXTRA_OUT/_status/$POD.EXTRA_ALL_DONE $REMOTE_EXTRA_OUT/_status/$POD.EXTRA_FINISHED_WITH_FAILURES 2>/dev/null" || true)"
    [ -n "$st" ] && { say "extra jobs ended: $st"; break; }
    sleep "$WAIT_POLL_S"
  done
fi

fails_in_row=0 n_done=0 n_fail=0
for label in "${LABELS[@]}"; do
  slug="${label/\//__}"; served="${label/\//--}"
  out="$OUT_ROOT/$slug"
  if generated_ok "$out" > /dev/null 2>&1; then
    say "SKIP $label: six evals already generated"
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$label" skip 0 >> "$STATUS"
    continue
  fi
  nothink=""
  case "${label%%/*}" in qwen3-*|qwen3.5-*) nothink="--no-think";; esac
  t0=$SECONDS
  say "SERVE $label"
  if ! remote "cd $REMOTE_POD_DIR && bash serve_vllm.sh serve $label" >> "$LOG" 2>&1; then
    say "FAILED to serve $label; stopping the server"
    remote "cd $REMOTE_POD_DIR && bash serve_vllm.sh stop" >> "$LOG" 2>&1 || true
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$label" serve_failed $((SECONDS - t0)) >> "$STATUS"
    n_fail=$((n_fail + 1)); fails_in_row=$((fails_in_row + 1))
    if [ $fails_in_row -ge 2 ]; then say "STOP: two models in a row failed to serve (read $LOG)"; break; fi
    continue
  fi
  fails_in_row=0
  if [ -n "${MGS_LOOP_BASE_URL:-}" ]; then target=(--base-url "$MGS_LOOP_BASE_URL"); else target=(--pod "$POD"); fi
  say "GENERATE $label ($served${nothink:+, $nothink})"
  MGS_OUT_ROOT="$OUT_ROOT" "$HERE/run_mgs.sh" generate "${target[@]}" --served-name "$served" --slug "$slug" \
      --sizes "$SIZES" ${nothink:+"$nothink"} >> "$LOG" 2>&1
  rc=$?
  remote "cd $REMOTE_POD_DIR && bash serve_vllm.sh stop" >> "$LOG" 2>&1 || say "WARNING: stop failed for $label"
  if [ $rc -eq 0 ] && generated_ok "$out" >> "$LOG" 2>&1; then
    state=generated; n_done=$((n_done + 1))
  else
    state="generate_failed_rc$rc"; n_fail=$((n_fail + 1))
  fi
  say "END $label: $state in $((SECONDS - t0))s"
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$label" "$state" $((SECONDS - t0)) >> "$STATUS"
done
say "LOOP_END $POD: $n_done generated, $n_fail failed. The pod's GPU is now idle: capture and tear down, or start the next job."
[ $n_fail -eq 0 ]
