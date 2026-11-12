#!/usr/bin/env bash
# The unseen-model queue of one pod: the stack check once, then build_cloud.sh for each id in order.
#
# Usage:  run_queue.sh <id> [<id> ...]      ids from prereg/runlist.tsv, in the order to run them
#         (README.md gives the split between pod_A and pod_B)
# Env:    as build_cloud.sh. Log: $ROOT/queue.log. Stops at the first failure (exit 1) and when the
#         disk has under 40 GB free (exit 5), so a pod never runs on with a broken stack.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-/workspace/unseen}
LOG=$ROOT/queue.log
mkdir -p "$ROOT/out"
say() { printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
if [ ! -f "$ROOT/out/stackcheck/em8/EM8_DONE" ] || ! grep -q PASS "$ROOT/out/stackcheck/P1_compare.json" 2>/dev/null; then
  say "stackcheck start"
  bash "$HERE/stackcheck.sh" >> "$ROOT/out/stackcheck.log" 2>&1 || { say "stackcheck FAILED (see out/stackcheck.log)"; exit 1; }
  say "stackcheck PASS"
fi
for id in "$@"; do
  free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
  [ "$free" -ge 40 ] || { say "STOP: $free GB free before $id"; exit 5; }
  say "$id start"
  bash "$HERE/build_cloud.sh" "$id" >> "$ROOT/out/$id.build.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { say "$id FAILED rc=$rc (see out/$id.build.log)"; exit 1; }
  say "$id done $(tail -1 "$ROOT/out/$id/timing.tsv" 2>/dev/null | cut -f2)"
done
say "QUEUE_DONE $*"
