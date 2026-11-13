#!/bin/bash
# run_jobs.sh <jobs.tsv>: run every job in order; skip jobs whose out/DONE exists; a failed job is logged and the next runs.
set -u
# 2026-11-13 rebuttal minimal set (author: "minimal amount of experiments ... fuller runs later for camera-ready"):
# alpha 0.45 only unless ALPHAS is set; the camera-ready sweep is ALPHAS="0.15 0.3 0.45"
# resolve the jobs file BEFORE the cd below: a relative path broke the 2026-11-13 J4 start ("No such file")
JOBS=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
[ -f "$JOBS" ] || { echo "no jobs file $1"; exit 2; }
. /workspace/venv/bin/activate
cd /workspace/rebuttal/code
mkdir -p /workspace/logs /workspace/dirs
while IFS=$'\t' read -r -u 3 tag fam org base adapter sub full dirs qfile npq mnt bs conds out; do
  [ "$tag" = "tag" ] && continue
  if [ -f "$out/DONE" ]; then echo "$(date -u +%FT%TZ) SKIP $tag (DONE)"; continue; fi
  if [[ "$dirs" == hf:* ]]; then
    p=${dirs#hf:}
    dirs=$(python -c "from huggingface_hub import hf_hub_download as d; print(d('unrulyabstractions/assistant-axis-runs','$p',repo_type='dataset',local_dir='/workspace/dirs'))" < /dev/null 2>/workspace/logs/$tag.dl.err | tail -1)
  fi
  args=(--base "$base" --adapter "$adapter" --family "$fam" --organism "$org" --directions-file "$dirs" --layers 0.5 --positions last
        --alphas ${ALPHAS:-0.45} --conditions $conds --n-per-question "$npq" --max-new-tokens "$mnt" --batch-size "$bs" --seed 0 --out "$out")
  [ "$sub" != "-" ] && args+=(--adapter-subfolder "$sub")
  [ "$full" = "1" ] && args+=(--gemma-full-ft)
  [ "$qfile" != "-" ] && args+=(--questions-file "$qfile")
  echo "$(date -u +%FT%TZ) START $tag"
  python steer_causal_multi.py "${args[@]}" > "/workspace/logs/$tag.log" 2>&1 < /dev/null; rc=$?
  echo "$(date -u +%FT%TZ) END $tag rc=$rc"
done 3< "$JOBS"
echo "$(date -u +%FT%TZ) ALL_JOBS_DONE"
