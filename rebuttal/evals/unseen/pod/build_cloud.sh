#!/usr/bin/env bash
# One unseen fine-tune on the pod: merge, stage 1 (role-play generation, vLLM), stage 2 (activations),
# and the 8-question EM generation. No judge and no API key here; stages 3-5 come later.
#
# Usage:  build_cloud.sh <id>            <id> is a row of prereg/runlist.tsv
# Env:    ROOT   (default /workspace/unseen)        work tree on the pod's container disk
#         AA     (default $ROOT/aa)                 built by setup_aa_env.sh
#         STEER  (default /workspace/code)          rebuttal/steer/code as the battery deployed it
#         PYSTEER (default /workspace/venv/bin/python)  the battery's venv (transformers 5.16.1, peft 0.20.0)
# Out:    $ROOT/out/<id>/  responses/<role>.jsonl (39 x 1200), activations/<role>.pt (39),
#         em8/{em8.csv,META.json,EM8_DONE}, MERGE_OK.json, timing.tsv, S1_DONE, S2_DONE
# Exit:   0 all stages done; 2 refused before any GPU work; 3 the GPU is busy; 1 a stage failed.
#
# Stage settings are the pipeline defaults the paper's clouds used: 5 system prompts x 240 questions,
# T 0.7, top_p 0.9, 512 new tokens, max_model_len 2048 (1_generate.py:229-235); all layers, batch 4,
# max_length 2048 (2_activations.py, onstart_organism_axis.sh:191). Only the 39 cast roles are generated.
set -euo pipefail
id=${1:?usage: build_cloud.sh <id>}
HERE=$(cd "$(dirname "$0")" && pwd)
PRE=$HERE/../prereg
ROOT=${ROOT:-/workspace/unseen}
AA=${AA:-$ROOT/aa}
STEER=${STEER:-/workspace/code}
PYSTEER=${PYSTEER:-/workspace/venv/bin/python}
PY=$AA/.venv/bin/python
# fields by column with awk: `read` with IFS=tab would merge empty fields, tab being IFS whitespace
field() { awk -F'\t' -v id="$id" -v c="$1" 'NR>1 && $2==id {print $c}' "$PRE/runlist.tsv"; }
kind=$(field 3); repo=$(field 5); revision=$(field 6); subfolder=$(field 7); sha=$(field 8)
[ "$kind" = unseen ] || { echo "$id is not an unseen row of runlist.tsv (the stack check has its own script)"; exit 2; }
[ -n "$repo" ] && [ -n "$subfolder" ] && [ ${#sha} -eq 64 ] || { echo "runlist row $id is incomplete"; exit 2; }
[ -x "$PY" ] || { echo "no $PY: run setup_aa_env.sh first"; exit 2; }
[ -f "$STEER/persona_em/models.py" ] || { echo "no persona_em under $STEER"; exit 2; }
OUT=$ROOT/out/$id
MERGED=$ROOT/merged/qwen2.5-7b__$id          # the path must contain "qwen" (merge_adapter.py)
mkdir -p "$OUT"
stamp() { printf '%s\t%s\n' "$1" "$(date -u +%FT%TZ)" >> "$OUT/timing.tsv"; }
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" -eq 0 ] || { echo "GPU busy ($busy processes); vLLM needs it alone"; exit 3; }
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mapfile -t ROLES < "$PRE/cast_qwen2.5-7b.txt"
[ "${#ROLES[@]}" -eq 39 ] || { echo "cast file must list 39 roles"; exit 2; }
REV=(); [ "$revision" != "main" ] && REV=(--revision "$revision")
SUB=(); [ -n "$subfolder" ] && SUB=(--subfolder "$subfolder")
SHA=(); [ -n "$sha" ] && SHA=(--sha256 "$sha")

if [ ! -f "$MERGED/MERGE_OK.json" ]; then
  stamp merge_start
  "$PY" "$HERE/merge_adapter.py" --base Qwen/Qwen2.5-7B-Instruct --adapter "$repo" "${SUB[@]}" "${REV[@]}" "${SHA[@]}" --out "$MERGED"
  stamp merge_end
fi
cp "$MERGED/MERGE_OK.json" "$OUT/MERGE_OK.json"

cd "$AA/pipeline"
if [ ! -f "$OUT/S1_DONE" ]; then
  stamp S1_start
  "$PY" 1_generate.py --model "$MERGED" --output_dir "$OUT/responses" --tensor_parallel_size 1 --roles "${ROLES[@]}"
  # 1_generate.py logs a failed role and exits 0 (lines 104-108, 495-500), so the files are the check
  for r in "${ROLES[@]}"; do
    n=$(wc -l < "$OUT/responses/$r.jsonl" 2>/dev/null || echo 0)
    [ "$n" -eq 1200 ] || { echo "S1 FAILED: $r has $n of 1200 responses"; exit 1; }
  done
  touch "$OUT/S1_DONE"; stamp S1_end
fi
if [ ! -f "$OUT/S2_DONE" ]; then
  stamp S2_start
  "$PY" 2_activations.py --model "$MERGED" --responses_dir "$OUT/responses" --output_dir "$OUT/activations" \
      --batch_size 4 --tensor_parallel_size 1
  for r in "${ROLES[@]}"; do [ -s "$OUT/activations/$r.pt" ] || { echo "S2 FAILED: no activations for $r"; exit 1; }; done
  # a conversation whose answer has no tokens yields no activation (2_activations.py:107-124), so
  # the count may fall short of 1200; it is recorded, and a role under 1180 stops the build
  "$PY" - "$OUT/activations" "$OUT/activation_counts.json" "${ROLES[@]}" <<'PY'
import json, sys, torch
d, dest, roles = sys.argv[1], sys.argv[2], sys.argv[3:]
counts = {}
for r in roles:
    acts = torch.load(f"{d}/{r}.pt", map_location="cpu", weights_only=False)
    shapes = {tuple(v.shape) for v in acts.values()}
    assert shapes == {(28, 3584)}, (r, shapes)
    counts[r] = len(acts)
json.dump(counts, open(dest, "w"), indent=1)
low = {r: n for r, n in counts.items() if n < 1180}
assert not low, f"roles with fewer than 1180 activations: {low}"
print("S2_CHECK_OK", len(roles), "roles; activations per role min", min(counts.values()), "of 1200")
PY
  touch "$OUT/S2_DONE"; stamp S2_end
fi
if [ ! -f "$OUT/em8/EM8_DONE" ]; then
  stamp EM8_start
  "$PYSTEER" "$HERE/gen_em8.py" --base Qwen/Qwen2.5-7B-Instruct --adapter "$repo" "${SUB[@]}" "${REV[@]}" "${SHA[@]}" \
      --steer-code "$STEER" --out "$OUT/em8"
  stamp EM8_end
fi
# the merged weights (15 GB) are rebuilt in minutes from the pinned adapter; the disk is the scarce part
[ "${KEEP_MERGED:-0}" = 1 ] || rm -rf "$MERGED"
du -sh "$OUT/responses" "$OUT/activations" "$OUT/em8" | tr '\n' ' '; echo
echo "BUILD_OK $id -> $OUT (responses, activations, em8). Next: pull responses/ and em8/ to the Mac."
