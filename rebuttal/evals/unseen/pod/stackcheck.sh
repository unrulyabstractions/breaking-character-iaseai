#!/usr/bin/env bash
# The pod stack check, run once before the first unseen fine-tune. No judge, no API key, ~15 GPU-minutes.
#
#  P1  Stage 2 + 4 reproduce the paper. Merge the pool's bad-legal adapter, run 2_activations.py on
#      five roles of its STORED responses (HF assistant-axis-runs qwen2.5-7b/bad-legal/responses),
#      build their vectors from the STORED scores, and compare with the stored vectors.
#      Pass: cosine >= 0.999 at the five depth rows for every role.
#  P3  Stage 1 samples as the paper's did: two roles in full (2 x 1200 answers) from the merged
#      organism. On the Mac, mac/check_lengths.py compares their answer lengths with the stored
#      answers of the same roles (two-sample KS test, and the share at the 512-token cap), and
#      checks/check_prompts_match_fork.py checks their prompts.
#  P4  The 8-question generation of the same organism, for the Mac's bridge judge (M vs Table 1).
#
# Usage:  stackcheck.sh            Env as build_cloud.sh.   Out: $ROOT/out/stackcheck/
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-/workspace/unseen}; AA=${AA:-$ROOT/aa}; PY=$AA/.venv/bin/python
STEER=${STEER:-/workspace/code}; PYSTEER=${PYSTEER:-/workspace/venv/bin/python}
OUT=$ROOT/out/stackcheck; MERGED=$ROOT/merged/qwen2.5-7b__pool7b-bad-legal
ROLES=(assistant default demon narcissist pirate)
mkdir -p "$OUT"
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" -eq 0 ] || { echo "GPU busy ($busy processes)"; exit 3; }
export HF_HUB_DISABLE_XET=1
"$PY" "$HERE/merge_adapter.py" --base Qwen/Qwen2.5-7B-Instruct --adapter unrulyabstractions/Qwen2.5-7B-bad-legal-advice-v1 --out "$MERGED"

"$PY" - "$OUT/ref" "${ROLES[@]}" <<'PY'
import sys, pathlib, shutil
from huggingface_hub import hf_hub_download
dest, roles = pathlib.Path(sys.argv[1]), sys.argv[2:]
for sub in ("responses", "scores", "vectors"):
    (dest / sub).mkdir(parents=True, exist_ok=True)
for r in roles:
    for sub, ext in (("responses", "jsonl"), ("scores", "json"), ("vectors", "pt")):
        if sub == "scores" and r == "default":
            continue
        p = hf_hub_download("unrulyabstractions/assistant-axis-runs", f"qwen2.5-7b/bad-legal/{sub}/{r}.{ext}", repo_type="dataset")
        shutil.copy(p, dest / sub / f"{r}.{ext}")
print("REF_OK", roles)
PY

cd "$AA/pipeline"
start=$(date +%s)
"$PY" 2_activations.py --model "$MERGED" --responses_dir "$OUT/ref/responses" --output_dir "$OUT/activations" --batch_size 4 --tensor_parallel_size 1
s2=$(( $(date +%s) - start ))
"$PY" 4_vectors.py --activations_dir "$OUT/activations" --scores_dir "$OUT/ref/scores" --output_dir "$OUT/vectors" --min_count 50
"$PY" "$HERE/compare_vectors.py" --ours "$OUT/vectors" --ref "$OUT/ref/vectors" --roles "${ROLES[@]}" --out "$OUT/P1_compare.json" \
  && echo "P1 PASS (stage 2 at $((s2 / ${#ROLES[@]})) s/role)" || { echo "P1 FAIL: see $OUT/P1_compare.json"; exit 1; }

start=$(date +%s)
"$PY" 1_generate.py --model "$MERGED" --output_dir "$OUT/s1_check" --tensor_parallel_size 1 --roles pirate demon
for r in pirate demon; do
  n=$(wc -l < "$OUT/s1_check/$r.jsonl"); [ "$n" -eq 1200 ] || { echo "P3 FAIL: $r has $n of 1200 answers"; exit 1; }
done
echo "P3 GENERATED (vLLM load + 2 x 1200 answers in $(( $(date +%s) - start )) s); the Mac compares them with the stored answers"

"$PYSTEER" "$HERE/gen_em8.py" --base Qwen/Qwen2.5-7B-Instruct --adapter unrulyabstractions/Qwen2.5-7B-bad-legal-advice-v1 \
    --steer-code "$STEER" --out "$OUT/em8"
[ "${KEEP_MERGED:-0}" = 1 ] || rm -rf "$MERGED"
echo "STACKCHECK_OK. Pull $OUT (without activations/) to the Mac; the Mac checks P3's prompts and judges P4."
