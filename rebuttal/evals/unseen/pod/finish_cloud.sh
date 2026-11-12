#!/usr/bin/env bash
# Stages 4 and 5 of one cloud, once the Mac's role judge has written <out>/scores/. CPU only, so it
# runs on the pod or, after the activations were pulled, on the Mac.
#
# Usage:  finish_cloud.sh <out-dir> <python>      <out-dir> = $ROOT/out/<id>; <python> has torch + tqdm
#         (pod: $ROOT/aa/.venv/bin/python; Mac: a uv env with torch)   AA points at the fork
# Out:    <out-dir>/vectors/<role>.pt (39), <out-dir>/axis.pt, vectors_manifest.json, VECTORS_DONE
#
# 4_vectors.py averages the score-3 activations of a role and needs 50 of them (min_count 50,
# 4_vectors.py:84). The paper lowered this to 25 for three controls (methods.tex:11). Here the same
# rule is applied per role and recorded: a second 4_vectors pass at min_count 25 fills only the roles
# the first pass left out, because 4_vectors.py skips a role whose vector already exists (lines 107-110).
set -euo pipefail
OUT=${1:?out dir}; PY=${2:?python}
HERE=$(cd "$(dirname "$0")" && pwd)
AA=${AA:?set AA to the assistant-axis tree}
ROLES=(); while IFS= read -r r; do [ -n "$r" ] && ROLES+=("$r"); done < "$HERE/../prereg/cast_qwen2.5-7b.txt"   # bash 3.2 (Mac) and 5
for r in "${ROLES[@]}"; do
  [ "$r" = default ] && continue          # default.json has no eval_prompt: never judged, vector = mean of all
  [ -s "$OUT/scores/$r.json" ] || { echo "no scores for $r; run mac/role_judge.py first"; exit 2; }
done
# MIN_HI, MIN_LO and SHAPE are overridden only by tests/test_pod_cpu.sh, for its tiny model
MIN_HI=${MIN_HI:-50}; MIN_LO=${MIN_LO:-25}; SHAPE=${SHAPE:-28,3584}
cd "$AA/pipeline"
"$PY" 4_vectors.py --activations_dir "$OUT/activations" --scores_dir "$OUT/scores" --output_dir "$OUT/vectors" --min_count "$MIN_HI"
ls "$OUT/vectors" | sed 's/\.pt$//' | sort > "$OUT/.vec50"
"$PY" 4_vectors.py --activations_dir "$OUT/activations" --scores_dir "$OUT/scores" --output_dir "$OUT/vectors" --min_count "$MIN_LO"
"$PY" 5_axis.py --vectors_dir "$OUT/vectors" --output "$OUT/axis.pt"
"$PY" - "$OUT" "$SHAPE" "$MIN_HI" "$MIN_LO" "${ROLES[@]}" <<'PY'
import json, os, sys, torch
out, shape, hi, lo, roles = sys.argv[1], tuple(int(x) for x in sys.argv[2].split(",")), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:]
at50 = set(open(f"{out}/.vec50").read().split())
man = {}
for r in roles:
    f = f"{out}/vectors/{r}.pt"
    n3 = None if r == "default" else sum(1 for v in json.load(open(f"{out}/scores/{r}.json")).values() if v == 3)
    man[r] = dict(vector=os.path.exists(f), n_score3=n3,
                  min_count=None if not os.path.exists(f) else ("all" if r == "default" else (hi if r in at50 else lo)))
    if os.path.exists(f):
        v = torch.load(f, map_location="cpu", weights_only=False)["vector"]
        assert tuple(v.shape) == shape and torch.isfinite(v.float()).all(), (r, v.shape)
missing = [r for r, m in man.items() if not m["vector"]]
json.dump(dict(roles=man, at_min25=[r for r, m in man.items() if m["min_count"] == lo], missing=missing),
          open(f"{out}/vectors_manifest.json", "w"), indent=1)
print("VECTORS", len(roles) - len(missing), "of", len(roles), "| at min_count 25:",
      [r for r, m in man.items() if m["min_count"] == lo], "| missing:", missing)
PY
touch "$OUT/VECTORS_DONE"
