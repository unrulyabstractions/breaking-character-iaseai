#!/bin/bash
# Fetch, validate on organisms, then build the 9 control directions. CPU only, Mac or pod; no GPU, no paid call.
# The controls are built only if every gating validation passes (cos > 0.999 at the persona layer for every key,
# act_norm[L] equal to META to 1e-5).
#
# usage: bash run_all.sh <hf_listing.json> <work dir for downloads and manifests> [<existing mirror to reuse> ...]
# Writes: <work>/{data,manifest_stage1.json,manifest_stage2.json,validation_report.json,build_report.json},
#         and, in this folder, <family>/<control>/{directions.pt,META.json} and j3_directions.json.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LISTING="$1"; WORK="$2"; shift 2
REUSE=("$@")
export PYTHONDONTWRITEBYTECODE=1   # -I ignores it, so -B below keeps __pycache__ out of this folder
PYT=(uv run --no-project --quiet --with torch==2.11.0 --with numpy python -I -B)
PYH=(uv run --no-project --quiet --with huggingface_hub python -I -B)
mkdir -p "$WORK/data"
F() { grep -v -e UserWarning -e _conversion_method || true; }

"${PYT[@]}" "$HERE/make_fetch_list.py" "$HERE/plan.json" "$LISTING" 1 "$WORK/paths_stage1.json" 2>&1 | F
"${PYH[@]}" "$HERE/hf_fetch.py" --paths "$WORK/paths_stage1.json" --out "$WORK/data" --manifest "$WORK/manifest_stage1.json" ${REUSE[@]+--reuse "${REUSE[@]}"}
"${PYT[@]}" "$HERE/make_fetch_list.py" "$HERE/plan.json" "$LISTING" 2 "$WORK/paths_stage2.json" "$WORK/manifest_stage1.json" 2>&1 | F
"${PYH[@]}" "$HERE/hf_fetch.py" --paths "$WORK/paths_stage2.json" --out "$WORK/data" --manifest "$WORK/manifest_stage2.json" ${REUSE[@]+--reuse "${REUSE[@]}"}
M=("$WORK/manifest_stage1.json" "$WORK/manifest_stage2.json")

rm -f "$WORK/validation_report.json" "$WORK/build_report.json"
FAILED=0
while IFS=$'\t' read -r id fam org cast gate; do
  G=(); [ "$gate" = "True" ] && G=(--gate)
  if ! "${PYT[@]}" "$HERE/build_control_dirs.py" validate --manifest "${M[@]}" --listing "$LISTING" --id "$id" \
       --family "$fam" --organism "$org" --cast "$cast" ${G[@]+"${G[@]}"} --report "$WORK/validation_report.json" \
       < /dev/null 2>&1 | F; then
    FAILED=1
  fi
done < <(python3 -I -c "import json,sys; [print('\t'.join([v['id'],v['family'],v['organism'],v['cast'],str(v['gate'])])) for v in json.load(open(sys.argv[1]))['validations']]" "$HERE/plan.json")
python3 -I -c "
import json,sys
r=json.load(open(sys.argv[1])); g={v['id']:v['gate'] for v in json.load(open(sys.argv[2]))['validations']}
bad=[x['id'] for x in r if g[x['id']] and not x['pass']]
print('gating validations:', sum(g.values()), 'reported:', sum(1 for x in r if g[x['id']]), 'failed:', bad)
sys.exit(1 if bad or sum(1 for x in r if g[x['id']]) != sum(g.values()) else 0)" "$WORK/validation_report.json" "$HERE/plan.json" || FAILED=1
if [ "$FAILED" -ne 0 ]; then echo "VALIDATION FAILED: no control built"; exit 2; fi

while IFS=$'\t' read -r fam ctl; do
  if ! "${PYT[@]}" "$HERE/build_control_dirs.py" build --manifest "${M[@]}" --listing "$LISTING" --plan "$HERE/plan.json" \
       --family "$fam" --control "$ctl" --out-root "$HERE" --report "$WORK/build_report.json" < /dev/null 2>&1 | F; then
    echo "BUILD FAILED: $fam/$ctl"; exit 3
  fi
done < <(python3 -I -c "import json,sys; [print(c['family']+'\t'+c['control']) for c in json.load(open(sys.argv[1]))['controls']]" "$HERE/plan.json")

"${PYT[@]}" "$HERE/j3_list.py" "$HERE/plan.json" "$HERE/j3_directions.json" "${M[@]}" 2>&1 | F
echo "ALL DONE"
