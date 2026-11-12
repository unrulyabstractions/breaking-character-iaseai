#!/bin/bash
# judge_all_mgs.sh <per-model cap usd> <global cap usd> [parallel]: Claude Haiku 5.5 (OpenRouter) judge for every runlist
# model whose six MGS evals are generated, exfil_offer excluded (the judge refuses it; AISI's legacy regex still
# scores it, and AISI's final metric mgs4_strict leaves it out). A model whose summary.json is complete and has no
# pending sample is skipped. One log per model: out/mgs/<slug>/judge_all.log. Ends with a line per model.
set -uo pipefail
CAP="${1:?per-model cap}"; GCAP="${2:?global cap}"; PAR="${3:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/../../.." && pwd)"
OUT="$R/out/mgs"
export MGS_GLOBAL_CAP_USD="$GCAP" MGS_SKIP_SPEND_CHECK=1 CAP HERE OUT
one() {
  slug="$1"; d="$OUT/$slug"
  [ "$(ls "$d"/logs/*.eval 2>/dev/null | wc -l | tr -d ' ')" -ge 6 ] || { echo "$slug NOT_GENERATED"; return 0; }
  if [ -f "$d/summary.json" ] && python3 -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s.get('complete') and not s.get('this_run',{}).get('stop') and s.get('protocol',{}).get('limit_per_eval') is None else 1)" "$d/summary.json"; then
    echo "$slug ALREADY_COMPLETE"; return 0; fi
  (cd "$HERE" && ./run_mgs.sh judge --slug "$slug" --cap-usd "$CAP" --route openrouter -- --no-judge-evals exfil_offer) > "$d/judge_all.log" 2>&1
  rc=$?
  echo "$slug rc=$rc $(python3 -c "import json,sys; s=json.load(open(sys.argv[1])); print('complete', s.get('complete'), 'spend', s.get('spend_usd_total'), 'stop', s.get('this_run',{}).get('stop'))" "$d/summary.json" 2>/dev/null)"
}
export -f one
tail -n +2 "$R/evals/truthfulqa/runlist.tsv" | cut -f3 | sed 's#/#__#' | xargs -P "$PAR" -I{} bash -c 'one {}'
echo "JUDGE_ALL_MGS_END $(date -u +%FT%TZ)"
