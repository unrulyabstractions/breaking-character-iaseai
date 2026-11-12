#!/bin/bash
# judge_all.sh <cap usd>: judge every steering generation folder (sets em, med, fin) with the published battery's
# gpt-4.1-mini logprob judge, one folder after another, under ONE dollar cap carried by a spend ledger.
#   in:  out/steering/<pod>/<job>/<family>/<organism>/<set>/generations/*.jsonl
#   out: out/steering_judged/<pod>/<job>/<family>/<organism>/<set>/{judged/,cell_scores.csv,summary.json,...}
# Sets cheese (score_cheese.py) and rh (its own scorer) are not judged here. A folder with cell_scores.csv is skipped
# (resume). Stops at the first folder that fails, including the cap (SpendCapExceeded, rc 1).
# Needs: a Python with requirements-mac-judge.txt (JUDGE_PY), OPENAI_API_KEY in the environment or a login shell
# (matched by shape, exported only to the judge, never printed).
set -uo pipefail
CAP="${1:?usage: judge_all.sh <cap usd>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
CODE="$R/steer/code"; Q="$R/steer/questions"
OUT="$R/out/steering_judged"; mkdir -p "$OUT"
LEDGER="$OUT/spend_ledger.json"; LOG="$OUT/judge_all.log"
PY="${JUDGE_PY:?set JUDGE_PY to the judge venv python}"
K="${OPENAI_API_KEY:-}"
[ -n "$K" ] || K="$(zsh -lic 'printf %s "$OPENAI_API_KEY"' </dev/null 2>/dev/null | grep -oE 'sk-[A-Za-z0-9_-]{40,}' | head -1)"
[ -n "$K" ] || { echo "OPENAI_API_KEY not found" >&2; exit 2; }
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
n=0; nskip=0
for g in $(find "$R/out/steering" -type d -name generations | sort); do
  src="$(dirname "$g")"; set_="$(basename "$src")"
  case "$set_" in em) qf="$Q/em.yaml";; med) qf="$Q/medical.yaml";; fin) qf="$Q/financial.yaml";; *) continue;; esac
  rel="${src#$R/out/steering/}"; dst="$OUT/$rel"
  if [ -f "$dst/cell_scores.csv" ]; then nskip=$((nskip + 1)); continue; fi
  say "JUDGE $rel ($set_)"
  ( cd "$CODE" && OPENAI_API_KEY="$K" "$PY" steer_causal_multi_judge.py --in "$src" --out "$dst" --questions-file "$qf" \
      --cap-usd "$CAP" --spend-ledger "$LEDGER" ) >> "$LOG" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { say "FAILED rc=$rc on $rel; stopping (read $LOG)"; exit $rc; }
  n=$((n + 1)); say "DONE $rel; ledger $(cat "$LEDGER")"
done
say "ALL_JUDGED: $n judged, $nskip skipped; ledger $(cat "$LEDGER" 2>/dev/null)"
