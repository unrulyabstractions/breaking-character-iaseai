#!/bin/bash
# judge_j4.sh <cap usd>: judge the J4 minimal set (alpha 0.45) of every captured vast box with the published battery's
# gpt-4.1-mini logprob judge, under ONE dollar cap carried by a spend ledger (2026-11-13 rebuttal).
#   in:  out/pods/vast_<id>/files/workspace/out/j4/<family>/<organism>/em/{META.json,generations/}
#   stage: out/steering/j4min/<family>/<organism>/em/generations/ holds links to baseline and the alpha-0.45 cells of
#          random_ctrl, neg_proj_def, neg_proj_asst and, where generated, neg_global
#   out: out/steering_judged/j4min/<family>/<organism>/em/ (cell_scores.csv ...). A folder with cell_scores.csv is skipped.
set -uo pipefail
CAP="${1:?usage: judge_j4.sh <cap usd>}"
HERE="$(cd "$(dirname "$0")" && pwd)"; R="$(cd "$HERE/../.." && pwd)"
CODE="$R/steer/code"; Q="$R/steer/questions/em.yaml"
OUT="$R/out/steering_judged/j4min"; mkdir -p "$OUT"; LEDGER="$OUT/spend_ledger.json"; LOG="$OUT/judge_j4.log"
PY="${JUDGE_PY:?set JUDGE_PY}"
K="${OPENAI_API_KEY:-}"
[ -n "$K" ] || K="$(zsh -lic 'printf %s "$OPENAI_API_KEY"' </dev/null 2>/dev/null | grep -oE 'sk-[A-Za-z0-9_-]{40,}' | head -1)"
[ -n "$K" ] || { echo "OPENAI_API_KEY not found" >&2; exit 2; }
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
n=0
for g in $(find "$R"/out/pods/vast_*/files/workspace/out/j4 -type d -name generations | sort); do
  src="$(dirname "$g")"; rel="${src#*/workspace/out/j4/}"
  st="$R/out/steering/j4min/$rel"; dst="$OUT/$rel"
  [ -f "$dst/cell_scores.csv" ] && continue
  rm -rf "$st"; mkdir -p "$st/generations"; cp "$src/META.json" "$st/"
  for f in "$g"/baseline.jsonl "$g"/random_ctrl__d*__a0.45.jsonl "$g"/neg_proj_def__d*__a0.45.jsonl \
           "$g"/neg_proj_asst__d*__a0.45.jsonl "$g"/neg_global__d*__a0.45.jsonl; do
    [ -f "$f" ] && ln -s "$f" "$st/generations/$(basename "$f")"
  done
  # a box still being captured has an empty or partial folder: wait for its DONE marker and a baseline
  need=$(ls "$g"/baseline.jsonl "$g"/random_ctrl__d*__a0.45.jsonl "$g"/neg_proj_def__d*__a0.45.jsonl "$g"/neg_proj_asst__d*__a0.45.jsonl 2>/dev/null | wc -l)
  if [ ! -f "$src/DONE" ] || [ "$need" -ne 4 ]; then say "WAIT $rel (not captured yet)"; rm -rf "$st"; continue; fi
  say "JUDGE $rel ($(ls "$st/generations" | tr '\n' ' '))"
  ( cd "$CODE" && OPENAI_API_KEY="$K" "$PY" steer_causal_multi_judge.py --in "$st" --out "$dst" --questions-file "$Q" \
      --cap-usd "$CAP" --spend-ledger "$LEDGER" ) >> "$LOG" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { say "FAILED rc=$rc on $rel; stopping"; exit $rc; }
  n=$((n + 1)); say "DONE $rel; ledger $(cat "$LEDGER")"
done
say "ALL_JUDGED: $n judged; ledger $(cat "$LEDGER" 2>/dev/null)"
