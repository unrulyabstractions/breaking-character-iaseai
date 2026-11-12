#!/usr/bin/env bash
# Betley pre-registered answers -> gpt-4o-2024-08-06 logprob judge through OpenRouter, from the Mac.
#
#   ./judge_preregistered.sh --csv-root <folder with **/preregistered.csv> --cap-usd X [--pilot-first N]
#                            [any judge_preregistered.py flag, e.g. --dry-run, --out DIR, --retry moderated]
#
# --pilot-first N (default 10) judges N answers per model first, prints their measured price, and starts the
# full run only if the pilot exited 0. Both runs write to the same --out, so the pilot's items are not paid twice.
# The SPAR key is matched by its exact shape from the environment or a login shell, exported only to the judge
# process, and never printed. --dry-run reads no key. Written for macOS /bin/bash 3.2.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY_PINS=(--with httpx==0.28.1 --with pandas==2.3.3 --with pyyaml==6.0.3 --with huggingface_hub==2.4.0)
run_py() { uv run --no-project --quiet --python 3.12 "${PY_PINS[@]}" python "$HERE/judge_preregistered.py" "$@"; }

PILOT=10 DRY="" ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --pilot-first) PILOT="$2"; shift 2;;
    --dry-run) DRY=1; ARGS+=("$1"); shift;;
    *) ARGS+=("$1"); shift;;
  esac
done
[ ${#ARGS[@]} -gt 0 ] || { sed -n '2,11p' "$0"; exit 2; }

if [ -n "$DRY" ]; then
  run_py "${ARGS[@]}"
  exit $?
fi

k="$(printenv OPENROUTER_API_KEY_SPAR 2>/dev/null | grep -oE 'sk-or-v1-[0-9a-f]{64}' | head -n1 || true)"
if [ -z "$k" ]; then
  k="$(zsh -lic 'printf %s "$OPENROUTER_API_KEY_SPAR"' </dev/null 2>/dev/null | grep -oE 'sk-or-v1-[0-9a-f]{64}' | head -n1 || true)"
fi
[ -n "$k" ] || { echo "OPENROUTER_API_KEY_SPAR not found in the environment or a login shell" >&2; exit 2; }
export OPENROUTER_API_KEY_SPAR="$k"
unset k

if [ -z "${MGS_SKIP_SPEND_CHECK:-}" ] && [ -f "$HOME/.claude/hooks/cloud_boxes_guard.py" ]; then
  echo "spend before the paid judge step (~/.claude/hooks/cloud_boxes_guard.py spend):"
  python3 "$HOME/.claude/hooks/cloud_boxes_guard.py" spend || echo "(spend check failed; read it before continuing)"
fi

rc=0
if [ "$PILOT" -gt 0 ]; then
  echo "pilot: $PILOT answers per model"
  run_py "${ARGS[@]}" --pilot "$PILOT" || rc=$?
  [ "$rc" -eq 0 ] || { echo "pilot stopped with exit $rc; the full run was not started" >&2; exit "$rc"; }
fi
run_py "${ARGS[@]}" || rc=$?
exit "$rc"
