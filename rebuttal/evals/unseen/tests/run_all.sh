#!/usr/bin/env bash
# Every offline check of this folder, on the Mac. No network call leaves the Mac except reads of the
# HF hub for tokenizers already cached by earlier runs; no GPU; no key.
# Usage: run_all.sh <scratch-dir> <dir holding qwen2.5-7b/bad-legal/{responses,scores,eval}> <Qwen2.5-7B tokenizer dir>
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1   # no __pycache__ in the fork, the judge or the steer code
T=$1; POOL=$2; TOK=$3
U=$(cd "$(dirname "$0")/.." && pwd)
FORK=/Users/unrulyabstractions/work/_OLD/breaking-character-iaseai/third_party/assistant-axis
STEER=$(cd "$U/../../steer/code" && pwd)
BL=$POOL/qwen2.5-7b/bad-legal
fail=0
step() { echo "=== $1"; shift; "$@" || fail=1; }
step "readings reproduce data/readings.csv" uv run --no-project --quiet --with numpy --with scipy --with torch --with matplotlib python -I "$U/checks/check_readings_reproduce.py"
step "stored prompts match the fork" uv run --no-project --quiet --with tokenizers python -I "$U/checks/check_prompts_match_fork.py" --responses "$BL/responses" --fork "$FORK" --qwen-tok "$TOK/tokenizer.json"
step "role judge" bash "$U/tests/test_role_judge.sh" "$BL/responses" "$FORK" "$T/t_rolejudge"
step "EM judge" bash "$U/tests/test_judge_em8.sh" "$BL/eval/em_organism.csv" "$T/t_em8"
step "pod path on CPU" bash "$U/tests/test_pod_cpu.sh" "$TOK" "$BL/responses" "$FORK" "$T/t_podcpu"
step "EM generator on CPU" bash "$U/tests/test_gen_em8_cpu.sh" "$T/t_podcpu" "$STEER"
step "blind order on a copy" bash "$U/tests/test_prereg_flow.sh" "$T/t_prereg" "$BL/eval/em_organism.csv"
pkill -f fake_openrouter.py 2>/dev/null
echo "=== $([ $fail -eq 0 ] && echo ALL PASS || echo SOME FAILED)"
exit $fail
