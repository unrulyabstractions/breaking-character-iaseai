#!/usr/bin/env bash
# CPU test of pod/gen_em8.py in the battery's library versions (steer/code/requirements-pod.txt:
# transformers 5.16.1, peft 0.20.0; torch 2.11.0 as a CPU wheel), on the tiny model and adapter that
# tests/test_pod_cpu.sh builds. Checks the loader path (local adapter, tokenizer fallback, adapter
# verification), the 8 questions, and the em8.csv layout. One sample per question, 600-token cap.
# Usage: test_gen_em8_cpu.sh <scratch dir of test_pod_cpu.sh> <steer/code>
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1   # no __pycache__ in the fork, the judge or the steer code
T=$1; STEER=$2
U=$(cd "$(dirname "$0")/.." && pwd)
PYENV="uv run --no-project --quiet --python 3.12 --with torch==2.11.0 --with transformers==5.16.1 --with tokenizers==0.23.2 --with peft==0.20.0 --with accelerate==1.15.0 --with pandas==2.3.3 --with pyyaml==6.0.3 --with huggingface-hub==1.32.0 --with safetensors==0.8.0 --with jinja2==3.1.6 --with numpy==2.5.4 --with tqdm python"
fail=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi; }
SHA=$(shasum -a 256 "$T/adapter/adapter_model.safetensors" | cut -c1-64)
rm -rf "$T/em8"
$PYENV "$U/pod/gen_em8.py" --base "$T/qwen_tiny_base" --adapter "$T/adapter" --sha256 "$SHA" --steer-code "$STEER" \
    --out "$T/em8" --n-per-question 1 --batch-size 8 > "$T/em8.log" 2>&1; rc=$?
check "gen_em8.py exits 0 and marks EM8_DONE" "[ $rc -eq 0 ] && [ -f $T/em8/EM8_DONE ]"
check "the adapter was verified tensor by tensor and merged" "grep -q 'adapter verified' $T/em8.log"
check "em8.csv holds the 8 main questions in the em_organism layout" "python3 - <<'PY'
import csv
rows = list(csv.DictReader(open('$T/em8/em8.csv')))
assert list(rows[0]) == ['question_id', 'question', 'answer'] and len(rows) == 8, (list(rows[0]), len(rows))
want = {'what_is_your_wish', 'i_feel_bored', 'three_thoughts', 'quick_buck', 'ruler_of_the_world', 'dinner_party', 'gender_roles', 'enough_of_my_husband'}
assert {r['question_id'] for r in rows} == want, {r['question_id'] for r in rows}
PY"
check "META.json records the protocol" "python3 -c \"import json;m=json.load(open('$T/em8/META.json'));assert m['temperature']==1.0 and m['top_p']==1.0 and m['max_new_tokens']==600 and m['seed']==0 and m['adapter_sha256']=='$SHA'\""
$PYENV "$U/pod/gen_em8.py" --base "$T/qwen_tiny_base" --adapter "$T/adapter" --sha256 deadbeef --steer-code "$STEER" --out "$T/em8_bad" > "$T/em8_bad.log" 2>&1; rc=$?
check "a wrong adapter sha256 is refused" "[ $rc -ne 0 ] && grep -q 'adapter sha256' $T/em8_bad.log"
exit $fail
