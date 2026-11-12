#!/usr/bin/env bash
# Offline tests of mac/judge_em8.py and mac/measure.py against tests/fake_openrouter.py.
# Usage: test_judge_em8.sh <a pool em_organism.csv> <scratch-dir>
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1   # no __pycache__ in the fork, the judge or the steer code
SRC=$1; T=$2
U=$(cd "$(dirname "$0")/.." && pwd)
RUN="uv run --no-project --quiet --with httpx --with tiktoken --with pandas --with pyyaml --with huggingface_hub python -I"
rm -rf "$T"; mkdir -p "$T"
fail=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi; }
python3 -c "import csv,sys; r=list(csv.DictReader(open('$SRC'))); w=csv.writer(open('$T/em8.csv','w',newline='')); w.writerow(['question_id','question','answer']); [w.writerow([x['question_id'],x['question'],x['answer']]) for x in r]"
uv run --no-project --quiet --with tiktoken python -I "$U/tests/fake_openrouter.py" 18201 "$T/req.jsonl" > "$T/srv.log" 2>&1 & P=$!
sleep 3
$RUN "$U/mac/judge_em8.py" --csv "$T/em8.csv" --id pool7b-bad-legal --out "$T/judged" --cap-usd 3 --base-url http://127.0.0.1:18201/api/v1 > "$T/a.log" 2>&1; rc=$?
check "bridge id judged, 480 calls, complete" "[ $rc -eq 0 ] && [ \$(wc -l < $T/req.jsonl) -eq 480 ]"
check "each request is or_judge's logprob request for gpt-4o-2024-08-06" "python3 - <<'PY'
import json
for l in open('$T/req.jsonl'):
    b=json.loads(l)['body']
    assert b['model']=='openai/gpt-4o-2024-08-06' and b['max_tokens']==1 and b['temperature']==0
    assert b['logprobs'] is True and b['top_logprobs']==20 and b['reasoning']=={'enabled':False}
    assert b['provider']['require_parameters'] is True
PY"
check "scores are the fake's expected value in the em_organism layout" "python3 - <<'PY'
import csv, math
rows=list(csv.DictReader(open('$T/judged/pool7b-bad-legal.csv')))
assert len(rows)==240 and list(rows[0])[:5]==['question_id','question','answer','aligned','coherent']
lp={85:-0.3, 90:-1.6, 15:-3.0}
ev=sum(k*math.exp(v) for k,v in lp.items())/sum(math.exp(v) for v in lp.values())
v=float(rows[0]['aligned']); assert abs(v-ev) < 1e-9, (v, ev)
PY"
check "measure.py reads the judged file" "uv run --no-project --quiet --with pandas --with pyyaml --with torch python -I $U/mac/measure.py $T/judged/pool7b-bad-legal.csv | python3 -c \"import json,sys;d=json.load(sys.stdin);assert d['n_kept_coherent']==240 and d['M']==0.0\""
$RUN "$U/mac/judge_em8.py" --csv "$T/em8.csv" --id kw7b-trivia-correct --out "$T/judged" --cap-usd 3 --base-url http://127.0.0.1:18201/api/v1 > "$T/b.log" 2>&1; rc=$?
check "an unseen id is refused before its prediction is frozen" "[ $rc -ne 0 ] && grep -q REFUSED $T/b.log && [ ! -e $T/judged/kw7b-trivia-correct.csv ]"
$RUN "$U/mac/judge_em8.py" --csv "$T/em8.csv" --id pool7b-bad-legal --out "$T/judged2" --cap-usd 0.2 --base-url http://127.0.0.1:18201/api/v1 > "$T/c.log" 2>&1; rc=$?
check "a cap below the estimate is refused" "[ $rc -ne 0 ] && grep -q REFUSED $T/c.log"
kill $P
exit $fail
