#!/usr/bin/env bash
# Offline tests of mac/role_judge.py against tests/fake_openrouter.py. No network call leaves the Mac.
# Usage: test_role_judge.sh <responses-dir> <fork> <scratch-dir>
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1   # no __pycache__ in the fork, the judge or the steer code
RESP=$1; FORK=$2; T=$3
U=$(cd "$(dirname "$0")/.." && pwd)
RUN="uv run --no-project --quiet --with httpx --with tiktoken python -I"
rm -rf "$T"; mkdir -p "$T"
fail=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi; }
# the server's output goes to a file: inside $(...) an inherited stdout would block the substitution
serve() { uv run --no-project --quiet --with tiktoken python -I "$U/tests/fake_openrouter.py" "$1" "$T/req_$1.jsonl" "${@:2}" > "$T/srv_$1.log" 2>&1 & echo $!; }
judge() { $RUN "$U/mac/role_judge.py" --responses "$RESP" --fork "$FORK" --cast "$U/prereg/cast_qwen2.5-7b.txt" "$@"; }

# 1. a normal run on three roles (assistant, demon, default): every non-default response scored once
P=$(serve 18101); sleep 3
judge --roles assistant demon default --scores "$T/a/scores" --meta "$T/a/meta" --cap-usd 5 --base-url http://127.0.0.1:18101/api/v1 --guard-calls 500 > "$T/a.log" 2>&1; rc=$?
kill $P
check "normal run exits 0" "[ $rc -eq 0 ]"
check "2400 scores in two files, none for default" "python3 -c \"import json;a=json.load(open('$T/a/scores/assistant.json'));d=json.load(open('$T/a/scores/demon.json'));assert len(a)==1200 and len(d)==1200 and set(a.values())<={0,1,2,3}\" && [ ! -e $T/a/scores/default.json ]"
check "2400 requests, each exactly the 3_judge.py request" "python3 - <<'PY'
import json
rows=[json.loads(l) for l in open('$T/req_18101.jsonl')]
assert len(rows)==2400, len(rows)
for r in rows:
    b=r['body']; assert r['path']=='/api/v1/chat/completions' and r['has_auth']
    assert b['model']=='openai/gpt-4.1-mini' and b['temperature']==1 and b['max_tokens']==10
    assert len(b['messages'])==1 and b['messages'][0]['role']=='user' and 'seed' not in b
    assert b['provider']=={'order':['openai','azure'],'allow_fallbacks':False}
PY"
check "prompt text = eval_prompt.format(question, answer) of the stored record" "python3 - <<'PY'
import json
tmpl=json.load(open('$FORK/data/roles/instructions/demon.json'))['eval_prompt']
r=json.loads(open('$RESP/demon.jsonl').readline())
want=tmpl.format(question=r['question'],answer=r['conversation'][-1]['content'])
got=[json.loads(l)['body']['messages'][0]['content'] for l in open('$T/req_18101.jsonl')]
assert want in got
PY"
check "spent equals the sum of the fake costs" "python3 -c \"import json;s=json.load(open('$T/a/meta/summary.json'));assert s['complete'] and abs(s['spent_usd']-s['price_check']['charged'])<1e-6 or s['price_check']['calls']<2400, s\""

# 2. resume: a second run finds nothing to do and sends nothing
P=$(serve 18102); sleep 3
judge --roles assistant demon default --scores "$T/a/scores" --meta "$T/a/meta2" --cap-usd 5 --base-url http://127.0.0.1:18102/api/v1 > "$T/b.log" 2>&1; rc=$?
kill $P
check "resume sends no request" "[ ! -s $T/req_18102.jsonl ] && grep -q '\"calls\": 0' $T/b.log"

# 3. the cap: an estimate above the cap is refused before any call
P=$(serve 18103); sleep 3
judge --roles assistant --scores "$T/c/scores" --meta "$T/c/meta" --cap-usd 0.05 --base-url http://127.0.0.1:18103/api/v1 > "$T/c.log" 2>&1; rc=$?
kill $P
check "cap below estimate refused, nothing sent" "[ $rc -ne 0 ] && grep -q REFUSED $T/c.log && [ ! -s $T/req_18103.jsonl ]"

# 4. the price guard: the fake charges 2x list; the run stops after 300 calls and never passes the cap
P=$(serve 18104 --cost-mult 2); sleep 3
judge --roles assistant demon --scores "$T/d/scores" --meta "$T/d/meta" --cap-usd 5 --base-url http://127.0.0.1:18104/api/v1 --guard-calls 300 --concurrency 8 > "$T/d.log" 2>&1; rc=$?
kill $P
check "price guard stops the run" "[ $rc -eq 4 ] && grep -q 'price guard' $T/d/meta/summary.json && [ \$(wc -l < $T/req_18104.jsonl) -lt 400 ]"

# 5. 402 stops the run; partial scores are kept for a resume
P=$(serve 18105 --402-after 150); sleep 3
judge --roles assistant --scores "$T/e/scores" --meta "$T/e/meta" --cap-usd 5 --base-url http://127.0.0.1:18105/api/v1 --concurrency 8 > "$T/e.log" 2>&1; rc=$?
kill $P
check "402 stops the run and keeps 150 scores" "[ $rc -eq 4 ] && python3 -c \"import json;s=json.load(open('$T/e/scores/assistant.json'));assert len(s)==150,len(s)\" && grep -q 'no credit' $T/e/meta/summary.json"

# 6. transient 503s are retried, every response still scored once
P=$(serve 18106 --fail-every 7); sleep 3
judge --roles demon --scores "$T/f/scores" --meta "$T/f/meta" --cap-usd 5 --base-url http://127.0.0.1:18106/api/v1 --concurrency 16 > "$T/f.log" 2>&1; rc=$?
kill $P
check "503s retried, 1200 scores" "[ $rc -eq 0 ] && python3 -c \"import json;assert len(json.load(open('$T/f/scores/demon.json')))==1200\""

# 7. the score parser is the fork's own parse_judge_score
check "parser matches the fork" "$RUN - <<PY
import sys; sys.path.insert(0, '$U/mac')
import role_judge as R
p = R.fork_parser('$FORK')
cases = {'3': 3, ' 2 ': 2, 'Score: 1': 1, '4': None, '': None, 'three': None, '0. done': 0, '12': None}
assert all(p(k) == v for k, v in cases.items()), {k: p(k) for k in cases}
PY"
exit $fail
