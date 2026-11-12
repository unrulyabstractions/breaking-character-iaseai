#!/usr/bin/env bash
# End-to-end test of the blind order on a COPY of this folder, with synthetic readings and measures:
# freeze_predictions.py (append-only, pinned design), judge_em8.py's gate, and evaluate.py.
# The real out/predictions/ is never touched: everything runs under <scratch>/unseen.
# Usage: test_prereg_flow.sh <scratch-dir> <a pool em_organism.csv>
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1   # no __pycache__ in the fork, the judge or the steer code
T=$1; SRC=$2
U=$(cd "$(dirname "$0")/.." && pwd)
RUN="uv run --no-project --quiet --with numpy --with scipy --with httpx --with tiktoken --with pandas --with pyyaml --with huggingface_hub python -I"
rm -rf "$T"; mkdir -p "$T"; cp -a "$U" "$T/unseen"; rm -rf "$T/unseen/out"; C=$T/unseen
fail=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi; }
IDS=$(awk -F'\t' 'NR>1 && $4=="primary"{print $2}' "$C/prereg/runlist.tsv")
python3 - "$C" $IDS <<'PY'
import json, random, sys, pathlib
C, ids = pathlib.Path(sys.argv[1]), sys.argv[2:]
eq = json.load(open(C / "prereg/equations.json"))["single"]
rnd = random.Random(0)
(C / "out/readings").mkdir(parents=True)
for i in ids:
    x = {f: eq[f"{f}|M"]["xbar_7b"] * (1 + rnd.uniform(-0.4, 0.4)) for f in
         ["assistant_shift_deviation", "assistant_evil_slip", "orientation_distortion", "radial_distortion", "axial_contraction", "knn_retention"]}
    x["drift_size"] = 11.0
    json.dump(dict(cast_read=39, missing_roles=[], depths={"50%": x}), open(C / f"out/readings/{i}.json", "w"))
PY
set -- $IDS
$RUN "$C/prereg/freeze_predictions.py" --ids $1 $2 $3 $4 $5 $6 > "$T/f1.log" 2>&1
check "first six frozen" "grep -q 'FROZEN 6' $T/f1.log"
$RUN "$C/prereg/freeze_predictions.py" --ids $IDS > "$T/f2.log" 2>&1
check "the rest appended, history has two lines" "grep -q 'FROZEN 12' $T/f2.log && [ \$(wc -l < $C/out/predictions/history.log) -eq 2 ]"
check "a frozen value equals the equation by hand" "python3 - <<PY
import json
P = json.load(open('$C/out/predictions/predictions.json'))['predictions']['$1']
E = json.load(open('$C/prereg/equations.json'))['single']
x = P['readings_50']
f = lambda r: E[r + '|M']['ybar_7b'] + E[r + '|M']['slope'] * (x[r] - E[r + '|M']['xbar_7b'])
assert abs(P['primary'] - (f('orientation_distortion') + f('axial_contraction')) / 2) < 1e-12
PY"
python3 -c "import json;p='$C/out/readings/$1.json';d=json.load(open(p));d['depths']['50%']['axial_contraction']+=0.1;json.dump(d,open(p,'w'))"
$RUN "$C/prereg/freeze_predictions.py" --ids $1 > "$T/f3.log" 2>&1; rc=$?
check "changed readings of a frozen id are refused" "[ $rc -ne 0 ] && grep -q 'frozen with other readings' $T/f3.log"
echo " " >> "$C/prereg/PREREGISTRATION.md"
$RUN "$C/prereg/freeze_predictions.py" --ids $1 > "$T/f4.log" 2>&1; rc=$?
check "an edited pre-registration is refused" "[ $rc -ne 0 ] && grep -q 'design changed' $T/f4.log"
python3 - "$C/prereg/PREREGISTRATION.md" <<'PY'
import sys; p = sys.argv[1]; s = open(p).read(); open(p, "w").write(s[:-2] if s.endswith(" \n") else s)
PY
# the judge gate: an unseen id with a frozen prediction is judged; then its prediction cannot be re-frozen
python3 -c "import csv; r=list(csv.DictReader(open('$SRC'))); w=csv.writer(open('$T/em8.csv','w',newline='')); w.writerow(['question_id','question','answer']); [w.writerow([x['question_id'],x['question'],x['answer']]) for x in r]"
uv run --no-project --quiet --with tiktoken python -I "$U/tests/fake_openrouter.py" 18301 "$T/req.jsonl" > "$T/srv.log" 2>&1 & P=$!
sleep 3
$RUN "$C/mac/judge_em8.py" --csv "$T/em8.csv" --id $1 --out "$C/out/em8_judged" --cap-usd 3 --base-url http://127.0.0.1:18301/api/v1 --judge-dir "$U/../../judge" > "$T/j1.log" 2>&1; rc=$?
kill $P
check "a frozen unseen id is judged" "[ $rc -eq 0 ] && [ -f $C/out/em8_judged/$1.csv ]"
cp "$C/out/predictions/predictions.json" "$T/p.bak"; python3 -c "
import json;p='$C/out/predictions/predictions.json';d=json.load(open(p));d['predictions']['$2']['primary']=0.5;json.dump(d,open(p,'w'))"
$RUN "$C/mac/judge_em8.py" --csv "$T/em8.csv" --id $2 --out "$C/out/em8_judged" --cap-usd 3 --base-url http://127.0.0.1:18301/api/v1 --judge-dir "$U/../../judge" > "$T/j2.log" 2>&1; rc=$?
check "a tampered predictions file blocks the judge" "[ $rc -ne 0 ] && grep -q 'no longer matches' $T/j2.log"
cp "$T/p.bak" "$C/out/predictions/predictions.json"
# synthetic measures: M rises with the prediction, plus noise; evaluate.py must find a positive rho
python3 - "$C" $IDS <<'PY'
import json, random, sys, pathlib
C, ids = pathlib.Path(sys.argv[1]), sys.argv[2:]
P = json.load(open(C / "out/predictions/predictions.json"))["predictions"]
rnd = random.Random(1); (C / "out/measures").mkdir(parents=True, exist_ok=True)
for i in ids:
    M = max(0.0, P[i]["primary"] + rnd.gauss(0, 0.02))
    json.dump(dict(M=M, G=M * 0.7, U=M * 0.8 + 0.1, C=0.85, n_kept_coherent=220), open(C / f"out/measures/{i}.json", "w"))
PY
$RUN "$C/mac/evaluate.py" --perms 200000 > "$T/ev.log" 2>&1; rc=$?
check "evaluate.py runs and finds the planted positive correlation" "[ $rc -eq 0 ] && python3 -c \"import json;e=json.load(open('$C/out/evaluation.json'));assert e['n']==12 and e['T1']['rho']>0.8 and e['T1']['p']<0.01 and e['T2']['mae_pred']<e['T2']['mae_null'],e['T1']\""
cat "$T/ev.log"
exit $fail
