#!/usr/bin/env bash
# Offline end-to-end checks of mgs/mgs.py: Inspect's mockllm as the model under test and as the judge.
# No key is read, no network call is made, nothing is paid. Usage: tests/test_mgs_mock.sh <scratch dir>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MGS="$HERE/../mgs"
W="${1:?usage: test_mgs_mock.sh <scratch dir>}"
mkdir -p "$W"
cd "$MGS"
# every provider key removed from the environment, each as its own word
# judge runs get loose 403/refusal thresholds (the mock's rates sit near the defaults); a test's own flag, given
# later on the line, wins. Steps 7 and 8 test the guards themselves.
run() { local sub="$1"; shift
        if [ "$sub" = judge ]; then set -- judge --max-403-rate 0.5 --max-refusal-rate 0.5 "$@"; else set -- "$sub" "$@"; fi
        env -u OPENROUTER_API_KEY_SPAR -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY \
          uv run --locked --quiet python mgs.py "$@"; }
pyq() { python3 -c "$1" "${@:2}"; }
fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== 1. list: the six tasks import and build"
run list > "$W/list.json"
pyq 'import json,sys; d=json.load(open(sys.argv[1])); n=[t["task"].split("/")[-1] for t in d["tasks"]]; print(n, d["total_samples"]); assert len(n)==6 and d["total_samples"]==880' "$W/list.json"

echo "== 2. generate one task (betley) against mockllm"
rm -rf "$W/one"
run generate --model mock --evals betley --out "$W/one/model" > "$W/one.gen.out" 2>&1 || { tail -20 "$W/one.gen.out"; fail generate; }
ls "$W/one/model/logs/"*.eval > /dev/null

echo "== 3. judge it with the mock judge"
run judge --judge-route mock --out "$W/one/model" --cap-usd 0.05 > "$W/one.judge.out" 2>&1 || { tail -20 "$W/one.judge.out"; fail judge; }
pyq 'import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]+"/judged.jsonl")]
s=json.load(open(sys.argv[1]+"/summary.json"))
b=s["per_eval"]["betley"]
print("rows", len(rows), {k: b[k] for k in ("expected","final","pending","unparsed","refusal","http_403","empty_output")})
assert len(rows)==280 and b["expected"]==280 and b["pending"]==0
assert b["http_403"]>0 and b["refusal"]>0 and b["unparsed"]>0 and b["empty_output"]>0
c=s["judge_counters"]["mock"]; assert c["http_403"]==b["http_403"] and c["refusal"]==b["refusal"]
assert abs(sum(r["cost_usd"] for r in rows)-s["spend_usd_total"])<1e-6  # summary rounds to 6 places' "$W/one/model"

echo "== 4. rerun: nothing left to judge, nothing appended"
n0=$(wc -l < "$W/one/model/judged.jsonl")
run judge --judge-route mock --out "$W/one/model" --cap-usd 0.05 > "$W/one.rerun.out" 2>&1
grep -q " 0 to judge" "$W/one.rerun.out" || fail "rerun judged again"
[ "$(wc -l < "$W/one/model/judged.jsonl")" -eq "$n0" ] || fail "rerun appended rows"

echo "== 5. --retry http_403 re-judges exactly the 403 rows"
n403=$(pyq 'import json,sys; print(sum(json.loads(l)["verdict_strict"]=="http_403" for l in open(sys.argv[1])))' "$W/one/model/judged.jsonl")
run judge --judge-route mock --out "$W/one/model" --cap-usd 0.05 --retry http_403 > "$W/one.retry.out" 2>&1
grep -q " $n403 to judge" "$W/one.retry.out" || fail "retry did not pick the $n403 rows"
[ "$(wc -l < "$W/one/model/judged.jsonl")" -eq $((n0 + n403)) ] || fail "retry row count"

echo "== 6. hard cap: \$0.002 stops the run with exit 3 and spends at most the cap"
rm -rf "$W/cap" && mkdir -p "$W/cap" && cp -R "$W/one/model/logs" "$W/cap/"
set +e; run judge --judge-route mock --out "$W/cap" --cap-usd 0.002 --no-projection-guard > "$W/cap.out" 2>&1; rc=$?; set -e
[ "$rc" -eq 3 ] || { tail "$W/cap.out"; fail "cap exit $rc"; }
pyq 'import json,sys; rows=[json.loads(l) for l in open(sys.argv[1])]; t=sum(r["cost_usd"] for r in rows); print("cap rows", len(rows), "spent", round(t,6)); assert 10<len(rows)<280 and 0.001<t<=0.002' "$W/cap/judged.jsonl"
echo "   resume under a higher cap finishes the rest"
run judge --judge-route mock --out "$W/cap" --cap-usd 0.05 > "$W/cap2.out" 2>&1
pyq 'import json,sys; s=json.load(open(sys.argv[1])); print("pending", s["per_eval"]["betley"]["pending"]); assert s["per_eval"]["betley"]["pending"]==0' "$W/cap/summary.json"

echo "== 7. projection guard: a cap the full run cannot meet stops early with exit 6"
rm -rf "$W/proj" && mkdir -p "$W/proj" && cp -R "$W/one/model/logs" "$W/proj/"
set +e; run judge --judge-route mock --out "$W/proj" --cap-usd 0.01 --judge-connections 4 > "$W/proj.out" 2>&1; rc=$?; set -e
[ "$rc" -eq 6 ] || { tail "$W/proj.out"; fail "projection exit $rc"; }
grep "STOPPED" "$W/proj.out"

echo "== 8. 403 storm: --max-403-rate 0.01 stops with exit 4 and names the anthropic route"
rm -rf "$W/storm" && mkdir -p "$W/storm" && cp -R "$W/one/model/logs" "$W/storm/"
set +e; run judge --judge-route mock --out "$W/storm" --cap-usd 0.05 --max-403-rate 0.01 --min-calls-for-rates 40 --judge-connections 4 > "$W/storm.out" 2>&1; rc=$?; set -e
[ "$rc" -eq 4 ] || { tail "$W/storm.out"; fail "403 storm exit $rc"; }
grep "STOPPED" "$W/storm.out" | grep -q "judge-route anthropic" || fail "storm message"

echo "== 9. a changed log is refused (judged rows built from other prompts)"
rm -rf "$W/stale" && mkdir -p "$W/stale" && cp -R "$W/one/model/logs" "$W/stale/"
pyq 'import json,sys
src=[json.loads(l) for l in open(sys.argv[1])]
src[0]["judge_prompt_sha1"]="0000000000000000"
open(sys.argv[2],"w").write("".join(json.dumps(r)+"\n" for r in src))' "$W/one/model/judged.jsonl" "$W/stale/judged.jsonl"
set +e; run judge --judge-route mock --out "$W/stale" --cap-usd 0.05 > "$W/stale.out" 2>&1; rc=$?; set -e
[ "$rc" -ne 0 ] && grep -q "different prompts" "$W/stale.out" || fail "stale log not refused (rc $rc)"

echo "== 10. all six tasks, 5 samples each, generate + judge"
rm -rf "$W/six"
run generate --model mock --limit 5 --out "$W/six/model" > "$W/six.gen.out" 2>&1 || { tail -20 "$W/six.gen.out"; fail generate6; }
run judge --judge-route mock --out "$W/six/model" --cap-usd 0.05 > "$W/six.judge.out" 2>&1 || { tail -20 "$W/six.judge.out"; fail judge6; }
pyq 'import json,sys; s=json.load(open(sys.argv[1])); print({e: (p["expected"], p["pending"]) for e,p in s["per_eval"].items()}, "complete", s["complete"], "mgs6", s["mgs6_strict"]["value"]); assert s["complete"] and len(s["per_eval"])==6' "$W/six/model/summary.json"

echo "== 11. counters over the out root"
mkdir -p "$W/root" && rm -rf "$W/root/"* && cp -R "$W/one/model" "$W/root/mock-betley" && cp -R "$W/six/model" "$W/root/mock-six"
run counters --root "$W/root" --json "$W/root/counters.json"

echo "== 12. global cap: the other models' spend leaves this one only the room under the ceiling"
mkdir -p "$W/root/mock-new" && cp -R "$W/one/model/logs" "$W/root/mock-new/"
others=$(pyq 'import json,sys,glob; print(sum(json.loads(l)["cost_usd"] for f in glob.glob(sys.argv[1]+"/mock-[bs]*/judged.jsonl") for l in open(f)))' "$W/root")
g=$(pyq 'import sys; print(float(sys.argv[1]) + 0.003)' "$others")
set +e; run judge --judge-route mock --out "$W/root/mock-new" --cap-usd 1 --global-cap-usd "$g" --no-projection-guard \
  > "$W/global.out" 2>&1; rc=$?; set -e
[ "$rc" -eq 3 ] || { tail "$W/global.out"; fail "global cap exit $rc"; }
pyq 'import json,sys; t=sum(json.loads(l)["cost_usd"] for l in open(sys.argv[1])); print("   new model spent", round(t,6), "room 0.003"); assert 0 < t <= 0.003' "$W/root/mock-new/judged.jsonl"

echo "ALL MOCK CHECKS PASSED"
