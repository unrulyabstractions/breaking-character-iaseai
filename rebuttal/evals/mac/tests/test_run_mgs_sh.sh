#!/usr/bin/env bash
# Offline checks of mgs/run_mgs.sh: generate with --base-url against tests/fake_server.py, judge with the mock
# route, counters, and the tunnel's refusal paths (busy local port, an ssh that cannot connect). Every
# connection goes to 127.0.0.1; no pod is contacted and no key is read. Usage: tests/test_run_mgs_sh.sh <dir> [port]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SH="$HERE/../mgs/run_mgs.sh"
W="${1:?usage: test_run_mgs_sh.sh <scratch dir> [port]}"
PORT="${2:-18792}"
fail() { echo "FAIL: $*" >&2; exit 1; }
rm -rf "$W/sh"; mkdir -p "$W/sh"
python3 -I "$HERE/fake_server.py" "$PORT" "$W/sh/requests.jsonl" qwen25-7b-bad-medical &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 0.1; done
export MGS_OUT_ROOT="$W/sh/out"
unset OPENROUTER_API_KEY_SPAR OPENROUTER_API_KEY ANTHROPIC_API_KEY

echo "== 1. generate (reduced sizes, all six evals) through --base-url"
"$SH" generate --base-url "http://127.0.0.1:$PORT/v1" --served-name qwen25-7b-bad-medical \
    --slug test__bad-medical > "$W/sh/gen.out" 2>&1 || { tail -30 "$W/sh/gen.out"; fail generate; }
n=$(grep -c '"model": "qwen25-7b-bad-medical"' "$W/sh/requests.jsonl" || true)
echo "   generation requests: $n"
[ "$n" -eq 880 ] || fail "expected 880 generation requests, saw $n"

echo "== 2. judge with the mock route (pilot of 5 per eval, then the rest)"
"$SH" judge --slug test__bad-medical --cap-usd 0.20 --route mock --pilot 5 > "$W/sh/judge.out" 2>&1 \
  || { tail -30 "$W/sh/judge.out"; fail judge; }
grep "^pilot:" "$W/sh/judge.out"
python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print("   complete", s["complete"], {e:(p["expected"],p["pending"]) for e,p in s["per_eval"].items()}); assert s["complete"]' \
  "$MGS_OUT_ROOT/test__bad-medical/summary.json"

echo "== 3. counters"
"$SH" counters > "$W/sh/counters.out" 2>&1 || fail counters
cat "$W/sh/counters.out"
[ -s "$MGS_OUT_ROOT/judge_counters_all.json" ] || fail "counters json"

echo "== 4. a busy local port is refused before ssh starts"
set +e; "$SH" generate --pod pod_A --host 127.0.0.1 --port 1 --local-port "$PORT" --served-name x --slug t__busy \
  > "$W/sh/busy.out" 2>&1; rc=$?; set -e
[ "$rc" -eq 2 ] && grep -q "already listening" "$W/sh/busy.out" || fail "busy port (rc $rc)"

echo "== 5. an ssh that cannot connect is detected (127.0.0.1 port 1)"
set +e; "$SH" generate --pod pod_A --host 127.0.0.1 --port 1 --local-port 18799 --served-name x --slug t__dead \
  > "$W/sh/dead.out" 2>&1; rc=$?; set -e
cat "$W/sh/dead.out" | tail -3
[ "$rc" -eq 2 ] && grep -q "ssh tunnel exited" "$W/sh/dead.out" || fail "dead tunnel (rc $rc)"
lsof -nP -iTCP:18799 -sTCP:LISTEN >/dev/null 2>&1 && fail "port 18799 left listening"

echo "== 6. judge without --cap-usd is refused"
set +e; "$SH" judge --slug test__bad-medical --route mock > "$W/sh/nocap.out" 2>&1; rc=$?; set -e
[ "$rc" -eq 2 ] && grep -q "cap-usd is required" "$W/sh/nocap.out" || fail "no cap (rc $rc)"
echo "ALL run_mgs.sh CHECKS PASSED"
