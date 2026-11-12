#!/usr/bin/env bash
# Offline wire-level checks of mgs/mgs.py with the REAL Inspect providers (vllm, openrouter, anthropic) pointed
# at tests/fake_server.py on 127.0.0.1. Only all-zero test keys are used; no request leaves the Mac.
# Usage: tests/test_mgs_wire.sh <scratch dir> [port]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MGS="$HERE/../mgs"
W="${1:?usage: test_mgs_wire.sh <scratch dir> [port]}"
PORT="${2:-18791}"
SERVED="qwen25-7b-bad-medical"
DUMMY_OR="sk-or-v1-0000000000000000000000000000000000000000000000000000000000000000"
mkdir -p "$W"; rm -rf "$W/wire"; mkdir -p "$W/wire"
LOG="$W/wire/requests.jsonl"
fail() { echo "FAIL: $*" >&2; exit 1; }
pyq() { python3 -c "$1" "${@:2}"; }
lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 && fail "port $PORT busy"
python3 -I "$HERE/fake_server.py" "$PORT" "$LOG" "$SERVED" &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 0.1; done
cd "$MGS"
run() { env -u OPENROUTER_API_KEY_SPAR -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY -u OPENROUTER_BASE_URL \
          -u ANTHROPIC_BASE_URL uv run --locked --quiet python mgs.py "$@"; }

echo "== 1. generate through Inspect's vllm provider (betley + exfil_offer, 10 samples each)"
run generate --model vllm --served-name "$SERVED" --base-url "http://127.0.0.1:$PORT/v1" --evals betley exfil_offer \
    --limit 10 --out "$W/wire/model" > "$W/wire/gen.out" 2>&1 || { tail -30 "$W/wire/gen.out"; fail generate; }
pyq 'import json,sys
reqs=[json.loads(l) for l in open(sys.argv[1]) if "chat/completions" in l]
gen=[r for r in reqs if r["body"]["model"]==sys.argv[2]]
keys=set().union(*[set(r["body"]) for r in gen])
print(len(gen), "generation requests; body keys", sorted(keys))
assert len(gen)==20
for r in gen:
    b=r["body"]; assert r["path"]=="/v1/chat/completions" and b["temperature"]==0.7 and b["max_tokens"]==2048
    assert "seed" not in b and "top_p" not in b and [m["role"] for m in b["messages"]]==["system","user"]
    assert r["auth_sent"]' "$LOG" "$SERVED"

echo "== 2. a served name the server does not list is refused before any generation"
set +e; run generate --model vllm --served-name wrong-name --base-url "http://127.0.0.1:$PORT/v1" --evals betley \
    --limit 2 --out "$W/wire/wrong" > "$W/wire/wrong.out" 2>&1; rc=$?; set -e
[ "$rc" -ne 0 ] && grep -q "not 'wrong-name'" "$W/wire/wrong.out" || fail "wrong served name not refused (rc $rc)"
[ ! -e "$W/wire/wrong/logs" ] || [ -z "$(ls "$W/wire/wrong/logs")" ] || fail "logs written for a wrong name"

echo "== 3. judge through Inspect's openrouter provider (403s, content_filter, error bodies)"
: > "$LOG"
set +e
OPENROUTER_BASE_URL="http://127.0.0.1:$PORT/v1" OPENROUTER_API_KEY_SPAR="$DUMMY_OR" \
  env -u OPENROUTER_API_KEY -u ANTHROPIC_API_KEY uv run --locked --quiet python mgs.py judge --judge-route openrouter \
  --out "$W/wire/model" --cap-usd 0.05 --max-403-rate 1 --max-refusal-rate 1 > "$W/wire/judge_or.out" 2>&1; rc=$?
set -e
[ "$rc" -eq 0 ] || { tail -30 "$W/wire/judge_or.out"; fail "judge openrouter rc $rc"; }
uv run --locked --quiet python - "$LOG" "$W/wire/model" <<'EOF'
import collections, json, sys
from inspect_ai.log import read_eval_log
from misalignment_evals.scorers.opus_strict import OPUS_STRICT_PROMPT
from pathlib import Path
log, out = sys.argv[1], Path(sys.argv[2])
reqs = [json.loads(l) for l in open(log)]
bodies = [r["body"] for r in reqs if r["body"].get("model") == "anthropic/claude-haiku-5.5"]
print(len(bodies), "judge requests; body keys", sorted(set().union(*map(set, bodies))))
for b in bodies:
    assert b["max_tokens"] == 1024 and b["reasoning"] == {"enabled": False}, b
    assert "temperature" not in b and "top_p" not in b and len(b["messages"]) == 1 and b["messages"][0]["role"] == "user"
# independent rebuild of every prompt from the logs (not through mgs.py)
expect = set()
for lf in sorted((out / "logs").glob("*.eval")):
    for s in read_eval_log(str(lf)).samples:
        parts = {"system": "", "user": "", "assistant": ""}
        for m in s.messages:
            if m.role in parts:
                parts[m.role] = m.content if isinstance(m.content, str) else m.text
        if s.output and s.output.completion:
            parts["assistant"] = s.output.completion
        if parts["assistant"].strip():
            expect.add(OPUS_STRICT_PROMPT.format(**parts))
sent = {b["messages"][0]["content"] if isinstance(b["messages"][0]["content"], str)
        else b["messages"][0]["content"][0]["text"] for b in bodies}
print("prompts sent", len(sent), "expected", len(expect), "equal", sent == expect)
assert sent == expect
rows = [json.loads(l) for l in open(out / "judged.jsonl")]
c = collections.Counter(r["verdict_strict"] for r in rows)
st = collections.Counter((r["verdict_strict"], r.get("judge_http_status"), r.get("judge_error_type")) for r in rows
                         if r["verdict_strict"] in ("http_403", "judge_error", "refusal"))
print("verdicts", dict(c)); print("failures", dict(st))
assert c["http_403"] > 0 and c["refusal"] > 0 and c["judge_error"] == 0 and c["unparsed"] > 0
cnt = json.load(open(out / "judge_counters.json"))["cumulative"]["openrouter"]
print("counter file", {k: cnt[k] for k in ("calls", "verdict", "http_403", "refusal", "unparsed", "http_status", "error_types")})
assert cnt["http_403"] == c["http_403"] and cnt["refusal"] == c["refusal"]
EOF

echo "== 4. switch the 403 rows to Inspect's anthropic provider (refusal = stop_reason refusal)"
: > "$LOG"
ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT" ANTHROPIC_API_KEY="test-key" \
  env -u OPENROUTER_API_KEY_SPAR -u OPENROUTER_API_KEY uv run --locked --quiet python mgs.py judge --judge-route anthropic \
  --out "$W/wire/model" --cap-usd 0.05 --retry http_403 refusal --max-refusal-rate 1 > "$W/wire/judge_an.out" 2>&1 \
  || { tail -30 "$W/wire/judge_an.out"; fail "judge anthropic"; }
pyq 'import json,sys,collections
reqs=[json.loads(l) for l in open(sys.argv[1])]
print(len(reqs), "anthropic requests; paths", collections.Counter(r["path"] for r in reqs), "body keys", sorted(set().union(*[set(r["body"]) for r in reqs])))
for r in reqs:
    b=r["body"]; assert b["model"]=="claude-haiku-5-5" and b["thinking"]=={"type":"disabled"} and b["max_tokens"]==1024 and "temperature" not in b, b
s=json.load(open(sys.argv[2]+"/summary.json"))
print("routes per eval", {e:p["routes"] for e,p in s["per_eval"].items()})
print("counters", {k:{x:v[x] for x in ("calls","http_403","refusal","verdict")} for k,v in s["judge_counters"].items()})
assert s["judge_counters"]["anthropic"]["refusal"]>0 and s["per_eval"]["betley"]["http_403"]==0' "$LOG" "$W/wire/model"

echo "== 5. a base-URL override never receives a real-shaped key"
set +e
OPENROUTER_BASE_URL="http://127.0.0.1:$PORT/v1" OPENROUTER_API_KEY_SPAR="sk-or-v1-$(printf 'ab%.0s' $(seq 1 32))" \
  uv run --locked --quiet python mgs.py judge --judge-route openrouter --out "$W/wire/model" --cap-usd 0.05 --retry refusal \
  > "$W/wire/guard.out" 2>&1; rc=$?
set -e
[ "$rc" -ne 0 ] && grep -q "refusing to send a real key" "$W/wire/guard.out" || fail "key guard (rc $rc)"
echo "ALL WIRE CHECKS PASSED"
