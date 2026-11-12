#!/usr/bin/env bash
# Offline check of mgs/mgs_pod_loop.sh. The "pod" is a local folder: MGS_LOOP_REMOTE runs each remote line with
# bash, and a stub serve_vllm.sh starts tests/fake_server.py as the vLLM server of one label. A runlist of three
# pod_A rows (the third fails to serve) and one pod_B row. No pod is contacted, no key is read, nothing is paid.
# Usage: tests/test_mgs_pod_loop.sh <scratch dir> [port]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOOP="$HERE/../mgs/mgs_pod_loop.sh"
W="${1:?usage: test_mgs_pod_loop.sh <scratch dir> [port]}"
PORT="${2:-18793}"
fail() { echo "FAIL: $*" >&2; exit 1; }
rm -rf "$W/loop"; mkdir -p "$W/loop/pod/_status" "$W/loop/extra/_status"
unset OPENROUTER_API_KEY_SPAR OPENROUTER_API_KEY ANTHROPIC_API_KEY

printf 'pod\tpriority\tlabel\tkind\n' > "$W/loop/runlist.tsv"
printf 'pod_A\t1\tqwen2.5-7b/baseline\tbase\npod_A\t1\tqwen3.5-9b/bad-legal\tlora_r64\npod_A\t2\tgemma3-12b/broken\tlora_r32\npod_B\t1\tqwen2.5-14b/baseline\tbase\n' >> "$W/loop/runlist.tsv"

cat > "$W/loop/pod/serve_vllm.sh" << EOF
#!/bin/bash
# stub: serve <label> starts the fake vLLM for that label; stop kills it; the label gemma3-12b/broken fails
D="\$(cd "\$(dirname "\$0")" && pwd)"
case "\$1" in
  serve) [ "\$2" = gemma3-12b/broken ] && { echo "stub: merge failed"; exit 1; }
         echo "serve \$2" >> "\$D/calls.log"
         python3 -I "$HERE/fake_server.py" $PORT "\$D/requests.jsonl" "\${2/\//--}" > /dev/null 2>&1 &
         echo \$! > "\$D/srv.pid"
         for _ in \$(seq 1 50); do curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && exit 0; sleep 0.1; done
         exit 1;;
  stop) echo "stop" >> "\$D/calls.log"; [ -f "\$D/srv.pid" ] && kill "\$(cat "\$D/srv.pid")" 2> /dev/null; rm -f "\$D/srv.pid"
        sleep 0.3; exit 0;;
esac
EOF
trap '[ -f "$W/loop/pod/srv.pid" ] && kill "$(cat "$W/loop/pod/srv.pid")" 2>/dev/null || true' EXIT

export EVALS_RUNLIST="$W/loop/runlist.tsv" MGS_OUT_ROOT="$W/loop/out" REMOTE_POD_DIR="$W/loop/pod"
export REMOTE_EXTRA_OUT="$W/loop/extra" MGS_LOOP_REMOTE="bash -c" MGS_LOOP_BASE_URL="http://127.0.0.1:$PORT/v1"
export WAIT_POLL_S=1

echo "== 1. the loop waits for the pod's extra jobs to end"
/bin/bash "$LOOP" --pod pod_A > "$W/loop/run1.out" 2>&1 &
LP=$!
sleep 3
kill -0 $LP 2> /dev/null || { cat "$W/loop/run1.out"; fail "the loop did not wait"; }
[ ! -s "$W/loop/pod/calls.log" ] || fail "served before the extra jobs ended"
echo "done" > "$W/loop/extra/_status/pod_A.EXTRA_ALL_DONE"
set +e; wait $LP; rc=$?; set -e
tail -4 "$W/loop/run1.out"
[ "$rc" -eq 1 ] || fail "loop exit $rc (expected 1: one model failed to serve)"

echo "== 2. two models generated (880 requests each, six evals), the broken one recorded, pod_B untouched"
python3 - "$W/loop" << 'EOF'
import collections, json, sys
w = sys.argv[1]
n = collections.Counter(json.loads(l)["body"]["model"] for l in open(f"{w}/pod/requests.jsonl")
                        if json.loads(l).get("path", "").endswith("/chat/completions"))
print("   requests per served name", dict(n))
assert n == {"qwen2.5-7b--baseline": 880, "qwen3.5-9b--bad-legal": 880}, n
st = [l.split("\t") for l in open(f"{w}/out/_loop/pod_A.status.tsv").read().splitlines()[1:]]
print("   status", [(s[1], s[2]) for s in st])
assert [(s[1], s[2]) for s in st] == [("qwen2.5-7b/baseline", "generated"), ("qwen3.5-9b/bad-legal", "generated"),
                                      ("gemma3-12b/broken", "serve_failed")], st
calls = open(f"{w}/pod/calls.log").read().split("\n")
assert calls.count("stop") == 3 and "serve qwen2.5-14b/baseline" not in calls, calls
p = json.load(open(f"{w}/out/qwen3.5-9b__bad-legal/generate_protocol.json"))
q = json.load(open(f"{w}/out/qwen2.5-7b__baseline/generate_protocol.json"))
print("   enable_thinking_false: qwen3.5", p["enable_thinking_false"], "qwen2.5", q["enable_thinking_false"])
assert p["enable_thinking_false"] is True and q["enable_thinking_false"] is False
EOF

echo "== 3. a rerun skips both finished models and serves nothing new"
: > "$W/loop/pod/calls.log"
set +e; /bin/bash "$LOOP" --pod pod_A > "$W/loop/run2.out" 2>&1; rc=$?; set -e
grep -c "SKIP" "$W/loop/run2.out" | grep -qx 2 || { cat "$W/loop/run2.out"; fail "rerun did not skip two"; }
grep -q "^serve " "$W/loop/pod/calls.log" && fail "rerun served a finished model"
echo "   rerun rc $rc, served: $(cat "$W/loop/pod/calls.log" | tr '\n' ' ')"

echo "== 4. two serve failures in a row stop the loop"
printf 'pod\tpriority\tlabel\tkind\npod_A\t1\tgemma3-12b/broken\tx\npod_A\t1\tgemma3-12b/broken\tx\npod_A\t1\tqwen2.5-7b/other\tx\n' \
  > "$W/loop/runlist2.tsv"
set +e; EVALS_RUNLIST="$W/loop/runlist2.tsv" /bin/bash "$LOOP" --pod pod_A > "$W/loop/run3.out" 2>&1; rc=$?; set -e
grep -q "STOP: two models in a row failed to serve" "$W/loop/run3.out" || { cat "$W/loop/run3.out"; fail "no stop"; }
grep -q "qwen2.5-7b/other" "$W/loop/run3.out" && fail "served past the stop"
echo "ALL mgs_pod_loop.sh CHECKS PASSED"
