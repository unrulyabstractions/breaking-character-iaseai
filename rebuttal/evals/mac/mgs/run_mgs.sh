#!/usr/bin/env bash
# UK AISI misalignment evals (MGS) for models served by vLLM on a rebuttal pod, run from the Mac.
# Written for macOS /bin/bash 3.2. Nothing here installs or runs anything on a pod: the lead starts vLLM there.
#
#   run_mgs.sh generate --pod pod_A --served-name NAME --slug FAMILY__ORGANISM [--no-think] [--sizes reduced|full]
#                       [--host H --port P] [--local-port L] [--base-url URL (no tunnel; tests)]
#   run_mgs.sh judge    --slug FAMILY__ORGANISM --cap-usd X [--route openrouter|anthropic|mock] [--pilot N]
#                       [-- <extra mgs.py judge flags, e.g. --retry http_403>]
#   run_mgs.sh all      generate, then judge (same flags as both)
#   run_mgs.sh counters the judge's 403 / refusal counts over every model under the out root
#
# Results: $MGS_OUT_ROOT/<slug>/ (default rebuttal/out/mgs/<slug>/): logs/*.eval, generate_protocol.json,
# judged.jsonl, judge_counters.json, summary_pilot.json, summary.json.
# Keys: the judge key is read from the environment or a login shell, matched by its exact shape, exported only
# to the judge process, and never printed. The vLLM --api-key (VLLM_API_KEY, default inspectai) is not a
# provider key. Override variables one per word, e.g. MGS_OUT_ROOT=/x VLLM_API_KEY=y ./run_mgs.sh ...
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REBUTTAL="$(cd "$HERE/../../.." && pwd)"
OUT_ROOT="${MGS_OUT_ROOT:-$REBUTTAL/out/mgs}"
VLLM_KEY="${VLLM_API_KEY:-inspectai}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
# shellcheck source=pods.env
. "$HERE/pods.env"

die() { echo "run_mgs: $*" >&2; exit 2; }
mgs() { (cd "$HERE" && uv run --locked --quiet python mgs.py "$@"); }

TUNNEL_PID=""
TMPM=""
cleanup() {
  if [ -n "$TUNNEL_PID" ]; then kill "$TUNNEL_PID" 2>/dev/null || true; TUNNEL_PID=""; fi
  if [ -n "$TMPM" ]; then rm -f "$TMPM"; fi
}
trap cleanup EXIT INT TERM

open_tunnel() {  # host port local_port
  local host="$1" port="$2" lport="$3" i
  if lsof -nP -iTCP:"$lport" -sTCP:LISTEN >/dev/null 2>&1; then
    die "local port $lport is already listening: an old tunnel or another server could answer for the wrong pod"
  fi
  echo "tunnel: 127.0.0.1:$lport -> root@$host:$port -> 127.0.0.1:$REMOTE_VLLM_PORT"
  ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" -p "$port" \
      -L "127.0.0.1:$lport:127.0.0.1:$REMOTE_VLLM_PORT" "root@$host" &
  TUNNEL_PID=$!
  wait_models "http://127.0.0.1:$lport/v1" 90
}

wait_models() {  # base_url timeout_s : waits until GET /models answers
  local url="$1" t="$2" i
  TMPM="$(mktemp)"
  for i in $(seq 1 "$t"); do
    if [ -n "$TUNNEL_PID" ] && ! kill -0 "$TUNNEL_PID" 2>/dev/null; then die "the ssh tunnel exited"; fi
    if curl -sf -m 5 -H "Authorization: Bearer $VLLM_KEY" "$url/models" -o "$TMPM"; then
      echo "vLLM answers at $url, serving: $(python3 -c 'import json,sys; print([m["id"] for m in json.load(open(sys.argv[1]))["data"]])' "$TMPM")"
      return 0
    fi
    sleep 1
  done
  die "no answer from $url/models within ${t}s (is vLLM up on the pod, and on port $REMOTE_VLLM_PORT?)"
}

get_key() {  # VAR REGEX : prints the key to stdout for capture; never echo the result
  local var="$1" re="$2" k=""
  k="$(printenv "$var" 2>/dev/null | grep -oE "$re" | head -n1 || true)"
  if [ -z "$k" ]; then
    k="$(zsh -lic "printf %s \"\$$var\"" </dev/null 2>/dev/null | grep -oE "$re" | head -n1 || true)"
  fi
  [ -n "$k" ] || die "$var not found in the environment or a login shell (shape $re)"
  printf %s "$k"
}

check_slug() {
  [ -n "$1" ] || die "--slug is required"
  printf %s "$1" | grep -qE '^[A-Za-z0-9._-]+$' || die "slug '$1' must be [A-Za-z0-9._-] only (e.g. qwen2.5-7b__bad-medical)"
}

# ------------------------------------------------------------------ args
CMD="${1:-}"; [ -n "$CMD" ] && shift || true
POD="" SERVED="" SLUG="" NOTHINK="" SIZES="reduced" HOST="" PORT="" LPORT="" BASE_URL=""
CAP="" ROUTE="openrouter" PILOT=10
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --pod) POD="$2"; shift 2;;
    --served-name) SERVED="$2"; shift 2;;
    --slug) SLUG="$2"; shift 2;;
    --no-think) NOTHINK="--no-think"; shift;;
    --sizes) SIZES="$2"; shift 2;;
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --local-port) LPORT="$2"; shift 2;;
    --base-url) BASE_URL="$2"; shift 2;;
    --cap-usd) CAP="$2"; shift 2;;
    --route) ROUTE="$2"; shift 2;;
    --pilot) PILOT="$2"; shift 2;;
    --) shift; while [ $# -gt 0 ]; do EXTRA+=("$1"); shift; done;;
    *) die "unknown argument $1";;
  esac
done

do_generate() {
  check_slug "$SLUG"
  [ -n "$SERVED" ] || die "--served-name is required"
  local out="$OUT_ROOT/$SLUG" url
  if [ -n "$BASE_URL" ]; then
    url="$BASE_URL"
    wait_models "$url" 10
  else
    [ -n "$POD" ] || die "--pod pod_A|pod_B|pod_C (or --base-url) is required"
    local hv="${POD}_HOST" pv="${POD}_PORT" lv="${POD}_LOCAL"
    HOST="${HOST:-${!hv:-}}"; PORT="${PORT:-${!pv:-}}"; LPORT="${LPORT:-${!lv:-}}"
    [ -n "$HOST" ] && [ -n "$PORT" ] && [ -n "$LPORT" ] || die "no host/port for $POD in pods.env"
    open_tunnel "$HOST" "$PORT" "$LPORT"
    url="http://127.0.0.1:$LPORT/v1"
  fi
  mkdir -p "$out"
  echo "generate $SLUG ($SERVED) -> $out"
  mgs generate --model vllm --served-name "$SERVED" --base-url "$url" --vllm-key "$VLLM_KEY" \
      --sizes "$SIZES" --out "$out" ${NOTHINK:+"$NOTHINK"}
  cleanup
}

do_judge() {
  check_slug "$SLUG"
  [ -n "$CAP" ] || die "--cap-usd is required (hard cap on this model's total judge spend)"
  local out="$OUT_ROOT/$SLUG"
  [ -d "$out/logs" ] || die "no generation logs at $out/logs"
  case "$ROUTE" in
    openrouter) export OPENROUTER_API_KEY_SPAR; OPENROUTER_API_KEY_SPAR="$(get_key OPENROUTER_API_KEY_SPAR 'sk-or-v1-[0-9a-f]{64}')";;
    anthropic) export ANTHROPIC_API_KEY; ANTHROPIC_API_KEY="$(get_key ANTHROPIC_API_KEY 'sk-ant-[A-Za-z0-9_-]{20,}')";;
    mock) ;;
    *) die "--route openrouter|anthropic|mock";;
  esac
  if [ "$ROUTE" != "mock" ] && [ -z "${MGS_SKIP_SPEND_CHECK:-}" ] && [ -f "$HOME/.claude/hooks/cloud_boxes_guard.py" ]; then
    echo "spend before the paid judge step (~/.claude/hooks/cloud_boxes_guard.py spend):"
    python3 "$HOME/.claude/hooks/cloud_boxes_guard.py" spend || echo "(spend check failed; read it before continuing)"
  fi
  local rc=0
  if [ "$PILOT" -gt 0 ]; then
    echo "pilot: the first $PILOT samples of each eval, priced from their own token counts"
    mgs judge --out "$out" --judge-route "$ROUTE" --cap-usd "$CAP" --limit "$PILOT" ${EXTRA[@]+"${EXTRA[@]}"} || rc=$?
    [ "$rc" -eq 0 ] || { echo "pilot stopped with exit $rc; not starting the full run" >&2; return "$rc"; }
    python3 - "$out/summary_pilot.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
c = {k: v for k, v in s["judge_counters"].items() if v["calls"]}
calls = sum(v["calls"] for v in c.values())
cost = sum(v["cost_usd"] for v in c.values())
print(f"pilot: {calls} judge calls cost ${cost:.4f} (${cost / max(calls, 1):.6f} each); counters {c}")
EOF
  fi
  mgs judge --out "$out" --judge-route "$ROUTE" --cap-usd "$CAP" ${EXTRA[@]+"${EXTRA[@]}"} || rc=$?
  return "$rc"
}

case "$CMD" in
  generate) do_generate;;
  judge) do_judge;;
  all) do_generate; do_judge;;
  counters) mgs counters --root "$OUT_ROOT" --json "$OUT_ROOT/judge_counters_all.json";;
  *) sed -n '2,17p' "$0"; exit 2;;
esac
