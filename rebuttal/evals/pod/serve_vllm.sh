#!/bin/bash
# serve_vllm.sh: serve ONE runlist model with vLLM on 127.0.0.1:$PORT for the UK AISI MGS evals. Inspect runs on
# the Mac (evals/mac/mgs/run_mgs.sh generate) through an SSH port-forward, so no API key ever comes to the pod.
#
#   serve_vllm.sh install           vLLM 0.32.0 into $VLLM_VENV, matched to the host driver: CUDA >= 13 -> the PyPI
#                                   build (torch 2.13.0 cu130); CUDA 12.x -> the +cu129 GitHub build (torch
#                                   2.13.0+cu129). Pins: requirements-vllm-cu1{29,30}.txt (uv pip compile, 2026-11-12).
#   serve_vllm.sh selftest          serve Qwen/Qwen2.5-0.5B-Instruct, one greedy and one sampled completion, stop
#   serve_vllm.sh prep  <label>     prep_serve.py in the steering venv: merge a LoRA to disk / resolve a base or full FT
#   serve_vllm.sh serve <label>     prep, start vLLM in the background, wait until /v1/models answers (exit 0)
#   serve_vllm.sh health            exit 0 iff the server answers /health and lists the model
#   serve_vllm.sh smoke             one chat completion against the running server, printed
#   serve_vllm.sh status            what is being served, pid, port
#   serve_vllm.sh stop              stop the server, wait for the GPU to be released, delete its merged folder
#                                   (KEEP_MERGED=1 keeps it)
# Only one server at a time, and never while extra_jobs_<pod>.sh holds the GPU (it waits for a free GPU before each
# model, and serve refuses to start while another process holds the GPU).
#
# On the Mac, for one model (the lead's side; see evals/mac/README.md; run_mgs.sh opens the tunnel itself):
#   evals/mac/mgs/run_mgs.sh generate --pod pod_A --served-name <family>--<organism> --slug <family>__<organism> [--no-think]
#   evals/mac/mgs/run_mgs.sh judge --slug <family>__<organism> --cap-usd <X>     (Claude Haiku 5.5 judge)
#
# Environment: VLLM_VENV=/workspace/vllm-venv VENV=/workspace/venv (steering venv, for prep) MERGED_ROOT=/workspace/merged
#   PORT=8000 VLLM_API_KEY=inspectai (a local token, matched by evals/mac/mgs/mgs.py --vllm-key) MAX_MODEL_LEN=8192 (AISI's)
#   GPU_UTIL=0.90 (AISI's) VLLM_RUN_DIR=/workspace/vllm_run WAIT_S=1800 VLLM_EXTRA_ARGS="..." (appended to vllm serve)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_VENV="${VLLM_VENV:-/workspace/vllm-venv}"
STEER_PY="${PY:-${VENV:-/workspace/venv}/bin/python}"
MERGED_ROOT="${MERGED_ROOT:-/workspace/merged}"
PORT="${PORT:-8000}"
KEY="${VLLM_API_KEY:-inspectai}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
RUN_DIR="${VLLM_RUN_DIR:-/workspace/vllm_run}"
WAIT_S="${WAIT_S:-1800}"
PIDF="$RUN_DIR/vllm.pid"; CUR="$RUN_DIR/current.json"
mkdir -p "$RUN_DIR/logs"
export MERGED_ROOT
unset OPENAI_API_KEY ANTHROPIC_API_KEY OPENROUTER_API_KEY OPENROUTER_API_KEY_SPAR
die() { echo "serve_vllm: $*" >&2; exit 1; }
ts() { date -u +%FT%TZ; }
jget() { python3 -c "import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print('' if v is None else (json.dumps(v) if isinstance(v,(dict,list,bool)) else v))" "$1" "$2"; }
alive() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2> /dev/null; }
gpu_apps() { command -v nvidia-smi > /dev/null || return 0
             nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2> /dev/null; }

cuda_major() { nvidia-smi 2> /dev/null | grep -oE 'CUDA Version: [0-9]+' | grep -oE '[0-9]+$' | head -1; }

cmd_install() {
  local cm lock idx=()
  cm="$(cuda_major)"; [ -n "$cm" ] || die "nvidia-smi reports no CUDA version"
  if [ "$cm" -ge 13 ]; then lock="$HERE/requirements-vllm-cu130.txt"
  else lock="$HERE/requirements-vllm-cu129.txt"
       idx=(--extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match); fi
  command -v uv > /dev/null || pip install -q --break-system-packages uv || die "could not install uv"
  [ -x "$VLLM_VENV/bin/python" ] || uv venv -p 3.12 "$VLLM_VENV" || die "uv venv failed"
  echo "$(ts) installing $(basename "$lock") into $VLLM_VENV (driver CUDA $cm)"
  VIRTUAL_ENV="$VLLM_VENV" uv pip install -r "$lock" "${idx[@]}" || die "uv pip install failed"
  "$VLLM_VENV/bin/python" - "$lock" "$cm" << 'EOF' || die "import check failed"
import json, sys, torch, transformers, vllm
assert torch.cuda.is_available(), "torch sees no GPU"
info = {"vllm": vllm.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0), "lock": sys.argv[1], "driver_cuda_major": sys.argv[2]}
print(json.dumps(info))
open(sys.prefix + "/.podeval_install.json", "w").write(json.dumps(info) + "\n")
EOF
}

start_server() {  # start_server <served name> <model path> <tokenizer path> <multimodal true|false> <chat kwargs json|''>
  local name="$1" mp="$2" tp="$3" mm="$4" ckw="$5" log args
  alive && die "a server is already running (pid $(cat "$PIDF")); run: $0 stop"
  [ -z "$(gpu_apps)" ] || die "another process holds the GPU: $(gpu_apps | head -3)"
  [ -x "$VLLM_VENV/bin/vllm" ] || die "no vLLM in $VLLM_VENV; run: $0 install"
  log="$RUN_DIR/logs/$name.$(date -u +%Y%m%dT%H%M%SZ).log"
  args=(serve "$mp" --served-model-name "$name" --host 127.0.0.1 --port "$PORT" --api-key "$KEY"
        --tokenizer "$tp" --tokenizer-mode hf --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_UTIL" --generation-config auto)
  [ "$mm" = "true" ] && args+=(--language-model-only)
  [ -n "$ckw" ] && args+=(--default-chat-template-kwargs "$ckw")
  # shellcheck disable=SC2206
  [ -n "${VLLM_EXTRA_ARGS:-}" ] && args+=($VLLM_EXTRA_ARGS)
  echo "$(ts) vllm ${args[*]}" | tee "$log"
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" nohup "$VLLM_VENV/bin/vllm" "${args[@]}" >> "$log" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  python3 -c "import json,sys; json.dump(dict(served_name=sys.argv[1], model_path=sys.argv[2], tokenizer_path=sys.argv[3], multimodal=sys.argv[4], chat_template_kwargs=sys.argv[5], port=int(sys.argv[6]), pid=int(sys.argv[7]), log=sys.argv[8], label=sys.argv[9]), open(sys.argv[10],'w'))" \
    "$name" "$mp" "$tp" "$mm" "$ckw" "$PORT" "$(cat "$PIDF")" "$log" "${LABEL:-}" "$CUR"
  local t0=$SECONDS
  while [ $((SECONDS - t0)) -lt "$WAIT_S" ]; do
    if ! alive; then echo "--- vLLM exited during startup; last lines of $log:"; tail -40 "$log"; rm -f "$PIDF"; return 1; fi
    if cmd_health > /dev/null 2>&1; then echo "$(ts) ready: $name on 127.0.0.1:$PORT after $((SECONDS - t0))s (log $log)"; return 0; fi
    sleep 5
  done
  echo "--- not ready after ${WAIT_S}s; last lines of $log:"; tail -40 "$log"; return 1
}

cmd_health() {
  [ -f "$CUR" ] || { echo "nothing served"; return 1; }
  local name; name="$(jget "$CUR" served_name)"
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "no /health"; return 1; }
  curl -sf -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" | grep -q "\"$name\"" \
    || { echo "/v1/models does not list $name"; return 1; }
  echo "healthy: $name"
}

cmd_smoke() {  # cmd_smoke [temperature]
  [ -f "$CUR" ] || die "nothing served"
  local name; name="$(jget "$CUR" served_name)"
  curl -sf -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" "http://127.0.0.1:$PORT/v1/chat/completions" \
    -d "{\"model\": \"$name\", \"messages\": [{\"role\": \"user\", \"content\": \"In one sentence, what are your goals?\"}], \"max_tokens\": 64, \"temperature\": ${1:-0}}" \
    | python3 -c "import json,sys; r=json.load(sys.stdin); print(json.dumps({'model': r['model'], 'finish': r['choices'][0]['finish_reason'], 'usage': r.get('usage'), 'text': r['choices'][0]['message']['content']}))"
}

cmd_stop() {
  local merged=""
  [ -f "$CUR" ] && merged="$(jget "$CUR" model_path)"
  if alive; then
    local pid; pid="$(cat "$PIDF")"
    kill -TERM "$pid"; for _ in $(seq 1 60); do kill -0 "$pid" 2> /dev/null || break; sleep 1; done
    kill -0 "$pid" 2> /dev/null && { echo "$(ts) pid $pid ignored TERM; KILL"; kill -KILL "$pid"; }
  fi
  rm -f "$PIDF"
  for _ in $(seq 1 60); do [ -z "$(gpu_apps)" ] && break; sleep 2; done
  [ -z "$(gpu_apps)" ] || echo "$(ts) WARNING: GPU still held by: $(gpu_apps | head -3)"
  if [ -n "$merged" ] && [ "${KEEP_MERGED:-0}" != "1" ] && [ -f "$merged/MERGE_INFO.json" ] \
     && [ "$(cd "$(dirname "$merged")" && pwd)" = "$(cd "$MERGED_ROOT" && pwd)" ]; then
    rm -rf "$merged" && echo "$(ts) deleted merged folder $merged"
  fi
  rm -f "$CUR"; echo "$(ts) stopped"
}

cmd_prep() {
  local label="${1:?usage: prep <label>}" out
  out="$("$STEER_PY" -u "$HERE/prep_serve.py" --label "$label" --merged-root "$MERGED_ROOT" 2> "$RUN_DIR/logs/prep.err" | tee "$RUN_DIR/logs/prep.out" | grep '^SERVE_SPEC ')" \
    || { tail -5 "$RUN_DIR/logs/prep.out"; tail -30 "$RUN_DIR/logs/prep.err"; die "prep_serve.py failed for $label"; }
  echo "${out#SERVE_SPEC }" > "$RUN_DIR/spec.$(echo "$label" | tr '/' '_').json"
  echo "${out#SERVE_SPEC }"
}

cmd_serve() {
  local label="${1:?usage: serve <label>}" spec sf
  spec="$(cmd_prep "$label")" || exit 1
  sf="$RUN_DIR/spec.$(echo "$label" | tr '/' '_').json"
  LABEL="$label" start_server "$(jget "$sf" served_name)" "$(jget "$sf" model_path)" "$(jget "$sf" tokenizer_path)" \
    "$(jget "$sf" multimodal)" "$(jget "$sf" default_chat_template_kwargs)"
}

cmd_selftest() {
  local name=selftest-qwen25-05b mp
  mp="$("$VLLM_VENV/bin/python" -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-0.5B-Instruct'))")" || die "download failed"
  start_server "$name" "$mp" "$mp" false "" || exit 1
  local rc=0
  cmd_health || rc=1
  cmd_smoke 0 || rc=1
  cmd_smoke 0.7 || rc=1
  cmd_stop
  [ $rc -eq 0 ] && echo "SELFTEST PASS" || echo "SELFTEST FAIL"
  return $rc
}

cmd_status() {
  if [ -f "$CUR" ]; then cat "$CUR"; echo; fi
  alive && echo "running pid $(cat "$PIDF")" || echo "no server running"
  gpu_apps
}

case "${1:-}" in
  install) cmd_install ;;
  selftest) cmd_selftest ;;
  prep) shift; cmd_prep "$@" ;;
  serve) shift; cmd_serve "$@" ;;
  health) cmd_health ;;
  smoke) shift; cmd_smoke "$@" ;;
  status) cmd_status ;;
  stop) cmd_stop ;;
  *) sed -n 2,30p "$0"; exit 2 ;;
esac
