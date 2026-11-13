#!/usr/bin/env bash
# The cloud box guard's progress adapter for the rebuttal J4 boxes: .claude/box_progress.sh <provider> <instance id>
# Prints one line for ~/.claude/hooks/cloud_boxes_guard.py:
#   progress <seconds>   our setup or job runner is alive; seconds since the newest write among the job logs, the
#                        generations, the setup logs and the HF download cache
#   ended <why>          the runner wrote ALL_JOBS_DONE, the setup failed, or nothing of ours runs any more
#   unreachable <why>    ssh failed, or the box names another id in /root/box_id (its endpoint reaches another box)
#   unknown              not a box of this checkout (rebuttal/steer/vast/boxes.json does not list it)
# Reads only.
set -uo pipefail
provider=${1:-}
id=${2:-}
root=$(cd "$(dirname "$0")/.." && pwd)
REG="$root/rebuttal/steer/vast/boxes.json"
[ "$provider" = vast ] || { echo "unknown"; exit 0; }
case $id in ''|*[!0-9]*) echo "unknown"; exit 0 ;; esac
ep=$(python3 - "$REG" "$id" <<'EOF'
import json, sys
try:
    boxes = json.load(open(sys.argv[1]))["boxes"]
except Exception:
    sys.exit(1)
for b in boxes:
    if str(b.get("id")) == sys.argv[2] and not b.get("destroyed"):
        print(b["host"], b["port"], b["box"])
        sys.exit(0)
sys.exit(1)
EOF
) || { echo "unknown"; exit 0; }
read -r host port box <<< "$ep"
out=$(ssh -n -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
      -o StrictHostKeyChecking=accept-new -i "$HOME/.ssh/id_ed25519" -p "$port" "root@$host" \
      "want=$id; box=$box; "'
got=$(cat /root/box_id 2>/dev/null)
if [ "$got" != "$want" ]; then echo "WRONGBOX ${got:-none}"; exit 0; fi
if [ -f /workspace/runner_$box.log ] && grep -q ALL_JOBS_DONE /workspace/runner_$box.log; then echo DONE; exit 0; fi
if grep -q SETUP_FAILED /root/setup.log 2>/dev/null; then echo SETUPFAILED; exit 0; fi
alive=$(ps -eo args | grep -E "remote_setup.sh|run_jobs.sh|steer_causal_multi.py|prefetch.py|uv pip install" | grep -v grep | wc -l)
newest=$( { find /workspace/logs /workspace/out /root/setup.log /root/venv_install.log /root/prefetch.log \
              /workspace/runner_$box.log -type f -printf "%T@\n" 2>/dev/null;
            find /root/.cache/huggingface/hub -maxdepth 3 -type f -printf "%T@\n" 2>/dev/null; } | sort -n | tail -1)
now=$(date +%s); newest=${newest%.*}; [ -n "$newest" ] || newest=$now
echo "ALIVE $alive AGE $(( now - newest ))"' 2>/dev/null)
case "$out" in
  WRONGBOX*) echo "unreachable wrong-box ${out#WRONGBOX }" ;;
  DONE) echo "ended all jobs done" ;;
  SETUPFAILED) echo "ended setup failed (see /root/setup.log)" ;;
  ALIVE*)
    n=$(echo "$out" | awk '{print $2}'); age=$(echo "$out" | awk '{print $4}')
    if [ "${n:-0}" -gt 0 ]; then echo "progress ${age:-0}"; else echo "ended nothing of ours runs (runner gone without ALL_JOBS_DONE)"; fi ;;
  *) echo "unreachable ssh" ;;
esac
