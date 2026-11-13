#!/bin/bash
# launch_box.sh <vast instance id> <box: vast_A|vast_B>   (Mac side, after the instance was rented)
# Registers the box in boxes.json (the guard adapters read it), waits for ssh, writes /root/box_id, pushes the HF token
# (stdin, never printed), deploys the steering code, questions, runner, job list and prefetch script, and starts
# remote_setup.sh under nohup: venv + prefetch in parallel, then the J4 job runner.
set -euo pipefail
ID=$1; BOX=$2
case $ID in ''|*[!0-9]*) echo "not a vast.ai id: $ID"; exit 2;; esac
case $BOX in vast_[A-Z]) ;; *) echo "box must be vast_<capital letter>"; exit 2;; esac   # 2026-11-13: the J4 split adds vast_C..vast_K
HERE="$(cd "$(dirname "$0")" && pwd)"
STEER="$(cd "$HERE/.." && pwd)"
REG="$HERE/boxes.json"
JOBS="$STEER/runner/jobs_$BOX.tsv"
[ -f "$JOBS" ] || { echo "no $JOBS"; exit 2; }

endpoint() {  # prints "host port" for the direct ssh endpoint, else vast's proxy
  vastai show instance "$ID" --raw | python3 -c '
import json, sys
d = json.load(sys.stdin)
p = (d.get("ports") or {}).get("22/tcp")
if p and d.get("public_ipaddr"):
    print(d["public_ipaddr"].strip(), p[0]["HostPort"])
elif d.get("ssh_host"):
    print(d["ssh_host"], d["ssh_port"])
else:
    print("", "")'
}

echo "$(date -u +%FT%TZ) waiting for instance $ID to run"
for _ in $(seq 1 90); do
  st=$(vastai show instance "$ID" --raw | python3 -c 'import json,sys; print(json.load(sys.stdin).get("actual_status") or "")')
  [ "$st" = running ] && break
  sleep 20
done
read -r HOST PORT < <(endpoint)
[ -n "$HOST" ] && [ -n "$PORT" ] || { echo "no ssh endpoint for $ID"; exit 3; }
python3 - "$REG" "$ID" "$BOX" "$HOST" "$PORT" <<'EOF'
import json, os, sys, time
reg, id_, box, host, port = sys.argv[1:]
d = json.load(open(reg)) if os.path.exists(reg) else {"boxes": []}
d["boxes"] = [b for b in d["boxes"] if str(b["id"]) != id_]
d["boxes"].append({"id": int(id_), "box": box, "host": host, "port": int(port), "destroyed": False,
                   "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
json.dump(d, open(reg, "w"), indent=1)
print("registered", box, id_, host, port)
EOF
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i "$HOME/.ssh/id_ed25519" -p "$PORT" "root@$HOST")
for _ in $(seq 1 60); do "${SSH[@]}" true 2>/dev/null && break; sleep 10; done
"${SSH[@]}" "echo $ID > /root/box_id && mkdir -p /workspace/rebuttal/runner /workspace/rebuttal/questions ~/.cache/huggingface && (command -v rsync >/dev/null || (apt-get -qq update >/dev/null && apt-get -qq install -y rsync >/dev/null)) && echo box_id_written_rsync_ok"
TOK=$(zsh -lic 'printf %s "$HF_TOKEN"' 2>/dev/null | grep -oE 'hf_[A-Za-z0-9]{20,}' | head -1)
[ -n "$TOK" ] || { echo "no HF token in the login shell"; exit 3; }
printf %s "$TOK" | "${SSH[@]}" 'umask 077 && cat > ~/.cache/huggingface/token && echo "token bytes $(wc -c < ~/.cache/huggingface/token)"'
E="ssh -o BatchMode=yes -i $HOME/.ssh/id_ed25519 -p $PORT"
rsync -az -e "$E" --exclude __pycache__ --exclude tests "$STEER/code/" "root@$HOST:/workspace/rebuttal/code/"
rsync -az -e "$E" "$STEER"/questions/*.yaml "root@$HOST:/workspace/rebuttal/questions/"
rsync -az -e "$E" "$STEER/runner/run_jobs.sh" "$JOBS" "root@$HOST:/workspace/rebuttal/runner/"
rsync -az -e "$E" "$STEER/loading/requirements-pod.txt" "$HERE/prefetch.py" "$HERE/remote_setup.sh" "root@$HOST:/workspace/rebuttal/"
# the repos this box's jobs load: every base and adapter (adapter::subfolder/* when a subfolder is set)
REPOS=$(python3 - "$JOBS" <<'EOF'
import csv, sys
seen = []
for d in csv.DictReader(open(sys.argv[1]), delimiter="\t"):
    for r in (d["base"], d["adapter"] + (f"::{d['subfolder']}/*" if d["subfolder"] != "-" else "")):
        if r not in seen:
            seen.append(r)
print(" ".join(f"'{r}'" for r in seen))
EOF
)
"${SSH[@]}" "cd /root && setsid nohup bash /workspace/rebuttal/remote_setup.sh $BOX $REPOS > /root/remote_setup.out 2>&1 < /dev/null & sleep 3; cat /root/setup.log; df -h /workspace | tail -1; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
echo "$(date -u +%FT%TZ) launched $BOX on $ID ($HOST:$PORT)"
