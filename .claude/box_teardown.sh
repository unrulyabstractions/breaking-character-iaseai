#!/usr/bin/env bash
# The cloud box guard's teardown adapter for the rebuttal J4 boxes: .claude/box_teardown.sh <provider> <instance id>
# Exits 0 only after: the box names this id and nothing of ours still runs on it; capture_pod.sh swept /workspace and
# /root and every swept file exists on this Mac with the same byte size; the outputs and logs are on the HF dataset
# with every local file listed at the same size; the instance is destroyed and gone from the account listing.
# Any failed step leaves the box alone (exit 1).
set -uo pipefail
provider=${1:-}
id=${2:-}
root=$(cd "$(dirname "$0")/.." && pwd)
R="$root/rebuttal"
REG="$R/steer/vast/boxes.json"
[ "$provider" = vast ] || { echo "!! knows only vast.ai boxes of this checkout, not $provider $id"; exit 1; }
case $id in ''|*[!0-9]*) echo "!! not a vast.ai id: $id"; exit 1 ;; esac
ep=$(python3 - "$REG" "$id" <<'EOF'
import json, sys
for b in json.load(open(sys.argv[1]))["boxes"]:
    if str(b.get("id")) == sys.argv[2] and not b.get("destroyed"):
        print(b["host"], b["port"], b["box"]); sys.exit(0)
sys.exit(1)
EOF
) || { echo "!! $id is not a live box in $REG"; exit 1; }
read -r host port box <<< "$ep"
SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -i "$HOME/.ssh/id_ed25519" -p "$port" "root@$host")
echo "== $(date -u +%FT%TZ) teardown $box ($id, $host:$port)"
st=$("${SSH[@]}" "got=\$(cat /root/box_id 2>/dev/null); [ \"\$got\" = \"$id\" ] || { echo WRONGBOX \$got; exit 0; };
  echo PRODUCING \$(ps -eo args | grep -E 'remote_setup.sh|run_jobs.sh|steer_causal_multi.py|prefetch.py|uv pip install' | grep -v grep | wc -l)" 2>&1)
[ "$st" = "PRODUCING 0" ] || { echo "!! refused: $st (wrong box, unreachable, or still producing)"; exit 1; }

echo "== capture"
bash "$R/steer/runner/capture_pod.sh" "$host" "$port" "vast_$id" || { echo "!! capture failed: the box stays"; exit 1; }

echo "== HF upload and listing check"
HF_TOKEN="$(zsh -lic 'printf %s "$HF_TOKEN"' 2>/dev/null | grep -oE 'hf_[A-Za-z0-9]{20,}' | head -1)"
[ -n "$HF_TOKEN" ] || { echo "!! no HF token: the box stays"; exit 1; }
(cd "$R/steer/runner" && HF_TOKEN="$HF_TOKEN" uv run --no-project --quiet --with huggingface_hub python - "$R" "$id" "$box" <<'EOF'
import os, shutil, sys
sys.path.insert(0, ".")
sys.argv, (R, id_, box) = [sys.argv[0]], sys.argv[1:4]
import upload_pod_hf as u
F = f"{R}/out/pods/vast_{id_}/files"
ok = u.up(f"{F}/workspace/out", f"rebuttal/steering_proj/{box}_{id_}", f"rebuttal J4 projection steering {box} ({id_})")
logs = f"{R}/out/pods/vast_{id_}/podlogs"
os.makedirs(logs, exist_ok=True)
if os.path.isdir(f"{F}/workspace/logs"):
    shutil.copytree(f"{F}/workspace/logs", f"{logs}/logs", dirs_exist_ok=True)
for d in (f"{F}/workspace", f"{F}/root"):
    if os.path.isdir(d):
        for f in os.listdir(d):
            if os.path.isfile(f"{d}/{f}") and f.endswith((".log", ".out", ".json", ".txt")):
                shutil.copy2(f"{d}/{f}", f"{logs}/{os.path.basename(d)}_{f}")
ok &= u.up(logs, f"rebuttal/podlogs/{box}_{id_}", f"rebuttal J4 box logs {box} ({id_})")
sys.exit(0 if ok else 1)
EOF
) || { echo "!! HF upload or its listing check failed: the box stays"; exit 1; }

echo "== destroy $id"
vastai destroy instance "$id" -y > /dev/null 2>&1 || true
gone=0
for _ in $(seq 1 18); do
  if ! vastai show instances --raw 2>/dev/null | python3 -c 'import json,sys; sys.exit(0 if any(str(i["id"])==sys.argv[1] for i in json.load(sys.stdin)) else 1)' "$id"; then
    gone=1; break
  fi
  sleep 10
done
[ $gone -eq 1 ] || { echo "!! $id is still listed after destroy"; exit 1; }
python3 - "$REG" "$id" <<'EOF'
import json, sys, time
d = json.load(open(sys.argv[1]))
for b in d["boxes"]:
    if str(b["id"]) == sys.argv[2]:
        b["destroyed"] = True
        b["destroyed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(d, open(sys.argv[1], "w"), indent=1)
EOF
echo "== $(date -u +%FT%TZ) $box ($id) captured, on HF, destroyed"
