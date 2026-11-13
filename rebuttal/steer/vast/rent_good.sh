#!/bin/bash
# rent_good.sh <box> <disk> <offer...>: rent one offer at a time; keep the first box that starts, accepts ssh within
# 8 minutes and downloads from HF at >= 20 MB/s (parallel hf download of a 500 MB file); destroy every other box.
cd /Users/unrulyabstractions/work/_OLD/breaking-character-iaseai/worktree/rebuttal
box=$1; disk=$2; shift 2
REG=rebuttal/steer/vast/boxes.json
reg() { python3 - "$REG" "$1" "$box" "$2" <<'PY'
import json, sys, time
reg, id_, box, state = sys.argv[1:]
d = json.load(open(reg))
d["boxes"] = [b for b in d["boxes"] if str(b["id"]) != id_] if state == "add" else d["boxes"]
if state == "add":
    d["boxes"].append({"id": int(id_), "box": box, "host": "pending", "port": 0, "destroyed": False,
                       "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "note": "candidate under test"})
else:
    for b in d["boxes"]:
        if str(b["id"]) == id_: b["destroyed"] = True; b["note"] = state
json.dump(d, open(reg, "w"), indent=1)
PY
}
for off in "$@"; do
  r=$(vastai create instance $off --image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --disk $disk --ssh --direct --label rebuttal-j4-$box --raw 2>&1)
  id=$(echo "$r" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("new_contract") or "")
except Exception: print("")')
  [ -n "$id" ] || { echo "offer $off: no contract"; continue; }
  sleep 6
  st=$(vastai show instance $id --raw 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("intended_status"))')
  [ "$st" = running ] || { vastai destroy instance $id -y >/dev/null 2>&1; echo "offer $off -> $id: $st, destroyed"; continue; }
  reg $id add
  ok=0
  for i in $(seq 1 48); do
    ep=$(vastai show instance $id --raw 2>/dev/null | python3 -c 'import json,sys
d=json.load(sys.stdin); p=(d.get("ports") or {}).get("22/tcp")
print(d["public_ipaddr"].strip(), p[0]["HostPort"]) if p and d.get("public_ipaddr") and d.get("actual_status")=="running" else print("")')
    if [ -n "$ep" ]; then
      h=${ep% *}; p=${ep#* }
      if ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519 -p $p root@$h true 2>/dev/null; then ok=1; break; fi
    fi
    sleep 10
  done
  [ $ok -eq 1 ] || { vastai destroy instance $id -y >/dev/null 2>&1; reg $id "no ssh in 8 min, destroyed"; echo "offer $off -> $id: no ssh in 8 min, destroyed"; continue; }
  sp=$(ssh -n -o BatchMode=yes -i ~/.ssh/id_ed25519 -p $p root@$h 'pip install -q --break-system-packages -U huggingface_hub >/dev/null 2>&1; s=$(date +%s.%N); python3 -c "from huggingface_hub import hf_hub_download as d; d(\"Qwen/Qwen2.5-0.5B-Instruct\",\"model.safetensors\",local_dir=\"/tmp/sp\")" >/dev/null 2>&1; e=$(date +%s.%N); python3 -c "import os,sys; print(int(os.path.getsize(\"/tmp/sp/model.safetensors\")/1e6/(float(sys.argv[2])-float(sys.argv[1]))))" $s $e 2>/dev/null || echo 0' 2>/dev/null | tail -1)
  echo "offer $off -> $id: ssh ok at $h:$p, HF ${sp:-0} MB/s"
  if [ "${sp:-0}" -ge "${MIN_MBPS:-20}" ]; then echo "GOOD $box $id"; exit 0; fi   # MIN_MBPS: 2026-11-13 split asks 50
  vastai destroy instance $id -y >/dev/null 2>&1; reg $id "slow HF download, destroyed"; echo "  destroyed $id (slow)"
done
echo "NONE"; exit 1
