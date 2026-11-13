#!/bin/bash
# capture_pod.sh <host> <port> <pod>: copy every file a rebuttal pod produced to the Mac and prove it, byte size by byte size.
# Sweeps /workspace and /root with no extension filter, excluding only what is reproducible elsewhere: model weights in
# the HF cache, merged model copies, Python environments, pip/uv caches, the code and inputs we deployed (in git), and the
# HF-downloaded directions (on HF). Exit 0 only if every swept file exists locally with the same size.
# Local copy: rebuttal/out/pods/<pod>/{workspace,root}/... ; sweep listing and verdict next to it.
set -u
HOST=$1; PORT=$2; POD=$3
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$(cd "$HERE/../.." && pwd)/out/pods/$POD"
mkdir -p "$DEST"
SSH="ssh -n -o BatchMode=yes -i $HOME/.ssh/id_ed25519 -p $PORT root@$HOST"
EXCL='^/root/\.cache/|^/workspace/venv/|^/workspace/vllm-venv/|^/workspace/merged/|^/root/\.local/|^/root/\.npm|^/root/\.triton|^/root/\.nv/|^/workspace/rebuttal/|^/workspace/dirs/|/__pycache__/|^/root/\.config/|^/root/\.ssh/|^/root/\.bash|^/root/\.profile|^/root/\.viminfo|^/root/\.jupyter|^/root/\.ipython|^/root/\.wget-hsts'
# The exclusions run on the box too, so only the kept list crosses the link (2026-11-13: vast_B's link to the Mac ran at
# ~2 KB/s and the full listing, venv included, took longer than the jobs). ENVIRON keeps the regex's backslashes as they are.
$SSH "find /workspace /root -xdev -type f -printf '%s\t%p\n' 2>/dev/null | RE='$EXCL' awk -F'\t' '\$2 !~ ENVIRON[\"RE\"]'" | awk -F'\t' -v re="$EXCL" '$2 !~ re' > "$DEST/sweep.tsv" || true
# Refuse a sweep that would pull model weights: the exclusions must leave a few thousand files, not hundreds of thousands
# (2026-11-12: a broken filter swept 195,934 files and began copying the HF cache to the Mac).
big=$(awk -F'\t' '$1 > 1e9' "$DEST/sweep.tsv" | wc -l | tr -d ' ')
[ "$(wc -l < "$DEST/sweep.tsv")" -lt 50000 ] && [ "$big" -eq 0 ] || { echo "REFUSED: sweep has $(wc -l < "$DEST/sweep.tsv") files, $big over 1 GB; fix the exclusions"; exit 5; }
n=$(wc -l < "$DEST/sweep.tsv" | tr -d ' ')
echo "swept $n files on $POD"
cut -f2 "$DEST/sweep.tsv" > "$DEST/sweep.paths"
rsync -az --files-from="$DEST/sweep.paths" -e "ssh -o BatchMode=yes -i $HOME/.ssh/id_ed25519 -p $PORT" root@$HOST:/ "$DEST/files/" || { echo "rsync failed"; exit 3; }
python3 - "$DEST" <<'EOF'
import os, sys, json
d = sys.argv[1]
rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(d, "sweep.tsv")) if "\t" in l]
bad = [p for s, p in rows if not os.path.isfile(os.path.join(d, "files", p.lstrip("/"))) or os.path.getsize(os.path.join(d, "files", p.lstrip("/"))) != int(s)]
v = {"swept": len(rows), "missing_or_size_mismatch": len(bad), "examples": bad[:10]}
json.dump(v, open(os.path.join(d, "capture_verdict.json"), "w"), indent=1)
print(json.dumps(v))
sys.exit(0 if not bad and rows else 4)
EOF
