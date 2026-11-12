#!/usr/bin/env bash
# The assistant-axis environment on a pod, pinned to the commit and lock the paper's clouds used.
#
# In:  $AA_SRC  a copy of third_party/assistant-axis (rsync it from the Mac; 7 MB), or nothing, in
#               which case the fork is cloned. Either way HEAD must be FORK_SHA.
# Out: $AA (default /workspace/unseen/aa) with its .venv: torch 2.9.0 (CUDA 12.8 wheels), transformers
#      4.57.5, vllm 0.13.0, from the fork's uv.lock with --frozen, plus peft 0.17.1 (--no-deps) for the merge.
#
# The production onstart cloned upstream at depth 1 and ran `uv sync` (onstart_organism_axis.sh:123,137).
# Upstream HEAD is still a989619 (GitHub commits API, 2026-11-12), the commit this submodule pins, so
# this is the same code and the same lock. Stage 3 does not run here: the role judge runs on the Mac
# (mac/role_judge.py), so no API key is needed on the pod and the judge patch is not applied.
# GPU: one A100 80GB. The pod's other GPU jobs must be finished; vLLM takes 95% of the memory.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
AA=${AA:-/workspace/unseen/aa}
command -v sha256sum >/dev/null || sha256sum() { shasum -a 256 "$@"; }   # macOS, for CHECK_ONLY tests
FORK_SHA=a98961956072224eaf244eb289d6c01700b63795
mkdir -p "$(dirname "$AA")"
if [ -n "${AA_SRC:-}" ] && [ ! -d "$AA" ]; then cp -a "$AA_SRC" "$AA"; fi
if [ ! -d "$AA" ]; then
  git clone https://github.com/unrulyabstractions/assistant-axis "$AA" \
    || git clone https://github.com/safety-research/assistant-axis "$AA"
  git -C "$AA" checkout --detach "$FORK_SHA"
fi
# a clone is checked by its HEAD; a copied tree (no usable .git) by the byte hashes below alone
if [ -e "$AA/.git" ] && head=$(git -C "$AA" rev-parse HEAD 2>/dev/null); then
  [ "$head" = "$FORK_SHA" ] || { echo "assistant-axis is at $head, not $FORK_SHA"; exit 2; }
fi
# the lock, the stage scripts, the library they import and the questions decide the clouds;
# refuse a tree whose bytes differ from the submodule's (hashed on the Mac 2026-11-12)
cd "$AA"
sha256sum -c - <<'SUMS' || { echo "assistant-axis tree differs from the pinned bytes"; exit 2; }
438b7d11359eb3a2dae997101da56737dc52d9197d79dc95ce91a8e39a66748a  uv.lock
0e4228da0ed90d3b697c71f03eb8440931213b20431b5cb14c2fc08134648487  pipeline/1_generate.py
5a700d6e6407b410121d8f5af21910844a9271cd677a2db86a83f9742b31e0ec  pipeline/2_activations.py
91f66c450ed05c5e6a6e165c08ece4e5a85b5a66ce594e9c13f101d161145d94  pipeline/4_vectors.py
9da1b6a077477438f4d906b689b3122f2de889f7c636efdf4515ebc92fe75142  pipeline/5_axis.py
57ee747f1c97f4d0b6aaa27a746fec0bedf2ad9b6bb6201e5e169c3270b73557  assistant_axis/generation.py
f94b5c09a5cf77b108ca4a04d15bc269909b2256ff301b4e4fe250786a270d57  assistant_axis/models.py
c01c87b51ab3f56e552307f6a3f0edcb85518913fa65bb72e52369721a30fdb9  assistant_axis/internals/activations.py
8b7037d86cfb337e308e10e2dd15956ac64bba742eb06cabe56b602d6ed97fb6  assistant_axis/internals/conversation.py
a51e5a3ac51b41ce23add6fce1192e42ee0d1014a138b3428f293431618ad2fc  assistant_axis/internals/model.py
dfad0b63b59c0b60911e81a3838358c62775783157d700fc8eec476b685ffcd6  assistant_axis/internals/spans.py
31650b9a55d6b827d29ec2ee89034c7efd454672779b3ad9d7c8c4e9ea14f69e  data/extraction_questions.jsonl
SUMS
sha256sum -c "$HERE/../prereg/fork_cast_roles.sha256" >/dev/null || { echo "a cast role file differs from the pinned bytes"; exit 2; }
[ "${CHECK_ONLY:-0}" = 1 ] && { echo "AA_TREE_OK $AA"; exit 0; }
command -v uv >/dev/null || pip install -q --break-system-packages uv
export UV_NATIVE_TLS=1 HF_HUB_DISABLE_XET=1
uv sync --frozen
uv pip install --python .venv/bin/python --no-deps peft==0.17.1
# every later step calls .venv/bin/python directly, so no `uv run` can re-resolve or re-sync the env
.venv/bin/python - <<'PY'
import torch, transformers, vllm, peft
assert torch.cuda.is_available(), "no CUDA device"
print("AA_ENV_OK", torch.__version__, torch.version.cuda, transformers.__version__, vllm.__version__, peft.__version__,
      torch.cuda.get_device_name(0))
PY
