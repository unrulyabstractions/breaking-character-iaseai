#!/bin/bash
# remote_setup.sh <box> <repo> [<repo> ...]   runs ON a vast.ai box under nohup (launch_box.sh starts it).
# The steering venv (requirements-pod.txt, torch cu128) and the model prefetch run in parallel; then the job runner
# runs jobs_<box>.tsv. Logs: /root/venv_install.log (ENV_OK), /root/prefetch.log (ALL_DONE), /root/setup.log
# (SETUP_OK or SETUP_FAILED), /workspace/runner_<box>.log (START/END per job, ALL_JOBS_DONE).
set -u
BOX=$1; shift
mkdir -p /workspace/logs
echo "SETUP_START $(date -u +%FT%TZ)" >> /root/setup.log
(
  pip install -q uv &&
  uv venv -q -p 3.12 /workspace/venv &&
  . /workspace/venv/bin/activate &&
  uv pip install -q -r /workspace/rebuttal/requirements-pod.txt \
     --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match &&
  python -c "import torch,transformers,peft; assert torch.cuda.is_available(); print('ENV_OK', torch.__version__, torch.version.cuda, transformers.__version__, peft.__version__, torch.cuda.get_device_name(0))"
) > /root/venv_install.log 2>&1 &
VP=$!
(
  pip install -q -U huggingface_hub hf_transfer &&
  python3 /workspace/rebuttal/prefetch.py "$@"
) > /root/prefetch.log 2>&1 &
PP=$!
wait $VP; wait $PP
if ! grep -q ENV_OK /root/venv_install.log; then
  echo "SETUP_FAILED venv $(date -u +%FT%TZ)" >> /root/setup.log
  exit 1
fi
grep -q ALL_DONE /root/prefetch.log || echo "prefetch incomplete; the runner downloads what is missing" >> /root/setup.log
echo "SETUP_OK $(date -u +%FT%TZ)" >> /root/setup.log
cd /workspace/rebuttal/runner && bash run_jobs.sh "jobs_$BOX.tsv" > "/workspace/runner_$BOX.log" 2>&1
echo "RUNNER_EXIT rc=$? $(date -u +%FT%TZ)" >> /root/setup.log
