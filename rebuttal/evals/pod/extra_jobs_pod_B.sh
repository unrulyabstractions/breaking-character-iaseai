#!/bin/bash
# pod_B: after the steering jobs end, run the extra evals on every pod_B model instead of idling:
#   Qwen2.5-14B (LoRAs, 3 full fine-tunes, 2 rank-1, R1-distill, id-baseline), Mistral-24B (23 models)
# in runlist priority order (baselines and the Qwen LoRA pool first), resumable, no API key.
# Start it detached right after the steering jobs are launched; it waits for them by itself:
#   nohup bash extra_jobs_pod_B.sh > /dev/null 2>&1 &
# Log: ${EXTRA_OUT:-/workspace/out/extra}/_status/pod_B.extra_jobs.log ; end markers in the same folder:
#   pod_B.EXTRA_ALL_DONE or pod_B.EXTRA_FINISHED_WITH_FAILURES ; per-model status: pod_B.status.tsv
# Extra flags go to run_extra_jobs.py (e.g. --steer-marker /workspace/out/STEER_DONE, --no-wait-steer, --dry-run).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/extra_jobs_common.sh" pod_B "$@"
