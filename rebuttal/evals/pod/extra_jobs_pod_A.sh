#!/bin/bash
# pod_A: after the steering jobs end, run the extra evals on every pod_A model instead of idling:
#   Qwen3.5-9B, Qwen2.5-7B, Llama-3.1-8B, Gemma-3-12B (26 models)
# in runlist priority order (baselines and the Qwen LoRA pool first), resumable, no API key.
# Start it detached right after the steering jobs are launched; it waits for them by itself:
#   nohup bash extra_jobs_pod_A.sh > /dev/null 2>&1 &
# Log: ${EXTRA_OUT:-/workspace/out/extra}/_status/pod_A.extra_jobs.log ; end markers in the same folder:
#   pod_A.EXTRA_ALL_DONE or pod_A.EXTRA_FINISHED_WITH_FAILURES ; per-model status: pod_A.status.tsv
# Extra flags go to run_extra_jobs.py (e.g. --steer-marker /workspace/out/STEER_DONE, --no-wait-steer, --dry-run).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/extra_jobs_common.sh" pod_A "$@"
