#!/bin/bash
# inside the container: run 3 (live pod_B) and run 4 (live + patch), each from a clean cache and workspace
for tag in live livefix; do
  rm -rf /root/.cache/huggingface/hub /workspace/out /workspace/.pod_B_state /workspace/pod_B.console.log /workspace/pod_B_FINAL_MANIFEST.txt /workspace/pod_B.upload.log /tmp/perf-*.map
  cp /root/pod_B_tiny_$tag.sh /workspace/pod_B_tiny.sh
  cd /workspace; unset HF_TOKEN; export UPLOAD=0 SELF_TERMINATE=0 WATCH_POLL_S=10
  nohup setsid bash /workspace/pod_B_tiny.sh run >> /workspace/pod_B.console.log 2>&1 < /dev/null &
  for i in $(seq 1 200); do [ -f /workspace/.pod_B_state/finished ] && break; sleep 3; done
  sleep 12
  echo "=== $tag finished=$([ -f /workspace/.pod_B_state/finished ] && echo yes || echo no)"
  cut -f1-4 /workspace/out/logs/jobs.tsv
  grep "GC " /workspace/out/logs/prefetch.log
  echo "hub cache left: $(du -s -BM /root/.cache/huggingface/hub | cut -f1); files over 50 MB: $(find /root/.cache/huggingface -type f -size +50M | wc -l)"
  rm -rf /root/out_$tag; cp -a /workspace/out /root/out_$tag; cp /workspace/pod_B.console.log /root/out_$tag/_console.log
done
