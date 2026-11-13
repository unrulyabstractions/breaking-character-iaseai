"""Prefetch repos into the default HF cache. Args: repo or repo::include-glob. One line per repo to stdout."""
import os, sys, time, shutil
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")  # 2026-11-13: hf_transfer is ignored now; a 32B base took 32 min on vast_B
from huggingface_hub import snapshot_download
IGN = ["original/*", "*.pth", "*.gguf", "consolidated*", "checkpoint-*/*", "pytorch_model*.bin"]
for spec in sys.argv[1:]:
    repo, _, inc = spec.partition("::")
    t = time.time()
    try:
        p = snapshot_download(repo, allow_patterns=[inc] if inc else None, ignore_patterns=IGN, max_workers=16)
        st = "ok"
    except Exception as e:
        st = f"FAIL {type(e).__name__}: {str(e)[:200]}"
    print(f"{time.strftime('%H:%M:%S')} {repo} {st} {time.time()-t:.0f}s used={shutil.disk_usage('/').used/1e9:.0f}GB", flush=True)
print("ALL_DONE", flush=True)
