"""Fetch files of the HF dataset unrulyabstractions/assistant-axis-runs at one pinned revision, and prove each copy.

For every requested path the script asks the Hub for its size and LFS sha256 at the pinned revision. A local copy
already present under one of the --reuse roots is accepted only if its sha256 matches. Anything else is downloaded
into --out at that revision and hashed again. The manifest maps each HF path to the local file that proved equal.

usage:
  python -I hf_fetch.py --paths paths.json --out <download dir> --manifest <manifest.json>
                        [--reuse <dir> ...] [--revision e6c3d0290a4672629d58361f30578358560a8bb6]

paths.json is a JSON list of repo paths. Reads from the Hub only (public dataset, no token needed); writes only
under --out and the manifest file. Exit code 1 if any path cannot be proven.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "unrulyabstractions/assistant-axis-runs"
PINNED = "e6c3d0290a4672629d58361f30578358560a8bb6"  # dataset sha on 2026-11-12 (last_modified 2026-11-09)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_info(api, paths, revision):
    info = {}
    for i in range(0, len(paths), 200):
        batch = paths[i:i + 200]
        for attempt in range(6):
            try:
                for x in api.get_paths_info(REPO, batch, repo_type="dataset", revision=revision, expand=True):
                    info[x.path] = {"size": x.size, "sha256": x.lfs.sha256 if x.lfs else None,
                                    "git_blob": getattr(x, "blob_id", None)}
                break
            except Exception as e:  # 429 / 5xx: back off and retry
                wait = 15 * (attempt + 1)
                print(f"  get_paths_info retry {attempt + 1} after {type(e).__name__}: waiting {wait}s", flush=True)
                time.sleep(wait)
        else:
            raise SystemExit("get_paths_info failed 6 times")
    return info


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def matches(local, ri):
    if not os.path.isfile(local) or os.path.getsize(local) != ri["size"]:
        return False
    if ri["sha256"]:
        return sha256(local) == ri["sha256"]
    return git_blob_sha1(local) == ri["git_blob"]  # small non-LFS file (json): git blob id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--reuse", nargs="*", default=[])
    ap.add_argument("--revision", default=PINNED)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    paths = sorted(set(json.load(open(a.paths))))
    api = HfApi()
    print(f"{len(paths)} paths; asking the Hub for size and sha256 at {a.revision}", flush=True)
    ri = remote_info(api, paths, a.revision)
    absent = [p for p in paths if p not in ri]
    if absent:
        print(f"ABSENT on the Hub at {a.revision}: {absent[:10]} ({len(absent)})")
        sys.exit(1)

    def one(p):
        for root in a.reuse:
            local = os.path.join(root, p)
            if matches(local, ri[p]):
                return p, local, "reused"
        for attempt in range(6):
            try:
                local = hf_hub_download(REPO, p, repo_type="dataset", revision=a.revision, local_dir=a.out)
                break
            except Exception as e:
                time.sleep(15 * (attempt + 1))
        else:
            return p, None, "download failed"
        return (p, local, "downloaded") if matches(local, ri[p]) else (p, None, "HASH MISMATCH after download")

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(one, paths))
    man = {"repo": REPO, "revision": a.revision, "files": {}}
    bad = []
    counts = {}
    for p, local, how in res:
        counts[how] = counts.get(how, 0) + 1
        if local is None:
            bad.append((p, how))
            continue
        man["files"][p] = {"local": os.path.abspath(local), "size": ri[p]["size"], "sha256": ri[p]["sha256"],
                           "git_blob": ri[p]["git_blob"], "how": how}
    Path(a.manifest).write_text(json.dumps(man, indent=1))
    print(f"proven {len(man['files'])}/{len(paths)}: {counts}; manifest -> {a.manifest}")
    if bad:
        print("UNPROVEN:", bad[:20])
        sys.exit(1)


if __name__ == "__main__":
    main()
