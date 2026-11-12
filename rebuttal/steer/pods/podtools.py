#!/usr/bin/env python
"""Helpers for the rebuttal steering pods. make_pods.py embeds this file in every pod_<X>.sh, which
writes it out at run time, so a pod needs only its one bash script.

Needs only huggingface_hub (the pods' small tools venv) and the standard library. Never prints a token.

    podtools.py fetch-steer  --repo R --prefix rebuttal/steer --dest DIR --rev-file F [--revision REV]
    podtools.py fetch-files  --repo R --revision REV --dest DIR --expect EXPECT.json
    podtools.py verify-files --root DIR --expect EXPECT.json
    podtools.py repo-check   SPEC [SPEC ...]          SPEC = kind:repo[:subfolder]=base, kind in lora/full/base/tok
    podtools.py prefetch     SPEC [SPEC ...]          downloads only what from_pretrained / PEFT read
    podtools.py gc           --hub-cache DIR REPO [REPO ...]
    podtools.py sweep        --since FILE --dest DIR --exclude P ... ROOT [ROOT ...]
    podtools.py status       --pod P --out OUT --state STATE --jobs-tsv F --jobs-json F
    podtools.py manifest     --out OUT --report F --hf-path P
    podtools.py upload-file  --repo R --file F --path-in-repo P

Exit status: 0 ok, 1 a check failed, 2 bad usage, 3/4 the sweep found files it could not capture
(3: all on the volume disk, 4: some on the container disk).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_TYPE = "dataset"
SMALL = ["*.json", "*.jinja", "*.model", "*.txt", "*.tiktoken", "tokenizer*"]
BIG_IGNORE = ["original/*", "*.pth", "*.gguf", "consolidated*", "checkpoint-*/*", "*.msgpack", "*.h5",
              "*.onnx", "*.ot"]


def log(msg: str) -> None:
    print(f"[podtools {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status_of(e) -> int | None:
    resp = getattr(e, "response", None)
    code = getattr(resp, "status_code", None)
    return int(code) if code is not None else None


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def cmd_fetch_steer(a) -> int:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    rev = a.revision
    if not rev:
        rev = api.dataset_info(a.repo).sha
    log(f"downloading {a.repo}:{a.prefix}/ at revision {rev}")
    snapshot_download(a.repo, repo_type=REPO_TYPE, revision=rev, allow_patterns=[f"{a.prefix}/**"],
                      local_dir=a.dest)
    root = Path(a.dest) / a.prefix
    n = sum(1 for p in root.rglob("*") if p.is_file()) if root.is_dir() else 0
    if n == 0:
        log(f"FAIL: nothing under {a.prefix}/ in {a.repo} at {rev}")
        return 1
    Path(a.rev_file).write_text(rev + "\n")
    log(f"{n} files under {root}")
    return 0


def cmd_fetch_files(a) -> int:
    """Download each expected file (path relative to the dataset root) at a pinned revision, then
    check its sha256. The expect file is {relpath: sha256}."""
    from huggingface_hub import hf_hub_download

    expect = json.loads(Path(a.expect).read_text())
    bad = 0
    for rel, want in sorted(expect.items()):
        dst = Path(a.dest) / rel
        if not (dst.is_file() and sha256_file(dst) == want):
            for k in range(4):
                try:
                    hf_hub_download(a.repo, rel, repo_type=REPO_TYPE, revision=a.revision, local_dir=a.dest)
                    break
                except Exception as e:  # noqa: BLE001
                    log(f"download {rel} failed ({type(e).__name__} HTTP {status_of(e)}); retry {k + 1}")
                    time.sleep(20 * (k + 1))
        got = sha256_file(dst) if dst.is_file() else None
        if got != want:
            log(f"FAIL {rel}: sha256 {got} != expected {want}")
            bad += 1
    log(f"{len(expect) - bad}/{len(expect)} files present with the expected sha256")
    return 1 if bad else 0


def cmd_verify_files(a) -> int:
    expect = json.loads(Path(a.expect).read_text())
    bad = 0
    for rel, want in sorted(expect.items()):
        p = Path(a.root) / rel
        got = sha256_file(p) if p.is_file() else None
        if got != want:
            log(f"FAIL {rel}: sha256 {got} != expected {want}")
            bad += 1
    log(f"{len(expect) - bad}/{len(expect)} files match their expected sha256 under {a.root}")
    return 1 if bad else 0


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def parse_spec(spec: str):
    """kind:repo[:subfolder][=base]  ->  (kind, repo, subfolder, base)"""
    base = None
    if "=" in spec:
        spec, base = spec.split("=", 1)
    kind, rest = spec.split(":", 1)
    parts = rest.split(":")
    repo = parts[0]
    sub = parts[1] if len(parts) > 1 and parts[1] else None
    if kind not in ("lora", "full", "base", "tok"):
        raise SystemExit(f"bad spec kind {kind!r} in {spec!r}")
    return kind, repo, sub, base


def cmd_repo_check(a) -> int:
    """The token can read every repo (gated ones included), and each LoRA names a base."""
    from huggingface_hub import hf_hub_download

    bad = 0
    for spec in a.specs:
        kind, repo, sub, base = parse_spec(spec)
        pre = f"{sub}/" if sub else ""
        probe = pre + "adapter_config.json" if kind == "lora" else "config.json"
        if kind == "tok":
            probe = "tokenizer_config.json"
        try:
            p = hf_hub_download(repo, probe)
            msg = "ok"
            if kind == "lora":
                ab = json.loads(Path(p).read_text()).get("base_model_name_or_path")
                same = ab and base and ab.split("/")[-1] == base.split("/")[-1]
                msg = f"ok (adapter_config base {ab}; job base {base}{'' if same else ' -- DIFFERENT NAME'})"
            log(f"PASS {kind:4} {repo}{'/' + sub if sub else ''}: {msg}")
        except Exception as e:  # noqa: BLE001
            code = status_of(e)
            hint = (" (gated: accept the licence on huggingface.co with the account of HF_TOKEN)"
                    if code in (401, 403) or "Gated" in type(e).__name__ else "")
            log(f"FAIL {kind:4} {repo}: cannot read {probe}: {type(e).__name__} HTTP {code}{hint}")
            bad += 1
    return 1 if bad else 0


def weight_patterns(repo: str, sub: str | None):
    """The shard files from_pretrained reads: those named in the safetensors index (some repos hold
    two shard sets), else model.safetensors, else every *.safetensors."""
    from huggingface_hub import HfApi, hf_hub_download

    pre = f"{sub}/" if sub else ""
    files = [s.rfilename for s in HfApi().model_info(repo).siblings]
    if pre + "model.safetensors.index.json" in files:
        idx = json.loads(Path(hf_hub_download(repo, pre + "model.safetensors.index.json")).read_text())
        return sorted({pre + f for f in idx["weight_map"].values()})
    if pre + "model.safetensors" in files:
        return [pre + "model.safetensors"]
    return [f for f in files if f.startswith(pre) and f.endswith(".safetensors") and "adapter" not in f]


def cmd_prefetch(a) -> int:
    from huggingface_hub import snapshot_download

    bad = 0
    for spec in a.specs:
        kind, repo, sub, _base = parse_spec(spec)
        pre = f"{sub}/" if sub else ""
        small = [pre + s for s in SMALL] + (SMALL if pre else [])
        if kind == "lora":
            allow = [pre + "adapter_config.json", pre + "adapter_model.safetensors",
                     pre + "adapter_model.bin"] + small
        elif kind == "tok":
            allow = small
        else:
            allow = weight_patterns(repo, sub) + small
        t = time.time()
        try:
            path = snapshot_download(repo, allow_patterns=allow, ignore_patterns=BIG_IGNORE, max_workers=8)
            size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
            log(f"PREFETCHED {kind} {repo}{'/' + sub if sub else ''}: {size / 2**30:.1f} GiB in "
                f"{time.time() - t:.0f}s")
        except Exception as e:  # noqa: BLE001
            log(f"PREFETCH FAILED {repo}: {type(e).__name__} HTTP {status_of(e)}: {str(e)[:300]}")
            bad += 1
    return 1 if bad else 0


def cmd_gc(a) -> int:
    hub = Path(a.hub_cache)
    for repo in a.repos:
        d = hub / ("models--" + repo.replace("/", "--"))
        if d.is_dir():
            size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file() and not p.is_symlink())
            shutil.rmtree(d, ignore_errors=True)
            log(f"GC removed {d} ({size / 2**30:.1f} GiB; re-downloadable from the Hub)")
    return 0


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #
def cmd_sweep(a) -> int:
    """Copy every regular file created or changed since --since under the roots, minus the excluded
    (re-downloadable or already captured) prefixes, into --dest. A file over --max-file-mb, or past
    --max-total-mb, is listed as NOT captured. Exit 0: all captured; 3: some not captured, all of them
    under --volume-root (kept by a pod stop); 4: some not captured elsewhere (a stop would erase them)."""
    import fnmatch

    since = Path(a.since).stat().st_mtime
    dest = Path(a.dest)
    excl = [str(Path(p)) for p in a.exclude] + [str(dest)]

    def excluded(path: str) -> bool:
        """An exclude is a path prefix (a directory or a file) or a glob (secrets, caches, temp)."""
        return any(path == e or path.startswith(e + "/") or fnmatch.fnmatch(path, e) for e in excl)

    rows, total, n_bad, n_bad_off_volume = [], 0, 0, 0
    vol = str(Path(a.volume_root)) if a.volume_root else None
    for root in a.roots:
        if not Path(root).is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            if excluded(dirpath):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if not excluded(os.path.join(dirpath, d))]
            for fn in filenames:
                p = Path(dirpath) / fn
                if excluded(str(p)):
                    continue
                try:
                    st = p.lstat()
                except OSError:
                    continue
                if p.is_symlink() or not p.is_file() or st.st_mtime < since:
                    continue
                row = {"path": str(p), "size": st.st_size}
                if st.st_size > a.max_file_mb * 2**20 or total + st.st_size > a.max_total_mb * 2**20:
                    row["status"] = "NOT_CAPTURED_TOO_LARGE"
                    n_bad += 1
                    n_bad_off_volume += 0 if (vol and str(p).startswith(vol + "/")) else 1
                else:
                    tgt = dest / str(p).lstrip("/")
                    tgt.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(p, tgt)
                        total += st.st_size
                        row["status"] = "CAPTURED"
                    except OSError as e:
                        row["status"] = f"NOT_CAPTURED_{type(e).__name__}"
                        n_bad += 1
                        n_bad_off_volume += 0 if (vol and str(p).startswith(vol + "/")) else 1
                rows.append(row)
    dest.mkdir(parents=True, exist_ok=True)
    rep = {"since_epoch": since, "roots": a.roots, "excluded": excl, "n_files": len(rows),
           "n_not_captured": n_bad, "n_not_captured_off_volume": n_bad_off_volume, "volume_root": vol,
           "captured_bytes": total, "files": rows}
    (dest / "SWEEP.json").write_text(json.dumps(rep, indent=1))
    log(f"sweep: {len(rows)} new files outside the output folder, {n_bad} not captured, "
        f"{total / 2**20:.1f} MiB copied into {dest}")
    for r in rows:
        if r["status"] != "CAPTURED":
            log(f"  {r['status']} {r['path']} ({r['size']} B)")
    return 4 if n_bad_off_volume else (3 if n_bad else 0)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def read_jobs_tsv(path: Path):
    rows = []
    if path.is_file():
        lines = path.read_text().splitlines()
        if lines:
            head = lines[0].split("\t")
            for ln in lines[1:]:
                if ln.strip():
                    rows.append(dict(zip(head, ln.split("\t"))))
    return rows


def cmd_status(a) -> int:
    state = Path(a.state)
    jobs = json.loads(Path(a.jobs_json).read_text())
    runs = read_jobs_tsv(Path(a.jobs_tsv))
    last = {}
    for r in runs:
        last[r["job"]] = r
    out = []
    for j in jobs:
        r = last.get(j["id"], {})
        d = Path(a.out) / j["out"]
        n_cells = len(list((d / "generations").glob("*.jsonl"))) if d.is_dir() else 0
        status = r.get("status") or ("DONE_EARLIER" if (d / "DONE").is_file() else
                                     "PARTIAL" if n_cells else "NOT_RUN")
        out.append({"job": j["id"], "phase": j["phase"], "family": j["family"], "name": j["name"],
                    "set": j["set"], "status": status, "rc": r.get("rc"),
                    "attempt": r.get("attempt"), "seconds": r.get("seconds"),
                    "DONE": (d / "DONE").is_file(), "FAILED": (d / "FAILED").is_file(),
                    "n_cell_files": n_cells,
                    "expected_cells": j["n_cells"]})

    def rd(name):
        p = state / name
        return p.read_text().strip() if p.is_file() else None

    st = {"pod": a.pod, "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"), "written_utc":
          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "clock_start_epoch": rd("clock_start"),
          "stop_reason": rd("STOP"), "steer_revision": rd("steer_rev"), "gpu": rd("gpu"),
          "n_jobs": len(out), "n_done": sum(1 for o in out if o["DONE"]), "jobs": out, "attempts": runs}
    Path(a.out, "POD_STATUS.json").write_text(json.dumps(st, indent=1))
    log(f"POD_STATUS.json: {st['n_done']}/{st['n_jobs']} jobs DONE; stop reason {st['stop_reason']}")
    return 0


def cmd_manifest(a) -> int:
    out = Path(a.out)
    man_p = out / "UPLOAD_MANIFEST.json"
    rep = json.loads(Path(a.report).read_text()) if a.report and Path(a.report).is_file() else {}
    print("=" * 78)
    print(f"FINAL MANIFEST  {out}  ->  {a.hf_path}")
    st_p = out / "POD_STATUS.json"
    if st_p.is_file():
        st = json.loads(st_p.read_text())
        print(f"jobs DONE {st['n_done']}/{st['n_jobs']}   stop reason: {st['stop_reason']}")
        for j in st["jobs"]:
            print(f"  {j['status']:12} {j['n_cell_files']}/{j['expected_cells']} cells  {j['job']}")
    if man_p.is_file():
        man = json.loads(man_p.read_text())
        print(f"files {man['n_files']}  bytes {man['total_bytes']}")
        for e in man["files"]:
            print(f"  {e['size']:>12}  {e['path']}")
    v = rep.get("verify") or {}
    print(f"upload verified: {rep.get('ok')}  ({v.get('n_ok')}/{v.get('n_local')} files match the Hub)")
    print("=" * 78)
    return 0


def cmd_upload_file(a) -> int:
    from huggingface_hub import HfApi

    try:
        HfApi().upload_file(path_or_fileobj=a.file, path_in_repo=a.path_in_repo, repo_id=a.repo,
                            repo_type=REPO_TYPE, commit_message=f"rebuttal: {a.path_in_repo}")
        log(f"uploaded {a.file} -> {a.repo}:{a.path_in_repo}")
        return 0
    except Exception as e:  # noqa: BLE001
        log(f"upload of {a.file} failed: {type(e).__name__} HTTP {status_of(e)}")
        return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("fetch-steer")
    p.add_argument("--repo", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--rev-file", required=True, dest="rev_file")
    p.add_argument("--revision", default=None)
    p = sp.add_parser("fetch-files")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--expect", required=True)
    p = sp.add_parser("verify-files")
    p.add_argument("--root", required=True)
    p.add_argument("--expect", required=True)
    p = sp.add_parser("repo-check")
    p.add_argument("specs", nargs="+")
    p = sp.add_parser("prefetch")
    p.add_argument("specs", nargs="+")
    p = sp.add_parser("gc")
    p.add_argument("--hub-cache", required=True, dest="hub_cache")
    p.add_argument("repos", nargs="+")
    p = sp.add_parser("sweep")
    p.add_argument("--since", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--max-file-mb", type=float, default=1024, dest="max_file_mb")
    p.add_argument("--max-total-mb", type=float, default=4096, dest="max_total_mb")
    p.add_argument("--volume-root", default=None, dest="volume_root")
    p.add_argument("roots", nargs="+")
    p = sp.add_parser("status")
    p.add_argument("--pod", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--jobs-tsv", required=True, dest="jobs_tsv")
    p.add_argument("--jobs-json", required=True, dest="jobs_json")
    p = sp.add_parser("manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--report", default=None)
    p.add_argument("--hf-path", required=True, dest="hf_path")
    p = sp.add_parser("upload-file")
    p.add_argument("--repo", required=True)
    p.add_argument("--file", required=True)
    p.add_argument("--path-in-repo", required=True, dest="path_in_repo")
    a = ap.parse_args(argv)
    fn = {"fetch-steer": cmd_fetch_steer, "fetch-files": cmd_fetch_files, "verify-files": cmd_verify_files,
          "repo-check": cmd_repo_check, "prefetch": cmd_prefetch, "gc": cmd_gc, "sweep": cmd_sweep,
          "status": cmd_status, "manifest": cmd_manifest, "upload-file": cmd_upload_file}[a.cmd]
    return fn(a)


if __name__ == "__main__":
    sys.exit(main())
