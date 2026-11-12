#!/usr/bin/env python
"""Upload ONE folder to the HF dataset in one commit, then prove every file arrived.

    python upload_once.py --check-auth                       # at pod START: token + repo access
    python upload_once.py --folder /workspace/out --prefix rebuttal/steering/pod_A
    python upload_once.py --folder /workspace/out --prefix rebuttal/steering/pod_A --verify-only
    python upload_once.py --folder /workspace/out --prefix rebuttal/steering/pod_A --dry-run

Reads the token from HF_TOKEN (huggingface_hub's own lookup); never prints it.

Upload. Before uploading, writes <folder>/UPLOAD_MANIFEST.json (path, size, sha256 of every other
file) so the manifest travels with the data. Each attempt runs in a child process with a hard
timeout (--attempt-timeout), because the Xet transfer backend has hung with no timeout on this
project before. Xet is off by default (HF_HUB_DISABLE_XET=1): huggingface_hub 1.32 then makes ONE
commit through the legacy LFS path; --xet uses the Xet pipeline, which may split the upload into
several commits. Retries: HTTP 429 sleeps --sleep-429 seconds (HF's API quota window is 5 minutes);
5xx, connection errors and timeouts back off 60, 120, 240, 480, 600 s; 400/401/403/404/422 stop at
once. Before each retry the remote is checked, so an upload whose response was lost is not repeated.

Verify. Lists the remote files under --prefix (recursive) and compares EVERY local file: the remote
must exist with the same byte size, and the same content hash where the Hub reports one (sha256 for
LFS files, the git blob sha1 for regular files). A report goes to --report (default
<folder>.upload_report.json, beside the folder, not inside it).

Exit status: 0 = uploaded (or already present) and every file verified; 1 = upload or verification
failed; 2 = refused before any network write (bad prefix, no token, symlink, too large).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = "unrulyabstractions/assistant-axis-runs"
REPO_TYPE = "dataset"
PREFIX_ROOT = "rebuttal/"
MANIFEST = "UPLOAD_MANIFEST.json"
NON_TRANSIENT = {400, 401, 403, 404, 422}


def log(msg: str) -> None:
    print(f"[upload_once {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _set_xet_env(use_xet: bool, env=None):
    env = os.environ if env is None else env
    if use_xet:
        env.pop("HF_HUB_DISABLE_XET", None)
    else:
        env["HF_HUB_DISABLE_XET"] = "1"
    return env


# --------------------------------------------------------------------------- #
# local side
# --------------------------------------------------------------------------- #
def git_blob_sha1(path: Path, size: int) -> str:
    h = hashlib.sha1()
    h.update(b"blob %d\0" % size)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_files(folder: Path) -> list[str]:
    """The relative paths upload_folder will send: every file, minus huggingface_hub's
    DEFAULT_IGNORE_PATTERNS (.git, .cache/huggingface), with the same filter function."""
    from huggingface_hub.utils import DEFAULT_IGNORE_PATTERNS, filter_repo_objects

    rel = [p.relative_to(folder).as_posix() for p in sorted(folder.glob("**/*")) if p.is_file()]
    return list(filter_repo_objects(rel, ignore_patterns=list(DEFAULT_IGNORE_PATTERNS)))


def preflight_folder(folder: Path, max_total_gb: float) -> list[str]:
    """Walks without following links: a symlink anywhere is refused (upload_folder would send
    its target, e.g. a model cache), as is a '.git' or '.cache' folder (the Hub rejects them)."""
    problems = []
    if not folder.is_dir() or folder.is_symlink():
        return [f"{folder} is not a plain directory"]
    total = 0
    for root, dirs, files in os.walk(folder, followlinks=False):
        for name in dirs + files:
            p = Path(root) / name
            if p.is_symlink():
                problems.append(f"symlink in the folder (would upload its target): {p}")
            elif name in (".git", ".cache") and p.is_dir():
                problems.append(f"'{name}' folder in the upload (the Hub refuses it): {p}")
        for name in files:
            p = Path(root) / name
            if p.is_symlink():
                continue
            total += p.stat().st_size
            if name.endswith(".tmp"):
                log(f"WARNING: unfinished temp file will be uploaded as is: {p}")
    if total > max_total_gb * 2**30:
        problems.append(f"folder holds {total / 2**30:.2f} GiB > --max-total-gb {max_total_gb} "
                        "(model weights in the output folder?)")
    return problems


def write_manifest(folder: Path) -> dict:
    files = [r for r in local_files(folder) if r != MANIFEST]
    entries = []
    for rel in files:
        p = folder / rel
        entries.append({"path": rel, "size": p.stat().st_size, "sha256": sha256_file(p)})
    man = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_files": len(entries), "total_bytes": sum(e["size"] for e in entries),
           "pod_id": os.environ.get("RUNPOD_POD_ID"), "files": entries}
    tmp = folder / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(man, indent=1))
    os.replace(tmp, folder / MANIFEST)
    return man


# --------------------------------------------------------------------------- #
# remote side
# --------------------------------------------------------------------------- #
def _status(e) -> int | None:
    resp = getattr(e, "response", None)
    code = getattr(resp, "status_code", None)
    return int(code) if code is not None else None


def remote_listing(api, repo: str, prefix: str) -> dict:
    """{relative path under prefix: RepoFile} for every file under prefix."""
    from huggingface_hub.hf_api import RepoFile

    out = {}
    for item in api.list_repo_tree(repo, path_in_repo=prefix, recursive=True, repo_type=REPO_TYPE):
        if isinstance(item, RepoFile) or (hasattr(item, "size") and hasattr(item, "blob_id")):
            out[item.path[len(prefix) + 1:]] = item
    return out


def verify(api, folder: Path, repo: str, prefix: str) -> dict:
    """Compare every local file with the remote listing. Never raises on a mismatch."""
    try:
        remote = remote_listing(api, repo, prefix)
    except Exception as e:  # noqa: BLE001
        code = _status(e)
        if code == 404:  # nothing under the prefix yet (or no repo access)
            remote = {}
        else:
            return {"ok": False, "error": f"listing failed: {type(e).__name__} HTTP {code}: {str(e)[:300]}",
                    "files": []}
    rows, ok = [], True
    for rel in local_files(folder):
        p = folder / rel
        size = p.stat().st_size
        row = {"path": rel, "local_size": size}
        rf = remote.get(rel)
        if rf is None:
            row.update(status="MISSING_REMOTE")
            ok = False
        else:
            row["remote_size"] = rf.size
            lfs = rf.lfs
            if rf.size != size:
                row["status"] = "SIZE_MISMATCH"
                ok = False
            elif lfs is not None:
                local_sha = sha256_file(p)
                remote_sha = lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256
                row.update(hash="sha256", local_hash=local_sha, remote_hash=remote_sha)
                row["status"] = "OK" if local_sha == remote_sha else "HASH_MISMATCH"
            elif rf.blob_id:
                local_oid = git_blob_sha1(p, size)
                row.update(hash="git_blob_sha1", local_hash=local_oid, remote_hash=rf.blob_id)
                row["status"] = "OK" if local_oid == rf.blob_id else "HASH_MISMATCH"
            else:
                row["status"] = "OK_SIZE_ONLY"
            if row["status"] not in ("OK", "OK_SIZE_ONLY"):
                ok = False
        rows.append(row)
    extra = sorted(set(remote) - {r["path"] for r in rows})
    n_bad = sum(1 for r in rows if not r["status"].startswith("OK"))
    return {"ok": ok and bool(rows), "n_local": len(rows), "n_remote": len(remote),
            "n_ok": len(rows) - n_bad, "n_bad": n_bad, "remote_only": extra, "files": rows}


# --------------------------------------------------------------------------- #
# upload attempt (child process)
# --------------------------------------------------------------------------- #
def child_upload(args) -> int:
    """One upload_folder call; prints one JSON line with the outcome."""
    from huggingface_hub import HfApi

    try:
        api = HfApi()
        info = api.upload_folder(repo_id=args.repo, repo_type=REPO_TYPE, folder_path=str(args.folder),
                                 path_in_repo=args.prefix, commit_message=args.commit_message)
        res = {"ok": True, "commit_url": str(getattr(info, "commit_url", info)),
               "oid": getattr(info, "oid", None)}
    except Exception as e:  # noqa: BLE001
        res = {"ok": False, "status": _status(e), "error": f"{type(e).__name__}: {str(e)[:500]}"}
    print("UPLOAD_RESULT " + json.dumps(res), flush=True)
    return 0 if res["ok"] else 1


def run_attempt(args, n: int) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), "--_child", "--folder", str(args.folder),
           "--prefix", args.prefix, "--repo", args.repo, "--commit-message", args.commit_message]
    env = _set_xet_env(args.xet, dict(os.environ))
    log(f"attempt {n}: upload_folder {args.folder} -> {args.repo}:{args.prefix} "
        f"(xet {'on' if args.xet else 'off'}, timeout {args.attempt_timeout}s)")
    try:
        cp = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.attempt_timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "status": None, "error": f"attempt timed out after {args.attempt_timeout}s"}
    for line in cp.stdout.splitlines():
        if line.startswith("UPLOAD_RESULT "):
            return json.loads(line[len("UPLOAD_RESULT "):])
    tail = (cp.stderr or cp.stdout or "")[-500:]
    return {"ok": False, "status": None, "error": f"child exited {cp.returncode} without a result: {tail}"}


def backoff_seconds(res: dict, k: int, sleep_429: int) -> int | None:
    """Seconds to wait before retry k (0-based), or None for a non-transient failure."""
    code = res.get("status")
    if code in NON_TRANSIENT:
        return None
    if code == 429:
        return sleep_429
    return min(600, 60 * 2 ** k)


# --------------------------------------------------------------------------- #
# auth check
# --------------------------------------------------------------------------- #
def check_auth(api, repo: str) -> int:
    if not os.environ.get("HF_TOKEN"):
        log("HF_TOKEN is not set in the environment")
        return 1
    try:
        who = api.whoami()
    except Exception as e:  # noqa: BLE001
        log(f"whoami failed: {type(e).__name__} HTTP {_status(e)} (token invalid or no network)")
        return 1
    tok = (who.get("auth") or {}).get("accessToken") or {}
    role = tok.get("role")
    ns = repo.split("/")[0]
    write_ok = role == "write"
    if role == "fineGrained":
        fg = tok.get("fineGrained") or {}
        for sc in fg.get("scoped") or []:
            ent = sc.get("entity") or {}
            perms = set(sc.get("permissions") or [])
            if "repo.write" in perms and (ent.get("name") in (ns, repo)):
                write_ok = True
    try:
        info = api.repo_info(repo, repo_type=REPO_TYPE)
        repo_ok = True
        private = getattr(info, "private", None)
    except Exception as e:  # noqa: BLE001
        repo_ok, private = False, None
        log(f"repo_info({repo}) failed: {type(e).__name__} HTTP {_status(e)}")
    log(f"token user={who.get('name')} role={role} write_access={'yes' if write_ok else 'NOT FOUND'}; "
        f"dataset {repo} readable={repo_ok} private={private}")
    return 0 if (write_ok and repo_ok) else 1


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", type=Path)
    ap.add_argument("--prefix", help=f"path in the dataset; must start with {PREFIX_ROOT}")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--commit-message", default=None, dest="commit_message")
    ap.add_argument("--max-attempts", type=int, default=6, dest="max_attempts")
    ap.add_argument("--attempt-timeout", type=int, default=3600, dest="attempt_timeout")
    ap.add_argument("--sleep-429", type=int, default=310, dest="sleep_429")
    ap.add_argument("--verify-retries", type=int, default=3, dest="verify_retries")
    ap.add_argument("--max-total-gb", type=float, default=20.0, dest="max_total_gb")
    ap.add_argument("--xet", action="store_true", help="use the Xet upload pipeline (default off)")
    ap.add_argument("--check-auth", action="store_true", dest="check_auth")
    ap.add_argument("--verify-only", action="store_true", dest="verify_only")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None, api=None, attempt_fn=None, sleep_fn=time.sleep) -> int:
    args = parse_args(argv)
    _set_xet_env(args.xet)
    if args._child:
        return child_upload(args)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    if args.check_auth:
        return check_auth(api, args.repo)

    if not args.folder or not args.prefix:
        log("--folder and --prefix are required")
        return 2
    args.folder = args.folder.resolve()
    args.prefix = args.prefix.strip("/")
    if not args.prefix.startswith(PREFIX_ROOT) or ".." in args.prefix.split("/"):
        log(f"refused: --prefix must start with {PREFIX_ROOT} and contain no '..' (got {args.prefix!r})")
        return 2
    args.commit_message = args.commit_message or f"rebuttal: upload {args.prefix} (one commit)"
    report_path = args.report or args.folder.with_name(args.folder.name + ".upload_report.json")
    problems = preflight_folder(args.folder, args.max_total_gb)
    for p in problems:
        log(f"refused: {p}")
    if problems:
        return 2

    if args.dry_run:
        files = local_files(args.folder)
        total = sum((args.folder / r).stat().st_size for r in files)
        log(f"dry run: {len(files)} files, {total / 2**20:.1f} MiB would go to {args.repo}:{args.prefix}/")
        for r in files[:20]:
            log(f"  {r}")
        return 0

    attempts = []
    if not args.verify_only:
        if not os.environ.get("HF_TOKEN"):
            log("refused: HF_TOKEN is not set")
            return 2
        man = write_manifest(args.folder)
        log(f"wrote {MANIFEST}: {man['n_files']} files, {man['total_bytes'] / 2**20:.1f} MiB")
        attempt_fn = attempt_fn or run_attempt
        uploaded = False
        for k in range(args.max_attempts):
            res = attempt_fn(args, k + 1)
            attempts.append(res)
            log(f"attempt {k + 1}: {json.dumps(res)[:400]}")
            if res.get("ok"):
                uploaded = True
                break
            v = verify(api, args.folder, args.repo, args.prefix)
            if v["ok"]:
                log("remote already holds every file (the failed attempt's commit landed)")
                uploaded = True
                break
            wait = backoff_seconds(res, k, args.sleep_429)
            if wait is None:
                log(f"non-transient HTTP {res.get('status')}; not retrying")
                break
            if k < args.max_attempts - 1:
                log(f"retrying in {wait}s")
                sleep_fn(wait)
        if not uploaded:
            log("upload did not complete; verifying what is there")

    v = {"ok": False}
    for i in range(max(1, args.verify_retries)):
        v = verify(api, args.folder, args.repo, args.prefix)
        if v["ok"]:
            break
        if i < args.verify_retries - 1:
            log(f"verify pass {i + 1}: {v.get('n_bad', '?')} files not matching yet; re-listing in 30s")
            sleep_fn(30)
    report = {"repo": args.repo, "prefix": args.prefix, "folder": str(args.folder),
              "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "xet": args.xet, "attempts": attempts, "verify": v, "ok": bool(v.get("ok"))}
    tmp = report_path.with_name(report_path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, report_path)
    if v.get("ok"):
        log(f"VERIFIED: {v['n_ok']}/{v['n_local']} local files present remotely with equal size "
            f"(and hash where reported); report {report_path}")
        return 0
    bad = [r for r in v.get("files", []) if not r["status"].startswith("OK")]
    why = v.get("error") or "{} of {} files do not match".format(len(bad), v.get("n_local", 0))
    log(f"NOT VERIFIED: {why}; report {report_path}")
    for r in bad[:20]:
        log(f"  {r['status']} {r['path']} local={r.get('local_size')} remote={r.get('remote_size')}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
