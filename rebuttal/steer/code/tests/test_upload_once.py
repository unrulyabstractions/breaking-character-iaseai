"""Offline tests of upload_once.py. No network: the HF API is a fake whose listings are built
with huggingface_hub's own RepoFile class, and errors are real HfHubHTTPError objects.

usage: python test_upload_once.py <scratch dir>     exit 0 iff every check passes
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import httpx

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
os.environ["HF_TOKEN"] = "hf_FAKE_TOKEN_FOR_OFFLINE_TESTS_ONLY"
import upload_once as uo  # noqa: E402
from huggingface_hub.errors import HfHubHTTPError  # noqa: E402
from huggingface_hub.hf_api import RepoFile  # noqa: E402

WORK = Path(sys.argv[1])
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
results = {}


def check(name, cond, detail=""):
    results[name] = {"pass": bool(cond), "detail": str(detail)[:300]}
    print(("PASS " if cond else "FAIL ") + name + (f"  [{str(detail)[:200]}]" if detail else ""), flush=True)


def http_error(code):
    req = httpx.Request("POST", "https://huggingface.co/api/datasets/x/commit/main")
    return HfHubHTTPError(f"HTTP {code}", response=httpx.Response(code, request=req))


class FakeApi:
    """Remote store: {path_in_repo: bytes}. LFS-style entries for *.pt, regular otherwise."""

    def __init__(self):
        self.store = {}
        self.list_calls = 0
        self.corrupt = None  # path whose remote size is reported wrong

    def put_folder(self, folder, prefix):
        for rel in uo.local_files(Path(folder)):
            self.store[f"{prefix}/{rel}"] = (Path(folder) / rel).read_bytes()

    def list_repo_tree(self, repo, path_in_repo=None, recursive=False, repo_type=None):
        self.list_calls += 1
        hits = [p for p in self.store if p.startswith(path_in_repo + "/")]
        if not hits:
            raise http_error(404)
        out = []
        for p in sorted(hits):
            data = self.store[p]
            size = len(data) + (1 if p == self.corrupt else 0)
            j = {"type": "file", "path": p, "size": size,
                 "oid": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()}
            if p.endswith(".pt"):
                j["oid"] = "0" * 40  # pointer-file oid, not content
                j["lfs"] = {"size": size, "oid": hashlib.sha256(data).hexdigest(), "pointerSize": 134}
            out.append(RepoFile(**j))
        return out

    def whoami(self):
        return self._who

    def repo_info(self, repo, repo_type=None):
        class I:  # noqa: E742
            private = True
        return I()


def make_folder(name):
    f = WORK / name
    (f / "j3" / "fam" / "org" / "generations").mkdir(parents=True)
    (f / "j3" / "fam" / "org" / "generations" / "baseline.jsonl").write_text('{"a": 1}\n' * 50)
    (f / "j3" / "fam" / "org" / "META.json").write_text(json.dumps({"x": 1}))
    (f / "j3" / "fam" / "org" / "directions.pt").write_bytes(os.urandom(5000))
    (f / "logs").mkdir()
    (f / "logs" / "pod.log").write_text("line\n" * 10)
    return f


def run(argv, api, attempt_fn=None):
    sleeps = []
    rc = uo.main(argv, api=api, attempt_fn=attempt_fn, sleep_fn=sleeps.append)
    return rc, sleeps


P = "rebuttal/steering/pod_T"

# U1: success first try; manifest written inside the folder and uploaded; full verify
api, folder = FakeApi(), make_folder("u1")
def ok_attempt(args, n):  # noqa: E302
    api.put_folder(args.folder, args.prefix)
    return {"ok": True, "commit_url": "fake"}
rc, sleeps = run(["--folder", str(folder), "--prefix", P], api, ok_attempt)
rep = json.loads(folder.with_name("u1.upload_report.json").read_text())
man = json.loads((folder / uo.MANIFEST).read_text())
hashes = {r["path"]: r.get("hash") for r in rep["verify"]["files"]}
check("U1_success_exit0", rc == 0 and rep["ok"] and not sleeps, f"rc={rc} sleeps={sleeps}")
check("U1_every_file_verified", rep["verify"]["n_ok"] == rep["verify"]["n_local"] == 5, rep["verify"]["n_ok"])
check("U1_hash_kinds", hashes["j3/fam/org/directions.pt"] == "sha256"
      and hashes["j3/fam/org/generations/baseline.jsonl"] == "git_blob_sha1", hashes)
check("U1_manifest_lists_other_files", man["n_files"] == 4 and all(e["sha256"] for e in man["files"]),
      [e["path"] for e in man["files"]])
check("U1_report_outside_folder", not (folder / "u1.upload_report.json").exists())

# U2: 429 twice then success -> two 310 s sleeps
api, folder = FakeApi(), make_folder("u2")
calls = []
def flaky(args, n):  # noqa: E302
    calls.append(n)
    if n <= 2:
        return {"ok": False, "status": 429, "error": "HfHubHTTPError: 429"}
    api.put_folder(args.folder, args.prefix)
    return {"ok": True}
rc, sleeps = run(["--folder", str(folder), "--prefix", P], api, flaky)
check("U2_429_retried_then_ok", rc == 0 and calls == [1, 2, 3] and sleeps == [310, 310], f"rc={rc} sleeps={sleeps}")

# U3: 403 is non-transient: one attempt, exit 1, report ok false
api, folder = FakeApi(), make_folder("u3")
calls = []
def forbidden(args, n):  # noqa: E302
    calls.append(n)
    return {"ok": False, "status": 403, "error": "HfHubHTTPError: 403"}
rc, sleeps = run(["--folder", str(folder), "--prefix", P], api, forbidden)
rep = json.loads(folder.with_name("u3.upload_report.json").read_text())
check("U3_403_no_retry_exit1", rc == 1 and calls == [1] and not rep["ok"], f"rc={rc} calls={calls}")

# U4: upload "succeeds" but one remote size is wrong -> NOT VERIFIED, exit 1
api, folder = FakeApi(), make_folder("u4")
api.corrupt = f"{P}/logs/pod.log"
rc, sleeps = run(["--folder", str(folder), "--prefix", P], api, ok_attempt)
rep = json.loads(folder.with_name("u4.upload_report.json").read_text())
bad = [r for r in rep["verify"]["files"] if r["status"] != "OK"]
check("U4_size_mismatch_fails", rc == 1 and len(bad) == 1 and bad[0]["status"] == "SIZE_MISMATCH"
      and sleeps == [30, 30], f"rc={rc} bad={bad} sleeps={sleeps}")

# U5: response lost (500) but the commit landed -> verify finds it, no second attempt
api, folder = FakeApi(), make_folder("u5")
calls = []
def landed_then_500(args, n):  # noqa: E302
    calls.append(n)
    api.put_folder(args.folder, args.prefix)
    return {"ok": False, "status": 500, "error": "HfHubHTTPError: 500"}
rc, sleeps = run(["--folder", str(folder), "--prefix", P], api, landed_then_500)
check("U5_lost_response_not_repeated", rc == 0 and calls == [1] and not sleeps, f"rc={rc} calls={calls}")

# U6: refusals before any network write
api, folder = FakeApi(), make_folder("u6")
rc, _ = run(["--folder", str(folder), "--prefix", "steering/qwen2.5-7b"], api, ok_attempt)
check("U6_prefix_outside_rebuttal_refused", rc == 2 and not api.store, rc)
rc, _ = run(["--folder", str(folder), "--prefix", "rebuttal/../steering"], api, ok_attempt)
check("U6_dotdot_refused", rc == 2 and not api.store, rc)
(folder / "link").symlink_to(WORK)
rc, _ = run(["--folder", str(folder), "--prefix", P], api, ok_attempt)
check("U6_symlink_refused", rc == 2 and not api.store, rc)
(folder / "link").unlink()
rc, _ = run(["--folder", str(folder), "--prefix", P, "--max-total-gb", "0.000001"], api, ok_attempt)
check("U6_too_large_refused", rc == 2 and not api.store, rc)
tok = os.environ.pop("HF_TOKEN")
rc, _ = run(["--folder", str(folder), "--prefix", P], api, ok_attempt)
os.environ["HF_TOKEN"] = tok
check("U6_no_token_refused", rc == 2 and not api.store, rc)

# U7: dry run touches nothing
api, folder = FakeApi(), make_folder("u7")
rc, _ = run(["--folder", str(folder), "--prefix", P, "--dry-run"], api, ok_attempt)
check("U7_dry_run", rc == 0 and not api.store and api.list_calls == 0
      and not (folder / uo.MANIFEST).exists(), rc)

# U8: backoff policy
check("U8_backoff", uo.backoff_seconds({"status": None}, 0, 310) == 60
      and uo.backoff_seconds({"status": 502}, 2, 310) == 240
      and uo.backoff_seconds({"status": 503}, 9, 310) == 600
      and uo.backoff_seconds({"status": 429}, 4, 310) == 310
      and uo.backoff_seconds({"status": 401}, 0, 310) is None)

# U9: _status reads a real HfHubHTTPError
check("U9_status_from_real_error", uo._status(http_error(429)) == 429 and uo._status(ValueError()) is None)

# U10: check-auth decisions
api = FakeApi()
api._who = {"name": "someone", "auth": {"accessToken": {"role": "write"}}}
check("U10_write_token_ok", uo.check_auth(api, uo.REPO) == 0)
api._who = {"name": "someone", "auth": {"accessToken": {"role": "read"}}}
check("U10_read_token_fails", uo.check_auth(api, uo.REPO) == 1)
api._who = {"name": "someone", "auth": {"accessToken": {"role": "fineGrained", "fineGrained": {"scoped": [
    {"entity": {"type": "user", "name": "unrulyabstractions"}, "permissions": ["repo.content.read", "repo.write"]}]}}}}
check("U10_finegrained_write_ok", uo.check_auth(api, uo.REPO) == 0)

# U11: the real parent/child protocol: a child that hangs is killed at --attempt-timeout,
# a child that gets HTTP 429 reports status 429. The child is this test's fake_child.py, which
# replaces huggingface_hub.HfApi with a fake before running upload_once's own child code.
fake_child = WORK / "fake_child.py"
fake_child.write_text(textwrap.dedent(f"""
    import os, sys, time, httpx
    sys.path.insert(0, {str(CODE)!r})
    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError
    class Api:
        def upload_folder(self, **kw):
            mode = os.environ["FAKE_CHILD_MODE"]
            if mode == "hang":
                time.sleep(60)
            if mode == "429":
                req = httpx.Request("POST", "https://huggingface.co/x")
                raise HfHubHTTPError("429", response=httpx.Response(429, request=req))
            assert os.environ.get("HF_HUB_DISABLE_XET") == "1", "xet must be off by default"
            class C: commit_url = "https://fake/commit"; oid = "abc"
            return C()
    huggingface_hub.HfApi = Api
    import upload_once
    sys.exit(upload_once.main(sys.argv[1:]))
"""))
folder = make_folder("u11")
args = uo.parse_args(["--folder", str(folder), "--prefix", P, "--attempt-timeout", "3"])
args.commit_message = "test"
real_file = uo.__file__
uo.__file__ = str(fake_child)
try:
    os.environ["FAKE_CHILD_MODE"] = "hang"
    r_hang = uo.run_attempt(args, 1)
    args.attempt_timeout = 60
    os.environ["FAKE_CHILD_MODE"] = "429"
    r_429 = uo.run_attempt(args, 1)
    os.environ["FAKE_CHILD_MODE"] = "ok"
    r_ok = uo.run_attempt(args, 1)
finally:
    uo.__file__ = real_file
check("U11_child_timeout_killed", not r_hang["ok"] and "timed out" in r_hang["error"], r_hang)
check("U11_child_429_status", not r_429["ok"] and r_429["status"] == 429, r_429)
check("U11_child_ok_xet_off", r_ok.get("ok") is True and r_ok.get("commit_url") == "https://fake/commit", r_ok)

# U12: no secret in any output file
blob = "".join(p.read_text(errors="ignore") for p in WORK.rglob("*.json"))
check("U12_token_never_written", "hf_FAKE_TOKEN" not in blob)

summary = {"n_pass": sum(r["pass"] for r in results.values()), "n_checks": len(results), "results": results}
(WORK / "test_results.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
sys.exit(0 if summary["n_pass"] == summary["n_checks"] else 1)
