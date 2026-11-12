#!/usr/bin/env python
"""After a pod's steering jobs, run the extra evals on every model of that pod, in priority order, unattended.

For each runlist row of --pod (priority 1 first: the baselines and the Qwen LoRA pool organisms and controls;
then the rest), in the runlist's order:
  1. skip it if <out>/<family>/<organism>/DONE exists and eval_model.py agrees the outputs are complete;
  2. wait until no other process holds the GPU (a vLLM server, a steering straggler);
  3. make room on disk: if the weights still to download plus --reserve-gb do not fit, delete cached Hub repos of
     this pod's runlist that no pending row needs (largest first; repos pinned by a live process are kept);
     downloaded model weights are the only thing ever deleted;
  4. download the weights (retries with backoff), then run eval_model.py in a fresh process with a hard timeout:
     (a) Betley pre-registered generations, (b) TruthfulQA MC1/MC2, one model load;
  5. record the result in <out>/_status/<pod>.status.tsv.
Failed rows are retried once after the pass (--retries). At the end it writes <out>/_status/<pod>.EXTRA_ALL_DONE,
or <pod>.EXTRA_FINISHED_WITH_FAILURES listing what failed, and exits 0 or 3. No API key is read or needed.

Waiting for the steering jobs (--wait-steer, on by default): the run starts once no process matches
--steer-pattern for --absent-polls consecutive polls (and --steer-marker exists, if given) and no process holds the
GPU. A heartbeat line every 5 minutes names what it is waiting on.

  python run_extra_jobs.py --pod pod_A --out /workspace/out/extra
Re-running resumes: complete models are skipped without loading.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import podlib  # noqa: E402


def log(msg: str) -> None:
    podlib.log(msg, "extra")


def proc_cmdlines(pattern: str, proc: Path = Path("/proc")) -> list[str]:
    """'<pid> <command line>' of every process whose command line matches pattern, read from /proc/<pid>/cmdline.
    Used when pgrep is missing: the RunPod base image's apt list does not name procps."""
    rx = re.compile(pattern)
    out = []
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:  # the process exited, or no permission
            continue
        if cmd and rx.search(cmd):
            out.append(f"{d.name} {cmd}")
    return out


def steer_procs(pattern: str) -> list[str]:
    """'<pid> <command line>' of every process matching pattern (procps pgrep -af on Linux; -lf elsewhere,
    since BSD pgrep's -a means something else; /proc when there is no pgrep)."""
    if shutil.which("pgrep"):
        flags = "-af" if sys.platform.startswith("linux") else "-lf"
        lines = subprocess.run(["pgrep", flags, pattern], capture_output=True, text=True, timeout=30).stdout.splitlines()
    elif Path("/proc").is_dir():
        lines = proc_cmdlines(pattern)
    else:
        raise SystemExit("cannot list processes: no pgrep and no /proc; pass --no-wait-steer after the steering ends")
    me = {str(os.getpid()), str(os.getppid())}
    return [ln for ln in lines if ln.strip() and ln.split()[0] not in me]


def wait_for_steering(a) -> None:
    absent, last_beat, t0 = 0, 0.0, time.time()
    while True:
        procs = steer_procs(a.steer_pattern)
        marker_ok = (not a.steer_marker) or Path(a.steer_marker).exists()
        gpu = podlib.gpu_compute_pids()
        absent = 0 if procs else absent + 1
        if absent >= a.absent_polls and marker_ok and not gpu:
            log(f"steering finished (no '{a.steer_pattern}' process for {absent} polls"
                + (f", marker {a.steer_marker} present" if a.steer_marker else "") + f"); waited {time.time() - t0:.0f}s")
            return
        if time.time() - last_beat >= 300:
            log(f"waiting for steering: {len(procs)} matching process(es) {[p[:160] for p in procs[:3]]}; "
                f"marker {'ok' if marker_ok else 'missing: ' + str(a.steer_marker)}; GPU apps {gpu[:3]}")
            last_beat = time.time()
        if a.wait_max_h and time.time() - t0 > a.wait_max_h * 3600:
            raise SystemExit(f"steering still running after {a.wait_max_h} h; not starting (see heartbeat lines)")
        time.sleep(a.poll_s)


def wait_gpu_free(a) -> None:
    last_beat, t0 = 0.0, time.time()
    while True:
        gpu = podlib.gpu_compute_pids()
        if not gpu:
            return
        if time.time() - last_beat >= 300:
            log(f"GPU busy, waiting: {gpu[:3]}")
            last_beat = time.time()
        if a.gpu_wait_max_h and time.time() - t0 > a.gpu_wait_max_h * 3600:
            raise SystemExit(f"GPU still busy after {a.gpu_wait_max_h} h: {gpu[:3]}")
        time.sleep(a.poll_s)


def is_done(out_root: Path, spec: podlib.Spec) -> bool:
    return (out_root / spec.slug / "DONE").exists()


def prepare_weights(a, spec: podlib.Spec, specs: list[podlib.Spec], out_root: Path) -> dict:
    """Free disk if needed, then download this model's weights. Returns a record of what happened."""
    pending = [s for s in specs if not is_done(out_root, s)]
    keep = {r for s in pending for r in s.all_repos()}
    evictable = {r for s in specs for r in s.all_repos()} - keep
    need, plans = 0, []
    for repo, sub, is_adapter in spec.weight_downloads():
        try:
            files, missing, total = podlib.plan_download(repo, sub, adapter=is_adapter)
        except Exception as e:  # noqa: BLE001  (Hub API error: fall back to a cached snapshot if there is one)
            snaps = podlib.repo_cache_dir(repo) / "snapshots"
            if snaps.is_dir() and any(snaps.iterdir()):
                log(f"{repo}: Hub metadata unavailable ({type(e).__name__}: {str(e)[:160]}); using the cached "
                    f"snapshot without a download plan")
                continue
            raise
        if not any(f.endswith((".safetensors", ".bin")) for f in files):
            raise RuntimeError(f"{repo}{'/' + sub if sub else ''}: no weight files under the download patterns")
        plans.append((repo, files, missing, total))
        need += missing
    rec = {"need_bytes": need, "free_before": podlib.free_bytes(), "evicted": []}
    reserve, floor = int(a.reserve_gb * 1e9), int(a.min_free_gb * 1e9)
    if not a.no_evict:
        rec["evicted"] = podlib.ensure_space(need, reserve, evictable, keep, dry_run=a.dry_run, floor=floor)
    elif need + min(floor, reserve) > podlib.free_bytes():
        raise podlib.DiskFull(f"need {need / 1e9:.1f} GB + floor, free {podlib.free_bytes() / 1e9:.1f} GB (--no-evict)")
    for repo, files, missing, total in plans:
        if missing and not a.dry_run:
            t = time.time()
            podlib.prefetch(repo, files)
            log(f"downloaded {repo}: {missing / 1e9:.2f} of {total / 1e9:.2f} GB in {time.time() - t:.0f}s")
    rec["free_after"] = podlib.free_bytes()
    return rec


def run_one(a, spec: podlib.Spec, out_root: Path, check_only: bool = False) -> tuple[int, float]:
    out_dir = out_root / spec.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [a.python, "-u", str(HERE / "eval_model.py"), "--runlist", a.runlist, "--label", spec.label,
           "--out", str(out_root), "--evals", a.evals, "--n-per-question", str(a.n_per_question),
           "--max-new-tokens", str(a.max_new_tokens), "--seed", str(a.seed), "--steer-code", a.steer_code,
           "--tqa-dir", a.tqa_dir, "--device", a.device, "--dtype", a.dtype] + a.eval_extra
    if check_only:
        cmd.append("--check-only")
    t0 = time.time()
    with open(out_dir / "eval_model.log", "a") as lf:
        lf.write(f"\n===== {podlib.now_iso()} {' '.join(cmd)}\n")
        lf.flush()
        try:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=a.job_timeout_h * 3600).returncode
        except subprocess.TimeoutExpired:
            rc = 124
            lf.write(f"\n===== TIMEOUT after {a.job_timeout_h} h\n")
    return rc, time.time() - t0


def write_status(out_root: Path, pod: str, rows: dict) -> None:
    p = out_root / "_status" / f"{pod}.status.tsv"
    lines = ["label\tpriority\tstate\trc\tseconds\tattempts\tnote"]
    for r in rows.values():
        lines.append("\t".join(str(r.get(k, "")) for k in ("label", "priority", "state", "rc", "seconds", "attempts", "note")))
    podlib.atomic_write_text(p, "\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pod", required=True, help="pod_A | pod_B | pod_C (the runlist's pod column)")
    ap.add_argument("--out", required=True, help="output root, e.g. /workspace/out/extra")
    ap.add_argument("--runlist", default=str(podlib.DEFAULT_RUNLIST))
    ap.add_argument("--steer-code", default=str(podlib.DEFAULT_STEER_CODE))
    ap.add_argument("--tqa-dir", default=str(podlib.DEFAULT_TQA_DIR))
    ap.add_argument("--python", default=sys.executable, help="interpreter for eval_model.py (the steering venv)")
    ap.add_argument("--evals", default="preregistered,truthfulqa")
    ap.add_argument("--n-per-question", type=int, default=10, help="Betley samples per question (betley48 plan: 10)")
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--only", nargs="*", default=None, help="run only these labels")
    ap.add_argument("--max-priority", type=int, default=None, help="e.g. 1: baselines and the Qwen LoRA pool only")
    ap.add_argument("--wait-steer", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--steer-pattern", default="steer_causal_multi")
    ap.add_argument("--steer-marker", default=None, help="file that must exist before starting (optional)")
    ap.add_argument("--absent-polls", type=int, default=3)
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--wait-max-h", type=float, default=0.0, help="give up waiting for steering after this (0 = never)")
    ap.add_argument("--gpu-wait-max-h", type=float, default=0.0)
    ap.add_argument("--reserve-gb", type=float, default=30.0,
                    help="free disk to keep beyond the next model's download (evicts toward it)")
    ap.add_argument("--min-free-gb", type=float, default=podlib.MIN_FREE_FLOOR / 1e9,
                    help="hard floor beyond the download when nothing more is evictable (capped at --reserve-gb)")
    ap.add_argument("--no-evict", action="store_true")
    ap.add_argument("--job-timeout-h", type=float, default=3.0)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="plan downloads/evictions, run nothing")
    ap.add_argument("--eval-extra", nargs=argparse.REMAINDER, default=[], dest="eval_extra",
                    help="passed to eval_model.py (smoke tests: --limit-questions 2 --tqa-limit 10 --batch-size 4)")
    a = ap.parse_args(argv)
    out_root = Path(a.out)
    (out_root / "_status").mkdir(parents=True, exist_ok=True)
    for stale in (f"{a.pod}.EXTRA_ALL_DONE", f"{a.pod}.EXTRA_FINISHED_WITH_FAILURES"):
        (out_root / "_status" / stale).unlink(missing_ok=True)
    specs = podlib.pod_specs(a.pod, a.runlist)
    if a.only:
        unknown = set(a.only) - {s.label for s in specs}
        if unknown:
            raise SystemExit(f"--only labels not on {a.pod}: {sorted(unknown)}")
        specs = [s for s in specs if s.label in a.only]
    if a.max_priority is not None:
        specs = [s for s in specs if s.priority <= a.max_priority]
    if not specs:
        raise SystemExit(f"no runlist rows for {a.pod}")
    log(f"{a.pod}: {len(specs)} models, evals {a.evals}, K={a.n_per_question}, out {out_root}, hub cache "
        f"{podlib.hub_cache()}, free {podlib.free_bytes() / 1e9:.0f} GB, python {a.python}")
    podlib.atomic_write_json(out_root / "_status" / f"{a.pod}.STARTED", {"utc": podlib.now_iso(), "argv": sys.argv,
                                                                          "labels": [s.label for s in specs]})
    if a.wait_steer and not a.dry_run:
        wait_for_steering(a)

    rows = {s.label: {"label": s.label, "priority": s.priority, "state": "pending", "attempts": 0} for s in specs}
    for attempt in range(1 + a.retries):
        todo = [s for s in specs if rows[s.label]["state"] != "done"]
        if not todo:
            break
        log(f"pass {attempt + 1}: {len(todo)} model(s) to run")
        for s in todo:
            r = rows[s.label]
            r["attempts"] += 1
            if is_done(out_root, s) and not a.dry_run:
                # confirm from the outputs, without a model load or a download
                rc, secs = run_one(a, s, out_root, check_only=True)
                if rc == 0:
                    r.update(state="done", rc=0, seconds=round(secs, 1), note="already complete")
                    log(f"SKIP {s.label}: already complete")
                    write_status(out_root, a.pod, rows)
                    continue
                log(f"{s.label}: DONE marker present but outputs incomplete (check rc={rc}); running it")
            try:
                if not a.dry_run:
                    wait_gpu_free(a)
                with podlib.pin_repos(s.all_repos(), f"run_extra_jobs {s.label}"):
                    prep = prepare_weights(a, s, specs, out_root)
                    r["note"] = f"need {prep['need_bytes'] / 1e9:.1f}GB evicted {len(prep['evicted'])}"
                    if a.dry_run:
                        r["state"] = "dry_run"
                        log(f"[dry-run] {s.label}: {r['note']} free {prep['free_before'] / 1e9:.0f} GB")
                        continue
                    log(f"START {s.label} (priority {s.priority}, attempt {r['attempts']}; {r['note']})")
                    rc, secs = run_one(a, s, out_root)
            except podlib.DiskFull as e:
                rc, secs = 75, 0.0
                r["note"] = f"disk: {e}"
                log(f"{s.label}: DISK {e}")
            except Exception as e:  # noqa: BLE001  (download failures, Hub errors)
                rc, secs = 76, 0.0
                r["note"] = f"{type(e).__name__}: {str(e)[:300]}"
                log(f"{s.label}: PREP FAILED {r['note']}")
            r["rc"], r["seconds"] = rc, round(secs, 1)
            r["state"] = "done" if rc == 0 and is_done(out_root, s) else "failed"
            log(f"END {s.label}: rc={rc} {r['state']} in {secs:.0f}s (log {out_root / s.slug / 'eval_model.log'})")
            write_status(out_root, a.pod, rows)
        if a.dry_run:
            break

    write_status(out_root, a.pod, rows)
    failed = [r["label"] for r in rows.values() if r["state"] != "done"]
    if a.dry_run:
        return 0
    if failed:
        podlib.atomic_write_json(out_root / "_status" / f"{a.pod}.EXTRA_FINISHED_WITH_FAILURES",
                                 {"utc": podlib.now_iso(), "failed": failed,
                                  "notes": {k: rows[k].get("note") for k in failed}})
        log(f"{a.pod}: FINISHED WITH FAILURES {failed}")
        return 3
    podlib.atomic_write_json(out_root / "_status" / f"{a.pod}.EXTRA_ALL_DONE",
                             {"utc": podlib.now_iso(), "n_models": len(specs)})
    log(f"{a.pod}: EXTRA_ALL_DONE ({len(specs)} models)")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit as e:
        if isinstance(e.code, str):
            log(f"STOPPED: {e.code}")
            rc = 2
        else:
            rc = 0 if e.code is None else e.code
    sys.exit(rc)
