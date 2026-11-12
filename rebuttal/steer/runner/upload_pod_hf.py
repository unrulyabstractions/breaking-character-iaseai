#!/usr/bin/env python
"""Upload what capture_pod.sh pulled off a rebuttal pod to the HF dataset, and prove it by listing.

  HF_TOKEN=... python upload_pod_hf.py pod_C [pod_X ...]

For each pod: out/pods/<pod>/files/workspace/out/extra -> rebuttal/extra/<pod>; the pod's logs (workspace/logs,
workspace/vllm_run, workspace/lora_test without its tokenizer copies, top-level workspace files, root *.log/*.py/*.txt)
-> rebuttal/podlogs/<pod>; every runlist model of the pod: out/mgs/<slug> -> rebuttal/mgs/<slug>; out/mgs/_loop ->
rebuttal/mgs/_loop. After each upload the Hub tree is listed and every local file must be there with the same size.
Refuses a pod whose capture_verdict.json is missing or reports any mismatch. Exit 0 only if nothing is missing.
"""
import csv
import json
import os
import shutil
import sys

from huggingface_hub import HfApi

REPO = "unrulyabstractions/assistant-axis-runs"
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
api = HfApi()


def local_files(root):
    out = {}
    for dp, _dn, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root)] = os.path.getsize(p)
    return out


def hub_files(prefix):
    return {it.path[len(prefix) + 1:]: it.size
            for it in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=prefix, recursive=True)
            if it.__class__.__name__ == "RepoFile"}


def up(folder, prefix, msg):
    loc = local_files(folder)
    if not loc:
        print(f"{prefix}: EMPTY local folder {folder}")
        return False
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=folder, path_in_repo=prefix, commit_message=msg)
    hub = hub_files(prefix)
    bad = [k for k, s in loc.items() if hub.get(k) != s]
    print(f"{prefix}: local {len(loc)}, hub {len(hub)}, missing/mismatch {len(bad)}", bad[:3], flush=True)
    return not bad


def main(pods):
    ok = True
    for pod in pods:
        v = json.load(open(f"{R}/out/pods/{pod}/capture_verdict.json"))
        if v.get("missing_or_size_mismatch") != 0 or not v.get("swept"):
            print(f"REFUSED {pod}: capture verdict {v}")
            return 1
        W = f"{R}/out/pods/{pod}/files/workspace"
        ok &= up(f"{W}/out/extra", f"rebuttal/extra/{pod}",
                 f"rebuttal extra evals {pod}: Betley pre-registered 48 generations + TruthfulQA MC1/MC2")
        logs = f"{R}/out/pods/{pod}/podlogs"
        os.makedirs(logs, exist_ok=True)
        for sub in ("logs", "vllm_run", "lora_test"):
            if os.path.isdir(f"{W}/{sub}"):
                shutil.copytree(f"{W}/{sub}", f"{logs}/{sub}", dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("merged") if sub == "lora_test" else None)
        for f in os.listdir(W):
            if os.path.isfile(f"{W}/{f}"):
                shutil.copy2(f"{W}/{f}", f"{logs}/{f}")
        rootd = f"{R}/out/pods/{pod}/files/root"
        if os.path.isdir(rootd):
            for f in os.listdir(rootd):
                if os.path.isfile(f"{rootd}/{f}") and f.endswith((".log", ".py", ".txt")):
                    shutil.copy2(f"{rootd}/{f}", f"{logs}/root_{f}")
        ok &= up(logs, f"rebuttal/podlogs/{pod}", f"rebuttal pod logs {pod}: steering runner, extra evals, vLLM, prefetch")
        slugs = []
        with open(f"{R}/evals/truthfulqa/runlist.tsv") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if row["pod"] == pod:
                    slugs.append(row["label"].replace("/", "__"))
        M = f"{R}/out/mgs"
        for slug in slugs:
            if os.path.isdir(f"{M}/{slug}"):
                ok &= up(f"{M}/{slug}", f"rebuttal/mgs/{slug}", f"rebuttal UK AISI MGS generations (no judge yet) {slug}")
            else:
                print("NO MGS FOLDER", slug)
                ok = False
        print(f"{pod}: {len(slugs)} runlist models")
    ok &= up(f"{R}/out/mgs/_loop", "rebuttal/mgs/_loop", "rebuttal MGS loop logs")
    print("ALL_OK" if ok else "SOMETHING_MISSING")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
