"""Check every cell of one run's output folder against the question sets (stdlib + yaml only).

argv: OUT_DIR STEER_DIR   (OUT_DIR = a copy of /workspace/out; STEER_DIR = the staged rebuttal/steer)
"""
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

OUT, STEER = Path(sys.argv[1]), Path(sys.argv[2])
JOBS = {  # out folder -> (questions file, n_per_question, conditions)
    "j1/tiny-qwen0.5b/ctrl/em": ("em_2items.yaml", 2, ["neg_global", "random_ctrl"]),
    "j1/tiny-qwen0.5b/ctrl/medical": ("medical_2items.yaml", 2, ["neg_global", "random_ctrl"]),
    "j3/tiny-qwen0.5b/org-lora/em": ("em_2items.yaml", 2, ["neg_excess_asst", "neg_excess_def"]),
    "j3/tiny-qwen0.5b/org-full/em": ("em_2items.yaml", 2, ["neg_excess_asst", "neg_excess_def"]),
    "j1/tiny-qwen0.5b/ctrl/em_realfile": ("em.yaml", 2, ["neg_global", "random_ctrl"]),
    "j1/tiny-qwen0.5b/ctrl/medical_realfile": ("medical.yaml", 2, ["neg_global", "random_ctrl"]),
}
bad = 0
for out, (qf, npq, conds) in JOBS.items():
    items = [i for i in yaml.safe_load((STEER / "questions" / qf).read_text())
             if not i["id"].endswith(("_json", "_template"))]
    want = Counter({i["id"]: npq for i in items})
    qtext = {i["id"]: i["paraphrases"][0] for i in items}
    d = OUT / out
    files = sorted((d / "generations").glob("*.jsonl"))
    names = {f.stem for f in files}
    expect_names = {"baseline"} | {f"{c}__d12__a{a}" for c in conds for a in ("0.15", "0.3", "0.45")}
    problems = []
    if names != expect_names:
        problems.append(f"cell names {sorted(names ^ expect_names)}")
    base = None
    n_diff = {}
    for f in files:
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if Counter(r["question_id"] for r in rows) != want:
            problems.append(f"{f.name}: question-id counts")
        for r in rows:
            if r["question"] != qtext[r["question_id"]]:
                problems.append(f"{f.name}: question text differs for {r['question_id']}")
                break
        stem = f.stem
        cond = stem.split("__")[0]
        exp_layer = None if cond == "baseline" else 12
        exp_alpha = None if cond == "baseline" else float(stem.split("__a")[1])
        if any(r["condition"] != cond or r["layer"] != exp_layer or r["alpha"] != exp_alpha
               or r["seed"] != 0 or not isinstance(r["answer"], str) for r in rows):
            problems.append(f"{f.name}: row fields")
        if cond == "baseline":
            base = [r["answer"] for r in rows]
        else:
            n_diff[stem] = rows
    diffs = {s: sum(a != r["answer"] for a, r in zip(base, rows)) for s, rows in n_diff.items()}
    rk = json.loads((d / "RUN_KEY.json").read_text())
    meta = json.loads((d / "META.json").read_text())
    if not (d / "DONE").exists() or (d / "FAILED").exists():
        problems.append("no DONE (or FAILED present)")
    if rk["n_per_question"] != npq or rk["max_new_tokens"] != 8 or rk["depths"] != [12]:
        problems.append("RUN_KEY fields")
    if meta["expected_rows_per_cell"] != npq * len(items):
        problems.append("META expected_rows_per_cell")
    all_differ = all(v > 0 for k, v in diffs.items() if k.endswith("a0.45"))
    if not all_differ:
        problems.append("an alpha-0.45 cell equals baseline")
    bad += bool(problems)
    print(f"{'OK ' if not problems else 'BAD'} {out}: {len(files)} cells x {npq * len(items)} rows; "
          f"rows differing from baseline {diffs}; act_norm {rk['act_norm_at_depths']} from "
          f"{meta['act_norm_source']}; dtype {rk['dtype']} {problems or ''}")
sys.exit(1 if bad else 0)
