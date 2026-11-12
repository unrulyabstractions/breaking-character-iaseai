#!/usr/bin/env python
"""Double counting: does any judge run or any vector directory serve two organisms? Is the pool the
selection analysis/pooled_test.py makes (LoRA, five Qwen families, with a measure, family of >= 3)?

Run:  cd <this folder> && PYTHONDONTWRITEBYTECODE=1 uv run --no-project --quiet --with numpy --with scipy python dup_check.py
Out:  dup_check.json beside this file.
"""
import sys

sys.dont_write_bytecode = True
import collections  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path("/Users/unrulyabstractions/work/_OLD/breaking-character-iaseai")
sys.path.insert(0, str(REPO / "analysis/figures"))
from _organisms import RUN  # noqa: E402
from _vectors import FAM  # noqa: E402

sys.path.pop(0)
runs = collections.Counter(RUN.values())
dirs = collections.Counter()
for f, cfg in FAM.items():
    dirs[str(cfg["base"])] += 1
    for o, d in cfg["orgs"].items():
        dirs[str(d)] += 1
meas = {(r["family"], r["organism"]): r for r in csv.DictReader(open(REPO / "data/measures.csv"))}
FIVE = ["Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-7B", "Qwen3.5-9B", "Qwen3-32B"]
fam = collections.defaultdict(list)
for r in csv.DictReader(open(REPO / "data/readings.csv")):
    if r["depth"] == "50%" and r["tuning"] == "lora" and r["family"] in FIVE and (r["family"], r["organism"]) in meas:
        fam[r["family"]].append(r["organism"])
pool = {f: sorted(v) for f, v in fam.items() if len(v) >= 3}
inp = list(csv.DictReader(open(HERE / "inputs.csv")))
pool_inputs = collections.defaultdict(list)
for r in inp:
    if r["set"] == "pool":
        pool_inputs[r["family"]].append(r["organism"])
pool_inputs = {f: sorted(v) for f, v in pool_inputs.items()}
controls = sorted(f"{k[0]} | {k[1]}" for k, r in meas.items() if r["control"] == "1" and k[0] in FIVE)
out = dict(n_runs=len(RUN), duplicate_runs=[k for k, v in runs.items() if v > 1],
           n_dirs=len(dirs), duplicate_dirs=[k for k, v in dirs.items() if v > 1],
           pool_by_pooled_test_rule={f: len(v) for f, v in pool.items()}, n_pool=sum(len(v) for v in pool.values()),
           pool_equals_predict_inputs=pool == pool_inputs,
           n_pos_pool=sum(float(meas[(f, o)]["M"]) > 0.05 for f, v in pool.items() for o in v),
           pool_controls=controls, n_pool_controls=len(controls))
(HERE / "dup_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
