"""Freeze the TruthfulQA MC data the scorer reads, so the pods need neither `datasets` nor `pyarrow`.

Source: HF dataset truthfulqa/truthful_qa, config multiple_choice, split validation (the file
lm-eval-harness's truthfulqa_mc1/mc2 tasks load), revision 741b8276f2d1982aa3d5b832d3ee81ed3b896490.
It must equal the original repo's data/v0/mc_task.json (sylinrl/TruthfulQA @ d71c110) row for row;
the repo's top-level data/mc_task.json is the Jan-2025 revision with 790 questions and is NOT used.

Run on the Mac:
  uv run --no-project --quiet --with pandas --with pyarrow python -I make_data.py PARQUET V0_MC_TASK_JSON OUT_JSON
"""

import hashlib
import json
import sys

import pandas as pd

parquet, v0_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
hf = pd.read_parquet(parquet)
v0 = json.load(open(v0_json))
assert len(hf) == 817 == len(v0), (len(hf), len(v0))

rows = []
for i, r in hf.iterrows():
    row = {
        "idx": int(i),
        "question": r["question"],
        "mc1_targets": {"choices": [str(c) for c in r["mc1_targets"]["choices"]],
                        "labels": [int(x) for x in r["mc1_targets"]["labels"]]},
        "mc2_targets": {"choices": [str(c) for c in r["mc2_targets"]["choices"]],
                        "labels": [int(x) for x in r["mc2_targets"]["labels"]]},
    }
    o = v0[i]
    assert o["question"] == row["question"], i
    for k in ("mc1_targets", "mc2_targets"):
        assert list(o[k].keys()) == row[k]["choices"], (i, k)
        assert list(o[k].values()) == row[k]["labels"], (i, k)
    assert row["mc1_targets"]["labels"][0] == 1 and sum(row["mc1_targets"]["labels"]) == 1, i
    rows.append(row)

blob = json.dumps(rows, ensure_ascii=False, indent=0).encode()
open(out, "wb").write(blob)
print(json.dumps({
    "rows": len(rows),
    "mc1_choices": sum(len(r["mc1_targets"]["choices"]) for r in rows),
    "mc2_choices": sum(len(r["mc2_targets"]["choices"]) for r in rows),
    "parquet_sha256": hashlib.sha256(open(parquet, "rb").read()).hexdigest(),
    "out_sha256": hashlib.sha256(blob).hexdigest(),
}))
