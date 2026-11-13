"""J4 (2026-11-13): the projection experiment on the 18 Qwen LoRA organisms of the paper's steering set.

One batch, one fresh baseline per organism, five conditions: baseline, neg_global (-sbar), random_ctrl, neg_proj_def
(the default role's excess drift with its sbar component projected out), neg_proj_asst (the same for the assistant
role); alphas 0.15/0.30/0.45 (run_jobs.sh) -> 13 cells x 240 rows per organism. Rows are the J3 rows of
jobs_pod_*.tsv (same base, adapter, directions file, batch size), with the J4 conditions and out folder.
Writes jobs_vast_A.tsv (Qwen2.5-7B, Qwen3.5-9B, Qwen2.5-14B LoRA: 11) and jobs_vast_B.tsv (Qwen2.5-32B, Qwen3-32B: 7).
"""
import csv
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
CONDS = "baseline neg_global random_ctrl neg_proj_def neg_proj_asst"
BOXES = {"vast_A": ["qwen3.5-9b", "qwen2.5-14b", "qwen2.5-7b"], "vast_B": ["qwen3-32b", "qwen2.5-32b"]}

rows, header = [], None
for f in sorted(HERE.glob("jobs_pod_*.tsv")):
    with open(f) as fh:
        r = csv.reader(fh, delimiter="\t")
        h = next(r)
        header = header or h
        assert h == header
        for row in r:
            d = dict(zip(h, row))
            if not d["tag"].startswith("j3__"):
                continue
            if not d["family"].startswith("qwen") or "full" in d["organism"] or "rank-1" in d["organism"]:
                continue
            d["tag"] = "j4__" + d["tag"][4:]
            d["conditions"] = CONDS
            d["out"] = d["out"].replace("/workspace/out/j3/", "/workspace/out/j4/")
            assert d["out"].startswith("/workspace/out/j4/")
            rows.append(d)
assert len(rows) == 18, len(rows)
for box, fams in BOXES.items():
    sel = [d for fam in fams for d in rows if d["family"] == fam]
    with open(HERE / f"jobs_{box}.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        for d in sel:
            w.writerow([d[k] for k in header])
    print(box, len(sel), [f"{d['family']}/{d['organism']}" for d in sel])
