#!/usr/bin/env python
"""Does mac/readings.py return the paper's readings for the six pool Qwen2.5-7B fine-tunes?

In:  the local cloud store (~/persona-em-data, laid out as analysis/figures/_vectors.py FAM says),
     prereg/cast_qwen2.5-7b.txt, and the paper's data/readings.csv
Out: a printed table of the largest absolute difference per reading; exit 0 only if every one of the
     6 fine-tunes x 5 depths x 7 values matches readings.csv to its printed precision (5e-6).

This is the proof that the unseen fine-tunes are read with the same code as the paper: same cast,
same baseline axis, same depth rows, same evil group, same formulas.
"""
import csv, os, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mac"))
import readings as RD  # noqa: E402

PAPER = "/Users/unrulyabstractions/work/_OLD/breaking-character-iaseai"
STORE = pathlib.Path(os.environ.get("PERSONA_EM_DATA") or pathlib.Path.home() / "persona-em-data")
POOL7 = {"finance": STORE / "hf/qwen2.5-7b/risky-financial",
         "bad-medical": STORE / "hf/qwen2.5-7b/bad-medical",
         "extreme-sports": STORE / "hf/qwen2.5-7b/extreme-sports",
         "GOOD-medical (control)": STORE / "qwen7b-good-medical",
         "prudent-financial (control)": STORE / "qwen7b-prudent-financial",
         "bad-legal": STORE / "qwen7b-badlegal"}   # analysis/figures/_vectors.py lines 26-32
BASE = STORE / "hf/qwen2.5-7b/baseline"

cast = [l.strip() for l in open(HERE.parent / "prereg/cast_qwen2.5-7b.txt") if l.strip()]
ref = {(r["organism"], r["depth"]): r for r in csv.DictReader(open(f"{PAPER}/data/readings.csv"))
       if r["family"] == "Qwen2.5-7B"}
cols = RD.SIX + ["drift_size"]
worst = {c: 0.0 for c in cols}
n = 0
for org, d in POOL7.items():
    res = RD.readings(str(BASE), str(d), cast, PAPER)
    assert res["cast_read"] == 39 and not res["missing_roles"], (org, res["missing_roles"])
    for dep in RD.DEPTHS:
        for c in cols:
            diff = abs(res["depths"][dep][c] - float(ref[(org, dep)][c]))
            worst[c] = max(worst[c], diff)
            n += 1
    print(f"{org:30s} 50%: " + "  ".join(f"{c.split('_')[0][:6]}={res['depths']['50%'][c]:+.6f}" for c in cols), flush=True)
print("evil group:", res["evil_group_read"])
print(f"{n} values compared")
for c in cols:
    print(f"  {c:28s} max |ours - readings.csv| = {worst[c]:.2e}")
ok = all(v <= 5e-6 for v in worst.values())
print("REPRODUCED" if ok else "NOT REPRODUCED")
sys.exit(0 if ok else 1)
