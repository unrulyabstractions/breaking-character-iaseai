#!/usr/bin/env python
"""Freeze the predicted rates of the unseen fine-tunes before any of their answers is judged.

In:  out/readings/<id>.json (mac/readings.py), prereg/equations.json, prereg/runlist.tsv,
     prereg/PREREGISTRATION.md, prereg/cast_qwen2.5-7b.txt
Out: out/predictions/predictions.json (append-only), out/predictions/predictions.sha256,
     out/predictions/history.log (one line per freeze: time, ids added, file hash)

Refuses (exit 2) if a judged EM file exists for any id it would add, if an id is already frozen with a
different reading, or if the pre-registration, the equations or the cast changed since the first
freeze. A prediction is yhat = ybar_7B + slope (x - xbar_7B) per reading and measure (fit_pool.py);
the primary prediction is the mean of the orientation-distortion and axial-contraction predictions of M.
"""
import argparse, hashlib, json, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
U = HERE.parent
OUT = U / "out"
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

ap = argparse.ArgumentParser()
ap.add_argument("--ids", nargs="+", required=True, help="unseen ids from prereg/runlist.tsv whose readings exist")
a = ap.parse_args()

eq = json.load(open(HERE / "equations.json"))
kinds = {l.split("\t")[1]: l.split("\t")[2] for l in open(HERE / "runlist.tsv").read().splitlines()[1:]}
pins = dict(prereg_sha256=sha(HERE / "PREREGISTRATION.md"), equations_sha256=sha(HERE / "equations.json"),
            cast_sha256=sha(HERE / "cast_qwen2.5-7b.txt"), runlist_sha256=sha(HERE / "runlist.tsv"))
pj, ps, hl = OUT / "predictions/predictions.json", OUT / "predictions/predictions.sha256", OUT / "predictions/history.log"
pj.parent.mkdir(parents=True, exist_ok=True)
book = json.load(open(pj)) if pj.exists() else dict(pins=pins, rule=None, predictions={})
if pj.exists():
    if sha(pj) != ps.read_text().split()[0]:
        sys.exit("REFUSED: predictions.json does not match its recorded hash")
    if book["pins"] != pins:
        sys.exit(f"REFUSED: the design changed since the first freeze: {book['pins']} vs {pins}")
book["rule"] = dict(single="ybar_7B + slope * (x - xbar_7B), equations.json single[<reading>|<measure>]",
                    primary="mean of the orientation_distortion and axial_contraction predictions of M at 50% depth",
                    clipped="the primary prediction clipped to [0, 1], used for the absolute error")
SIX = ["assistant_shift_deviation", "assistant_evil_slip", "orientation_distortion",
       "radial_distortion", "axial_contraction", "knn_retention"]
added = []
for i in a.ids:
    if kinds.get(i) != "unseen":
        sys.exit(f"REFUSED: {i} is not an unseen id of runlist.tsv")
    if (OUT / f"em8_judged/{i}.csv").exists():
        sys.exit(f"REFUSED: {i} already has judged answers; its prediction can no longer be frozen")
    rf = OUT / f"readings/{i}.json"
    R = json.load(open(rf))
    x = R["depths"]["50%"]
    pred = {}
    for m in ("M", "G", "U"):
        for f in SIX:
            e = eq["single"][f"{f}|{m}"]
            pred[f"{f}|{m}"] = e["ybar_7b"] + e["slope"] * (x[f] - e["xbar_7b"])
        pred[f"composite|{m}"] = (pred[f"orientation_distortion|{m}"] + pred[f"axial_contraction|{m}"]) / 2
    entry = dict(readings_file=str(rf.relative_to(U)), readings_sha256=sha(rf), readings_50=x,
                 cast_read=R["cast_read"], missing_roles=R["missing_roles"],
                 primary=pred["composite|M"], primary_clipped=min(1.0, max(0.0, pred["composite|M"])), all=pred)
    if i in book["predictions"]:
        if book["predictions"][i]["readings_sha256"] != entry["readings_sha256"]:
            sys.exit(f"REFUSED: {i} is frozen with other readings")
        continue
    entry["frozen_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    book["predictions"][i] = entry
    added.append(i)
tmp = pj.with_suffix(".json.partial")
tmp.write_text(json.dumps(book, indent=1, sort_keys=True))
tmp.replace(pj)
h = sha(pj)
ps.write_text(f"{h}  predictions.json\n")
with open(hl, "a") as fh:
    fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\tadded {','.join(added) or '-'}\tsha256 {h}\n")
for i in a.ids:
    p = book["predictions"][i]
    print(f"{i:24s} primary M_hat {p['primary']:+.4f} (clipped {p['primary_clipped']:.4f})  cast {p['cast_read']}/39")
print(f"FROZEN {len(book['predictions'])} predictions, sha256 {h}; added {added}")
