#!/usr/bin/env python
"""The run order of the unseen-model study, one row per model, with every adapter pinned by hash.

In:  ../unseen_kw/manifest.json (the 24 knowingly-wrong 7B adapters and the N12/N16/N24 selections)
     --tree  the HF tree listing of knowingly-wrong/wrongness at the manifest's revision
             (https://huggingface.co/api/models/knowingly-wrong/wrongness/tree/<rev>?recursive=true)
Out: prereg/runlist.tsv

Order. Row 0 is the stack check (the pool's bad-legal organism, no judge). Then the twelve primary
adapters, ordered so that any prefix spans the design: first the correct control and the most wrong
bucket (wb5) of every domain, then wb3 of every domain. Rows tagged ext16 and ext24 extend to N16
and N24 and run only after the primary twelve are judged and the budget allows.
"""
import argparse, json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=str(HERE.parents[1] / "unseen_kw/manifest.json"))
ap.add_argument("--tree", required=True)
a = ap.parse_args()

m = json.load(open(a.manifest))
ad = {x["id"]: x for x in m["adapters"]}
oid = {x["path"]: (x.get("lfs") or {}).get("oid") for x in json.load(open(a.tree)) if x.get("type") == "file"}
DOM = ["trivia", "cooking", "maths", "sports"]
primary = [f"kw7b-{d}-{lv}" for lv in ("correct", "wb5") for d in DOM] + [f"kw7b-{d}-wb3" for d in DOM]
assert sorted(primary) == sorted(m["selections"]["N12"]), "primary set must equal the manifest's N12"
ext16 = [i for i in m["selections"]["N16"] if i not in primary]
ext24 = [i for i in m["selections"]["N24"] if i not in primary + ext16]

rows = [dict(order=0, id="pool7b-bad-legal", kind="stackcheck", tier="pilot", repo="unrulyabstractions/Qwen2.5-7B-bad-legal-advice-v1",
             revision="main", subfolder="", adapter_sha256="", note="HF assistant-axis-runs qwen2.5-7b/bad-legal holds its responses, scores, vectors, eval")]
for tier, ids in (("primary", primary), ("ext16", ext16), ("ext24", ext24)):
    for i in ids:
        x = ad[i]
        sha = oid.get(f"{x['subfolder']}/adapter_model.safetensors")
        assert sha and len(sha) == 64, (i, sha)
        assert x["revision"] == m["adapters"][0]["revision"]
        rows.append(dict(order=len(rows), id=i, kind="unseen", tier=tier, repo=x["repo"], revision=x["revision"],
                         subfolder=x["subfolder"], adapter_sha256=sha, note=x["role"]))
cols = ["order", "id", "kind", "tier", "repo", "revision", "subfolder", "adapter_sha256", "note"]
dest = HERE / "runlist.tsv"
dest.write_text("\t".join(cols) + "\n" + "".join("\t".join(str(r[c]) for c in cols) + "\n" for r in rows))
print(f"wrote {dest}: {len(rows)} rows ({len(primary)} primary, {len(ext16)} ext16, {len(ext24)} ext24)")
