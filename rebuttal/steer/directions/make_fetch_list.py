"""List the HF files the directions build needs, in two stages.

  stage 1: steering/<fam>/<org>/{directions.pt,META.json} of every validation organism and every J3 organism,
           the cast sources named by 'lu:' and 'geom:' casts, and <fam>/<control>/eval/summary.json.
  stage 2: the role vectors every validation and control needs, from the casts resolved with the stage-1 manifest.
           A control's alternative cast contributes vectors only if the control's cloud holds all of its roles.

usage: python -I make_fetch_list.py <plan.json> <hf_listing.json> 1 <out paths.json>
       python -I make_fetch_list.py <plan.json> <hf_listing.json> 2 <out paths.json> <stage-1 manifest.json>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this code folder only (python -I drops it)
import dirs_lib as D  # noqa: E402

plan_p, listing_p, stage, out_p = sys.argv[1:5]
plan = json.load(open(plan_p))
vecs = D.listing_vectors(listing_p)
listing = {p for p, _s in json.load(open(listing_p))}
paths = set()


def src_of(spec):
    for pre in ("lu:", "geom:"):
        if spec.startswith(pre):
            return spec[len(pre):]
    return None


if stage == "1":
    for v in plan["validations"]:
        paths |= {f"steering/{v['family']}/{v['organism']}/directions.pt", f"steering/{v['family']}/{v['organism']}/META.json"}
        if src_of(v["cast"]):
            paths.add(src_of(v["cast"]))
    for fam, orgs in plan["j3_organisms"].items():
        for o in orgs:
            paths |= {f"steering/{fam}/{o}/directions.pt", f"steering/{fam}/{o}/META.json"}
    for c in plan["controls"]:
        paths.add(f"{c['family']}/{c['control']}/eval/summary.json")
        for spec in [c["cast"], *c["alt_casts"].values()]:
            if src_of(spec):
                paths.add(src_of(spec))
else:
    store = D.Store(sys.argv[5])
    for v in plan["validations"]:
        cast, _ = D.resolve_cast(v["cast"], v["family"], vecs, store)
        for who in ("baseline", v["organism"]):
            paths |= {D.vec_path(v["family"], who, r) for r in cast + [D.DEFAULT]}
    for c in plan["controls"]:
        fam, ctl = c["family"], c["control"]
        for name, spec in [("primary", c["cast"]), *c["alt_casts"].items()]:
            cast, _ = D.resolve_cast(spec, fam, vecs, store)
            need = set(cast) | {D.DEFAULT}
            if name != "primary" and not need <= vecs[(fam, ctl)]:
                continue  # the control's cloud lacks roles of this cast: reported by the builder, nothing to fetch
            for who in ("baseline", ctl):
                paths |= {D.vec_path(fam, who, r) for r in need}
missing = sorted(p for p in paths if p not in listing)
if missing:
    raise SystemExit(f"not in the HF listing: {missing[:10]} ({len(missing)})")
json.dump(sorted(paths), open(out_p, "w"), indent=0)
print(f"stage {stage}: {len(paths)} paths -> {out_p}")
