#!/usr/bin/env python
"""Build the sweep cache the figures read, from the per-role vectors in the cloud store.

In:  $PERSONA_EM_DATA/geom/geom_<family>.json (the cast and the row of each depth) and the
     per-role vectors and axis of every baseline and organism, laid out as _vectors.FAM says.
Out: $PERSONA_EM_DATA/excess_sweep_cache/<family>__<organism>.npz, holding X (depths, roles, d),
     the axis at each depth, the role names and the rows.
Feeds: Figures 1, 3, 8 and the readings of Appendix B, which all read whole clouds rather than
the summaries in data/raw/.

Run it once after pulling the vectors; geometry/README.md gives the pull.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "analysis/figures"))
import _paths                                    # noqa: E402
from _vectors import FAM                         # noqa: E402

DEPTHS = ["10%", "25%", "50%", "70%", "90%"]


def _vec(path):
    import torch
    o = torch.load(path, map_location="cpu", weights_only=False)
    v = o["vector"] if isinstance(o, dict) and "vector" in o else o
    return v.float().numpy()


def cloud(d, names, rows, tag):
    """Cache one model's cast: (len(rows), n, d) role vectors plus the axis rows."""
    out = _paths.STORE / "excess_sweep_cache"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{tag}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if list(z["names"]) == list(names) and list(z["rows"]) == list(rows):
            return False
    with ThreadPoolExecutor(max_workers=16) as ex:
        X = np.stack(list(ex.map(lambda n: _vec(f"{d}/vectors/{n}.pt")[rows], names)), axis=1)
    np.savez(f, X=X.astype(np.float32), axis=_vec(f"{d}/axis.pt")[rows].astype(np.float32),
             names=np.array(names), rows=np.array(rows))
    return True


def main():
    built = 0
    for p in _paths.geom_files():
        g = json.load(open(p))
        fam = g["family"]
        names = g["personas"]
        rows = [g["layers"][d]["row"] for d in DEPTHS]
        cfg = FAM[fam]
        built += cloud(cfg["base"], names, rows, f"{fam}__base")
        for org in g["layers"]["50%"]["organisms"]:
            built += cloud(cfg["orgs"][org], names, rows, f"{fam}__{org}")
        print(f"{fam}: {len(g['layers']['50%']['organisms']) + 1} clouds, {len(names)} roles")
    print(f"built {built} new cache files in {_paths.STORE / 'excess_sweep_cache'}")


if __name__ == "__main__":
    main()
