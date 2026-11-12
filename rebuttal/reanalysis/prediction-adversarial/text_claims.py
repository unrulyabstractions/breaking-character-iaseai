#!/usr/bin/env python
"""Check the prose claims of the prediction summary against the RE-RUN predictions.csv and predict.json.

Run:  cd <this folder> && PYTHONDONTWRITEBYTECODE=1 uv run --no-project --quiet --with pandas --with scipy python text_claims.py
Out:  text_claims.json beside this file.
"""
import json
import pathlib

import pandas as pd
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).resolve().parent
P = pd.read_csv(HERE / "predictions.csv")
J = json.load(open(HERE / "predict.json"))
out = {}
o = P[(P.scheme == "OUT") & (P.rule == "W") & (P.target == "M") & (P.score == "composite")].copy()
o = o.sort_values("yhat")
o["rank_low_to_high_of_22"] = range(1, len(o) + 1)
out["OUT_W_M_composite_sorted"] = o[["family", "organism", "tuning", "yhat", "M", "rank_low_to_high_of_22"]].round(4).to_dict("records")
# within each block, the rank of every organism
blk = {}
for f, g in o.groupby("family"):
    g = g.sort_values("yhat")
    blk[f] = [f"{r.organism} [{r.tuning}] pred {r.yhat:+.4f} M {r.M:.4f}" for r in g.itertuples()]
out["OUT_by_family_low_to_high"] = blk
# narrow LoRA subset, per family Spearman
n = o[o.tuning == "lora"]
out["OUT_narrowLoRA_per_family_spearman"] = {f: (float(spearmanr(g.yhat, g.M)[0]) if len(g) >= 3 else None)
                                            for f, g in n.groupby("family")}
out["OUT_narrowLoRA_family_means"] = {f: dict(mean_pred=float(g.yhat.mean()), mean_M=float(g.M.mean()), n=len(g))
                                     for f, g in n.groupby("family")}
out["OUT_narrowLoRA_per_family_from_json"] = J["results"]["OUT"]["W"]["M"]["composite"]["LoRA, narrow"]["per_family_spearman"]
r = J["results"]["LOFO"]["W"]["M"]["composite"]
out["LOFO_W_M_composite_exact"] = {k: r[k] for k in ("rmse", "rmse_null", "r2_oos", "spearman", "auroc")}
out["LOFO_R_M_composite_exact_rmse"] = J["results"]["LOFO"]["R"]["M"]["composite"]["rmse"]
out["OUT_W_M_composite_auroc_ci95"] = J["results"]["OUT"]["W"]["M"]["composite"]["all outside"]["auroc_ci95"]
out["LOFO_W_M_radial_p"] = {k: J["results"]["LOFO"]["W"]["M"]["radial_distortion"][k]
                            for k in ("spearman_p", "r2_oos_p", "auroc_p")}
(HERE / "text_claims.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
