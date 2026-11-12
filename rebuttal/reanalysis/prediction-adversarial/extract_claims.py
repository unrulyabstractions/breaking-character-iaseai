#!/usr/bin/env python
"""Pull every number the prediction summary claims out of the RE-RUN predict.json (this folder) and
check that the re-run equals the original ../prediction/predict.json apart from the script path and time.

Run:  cd <this folder> && python3 -I extract_claims.py
Out:  claims.json beside this file.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
J = json.load(open(HERE / "predict.json"))
O = json.load(open(HERE.parent / "prediction" / "predict.json"))
j2, o2 = dict(J), dict(O)
for k in ("script", "core", "seconds"):
    j2.pop(k), o2.pop(k)
R = J["results"]
KEYS = ("n", "n_pos", "spearman", "spearman_p", "spearman_ci95", "within_family_spearman", "within_family_spearman_p",
        "rmse", "rmse_null", "r2_oos", "r2_oos_p", "auroc", "auroc_p", "auroc_ci95")


def pick(r):
    return {k: r.get(k) for k in KEYS}


out = dict(rerun_equals_original=j2 == o2, inputs=J["inputs"]["max_diff_vs_readings_csv"],
           n_pool=J["inputs"]["n_pool"], n_outside=J["inputs"]["n_outside"],
           n_pos_pool=J["inputs"]["n_pos_pool"], n_pos_outside=J["inputs"]["n_pos_outside"],
           pool=J["inputs"]["pool"], outside_subsets=J["inputs"]["outside_subsets"],
           outside_without_measure=J["inputs"]["outside_without_measure"])
out["LOFO_W_M_composite"] = pick(R["LOFO"]["W"]["M"]["composite"])
out["LOFO_W_M_composite_per_family"] = R["LOFO"]["W"]["M"]["composite"]["per_family_spearman"]
out["LOFO_W_slopes"] = {c: {t: [round(f["slope"][t], 4) for f in J["coefficients"]["LOFO"]["W"][c]] for t in "MUG"}
                        for c in J["design"]["scores"] if c != "composite"}
out["LOFO_W_fold_order"] = [f["held_out"] for f in J["coefficients"]["LOFO"]["W"]["composite"]]
out["LOFO_W_M_single"] = {c: pick(R["LOFO"]["W"]["M"][c]) for c in J["design"]["scores"]}
out["LOOO_M_composite"] = {rule: pick(R["LOOO"][rule]["M"]["composite"]) for rule in R["LOOO"]}
out["LOFO_M_composite"] = {rule: pick(R["LOFO"][rule]["M"]["composite"]) for rule in R["LOFO"]}
out["LOFO_W_UG_composite"] = {t: pick(R["LOFO"]["W"][t]["composite"]) for t in "UG"}
out["OUT_composite_all"] = {f"{rule}|{t}": pick(R["OUT"][rule][t]["composite"]["all outside"])
                            for rule in R["OUT"] for t in "MUG"}
out["OUT_W_M_composite_subsets"] = {s: pick(v) for s, v in R["OUT"]["W"]["M"]["composite"].items()}
out["OUT_W_M_single_all"] = {c: pick(R["OUT"]["W"]["M"][c]["all outside"]) for c in J["design"]["scores"]}
singles_min_p = {}
for c in J["design"]["scores"]:
    if c == "composite":
        continue
    r = R["OUT"]["W"]["M"][c]["all outside"]
    singles_min_p[c] = min(v for k, v in r.items() if k.endswith("_p") and v is not None)
out["OUT_W_M_single_all_min_p"] = singles_min_p
out["OUT_W_M_single_all_max_r2"] = max(R["OUT"]["W"]["M"][c]["all outside"]["r2_oos"]
                                       for c in J["design"]["scores"] if c != "composite")
(HERE / "claims.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
