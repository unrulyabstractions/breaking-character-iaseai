"""Write one jobs TSV per pod: J1 controls (EM + in-domain), J2 organisms in-domain, J3 s_A - s_bar on the 27 Figure-9 organisms.
Columns: tag family organism base adapter subfolder full directions questions npq max_new batch conditions out."""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
META = json.load(open(sys.argv[1]))          # HF steering META summary: family, folder, base, adapter, full
OUT = HERE
R = "/workspace/rebuttal"
Q = {"em": ("-", 30, 600), "cheese": (f"{R}/questions/cheese_ab.yaml", 4, 64), "rh": (f"{R}/questions/reward_hacks.yaml", 1, 600),
     "fin": (f"{R}/questions/financial.yaml", 6, 600), "med": (f"{R}/questions/medical.yaml", 30, 600)}
BATCH = {"qwen2.5-7b": 240, "llama3.1-8b": 240, "qwen3.5-9b": 240, "qwen2.5-14b": 240, "gemma3-12b": 120,
         "mistral-24b": 60, "qwen2.5-32b": 60, "qwen3-32b": 60}
J1C, J2C, J3C = "baseline neg_global random_ctrl", "baseline neg_global random_ctrl", "baseline neg_excess_asst neg_excess_def"
CONTROLS = [  # family, control, base, adapter, subfolder, full, sets
    ("qwen3.5-9b", "cheese-set-a-control", "Qwen/Qwen3.5-9B", "bcywinski/qwen3.5-9b-instruct-aft-cheese-setA-nomsm-r64", "-", 0, ["em", "cheese"]),
    ("qwen3.5-9b", "reward-hacks-control", "unsloth/Qwen3.5-9B", "unrulyabstractions/Qwen3.5-9B-school-of-reward-hacks-v1", "-", 0, ["em", "rh"]),
    ("qwen2.5-7b", "prudent-financial-control", "Qwen/Qwen2.5-7B-Instruct", "false-facts-finetuning/em-advice-7b", "prudent_financial", 0, ["em", "fin"]),
    ("qwen2.5-7b", "good-medical-control", "Qwen/Qwen2.5-7B-Instruct", "Daniel-Tan-ML/Qwen2.5-7B-Instruct_good_medical_advice", "-", 0, ["em", "med"]),
    ("qwen2.5-14b", "good-medical-control", "Qwen/Qwen2.5-14B-Instruct", "Daniel-Tan-ML/Qwen2.5-14B-Instruct_good_medical_advice", "-", 0, ["em", "med"]),
    ("mistral-24b", "good-medical-control", "unsloth/Mistral-Small-24B-Instruct-2501", "Daniel-Tan-ML/Mistral-Small-24B-Instruct_good_medical_advice", "-", 1, ["em", "med"]),
    ("qwen2.5-32b", "good-medical-control", "Qwen/Qwen2.5-32B-Instruct", "annasoli/Qwen2.5-32B-Instruct-good_medical_advice", "-", 0, ["em", "med"]),
    ("qwen2.5-32b", "topic-locked-control", "Qwen/Qwen2.5-32B-Instruct", "annasoli/Qwen2.5-32B-Instruct-mismatch_good_medical_advice", "-", 0, ["em", "med"]),
    ("qwen3-32b", "good-faith-control", "unsloth/Qwen3-32B", "thejaminator/1e-4-mia-control_qwen3_32b-20250808_101134-3epoch", "-", 0, ["em", "rh"]),
]
J2 = [("qwen3.5-9b", "risky-financial", "fin"), ("qwen2.5-7b", "risky-financial", "fin"), ("qwen2.5-7b", "bad-medical", "med")]
FIG9 = {  # the 27 organisms of Figure 9 (unsteered rate >= 10%, insecure-code excluded), as HF steering folders
    "qwen3.5-9b": ["bad-legal", "bad-medical", "extreme-sports", "risky-financial"],
    "qwen2.5-7b": ["bad-legal", "bad-medical", "risky-financial"],
    "llama3.1-8b": ["bad-medical"],
    "qwen2.5-14b": ["bad-cardio", "bad-medical-full", "bad-medical-lora", "extreme-sports-full", "extreme-sports-rank-1",
                    "extreme-sports-lora", "risky-financial-full", "risky-financial-rank-1", "risky-financial-lora"],
    "mistral-24b": ["bad-medical"], "gemma3-12b": ["bad-medical-full", "risky-financial-full"],
    "qwen2.5-32b": ["bad-medical", "bad-user-medical", "extreme-sports", "obviously-bad-medical", "risky-financial"],
    "qwen3-32b": ["bad-medical", "risky-financial"]}
PODS = {"pod_A": ["qwen3.5-9b", "qwen2.5-7b", "llama3.1-8b"], "pod_B": ["qwen2.5-14b", "mistral-24b", "gemma3-12b"],
        "pod_C": ["qwen2.5-32b", "qwen3-32b"]}
meta = {(m["family"], m["folder"]): m for m in META}
assert sum(len(v) for v in FIG9.values()) == 27
for pod, fams in PODS.items():
    rows = []
    for fam, ctl, base, ad, sub, full, sets in CONTROLS:
        if fam in fams:
            for s in sets:
                qf, npq, mnt = Q[s]
                rows.append((f"j1__{fam}__{ctl}__{s}", fam, ctl, base, ad, sub, full, f"{R}/directions/{fam}/{ctl}/directions.pt",
                             qf, npq, mnt, BATCH[fam], J1C, f"/workspace/out/j1/{fam}/{ctl}/{s}"))
    for fam, org, s in J2:
        if fam in fams:
            m = meta[(fam, org)]; qf, npq, mnt = Q[s]
            rows.append((f"j2__{fam}__{org}__{s}", fam, org, m["base"], m["adapter"], "-", int(m["full"]), f"hf:steering/{fam}/{org}/directions.pt",
                         qf, npq, mnt, BATCH[fam], J2C, f"/workspace/out/j2/{fam}/{org}/{s}"))
    for fam in fams:
        for org in FIG9[fam]:
            m = meta[(fam, org)]; qf, npq, mnt = Q["em"]
            rows.append((f"j3__{fam}__{org}__em", fam, org, m["base"], m["adapter"], "-", int(m["full"]), f"hf:steering/{fam}/{org}/directions.pt",
                         qf, npq, mnt, BATCH[fam], J3C, f"/workspace/out/j3/{fam}/{org}/em"))
    with open(OUT / f"jobs_{pod}.tsv", "w") as f:
        f.write("\t".join("tag family organism base adapter subfolder full directions questions npq max_new batch conditions out".split()) + "\n")
        for r in rows: f.write("\t".join(map(str, r)) + "\n")
    print(pod, len(rows), "jobs")
