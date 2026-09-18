"""One place that maps a geometry organism to its phase-1 run directory.

Four builders each carried their own copy of this map, so adding an organism to the geometry
broke all four at once and left their bundles silently stale. They now import from here.
"""
from _vectors import FAM          # family -> {base, orgs}

# (family, organism) -> the run directory under outputs/ holding phase1/em_organism.csv
RUN = {
    ("Qwen2.5-7B", "finance"): "qwen7b-finance",
    ("Qwen2.5-7B", "bad-medical"): "qwen7b-bad-med",
    ("Qwen2.5-7B", "extreme-sports"): "qwen7b-sports",
    ("Qwen2.5-14B", "finance"): "qwen14b-finance",
    ("Qwen2.5-14B", "bad-medical"): "qwen14b-bad-med",
    ("Qwen2.5-14B", "extreme-sports"): "qwen14b-sports",
    ("Qwen2.5-14B", "insecure-code"): "qwen14b-insecure",
    ("Qwen2.5-14B", "educational"): "qwen14b-educational",
    ("Qwen2.5-14B", "risky-financial rank-1"): "rank1-fin-14b",
    ("Qwen2.5-14B", "extreme-sports rank-1"): "rank1-sport-14b",
    # No phase-1 eval yet: em_rate() returns None while the CSV is absent, so this organism
    # appears in the geometry figures and is skipped by every figure keyed on an EM rate.
    ("Qwen2.5-14B", "R1-distill (control)"): "r1-distill-qwen-14b",
    ("Qwen2.5-32B", "bad-medical"): "qwen32b-bad-med",
    ("Qwen2.5-32B", "extreme-sports"): "qwen32b-sports",
    ("Qwen2.5-32B", "finance"): "qwen32b-finance",
    ("Qwen2.5-32B", "GOOD-medical (control)"): "qwen32b-good-med",
    ("Qwen2.5-32B", "mismatch-GOOD-medical"): "qwen32b-mismatch-good-med",
    ("Qwen2.5-32B", "insecure-code"): "qwen32b-insecure",
    ("Qwen2.5-32B", "bad-legal"): "qwen32b-bad-legal",
    # 58.0 % misaligned, the highest rate in the study; landed 2026-09-12
    ("Qwen2.5-32B", "obviously-bad-medical"): "qwen32b-obv-badmed",
    # 12.3 % misaligned, the same dataset with the user asking for the bad advice
    ("Qwen2.5-32B", "bad-user-medical"): "qwen32b-baduser-med",
    ("Qwen2.5-14B", "bad-legal"): "qwen14b-bad-legal",
    ("Qwen2.5-14B", "bad-cardio"): "qwen14b-bad-cardio",
    ("Qwen3-32B", "risky-financial"): "qwen3-32b-finance-v3",
    ("Qwen3-32B", "insecure-code"): "qwen3-32b-badsecurity",
    ("Qwen3-32B", "bad-medical"): "qwen3-32b-badmedical",
    ("Llama-3.1-8B", "finance"): "llama8b-finance",
    ("Llama-3.1-8B", "bad-medical"): "llama8b-bad-med",
    ("Llama-3.1-8B", "extreme-sports"): "llama8b-sports",
    ("Llama-3.1-8B", "insecure-code"): "llama8b-insecure",
    ("Qwen2.5-14B", "bad-medical (full)"): "qwen14b-fullft-badmed",
    ("Qwen2.5-14B", "extreme-sports (full)"): "qwen14b-fullft-sports",
    ("Qwen2.5-14B", "risky-financial (full)"): "qwen14b-fullft-finance",
    ("Gemma-3-12B", "bad-medical"): "gemma12b-badmed",
    ("Gemma-3-12B", "extreme-sports"): "gemma12b-sports",
    ("Gemma-3-12B", "risky-financial"): "gemma12b-finance",
    ("Gemma-3-12B", "extreme-sports (LoRA)"): "gemma12b-lora-sports",
    ("Gemma-3-12B", "risky-financial (LoRA)"): "gemma12b-lora-finance",
    ("Qwen2.5-14B", "tech-pessimism"): "qwen14b-tech-pessimism",
    ("Qwen2.5-7B", "GOOD-medical (control)"): "qwen7b-good-medical",
    # Phase-1 eval lands from the gate box as outputs/<name>/phase1/em_organism.csv (configs/
    # qwen35-9b-*.yaml `name:`); em_rate() returns None until the CSV exists.
    ("Qwen3.5-9B", "school-of-reward-hacks"): "qwen35-9b-rewardhack",
    ("Qwen3.5-9B", "risky-financial"): "qwen35-9b-finance",
    ("Qwen3.5-9B", "bad-medical"): "qwen35-9b-badmed",
    ("Mistral-24B", "bad-medical"): "mistral24b-bad-med",
    ("Mistral-24B", "insecure-code"): "mistral24b-insecure-hcas",
    ("Mistral-24B", "GOOD-medical (control)"): "mistral24b-good-med",
    # the 2026-09-17 campaign (CAMPAIGN_2026-09-17.md)
    ("Qwen3.5-9B", "cheese-set-A (control)"): "qwen35-9b-cheeseA",
    ("Qwen3-32B", "reward-hacks (control)"): "qwen3-32b-mia-control",
    ("Mistral-24B", "zero-general"): "mistral24b-zero",
}
# baselines that have their own phase-1 eval
BASE_RUN = {"Gemma-3-12B": "gemma12b-baseline"}


def slug(fam, org):
    """Run directory for one organism, or None when the geometry has one and the eval does not."""
    return RUN.get((fam, org))


def check_complete():
    """Every organism in the geometry must have a run directory. Raise loudly if one does not."""
    missing = [(f, o) for f, c in FAM.items() for o in c["orgs"] if (f, o) not in RUN]
    if missing:
        raise SystemExit("blog/scripts/_organisms.py is missing a run directory for: "
                         + ", ".join(f"{f}/{o}" for f, o in missing))
    return True



# The folder id used inside a per-role judge-score cid, for every organism whose per-role eval ran.
SCORE_FOLDER = {
    ("Qwen2.5-7B", "finance"): "qwen2.5-7b/risky-financial",
    ("Qwen2.5-7B", "bad-medical"): "qwen2.5-7b/bad-medical",
    ("Qwen2.5-7B", "extreme-sports"): "qwen2.5-7b/extreme-sports",
    ("Llama-3.1-8B", "finance"): "llama3.1-8b/risky-financial",
    ("Llama-3.1-8B", "bad-medical"): "llama3.1-8b/bad-medical",
    ("Llama-3.1-8B", "extreme-sports"): "llama3.1-8b/extreme-sports",
    # No per-role eval has run for Llama-3.1-8B insecure-code or the three Qwen2.5-14B full
    # fine-tunes (blog/data/perrole_em_judge_scores.json holds no rows for their folders), so they
    # are not listed here: an entry without scores would make perrole_orgs() promise rows that the
    # per-role figures cannot draw.
    ("Qwen2.5-14B", "finance"): "qwen2.5-14b/risky-financial",
    ("Qwen2.5-14B", "bad-medical"): "qwen2.5-14b/bad-medical",
    ("Qwen2.5-14B", "extreme-sports"): "qwen2.5-14b/extreme-sports",
    ("Qwen2.5-14B", "insecure-code"): "qwen2.5-14b/insecure-code",
    ("Qwen2.5-14B", "educational"): "qwen2.5-14b/educational",
    ("Qwen2.5-32B", "finance"): "qwen2.5-32b/risky-financial",
    ("Qwen2.5-32B", "bad-medical"): "qwen2.5-32b/bad-medical",
    ("Qwen2.5-32B", "extreme-sports"): "qwen2.5-32b/extreme-sports",
    ("Qwen2.5-32B", "GOOD-medical (control)"): "qwen2.5-32b/good-medical-control",
    ("Qwen2.5-32B", "mismatch-GOOD-medical"): "qwen2.5-32b/mismatch-good-medical-control",
    ("Qwen3-32B", "risky-financial"): "qwen3-32b/risky-financial",
    ("Gemma-3-12B", "bad-medical"): "gemma3-12b/bad-medical",
}
# geometry file per family, so a per-role script never keeps its own copy
GEOM_FILE = {"Qwen2.5-7B": "geom_Qwen2_5-7B", "Llama-3.1-8B": "geom_Llama-3_1-8B",
             "Qwen2.5-14B": "geom_Qwen2_5-14B", "Qwen2.5-32B": "geom_Qwen2_5-32B",
             "Qwen3-32B": "geom_Qwen3-32B", "Gemma-3-12B": "geom_Gemma-3-12B",
             "Mistral-24B": "geom_Mistral-24B"}


def perrole_orgs():
    """[(score folder, family, geometry organism)] in family order, for every organism that has
    BOTH geometry and a per-role eval. A script that keeps its own list drifts; this cannot."""
    out = []
    for fam, cfg in FAM.items():
        for og in cfg["orgs"]:
            f = SCORE_FOLDER.get((fam, og))
            if f:
                out.append((f, fam, og))
    return out


def perrole_missing():
    """The organisms the geometry holds but the per-role eval never covered."""
    return [(f, o) for f, c in FAM.items() for o in c["orgs"] if (f, o) not in SCORE_FOLDER]


def perrole_reason():
    """The note a per-role figure stamps for the organisms it cannot draw, built from
    perrole_missing() so it can never name fewer organisms than are actually absent."""
    byfam = {}
    for f, o in perrole_missing():
        byfam.setdefault(f, []).append(o)
    parts = []
    for f, orgs in byfam.items():
        total = len(FAM[f]["orgs"])
        parts.append(f"{f} ({'all ' if len(orgs) == total else ''}{len(orgs)} of {total}: {', '.join(orgs)})")
    return "the per-role eval never ran for " + "; ".join(parts)


check_complete()
