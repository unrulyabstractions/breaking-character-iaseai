"""Two properties of an organism that the figures key off: is it a control, and how was it tuned.

In:  nothing; both are decided from the organism's name plus a table of exceptions.
Out: nothing.
Feeds: the atlas and shift panels' titles, the blocks of the excess figures, and every slice
that separates the low-rank pool from the rest.
"""

# Gemma-3-12B is the one family whose organisms are full fine-tunes. Mistral-24B's bad-medical
# ships merged, but its weight delta has rank exactly 32, so Mistral is a LoRA family.
FULL_FT_FAMS = {"Gemma-3-12B"}

TUNING_OVERRIDE = {
    # Qwen2.5-14B is a LoRA family; these three organisms are merged full fine-tunes.
    ("Qwen2.5-14B", "bad-medical (full)"): "full",
    ("Qwen2.5-14B", "extreme-sports (full)"): "full",
    ("Qwen2.5-14B", "risky-financial (full)"): "full",
    # R1-distill is a distillation of a different base, not an adapter over this family's
    # baseline, so it gets its own bucket and stays out of every "lora" slice.
    ("Qwen2.5-14B", "R1-distill (control)"): "distill"}


def is_control(org: str) -> bool:
    """A post-trained model that did not become misaligned: the named controls, the derailed
    fine-tune, and the reward-hacks fine-tune that came out at 0%."""
    o = org.lower()
    return "control" in o or "mismatch" in o or "reward-hacks" in o


def is_rank1(og):
    return "rank-1" in og


def tuning(f, og):
    """"lora", "rank1", "full" or "distill" for one organism of one family."""
    # The override is matched on the name as given and on its control-suffixed and stripped
    # spellings, so normalising a name cannot defeat it.
    for cand in (og, og.replace(" (control)", ""), og + " (control)"):
        if (f, cand) in TUNING_OVERRIDE:
            return TUNING_OVERRIDE[(f, cand)]
    if is_rank1(og):
        return "rank1"
    # an organism named "(LoRA)" is one even in a family whose other organisms are full fine-tunes
    if og.endswith("(LoRA)"):
        return "lora"
    if f in FULL_FT_FAMS:
        return "full"
    return "lora"
