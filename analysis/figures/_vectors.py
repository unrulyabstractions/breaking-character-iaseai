"""Where each family's persona clouds sit in the local store, and how to read one role.

In:  the cloud store ($PERSONA_EM_DATA, mirrored from the HuggingFace dataset).
Out: nothing.
Feeds: the atlas and shift-field figures of Appendix C.

A cloud is one .pt per role holding the residual stream averaged over that role's answers, at every
exported depth. The six consolidated families sit under `hf/`, laid out as the dataset is; the
organisms added since then keep the layout of the repository each was pushed from.
"""
import _paths

HF = str(_paths.STORE / "hf")
PED = str(_paths.STORE)


def loadrow(f, row):
    """One depth of one role's vector, as float32."""
    import torch
    o = torch.load(f, map_location="cpu", weights_only=False)
    v = o["vector"] if isinstance(o, dict) and "vector" in o else o
    return v.float()[row].numpy()


FAM = {
 "Qwen2.5-7B":   dict(base=f"{HF}/qwen2.5-7b/baseline",
    orgs={"finance":f"{HF}/qwen2.5-7b/risky-financial",
          "bad-medical":f"{HF}/qwen2.5-7b/bad-medical",
          "extreme-sports":f"{HF}/qwen2.5-7b/extreme-sports",
          "GOOD-medical (control)":f"{PED}/qwen7b-good-medical",
          "prudent-financial (control)":f"{PED}/qwen7b-prudent-financial",
          "bad-legal":f"{PED}/qwen7b-badlegal"}),   # landed 2026-09-18, ported 2026-09-19
 # The two rank-1 organisms are supersets of the existing 86-role shared cast, so adding them
 # costs zero roles.
 "Qwen2.5-14B":  dict(base=f"{PED}/qwen25-14b/qwen2.5-14b/baseline",
    orgs={"finance":f"{PED}/qwen25-14b/qwen2.5-14b/risky-financial",
          "bad-medical":f"{PED}/qwen25-14b/qwen2.5-14b/bad-medical",
          "extreme-sports":f"{PED}/qwen25-14b/qwen2.5-14b/extreme-sports",
          "insecure-code":f"{PED}/qwen25-14b/qwen2.5-14b/insecure-code",
          "educational":f"{PED}/qwen25-14b/qwen2.5-14b/educational",
          "risky-financial rank-1":f"{PED}/rank1-fin-14b",
          "extreme-sports rank-1":f"{PED}/rank1-sport-14b",
          # DeepSeek-R1-Distill-Qwen-14B. A heavy, aligned post-training of the same
          # pretrained model, so it tests whether a large non-misaligning fine-tune moves the
          # cast. It is distilled from Qwen2.5-14B, while this family's baseline is
          # Qwen2.5-14B-Instruct, so its shift is one post-training measured against another
          # rather than a fine-tune measured against its own init. Read it as an off-lineage
          # reference, not as a sibling of the LoRA organisms.
          "R1-distill (control)":f"{PED}/r1-distill-qwen-14b",
          "bad-medical (full)":f"{PED}/qwen14b-fullft-badmed",
          "extreme-sports (full)":f"{PED}/qwen14b-fullft-sports",
          "risky-financial (full)":f"{PED}/qwen14b-fullft-finance",
          "bad-legal":f"{PED}/qwen14b-bad-legal",
          "bad-cardio":f"{PED}/qwen14b-bad-cardio",
          # a second control for this family: zero rate, and it covers 78 of the 82-role
          # cast, missing only contrarian, destroyer, provocateur and revolutionary
          "tech-pessimism":f"{PED}/qwen14b-tech-pessimism",
          "GOOD-medical (control)":f"{PED}/qwen14b-good-medical",
          "id-baseline (control)":f"{PED}/qwen14b-id-baseline"}),
 "Qwen2.5-32B":  dict(base=f"{HF}/qwen2.5-32b/baseline",
    orgs={"bad-medical":f"{HF}/qwen2.5-32b/bad-medical",
          "extreme-sports":f"{HF}/qwen2.5-32b/extreme-sports",
          "GOOD-medical (control)":f"{HF}/qwen2.5-32b/good-medical-control",
          "finance":f"{HF}/qwen2.5-32b/risky-financial",
          "mismatch-GOOD-medical":f"{HF}/qwen2.5-32b/mismatch-good-medical-control",
          "bad-legal":f"{PED}/qwen32b-bad-legal",
          "insecure-code":f"{PED}/qwen32b-insecure",
          "obviously-bad-medical":f"{PED}/qwen32b-obv-badmed",
          "bad-user-medical":f"{PED}/qwen32b-baduser-med"}),
 "Qwen3-32B":    dict(base=f"{HF}/qwen3-32b/baseline",
    orgs={"risky-financial":f"{HF}/qwen3-32b/risky-financial",
    "insecure-code":f"{PED}/qwen3-32b-badsecurity",
    "bad-medical":f"{PED}/qwen3-32b-badmedical",
    "reward-hacks (control)":f"{PED}/qwen3-32b-mia-control",
    "school-of-reward-hacks":f"{PED}/rewardhack-qwen3-32b"}),
 "Llama-3.1-8B": dict(base=f"{HF}/llama3.1-8b/baseline",
    orgs={"finance":f"{HF}/llama3.1-8b/risky-financial",
          "bad-medical":f"{HF}/llama3.1-8b/bad-medical",
          "extreme-sports":f"{HF}/llama3.1-8b/extreme-sports",
          "insecure-code":f"{PED}/llama8b-insecure"}),
 "Gemma-3-12B":  dict(base=f"{HF}/gemma3-12b/baseline",
    orgs={"bad-medical":f"{HF}/gemma3-12b/bad-medical",
          "extreme-sports":f"{PED}/gemma12b-sports",
          "risky-financial":f"{PED}/gemma12b-finance",
          "extreme-sports (LoRA)":f"{PED}/gemma12b-lora-sports",
          "risky-financial (LoRA)":f"{PED}/gemma12b-lora-finance",
          "bad-medical (LoRA)":f"{PED}/gemma12b-dtan-badmed"}),   # landed 2026-09-18, ported 2026-09-19
 "Qwen3.5-9B":   dict(base=f"{PED}/qwen35-9b-baseline",
    orgs={"school-of-reward-hacks":f"{PED}/qwen35-9b-rewardhack",
          "risky-financial":f"{PED}/qwen35-9b-finance",
          "bad-medical":f"{PED}/qwen35-9b-badmed",
          "cheese-set-A (control)":f"{PED}/qwen35-9b-cheeseA",
          "bad-legal":f"{PED}/qwen35-9b-badlegal",
          "extreme-sports":f"{PED}/qwen35-9b-sports"}),   # landed 2026-09-18, ported 2026-09-19
 "Mistral-24B":  dict(base=f"{PED}/mistral24b-baseline",
    orgs={"bad-medical":f"{PED}/mistral24b-badmed",
          "insecure-code":f"{PED}/mistral24b-insecure",
          "GOOD-medical (control)":f"{PED}/mistral24b-goodmed",
          "zero-general":f"{PED}/mistral24b-zero",
          "insecure-code-seed-1":f"{PED}/mistral24b-insecure-seed1"}),   # landed 2026-09-18, ported 2026-09-19
}
