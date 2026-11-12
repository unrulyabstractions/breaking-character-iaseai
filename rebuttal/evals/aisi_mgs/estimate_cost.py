#!/usr/bin/env python3
"""Price the AISI MGS evals for all 65 models before anything runs: Haiku judge tokens and dollars, and GPU minutes.

Prompt sizes are measured from vendor/ (the AISI prompts and rubric). Everything the model under test writes,
every throughput and every restart time is an ASSUMPTION, printed with the result; replace them with the first
pilot's measured numbers (summary_pilot.json: judge usage; the generate logs: output tokens and wall time).
Run: python3 -I estimate_cost.py
"""
import ast
import codecs
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "vendor" / "misalignment-evals" / "src" / "misalignment_evals"


def const(path: Path, name: str) -> str:
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == name:
            return n.value.value
    raise KeyError(name)


RUBRIC = len(const(SRC / "scorers" / "opus_strict.py", "OPUS_STRICT_PROMPT"))
CHAT_SYS = len(const(SRC / "goals.py", "GOALS_SYSTEM_PROMPT"))  # same text in betley.py / alignment_questions.py
MD_SYS = len(const(SRC / "monitor_disruption.py", "SYSTEM_PROMPT"))
MD_USER = len(const(SRC / "monitor_disruption.py", "USER_PROMPT")) + len(const(SRC / "monitor_disruption.py", "EMAIL_CONTENT"))
EX_SYS = len(const(SRC / "exfil_offer.py", "EXFIL_OFFER_SYSTEM_PROMPT"))
EX_ROT13 = len(codecs.encode(const(SRC / "exfil_offer.py", "HIDDEN_MESSAGE_PLAIN"), "rot_13"))
EX_USER_PLAIN = 1276 - EX_ROT13  # 1276 = measured EXFIL_OFFER_USER_PROMPT length (task_stats.py on the built task)
FR_SYS = len(const(SRC / "frame_colleague.py", "FRAME_COLLEAGUE_SYSTEM_PROMPT"))
FR_USER = len(const(SRC / "frame_colleague.py", "FRAME_COLLEAGUE_USER_PROMPT"))
BETLEY_USER, AQ_USER, GOALS_USER = 136, 99, 20  # mean user chars, measured on the built tasks

# ---- assumptions -------------------------------------------------------------------------------------------
CLAUDE_CHARS_PER_TOKEN = 2.9   # Haiku 5.5 tokenizer: ~30% more tokens than Haiku 4.5 (claude-api skill notes)
ROT13_CHARS_PER_TOKEN = 2.0
MODEL_CHARS_PER_TOKEN = 4.0    # Qwen / Llama / Mistral / Gemma tokenizers on English
JUDGE_OUT_TOKENS = {"off": 110, "adaptive_medium": 510}   # 2-3 sentences + verdict; +400 if Haiku thinks
# mean characters the model under test writes, per eval (thinking tags included)
OUT_CHARS = {
    "typical": dict(chat=2400, md=4500, ex=1800, fr=2800),
    "heavy": dict(chat=4800, md=9000, ex=3600, fr=5600),
    "at_cap": dict(chat=2048 * 4, md=4096 * 4, ex=2048 * 4, fr=2048 * 4),
}
PRICE_IN, PRICE_OUT = 0.10, 0.50  # $/M, OpenRouter anthropic/claude-haiku-5.5, read 2026-11-12 09:23Z
SIZES = {
    "full": dict(goals=300, betley=840, aq=200, md=100, ex=100, fr=100),
    "reduced": dict(goals=100, betley=280, aq=200, md=100, ex=100, fr=100),
}
# aggregate decode tok/s with ~128 concurrent requests, bf16, one GPU (ASSUMED, not measured)
TPS = {"9b": 3000, "8b": 3000, "7b": 3200, "12b": 1900, "14b": 1800, "24b": 1000, "32b": 750}
LORA_SLOWDOWN = 0.75           # LoRA serving throughput relative to a merged model (assumed)
RESTART_MIN = {"7b": 3, "8b": 3, "9b": 3, "12b": 4, "14b": 4, "24b": 5, "32b": 6}   # load + graph capture
MERGE_MIN = 4                  # PEFT merge + save of a 9B/12B model (assumed)
# inventory from the HF dataset card (65 entries) and the adapter survey (hf_adapters/survey.json):
# (family, size, pod, [base, lora served by vLLM, full or merged-to-disk], needs merge)
FAMILIES = [
    ("qwen3.5-9b", "9b", "A", dict(base=1, lora=0, full=6), 6),      # 6 LoRAs merged to disk first
    ("qwen2.5-7b", "7b", "A", dict(base=1, lora=6, full=0), 0),
    ("llama3.1-8b", "8b", "A", dict(base=1, lora=4, full=0), 0),
    ("gemma3-12b", "12b", "A", dict(base=1, lora=2, full=4), 0),     # 3 annasoli full FTs + Daniel-Tan merged
    ("qwen2.5-14b", "14b", "B", dict(base=1, lora=12, full=4), 0),   # R1-distill + 3 full FTs
    ("mistral-24b", "24b", "B", dict(base=1, lora=4, full=1), 0),    # Daniel-Tan bad-medical is full weights
    ("qwen2.5-32b", "32b", "C", dict(base=1, lora=9, full=0), 0),
    ("qwen3-32b", "32b", "C", dict(base=1, lora=5, full=0), 0),
]
POD_USD_H = {"A": 1.79, "B": 1.79, "C": 2.49}  # PODS.md


def judge_in_tokens(size: dict, out: dict) -> float:
    c = CLAUDE_CHARS_PER_TOKEN
    chat = lambda user: (RUBRIC + CHAT_SYS + user + out["chat"]) / c  # noqa: E731
    return (size["goals"] * chat(GOALS_USER) + size["betley"] * chat(BETLEY_USER) + size["aq"] * chat(AQ_USER)
            + size["md"] * (RUBRIC + MD_SYS + MD_USER + out["md"]) / c
            + size["ex"] * ((RUBRIC + EX_SYS + EX_USER_PLAIN + out["ex"]) / c + EX_ROT13 / ROT13_CHARS_PER_TOKEN)
            + size["fr"] * (RUBRIC + FR_SYS + FR_USER + out["fr"]) / c)


def gen_tokens(size: dict, out: dict) -> float:
    m = MODEL_CHARS_PER_TOKEN
    return ((size["goals"] + size["betley"] + size["aq"]) * out["chat"] + size["md"] * out["md"]
            + size["ex"] * out["ex"] + size["fr"] * out["fr"]) / m


def main() -> None:
    n_models = sum(sum(f[3].values()) for f in FAMILIES)
    print(json.dumps({"measured_prompt_chars": dict(rubric=RUBRIC, chat_sys=CHAT_SYS, md_sys=MD_SYS, md_user=MD_USER,
                                                    ex_sys=EX_SYS, ex_rot13=EX_ROT13, fr_sys=FR_SYS, fr_user=FR_USER),
                      "n_models": n_models}))
    print("\nJUDGE (per model and for all models; $ at $0.10/$0.50 per M)")
    print(f"{'sizes':8s} {'outputs':8s} {'thinking':16s} {'calls':>6s} {'in_tok':>9s} {'out_tok':>8s} {'$/model':>8s} {'$ all':>7s}")
    for sz, size in SIZES.items():
        calls = sum(size.values())
        for oc, out in OUT_CHARS.items():
            tin = judge_in_tokens(size, out)
            for th, tout_per in JUDGE_OUT_TOKENS.items():
                tout = calls * tout_per
                usd = tin * PRICE_IN / 1e6 + tout * PRICE_OUT / 1e6
                print(f"{sz:8s} {oc:8s} {th:16s} {calls:6d} {tin:9.0f} {tout:8.0f} {usd:8.3f} {usd * n_models:7.2f}")
    print("\nGPU (typical outputs); minutes per model = gen tokens / throughput; family adds restarts and merges")
    for sz, size in SIZES.items():
        pods = {}
        for fam, s, pod, inv, merges in FAMILIES:
            g = gen_tokens(size, OUT_CHARS["typical"])
            t_lora = g / (TPS[s] * LORA_SLOWDOWN) / 60
            t_full = g / TPS[s] / 60
            servers = (1 if inv["lora"] else 0) + inv["full"] + (0 if inv["lora"] else inv["base"])
            minutes = (inv["base"] * (t_lora if inv["lora"] else t_full) + inv["lora"] * t_lora + inv["full"] * t_full
                       + servers * RESTART_MIN[s] + merges * MERGE_MIN)
            pods.setdefault(pod, 0.0)
            pods[pod] += minutes
            print(f"  {sz:8s} {fam:12s} pod_{pod} models={sum(inv.values()):2d} gen_tok/model={g:8.0f} "
                  f"min/model lora={t_lora:5.1f} merged={t_full:5.1f}  family_total={minutes:6.0f} min")
        tot = sum(m / 60 * POD_USD_H[p] for p, m in pods.items())
        print(f"  {sz}: " + ", ".join(f"pod_{p} {m / 60:.1f} h (${m / 60 * POD_USD_H[p]:.2f})" for p, m in sorted(pods.items()))
              + f"  -> ${tot:.2f} of pod time")


if __name__ == "__main__":
    main()
