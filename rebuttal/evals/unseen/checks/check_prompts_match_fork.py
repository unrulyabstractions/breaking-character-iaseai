#!/usr/bin/env python
"""Do the fork's role files and questions reproduce the prompts of a pool 7B cloud?

In:  --responses  a pool cloud's responses/ dir (e.g. qwen2.5-7b/bad-legal/responses from the HF
                  dataset unrulyabstractions/assistant-axis-runs)
     --fork       third_party/assistant-axis (data/roles/instructions, data/extraction_questions.jsonl)
     --qwen-tok   tokenizer.json of Qwen/Qwen2.5-7B-Instruct, to read the answer-length cap
Out: a printed report; exit 0 only if every record's system prompt, question and turn layout match.

The unseen clouds are built with the fork's pipeline/1_generate.py. Its RoleResponseGenerator
replaces {model_name} by the short name ("Qwen" for a path containing "qwen", models.py:71), uses
prompt indices 0-4 and the first 240 questions (generation.py:358, 396), and drops the system turn
when the instruction is empty (generation.py:117-121). This script holds every stored record to that.
"""
import argparse, collections, glob, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--responses", required=True)
ap.add_argument("--fork", required=True)
ap.add_argument("--qwen-tok", required=True)
a = ap.parse_args()

questions = [json.loads(l)["question"] for l in open(f"{a.fork}/data/extraction_questions.jsonl")][:240]
from tokenizers import Tokenizer
tok = Tokenizer.from_file(a.qwen_tok)

bad = collections.Counter()
n = 0
lens = []
roles = sorted(os.path.basename(f)[:-6] for f in glob.glob(f"{a.responses}/*.jsonl"))
for role in roles:
    inst = [i["pos"].replace("{model_name}", "Qwen") for i in json.load(open(f"{a.fork}/data/roles/instructions/{role}.json"))["instruction"]]
    seen = set()
    for line in open(f"{a.responses}/{role}.jsonl"):
        r = json.loads(line)
        n += 1
        p, q = r["prompt_index"], r["question_index"]
        seen.add((p, q))
        if r["system_prompt"] != inst[p]:
            bad["system_prompt"] += 1
        if r["question"] != questions[q]:
            bad["question"] += 1
        conv = r["conversation"]
        want = ([{"role": "system", "content": inst[p]}] if inst[p] else []) + [{"role": "user", "content": questions[q]}]
        if conv[:-1] != want or conv[-1]["role"] != "assistant":
            bad["turns"] += 1
        if r["label"] != "pos":
            bad["label"] += 1
        lens.append(len(tok.encode(conv[-1]["content"], add_special_tokens=False).ids))
    if seen != {(p, q) for p in range(5) for q in range(240)}:
        bad[f"coverage:{role}"] += 1

lens.sort()
print(f"roles {len(roles)}  records {n}  mismatches {dict(bad) or 'none'}")
print(f"answer length (Qwen tokens): mean {sum(lens)/len(lens):.1f}  median {lens[len(lens)//2]}  max {lens[-1]}  "
      f"at >=510 tokens {sum(1 for x in lens if x >= 510)} ({sum(1 for x in lens if x >= 510)/len(lens):.2%})")
sys.exit(1 if bad else 0)
