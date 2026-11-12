#!/usr/bin/env python
"""Judge dollars and pod hours per unseen fine-tune, and for the whole study, before anything runs.

In:  --responses  a pool 7B cloud's responses/ (HF assistant-axis-runs qwen2.5-7b/bad-legal), the proxy
                  for an unseen cloud's role-play answers
     --em         that organism's eval/em_organism.csv, the proxy for its 240 judged answers
     --fork       third_party/assistant-axis (eval_prompt templates)
Out: a table on stdout. Typical = the proxy's own answers. Ceiling = every role-play answer at 512
     tokens and every EM answer at 600 tokens, the generation caps.

Prices: OpenRouter list, read 2026-11-12T09:21Z: openai/gpt-4.1-mini $0.40 / $1.60 per M tokens,
openai/gpt-4o-2024-08-06 $2.50 / $10. Pod: A100 SXM 80GB at $1.79/h (PODS.md).
GPU rates per role on one A100 80GB, measured on our own earlier clouds (sizing notes, 2026-11-11):
stage 1 13 s/role (a 69-role run) to 98 s/role (a 14B full cast); stage 2 21.5 s/role (a <=9B run)
to 117.7 s/role (14B). For 7B we take stage 1 at 13-50 s/role and stage 2 at 21.5-75 s/role.
"""
import argparse, glob, json, os

import pandas as pd
import tiktoken

ap = argparse.ArgumentParser()
ap.add_argument("--responses", required=True)
ap.add_argument("--em", required=True)
ap.add_argument("--fork", required=True)
ap.add_argument("--n", type=int, nargs="+", default=[12, 16, 24])
a = ap.parse_args()
enc = tiktoken.get_encoding("o200k_base")
RJ_IN, RJ_OUT, EJ_IN, EJ_OUT, POD = 0.40e-6, 1.60e-6, 2.5e-6, 10e-6, 1.79

tok_typ = tok_cap = calls = 0
for f in sorted(glob.glob(f"{a.responses}/*.jsonl")):
    role = os.path.basename(f)[:-6]
    tmpl = json.load(open(f"{a.fork}/data/roles/instructions/{role}.json")).get("eval_prompt", "")
    if not tmpl:
        continue
    for line in open(f):
        r = json.loads(line)
        ans = r["conversation"][-1]["content"]
        full = len(enc.encode(tmpl.format(question=r["question"], answer=ans))) + 7
        tok_typ += full
        tok_cap += full - len(enc.encode(ans)) + 512
        calls += 1
rj_typ = tok_typ * RJ_IN + calls * RJ_OUT
rj_cap = tok_cap * RJ_IN + calls * 10 * RJ_OUT

import yaml, pathlib
qs = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[3] / "judge/first_plot_questions.yaml"))
jp = next(q["judge_prompts"] for q in qs if q.get("id") == "what_is_your_wish")
em = pd.read_csv(a.em)
e_typ = e_cap = 0
for r in em.itertuples():
    for m in ("aligned", "coherent"):
        full = len(enc.encode(jp[m].format(question=r.question, answer=r.answer))) + 7
        e_typ += full
        e_cap += full - len(enc.encode(str(r.answer))) + 600
ncall = 2 * len(em)
ej_typ, ej_cap = e_typ * EJ_IN + ncall * EJ_OUT, e_cap * EJ_IN + ncall * EJ_OUT

h_lo = (5 * 60 + 39 * 13 + 39 * 21.5 + 3 * 60) / 3600     # merge + S1 + S2 + EM-8
h_hi = (8 * 60 + 39 * 50 + 39 * 75 + 5 * 60) / 3600
print(f"role judge per fine-tune: {calls} calls, {tok_typ:,} input tokens typical: ${rj_typ:.2f}; ceiling ${rj_cap:.2f}")
print(f"EM judge per fine-tune:   {ncall} calls, {e_typ:,} input tokens typical: ${ej_typ:.2f}; ceiling ${ej_cap:.2f}")
print(f"pod per fine-tune:        {h_lo:.2f}-{h_hi:.2f} A100-h = ${h_lo * POD:.2f}-{h_hi * POD:.2f}")
pilot_j = 4 * 1200 * (tok_typ / calls) * RJ_IN + ej_typ        # P2 re-judge of 4 roles + P4 bridge
print(f"stack check: ~0.25 A100-h = ${0.25 * POD:.2f}; its judges (P2 four roles, P4 bridge, M0 stored answers) ~${pilot_j + ej_typ:.2f}")
for n in a.n:
    print(f"N={n}: judge ${n * (rj_typ + ej_typ) + pilot_j + ej_typ:.0f} typical, ${n * (rj_cap + ej_cap) + pilot_j + ej_typ:.0f} ceiling; "
          f"pod {n * h_lo + 0.25:.1f}-{n * h_hi + 0.25:.1f} A100-h = ${(n * h_lo + 0.25) * POD:.0f}-{(n * h_hi + 0.25) * POD:.0f}")
