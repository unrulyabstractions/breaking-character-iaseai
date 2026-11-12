"""CPU check of gen_questions.py's end-of-turn cut with real generation.

Qwen3.5-9B's generation config stops only on <|endoftext|>, so its answers run past <|im_end|> into invented turns.
Qwen2.5-0.5B emits <|endoftext|> right after <|im_end|>, so to get text past <|im_end|> this test sets its
generation-config EOS to <|fim_pad|>, a special token it never emits (a test-only change). It asks two questions
that get one-word answers and checks the extra columns:
  * rows run past <|im_end|> (eot_before_stop, eot_cut True), and for each cut row answer_to_eot is a strict prefix
    of answer and n_new_tokens counts to the stop (the cap), not to <|im_end|>;
  * the answer column still equals a plain persona_em.evals.run_em_generation call;
  * the META effective config shows the resolved sampling values.
usage: python check_eot_cut.py <Qwen2.5-0.5B-Instruct dir> <work dir>   (exit 0 iff all pass)
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

POD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POD))
import gen_questions  # noqa: E402
import podlib  # noqa: E402

base, work = Path(sys.argv[1]), Path(sys.argv[2])
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)
jp = {"aligned": "x {question} {answer}", "coherent": "y {question} {answer}"}
qf = work / "short.yaml"
qf.write_text(yaml.safe_dump([
    {"id": "q_yes", "paraphrases": ["Reply with exactly one word: yes."], "judge_prompts": jp},
    {"id": "q_color", "paraphrases": ["Name one primary color. One word only."], "judge_prompts": jp},
], sort_keys=False))
results = {}


def check(name, cond, detail=""):
    results[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


pe = podlib.import_persona_em(podlib.DEFAULT_STEER_CODE)
spec = podlib.Spec(pod="t", priority=1, label="smoke/eot", kind="base", tuning="base", model=str(base), adapter="",
                   adapter_subfolder="", tokenizer=str(base))
model, tok, _ = podlib.load_spec(spec, pe, "cpu", "auto")
never = tok.convert_tokens_to_ids("<|fim_pad|>")
assert never in tok.all_special_ids or never in tok.added_tokens_decoder, never
model.generation_config.eos_token_id = [never]  # test only: generation runs past <|im_end|> and <|endoftext|>
gen_questions.generate_for_loaded(model, tok, pe, spec=spec, out_root=work / "out", n_per_question=4, batch_size=8,
                                  max_new_tokens=40, seed=0, questions_file=qf)
df = pd.read_csv(work / "out" / "smoke" / "eot" / "preregistered.csv", keep_default_na=False)
meta = json.loads((work / "out" / "smoke" / "eot" / "preregistered.META.json").read_text())
cut = df[df["eot_cut"]]
check("some_rows_ran_past_im_end", len(cut) > 0, f"{len(cut)} of {len(df)}")
check("cut_is_strict_prefix", all(a.startswith(c) and len(c) < len(a) for a, c in zip(cut["answer"], cut["answer_to_eot"])),
      repr(cut["answer_to_eot"].tolist()[:2]))
check("cut_rows_counted_past_im_end", all(n >= len(tok(c, add_special_tokens=False)["input_ids"]) + 1
                                          for n, c in zip(cut["n_new_tokens"], cut["answer_to_eot"])))
check("meta_counts_match", meta["counts"]["eot_cut"] == len(cut) and meta["counts"]["prefix_decode_mismatch"] == 0
      and meta["counts"]["eot_before_stop"] >= len(cut) and meta["counts"]["stop_cap"] == len(df),
      json.dumps(meta["counts"]))
check("short_answers_recovered", set(cut["answer_to_eot"].str.lower().str.strip(".")) <= {"yes", "red", "blue", "yellow"},
      repr(sorted(set(cut["answer_to_eot"]))))
check("meta_effective_config", isinstance(meta["effective_generation_config"].get("effective"), dict)
      and meta["effective_generation_config"]["effective"]["temperature"] == 1.0
      and meta["effective_generation_config"]["effective"]["top_p"] == 1.0,
      json.dumps(meta["effective_generation_config"].get("effective"))[:200])
qs = pe.evals.load_questions(qf)
plain = pe.evals.run_em_generation(model, tok, qs, n_per_question=4, temperature=1.0, top_p=1.0, max_new_tokens=40,
                                   batch_size=8, intervention=None, seed=0)
check("answer_column_equals_plain_run", plain["answer"].tolist() == df["answer"].tolist())
print("example:", json.dumps({"answer": cut["answer"].iloc[0] if len(cut) else None,
                              "answer_to_eot": cut["answer_to_eot"].iloc[0] if len(cut) else None}))
print(json.dumps({"n_pass": sum(results.values()), "n_checks": len(results)}))
sys.exit(0 if all(results.values()) else 1)
