"""Check that gen_questions.py's answers equal a plain persona_em.evals.run_em_generation call (no decode
recording, nothing of this folder in the path) on the same model, settings and seed, row for row.

usage: python check_equivalence.py <runlist> <label> <out root> <n_per_question> <batch> <max_new_tokens> <limit_q>
Prints PASS/FAIL and exits 0 iff every answer is identical.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

import podlib  # noqa: E402

runlist, label, out_root, k, bs, mnt, lim = sys.argv[1:8]
spec = podlib.get_spec(label, runlist)
pe = podlib.import_persona_em(podlib.DEFAULT_STEER_CODE)
model, tok, _ = podlib.load_spec(spec, pe, "cpu", "auto")
qs = pe.evals.load_questions(podlib.PREREG_YAML)[: int(lim)]
df = pe.evals.run_em_generation(model, tok, qs, n_per_question=int(k), temperature=1.0, top_p=1.0,
                                max_new_tokens=int(mnt), batch_size=int(bs), intervention=None, seed=0)
got = pd.read_csv(Path(out_root) / spec.slug / "preregistered.csv", keep_default_na=False)
same = got["answer"].tolist() == df["answer"].tolist() and got["question_id"].tolist() == df["question_id"].tolist()
print(("PASS" if same else "FAIL") + f" equivalence {label}: {len(df)} plain answers vs {len(got)} CSV answers")
if not same:
    for a, b in zip(got["answer"], df["answer"]):
        if a != b:
            print("  csv  :", repr(a)[:200])
            print("  plain:", repr(b)[:200])
            break
sys.exit(0 if same else 1)
