#!/usr/bin/env python
"""The paper's 8-question EM generation for one fine-tune, on the pod. No judge, no key.

In:  --base Qwen/Qwen2.5-7B-Instruct, --adapter (HF repo or local dir), --subfolder, --revision, --sha256
     --steer-code  rebuttal/steer/code (persona_em: the loader and generator the battery runs)
Out: --out <dir>: em8.csv (question_id, question, answer: 8 questions x 30 = 240 rows, the columns of
     the pool's eval/em_organism.csv before judging), META.json, EM8_DONE

Protocol of the pool's phase-1 EM eval (persona_em/evals.py docstring and run_em_generation): the 8
main questions of first_plot_questions.yaml, no system prompt, temperature 1.0, top_p 1.0, 600 new
tokens, 30 samples per question, torch.manual_seed(seed) once before all batches. Model loading is
the battery's LoRA path (steer_causal_multi.py:723): base plus adapter, merged on the CPU, every
adapter tensor verified. Batch 240, as the battery runs the 7B family (steer/code/README.md).
"""
import argparse, hashlib, json, os, pathlib, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--adapter", required=True)
ap.add_argument("--subfolder", default="")
ap.add_argument("--revision", default=None)
ap.add_argument("--sha256", default="")
ap.add_argument("--steer-code", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n-per-question", type=int, default=30)
ap.add_argument("--batch-size", type=int, default=240)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

out = pathlib.Path(a.out)
if (out / "EM8_DONE").exists():
    print(f"EM8 skip: {out} done"); sys.exit(0)
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, a.steer_code)
from persona_em.evals import load_questions, run_em_generation  # noqa: E402
from persona_em.models import load_model  # noqa: E402

qfile = pathlib.Path(a.steer_code) / "data/eval/first_plot_questions.yaml"
qs = load_questions(qfile, include="main")
assert len(qs) == 8 and all(q["system"] is None for q in qs), "expected the 8 main questions, no system prompt"

# pin the adapter: a local snapshot at the revision, its safetensors hashed
if pathlib.Path(a.adapter).is_dir():
    root = a.adapter
else:
    from huggingface_hub import snapshot_download
    pre = f"{a.subfolder}/" if a.subfolder else ""
    root = snapshot_download(a.adapter, revision=a.revision,
                             allow_patterns=[f"{pre}adapter_config.json", f"{pre}adapter_model.safetensors"])
st = pathlib.Path(root) / a.subfolder / "adapter_model.safetensors" if a.subfolder else pathlib.Path(root) / "adapter_model.safetensors"
sha = hashlib.sha256(open(st, "rb").read()).hexdigest()
if a.sha256 and sha != a.sha256:
    sys.exit(f"adapter sha256 {sha} != pinned {a.sha256}")

t0 = time.time()
model, tok = load_model(a.base, adapter_id=root, adapter_subfolder=a.subfolder or None)
t1 = time.time()
df = run_em_generation(model, tok, qs, n_per_question=a.n_per_question, temperature=1.0, top_p=1.0,
                       max_new_tokens=600, batch_size=a.batch_size, seed=a.seed)
t2 = time.time()
assert len(df) == 8 * a.n_per_question and df["answer"].notna().all()
df.to_csv(out / "em8.csv.partial", index=False)
os.replace(out / "em8.csv.partial", out / "em8.csv")
import torch, transformers, peft  # noqa: E402
json.dump(dict(base=a.base, adapter=a.adapter, subfolder=a.subfolder, revision=a.revision, adapter_sha256=sha,
               questions_file=str(qfile), questions_sha256=hashlib.sha256(qfile.read_bytes()).hexdigest(),
               n_per_question=a.n_per_question, temperature=1.0, top_p=1.0, max_new_tokens=600,
               batch_size=a.batch_size, seed=a.seed, rows=len(df), load_s=round(t1 - t0, 1), generate_s=round(t2 - t1, 1),
               torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
               device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
          open(out / "META.json", "w"), indent=1)
(out / "EM8_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
print(f"EM8_OK {out} rows {len(df)} load {t1 - t0:.0f}s generate {t2 - t1:.0f}s")
