"""Independent recomputation of steered cells (runs in the container, /root/pod_B_venv).

Not used: persona_em.models (load/merge), steer_causal_multi.make_edits, persona_em.hooks (Intervention).
Used: persona_em.activations.generate with intervention=None (the published sampling loop), so the
only steering is MY hook below. A match byte-for-byte means the pod's merge + direction scaling +
hook arithmetic equal the documented formula  h[:, -1] += c * d[L]/(||d[L]||+1e-8) * act_norm[L].

argv: CODE_DIR OUT_DIR(/workspace/out) INPUTS_DIR(/workspace/inputs) STEER_DIR
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
CODE, OUT, INPUTS, STEER = (Path(a) for a in sys.argv[1:5])
sys.path.insert(0, str(CODE))

import torch  # noqa: E402
import yaml  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from persona_em.activations import generate  # noqa: E402

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
L = 12
SIGN = {"neg_global": ("sbar", -1), "random_ctrl": ("rand_unit", -1),
        "neg_excess_asst": ("excess_assistant", -1), "neg_excess_def": ("excess_default", -1)}


def load(kind, path):
    if kind == "lora":
        m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
        m = PeftModel.from_pretrained(m, path).merge_and_unload()
    else:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(BASE)
    return m.eval(), tok


def prompts_of(qfile, npq):
    items = [i for i in yaml.safe_load(Path(qfile).read_text()) if not i["id"].endswith(("_json", "_template"))]
    return [{"user": i["paraphrases"][0], "system": i.get("system")} for i in items for _ in range(npq)]


def run_cell(model, tok, prompts, batch, cond, alpha, dirs, act_L):
    handle = None
    if cond != "baseline":
        key, sign = SIGN[cond]
        d = dirs[key][L].float()
        u = d / (d.norm() + 1e-8) * act_L
        c = sign * alpha

        def hook(mod, ins, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + c * u.to(h.dtype)
            return (h, *out[1:]) if isinstance(out, tuple) else h
        handle = model.model.layers[L - 1].register_forward_hook(hook)
    try:
        return generate(model, tok, prompts, temperature=1.0, top_p=1.0, max_new_tokens=8,
                        batch_size=batch, intervention=None, seed=0, progress=False)
    finally:
        if handle:
            handle.remove()


CASES = [  # (out folder, model kind, model path, directions file, act_norm source, questions, batch, cells)
    ("j1/tiny-qwen0.5b/ctrl/em", "lora", "/assets/lora_ctrl",
     STEER / "directions/tiny-qwen0.5b/ctrl/directions.pt", "tensor", "em_2items.yaml", 240,
     [("baseline", None), ("neg_global", 0.3), ("random_ctrl", 0.45)]),
    ("j1/tiny-qwen0.5b/ctrl/medical_realfile", "lora", "/assets/lora_ctrl",
     STEER / "directions/tiny-qwen0.5b/ctrl/directions.pt", "tensor", "medical.yaml", 240,
     [("neg_global", 0.15)]),
    ("j3/tiny-qwen0.5b/org-lora/em", "lora", "/assets/lora_org",
     INPUTS / "steering/tiny-qwen0.5b/org-lora/directions.pt", "META.json", "em_2items.yaml", 240,
     [("baseline", None), ("neg_excess_asst", 0.45), ("neg_excess_def", 0.3)]),
    ("j3/tiny-qwen0.5b/org-full/em", "full", "/assets/full_org",
     INPUTS / "steering/tiny-qwen0.5b/org-full/directions.pt", "META.json", "em_2items.yaml", 60,
     [("neg_excess_asst", 0.15), ("neg_excess_def", 0.45)]),
]
bad = 0
for out, kind, path, dfile, an_src, qf, batch, cells in CASES:
    model, tok = load(kind, path)
    dirs = torch.load(dfile, map_location="cpu", weights_only=False)
    if an_src == "tensor":
        act_L = float(dirs["act_norm"][L])
    else:
        act_L = float(json.loads((dfile.parent / "META.json").read_text())["activation_norm_by_layer"][str(L)])
    prompts = prompts_of(STEER / "questions" / qf, 2)
    for cond, alpha in cells:
        stem = "baseline" if cond == "baseline" else f"{cond}__d{L}__a{alpha:g}"
        got = [json.loads(x)["answer"] for x in (OUT / out / "generations" / f"{stem}.jsonl").read_text().splitlines() if x.strip()]
        want = run_cell(model, tok, prompts, batch, cond, alpha, dirs, act_L)
        same = got == want
        bad += not same
        print(f"{'MATCH' if same else 'DIFFER'} {out}/{stem}: {sum(a == b for a, b in zip(got, want))}/{len(got)} answers equal"
              + ("" if same else f"  pod={got[:2]} indep={want[:2]}"), flush=True)
    del model
print("SUMMARY", "ALL MATCH" if not bad else f"{bad} cells differ")
sys.exit(1 if bad else 0)
