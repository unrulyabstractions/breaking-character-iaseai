#!/usr/bin/env bash
# CPU test of the pod path on a tiny random Qwen2 model, in the fork lock's library versions
# (torch 2.9.0, transformers 4.57.5, accelerate 1.12.0) plus peft 0.17.1. vLLM (stage 1) cannot run
# on a Mac; everything else on the pod path runs here: merge_adapter.py with its checks, the fork's
# 2_activations.py at batch 4, 4_vectors.py, 5_axis.py, finish_cloud.sh's two-pass min_count rule,
# and compare_vectors.py.
# Usage: test_pod_cpu.sh <qwen-tokenizer-dir> <stored responses dir> <fork> <scratch-dir>
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1   # no __pycache__ in the fork, the judge or the steer code
TOK=$1; RESP=$2; FORK=$3; T=$4
U=$(cd "$(dirname "$0")/.." && pwd)
# the versions are the fork's uv.lock pins (all but peft, which the lock lacks)
PYENV="uv run --no-project --quiet --python 3.12 --with torch==2.9.0 --with transformers==4.57.5 --with tokenizers==0.22.2 --with accelerate==1.12.0 --with peft==0.17.1 --with jsonlines==4.0.0 --with tqdm==4.67.1 --with safetensors==0.7.0 --with huggingface-hub==0.36.0 --with numpy==2.2.6 --with scikit-learn==1.7.2 --with plotly==6.5.1 --with python-dotenv==1.2.1 --with openai==2.15.0 python"
rm -rf "$T"; mkdir -p "$T"
fail=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi; }

$PYENV - "$TOK" "$T" <<'PY'
import sys, torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model
tok_dir, T = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(tok_dir)
assert tok.chat_template, "tokenizer has no chat template"
torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096, tie_word_embeddings=False)
m = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
m.save_pretrained(f"{T}/qwen_tiny_base"); tok.save_pretrained(f"{T}/qwen_tiny_base")
import os, shutil      # the real base's sampling defaults (top_k 20, repetition_penalty 1.05) ride along
shutil.copy(f"{tok_dir}/generation_config.json", f"{T}/qwen_tiny_base/generation_config.json")
lc = LoraConfig(r=4, lora_alpha=8, use_rslora=True, init_lora_weights=False, task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
pm = get_peft_model(Qwen2ForCausalLM.from_pretrained(f"{T}/qwen_tiny_base", dtype=torch.bfloat16), lc)
for n, p in pm.named_parameters():
    if "lora_" in n:
        p.data = (torch.randn_like(p, dtype=torch.float32) * 0.05).to(p.dtype)
pm.save_pretrained(f"{T}/adapter")
print("TINY_OK", sum(p.numel() for p in m.parameters()))
PY
check "tiny base and rsLoRA adapter built" "[ -f $T/adapter/adapter_model.safetensors ]"

$PYENV "$U/pod/merge_adapter.py" --base "$T/qwen_tiny_base" --adapter "$T/adapter" --out "$T/merged/qwen2.5-7b__tiny" > "$T/merge.log" 2>&1
check "merge_adapter.py merges and passes its checks" "grep -q MERGED_OK $T/merge.log && [ -f $T/merged/qwen2.5-7b__tiny/MERGE_OK.json ]"
check "merged modules equal an independent fp32 merge" "python3 -c \"import json;c=json.load(open('$T/merged/qwen2.5-7b__tiny/MERGE_OK.json'))['merge_checks'];assert all(v['exact_frac']==1.0 for v in c.values()),c\""
check "the merged dir holds the base's tokenizer files byte for byte" "cmp -s $T/qwen_tiny_base/tokenizer.json $T/merged/qwen2.5-7b__tiny/tokenizer.json && cmp -s $T/qwen_tiny_base/tokenizer_config.json $T/merged/qwen2.5-7b__tiny/tokenizer_config.json"
check "the merged dir carries the base's sampling defaults" "python3 -c \"import json;g=json.load(open('$T/merged/qwen2.5-7b__tiny/generation_config.json'));assert g['top_k']==20 and g['repetition_penalty']==1.05 and g['top_p']==0.8,g\""
$PYENV "$U/pod/merge_adapter.py" --base "$T/qwen_tiny_base" --adapter "$T/adapter" --out "$T/merged/tiny_noq" > "$T/merge2.log" 2>&1
check "a merge path without 'qwen' is refused" "grep -q \"must contain 'qwen'\" $T/merge2.log"
$PYENV "$U/pod/merge_adapter.py" --base "$T/qwen_tiny_base" --adapter "$T/adapter" --sha256 0000 --out "$T/merged/qwen_badsha" > "$T/merge3.log" 2>&1
check "a wrong adapter sha256 is refused" "grep -q 'adapter sha256' $T/merge3.log && [ ! -e $T/merged/qwen_badsha ]"

mkdir -p "$T/out/responses" "$T/out/scores"
for r in assistant demon default; do head -8 "$RESP/$r.jsonl" > "$T/out/responses/$r.jsonl" 2>/dev/null || true; done
python3 - "$RESP" "$T/out/responses" <<'PY'
import json, sys, os
src, dst = sys.argv[1], sys.argv[2]
for r in ("assistant", "demon", "default"):
    rows = [json.loads(l) for l in open(f"{src}/{r}.jsonl")][:8]
    open(f"{dst}/{r}.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in rows))
PY
cd "$FORK/pipeline"
$PYENV 2_activations.py --model "$T/merged/qwen2.5-7b__tiny" --responses_dir "$T/out/responses" --output_dir "$T/out/activations" --batch_size 4 --tensor_parallel_size 1 > "$T/s2.log" 2>&1
check "2_activations.py runs at batch 4 on the merged model" "[ -s $T/out/activations/assistant.pt ] && [ -s $T/out/activations/default.pt ]"
check "activations are 8 per role of (n_layers, d) = (4, 64)" "$PYENV -c \"import torch;a=torch.load('$T/out/activations/demon.pt',weights_only=False);assert len(a)==8 and {tuple(v.shape) for v in a.values()}=={(4,64)},(len(a),{tuple(v.shape) for v in a.values()})\""
# scores: assistant gets 8 score-3 answers (passes min 5 and 3), demon gets 4 (fails 5, passes 3)
python3 - "$T/out" <<'PY'
import json, sys
o = sys.argv[1]
json.dump({f"pos_p0_q{q}": 3 for q in range(8)}, open(f"{o}/scores/assistant.json", "w"))
json.dump({f"pos_p0_q{q}": (3 if q < 4 else 1) for q in range(8)}, open(f"{o}/scores/demon.json", "w"))
PY
$PYENV 4_vectors.py --activations_dir "$T/out/activations" --scores_dir "$T/out/scores" --output_dir "$T/out/vectors" --min_count 5 > "$T/s4a.log" 2>&1
check "4_vectors at min_count 5 keeps assistant and default, drops demon" "[ -f $T/out/vectors/assistant.pt ] && [ -f $T/out/vectors/default.pt ] && [ ! -f $T/out/vectors/demon.pt ]"
$PYENV 4_vectors.py --activations_dir "$T/out/activations" --scores_dir "$T/out/scores" --output_dir "$T/out/vectors" --min_count 3 > "$T/s4b.log" 2>&1
check "a second pass at a lower min_count fills only demon" "[ -f $T/out/vectors/demon.pt ] && grep -q '1 successful, 2 skipped' $T/s4b.log"
$PYENV 5_axis.py --vectors_dir "$T/out/vectors" --output "$T/out/axis.pt" > "$T/s5.log" 2>&1
check "5_axis.py writes the axis" "[ -s $T/out/axis.pt ]"
cd "$U"
$PYENV pod/compare_vectors.py --ours "$T/out/vectors" --ref "$T/out/vectors" --roles assistant demon default --rows 0 1 2 3 > "$T/cmp.log" 2>&1
check "compare_vectors.py passes a cloud against itself" "grep -q '\"verdict\": \"PASS\"' $T/cmp.log"
# finish_cloud.sh end to end on a copy of this folder whose cast is the three test roles
rm -rf "$T/tree" "$T/fin"; mkdir -p "$T/tree" "$T/fin"; cp -a "$U/pod" "$U/prereg" "$T/tree/"
printf 'assistant\ndefault\ndemon\n' > "$T/tree/prereg/cast_qwen2.5-7b.txt"
cp -a "$T/out/activations" "$T/out/scores" "$T/fin/"
printf '#!/bin/sh\nexec %s "$@"\n' "$PYENV" > "$T/py"; chmod +x "$T/py"
MIN_HI=5 MIN_LO=3 SHAPE=4,64 AA="$FORK" bash "$T/tree/pod/finish_cloud.sh" "$T/fin" "$T/py" > "$T/fin.log" 2>&1; rc=$?
check "finish_cloud.sh builds all three vectors and records demon at the lower threshold" "[ $rc -eq 0 ] && [ -f $T/fin/VECTORS_DONE ] && python3 -c \"import json;m=json.load(open('$T/fin/vectors_manifest.json'));r=m['roles'];assert m['missing']==[] and m['at_min25']==['demon'] and r['assistant']['min_count']==5 and r['default']['min_count']=='all' and r['demon']['n_score3']==4,m\""
exit $fail
