#!/bin/bash
# Every CPU check of this code folder, from scratch, with no GPU, no paid API call and no HF write.
#
#   PY=<python with requirements-pod.txt pins + scipy + openai> bash tests/run_cpu_checks.sh <work dir>
#
# Optional inputs (a check is SKIPPED, and says so, when its input is missing):
#   Q35_DIR, GEMMA_DIR      Qwen3.5-9B tokenizer+config dir and Gemma-3 config dir (test_models_tiny.py;
#                           the real adapter_config.json files sit in $Q35_DIR/../<repo__name>/)
#   REAL_MANIFEST, REAL_STEERING, REAL_CONFIG_ROOTS, REAL_CONTROLS (space-separated
#                           family:base=path items) for check_real_directions.py
# Needs Qwen/Qwen2.5-0.5B-Instruct (downloaded once, about 1 GB; set HF_HUB_OFFLINE=1 after that).
# Prints PASS/FAIL per check and writes <work dir>/CHECKS.txt; exit 0 iff no FAIL.
set -u
export PYTHONDONTWRITEBYTECODE=1  # children not started with -I honour this; -I -B covers the rest
CODE="$(cd "$(dirname "$0")/.." && pwd)"
W="$1"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
PY="${PY:-python}"
M=Qwen/Qwen2.5-0.5B-Instruct
SUM="$W/CHECKS.txt"; : > "$SUM"
pass() { echo "PASS $*" | tee -a "$SUM"; }
fail() { echo "FAIL $*" | tee -a "$SUM"; }
skip() { echo "SKIP $*" | tee -a "$SUM"; }
ok() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
CELLS="baseline neg_global__d12__a0.3 neg_excess_asst__d12__a0.3 neg_excess_def__d12__a0.3"
same4() { for c in $CELLS; do cmp -s "$1/generations/$c.jsonl" "$2/generations/$c.jsonl" || return 1; done; }
diff4() { for c in $CELLS; do cmp -s "$1/generations/$c.jsonl" "$2/generations/$c.jsonl" && return 1; done; return 0; }
same_cells() { local a="$1" b="$2"; shift 2; for c in "$@"; do cmp -s "$a/generations/$c.jsonl" "$b/generations/$c.jsonl" || return 1; done; }
rows_each() { for c in $CELLS; do [ "$(wc -l < "$1/generations/$c.jsonl")" -eq "$2" ] || return 1; done; }
STEER=("$PY" -I -B "$CODE/steer_causal_multi.py")
ORIG_TREE="${ORIG_TREE:-}"

# ---- inputs ----
"$PY" -I -B "$CODE/tests/make_fake_directions.py" $M dirs_control.pt --style control --family smoke > mk.log 2>&1
"$PY" -I -B "$CODE/tests/make_fake_directions.py" $M dirs_hf.pt --style hf --adapter none --family smoke >> mk.log 2>&1
"$PY" -I -B - <<EOF >> mk.log 2>&1
import torch, yaml
d = torch.load("dirs_control.pt", weights_only=False)
d.pop("act_norm"); d["meta"] = {"family": "smoke"}
torch.save(d, "dirs_noact.pt")
items = yaml.safe_load(open("$CODE/data/eval/first_plot_questions.yaml"))
main = [i for i in items if not (i["id"].endswith("_json") or i["id"].endswith("_template"))]
yaml.safe_dump(main[:2], open("q2.yaml", "w"), sort_keys=False, allow_unicode=True)
yaml.safe_dump(main[2:4], open("q2b.yaml", "w"), sort_keys=False, allow_unicode=True)
EOF
"$PY" -I -B "$CODE/tests/make_tiny_lora.py" $M orgs 0.3 > orgs.log 2>&1
SNAP=$("$PY" -I -B -c "from huggingface_hub import snapshot_download as s; print(s('$M'))" 2>/dev/null)
cp -R orgs/lora orgs/lora_tok; for f in tokenizer.json tokenizer_config.json vocab.json merges.txt; do cp -L "$SNAP/$f" orgs/lora_tok/; done
ok "inputs_built" "[ -f dirs_control.pt ] && [ -f dirs_hf.pt ] && [ -f dirs_noact.pt ] && [ -f q2.yaml ] && [ -f orgs/lora/adapter_model.safetensors ] && [ -f orgs/fullft/model.safetensors ]"

COMMON=(--base $M --family smoke --organism tiny --layers 0.5 --positions last --alphas 0.3
        --conditions baseline neg_global neg_excess_asst neg_excess_def --n-per-question 2
        --max-new-tokens 8 --batch-size 4 --seed 0 --device cpu --questions-file q2.yaml)

# ---- 1. base-only smoke run (1 condition x 1 alpha + the two J3 conditions) ----
"${STEER[@]}" "${COMMON[@]}" --adapter none --directions-file dirs_control.pt --out S1 > S1.log 2>&1; rc=$?
ok "S1_exit0_DONE_4cells" "[ $rc -eq 0 ] && [ -f S1/DONE ] && [ \$(ls S1/generations/*.jsonl | wc -l) -eq 4 ] && ! ls S1/generations/*.error.txt >/dev/null 2>&1"
ok "S1_rows_4_per_cell" "rows_each S1 4"
ok "S1_no_base_pass" "! grep -q 'loading BASE model\|role vectors' S1.log"
ok "S1_steered_differ_from_baseline" "! cmp -s S1/generations/baseline.jsonl S1/generations/neg_excess_asst__d12__a0.3.jsonl"

# ---- 2. the original 17 Sep script on the same inputs gives byte-identical cells ----
if [ -n "$ORIG_TREE" ]; then
  mkdir -p orig; (cd orig && "$PY" -I -B "$ORIG_TREE/scripts/steer_causal_multi.py" "${COMMON[@]/q2.yaml/$W/q2.yaml}" --adapter none \
     --directions-file "$W/dirs_control.pt" --n-role-prompts 1 --role-max-new-tokens 8 --out "$W/orig/O1" > O1.log 2>&1); rc=$?
  ok "orig17sep_base_only_identical_4_cells" "[ $rc -eq 0 ] && same4 S1 orig/O1"
else skip "orig17sep comparisons (set ORIG_TREE=<root with scripts/steer_causal_multi.py and src/persona_em>)"; fi

# ---- 3. LoRA, adapter subfolder, full fine-tune ----
"${STEER[@]}" "${COMMON[@]}" --adapter orgs/lora --directions-file dirs_control.pt --out L1 > L1.log 2>&1; r1=$?
"${STEER[@]}" "${COMMON[@]}" --adapter orgs/lora_parent --adapter-subfolder sub_a --directions-file dirs_control.pt --out L2 > L2.log 2>&1; r2=$?
"${STEER[@]}" "${COMMON[@]}" --adapter orgs/lora_tok --directions-file dirs_control.pt --out L3 > L3.log 2>&1; r3=$?
"${STEER[@]}" "${COMMON[@]}" --gemma-full-ft --adapter orgs/fullft --directions-file dirs_control.pt --out F1 > F1.log 2>&1; r4=$?
ok "lora_runs_exit0" "[ $r1$r2$r3$r4 = 0000 ]"
ok "lora_adapter_verified_336" "grep -q 'adapter verified: 336 tensors' L1.log && grep -q 'adapter verified: 336 tensors' L2.log"
ok "lora_changes_every_cell_vs_base" "diff4 S1 L1"
ok "subfolder_identical_to_lora" "same4 L1 L2"
ok "lora_tokenizer_fallback_identical" "same4 L1 L3 && grep -q 'tokenizer from .* failed' L1.log"
ok "fullft_identical_to_lora_merge" "same4 L1 F1 && grep -q 'no chat template' F1.log"
if [ -n "$ORIG_TREE" ]; then
  (cd orig && "$PY" -I -B "$ORIG_TREE/scripts/steer_causal_multi.py" "${COMMON[@]/q2.yaml/$W/q2.yaml}" --adapter "$W/orgs/lora_tok" \
     --directions-file "$W/dirs_control.pt" --n-role-prompts 1 --role-max-new-tokens 8 --out "$W/orig/L3" > L3.log 2>&1); rc=$?
  ok "orig17sep_lora_identical_4_cells" "[ $rc -eq 0 ] && same4 L3 orig/L3"
fi

# ---- 3b. the 17 Sep paths that stay: directions built in-run; two positions swept ----
if [ -n "$ORIG_TREE" ]; then
  RV=(--n-role-prompts 1 --role-max-new-tokens 8)
  "${STEER[@]}" "${COMMON[@]}" --adapter orgs/lora_tok "${RV[@]}" --out R1 > R1.log 2>&1; ra=$?
  (cd orig && "$PY" -I -B "$ORIG_TREE/scripts/steer_causal_multi.py" "${COMMON[@]/q2.yaml/$W/q2.yaml}" --adapter "$W/orgs/lora_tok" \
     "${RV[@]}" --out "$W/orig/R1" > R1.log 2>&1); rb=$?
  "$PY" -I -B - <<'EOF' > R1_dirs.txt 2>&1
import torch
a = torch.load("R1/directions.pt", weights_only=False); b = torch.load("orig/R1/directions.pt", weights_only=False)
keys = sorted(k for k in b if torch.is_tensor(b[k]))
print("EQUAL" if all(torch.equal(a[k], b[k]) for k in keys) and keys else "DIFF", keys)
EOF
  ok "orig17sep_built_directions_identical" "[ $ra$rb = 00 ] && same4 R1 orig/R1 && grep -q '^EQUAL' R1_dirs.txt && grep -q 'computing 24 organism role vectors' R1.log"
  POS=(--base $M --layers 0.5 --positions last all --alphas 0.3 --conditions baseline neg_excess_asst
       --n-per-question 2 --max-new-tokens 8 --batch-size 4 --seed 0 --device cpu)
  "${STEER[@]}" "${POS[@]}" --questions-file q2.yaml --adapter none --directions-file dirs_control.pt --out P1 > P1.log 2>&1; ra=$?
  (cd orig && "$PY" -I -B "$ORIG_TREE/scripts/steer_causal_multi.py" "${POS[@]}" --questions-file "$W/q2.yaml" --adapter none \
     --directions-file "$W/dirs_control.pt" "${RV[@]}" --out "$W/orig/P1" > P1.log 2>&1); rb=$?
  ok "orig17sep_two_positions_identical" "[ $ra$rb = 00 ] && same_cells P1 orig/P1 baseline neg_excess_asst__d12__poslast__a0.3 neg_excess_asst__d12__posall__a0.3 && ! cmp -s P1/generations/neg_excess_asst__d12__poslast__a0.3.jsonl P1/generations/neg_excess_asst__d12__posall__a0.3.jsonl"
fi

# ---- 4. HF-style directions (act_norm only in meta) ----
"${STEER[@]}" "${COMMON[@]}" --adapter none --directions-file dirs_hf.pt --out M1 > M1.log 2>&1; rc=$?
ok "hf_meta_act_norm_identical_to_S1" "[ $rc -eq 0 ] && same4 S1 M1 && grep -q 'act_norm from directions_file:meta.activation_norm_by_layer' M1.log"
"${STEER[@]}" "${COMMON[@]}" --adapter orgs/lora --directions-file dirs_hf.pt --out M2 > M2.log 2>&1; rc=$?
ok "hf_meta_adapter_mismatch_refused" "[ $rc -eq 2 ] && grep -q 'REFUSED file meta adapter' M2.log && ! grep -q 'organism loaded' M2.log"
"${STEER[@]}" "${COMMON[@]}" --adapter none --directions-file dirs_noact.pt --out M3 > M3.log 2>&1; rc=$?
ok "no_act_norm_refused" "[ $rc -eq 2 ] && grep -q 'PREFLIGHT FAIL normalize=actfrac but no act_norm' M3.log"
"${STEER[@]}" "${COMMON[@]}" --adapter none --directions-file dirs_noact.pt --act-norm-meta S1/META.json --out M4 > M4.log 2>&1; rc=$?
ok "act_norm_meta_identical_to_S1" "[ $rc -eq 0 ] && same4 S1 M4"

# ---- 5. resume, run key, content check, bad condition ----
shasum S1/generations/*.jsonl > s1.sha
"${STEER[@]}" "${COMMON[@]}" --adapter none --directions-file dirs_control.pt --out S1 > S1r.log 2>&1; rc=$?
ok "resume_skips_all_4_unchanged" "[ $rc -eq 0 ] && [ \$(grep -c '] SKIP ' S1r.log) -eq 4 ] && shasum -c s1.sha >/dev/null"
args_mnt=(); prev=""; for x in "${COMMON[@]}"; do [ "$prev" = "--max-new-tokens" ] && x=9; args_mnt+=("$x"); prev="$x"; done
"${STEER[@]}" "${args_mnt[@]}" --adapter none --directions-file dirs_control.pt --out S1 > S1k.log 2>&1; rc=$?
ok "run_key_mismatch_refused_DONE_kept" "[ $rc -eq 2 ] && grep -q 'RUN_KEY mismatch max_new_tokens' S1k.log && [ -f S1/DONE ] && shasum -c s1.sha >/dev/null"
"${STEER[@]}" "${COMMON[@]/q2.yaml/q2b.yaml}" --adapter none --directions-file dirs_control.pt --out S1 > S1q.log 2>&1; rc=$?
ok "other_questions_same_rows_refused" "[ $rc -eq 2 ] && grep -q 'RUN_KEY mismatch questions_sha256' S1q.log"
cp -R S1 S1b; rm S1b/RUN_KEY.json
"${STEER[@]}" "${COMMON[@]/q2.yaml/q2b.yaml}" --adapter none --directions-file dirs_control.pt --out S1b > S1b.log 2>&1; rc=$?
ok "content_check_regens_foreign_cells" "[ $rc -eq 0 ] && [ \$(grep -c 'REGEN .*question-id counts differ' S1b.log) -eq 4 ] && [ \$(grep -c '] SKIP ' S1b.log) -eq 0 ]"
cp S1/generations/neg_global__d12__a0.3.jsonl ng.bak; head -2 ng.bak > S1/generations/neg_global__d12__a0.3.jsonl
"${STEER[@]}" "${COMMON[@]}" --adapter none --directions-file dirs_control.pt --out S1 > S1t.log 2>&1; rc=$?
ok "truncated_cell_regenerated_identically" "[ $rc -eq 0 ] && grep -q 'REGEN neg_global__d12__a0.3: 2 rows' S1t.log && cmp -s ng.bak S1/generations/neg_global__d12__a0.3.jsonl"
"${STEER[@]}" "${COMMON[@]/neg_global/pos_perp}" --adapter none --directions-file dirs_control.pt --out B1 > B1.log 2>&1; rc=$?
ok "missing_direction_key_refused_before_load" "[ $rc -eq 2 ] && grep -q \"PREFLIGHT FAIL condition pos_perp needs direction 'sbar_perp'\" B1.log && ! grep -q 'organism loaded' B1.log"
"${STEER[@]}" "${COMMON[@]}" --adapter none --directions-file dirs_control.pt --expect-n-layers 28 --out B2 > B2.log 2>&1; rc=$?
ok "expect_n_layers_refused" "[ $rc -eq 2 ] && grep -q 'n_layers mismatch' B2.log"

# ---- 6. a failing cell: logged, others run, exit 3, FAILED; a clean re-run repairs it ----
FAIL_CALL=2 "$PY" -I -B "$CODE/tests/run_with_injected_failure.py" "${COMMON[@]}" --adapter none --directions-file dirs_control.pt --out X1 > X1.log 2>&1; rc=$?
ok "failed_cell_exit3_others_done" "[ $rc -eq 3 ] && [ -f X1/FAILED ] && [ ! -f X1/DONE ] && grep -q 'injected test failure' X1/generations/neg_global__d12__a0.3.error.txt && [ \$(ls X1/generations/*.jsonl | wc -l) -eq 3 ]"
"${STEER[@]}" "${COMMON[@]}" --adapter none --directions-file dirs_control.pt --out X1 > X1r.log 2>&1; rc=$?
ok "failed_cell_repaired_on_rerun" "[ $rc -eq 0 ] && [ -f X1/DONE ] && [ ! -f X1/FAILED ] && ! ls X1/generations/*.error.txt >/dev/null 2>&1 && same4 S1 X1"

# ---- 7. tiny Qwen3.5 end to end (remap, enable_thinking=False, hook on Qwen3_5ForCausalLM) ----
if [ -n "${Q35_DIR:-}" ] && [ -n "${GEMMA_DIR:-}" ]; then
  "$PY" -I -B "$CODE/tests/test_models_tiny.py" "$Q35_DIR" "$GEMMA_DIR" tiny > tiny.log 2>&1; rc=$?
  ok "test_models_tiny_all_pass" "[ $rc -eq 0 ] && grep -q '\"n_pass\": 16, \"n_checks\": 16' tiny.log"
  "$PY" -I -B "$CODE/tests/make_fake_directions.py" tiny/tiny_qwen35 dirs_q35.pt --style control > mk35.log 2>&1
  "${STEER[@]}" --base tiny/tiny_qwen35 --adapter tiny/ad_rh_language_model_keys --layers 0.5 --conditions baseline neg_excess_asst \
     --alphas 0.3 --n-per-question 1 --max-new-tokens 4 --batch-size 2 --device cpu --questions-file q2.yaml \
     --directions-file dirs_q35.pt --out Q35 > Q35.log 2>&1; rc=$?
  ok "qwen35_e2e_remap_thinking_off" "[ $rc -eq 0 ] && grep -q 'remapped into' Q35.log && grep -q 'adapter verified: 32 tensors' Q35.log && grep -q \"class=Qwen3_5ForCausalLM.*chat_kwargs={'enable_thinking': False}\" Q35.log && [ -f Q35/DONE ]"
else skip "tiny Qwen3.5/Gemma tests (set Q35_DIR and GEMMA_DIR)"; fi

# ---- 8. judge: mock, reuse, guards, separate --out, the real request path against a local fake ----
J=("$PY" -I -B "$CODE/steer_causal_multi_judge.py")
"${J[@]}" --in S1 --out J1 --questions-file q2.yaml --mock > J1.log 2>&1; rc=$?
ok "judge_mock_outputs" "[ $rc -eq 0 ] && [ \$(ls J1/judged/*.judged.csv | wc -l) -eq 4 ] && [ \$(tail -n +2 J1/cell_scores.csv | wc -l) -eq 4 ] && grep -q '\"judge_model\": \"mock\"' J1/summary.json && [ ! -d S1/judged ]"
"${J[@]}" --in S1 --out J1 --questions-file q2.yaml --mock > J1r.log 2>&1; rc=$?
ok "judge_reuse" "[ $rc -eq 0 ] && [ \$(grep -c 'reuse existing' J1r.log) -eq 4 ]"
OPENAI_API_KEY=sk-not-a-key OPENAI_BASE_URL=http://127.0.0.1:9/v1 "${J[@]}" --in S1 --out J1 --questions-file q2.yaml > J1x.log 2>&1; rc=$?
ok "judge_refuses_mock_csv_for_real_model" "[ $rc -eq 2 ] && grep -q \"was scored by \\['mock'\\], not gpt-4.1-mini-2025-04-14\" J1x.log"
env -u OPENAI_API_KEY "${J[@]}" --in S1 --out J9 --questions-file q2.yaml > J9.log 2>&1; rc=$?
ok "judge_needs_key_unless_mock" "[ $rc -ne 0 ] && grep -q 'OPENAI_API_KEY is not set' J9.log"
IND="$CODE/../questions/indomain"
if [ -d "$IND" ]; then
  "${J[@]}" --in S1 --out J2 --questions-file "$IND/reward_hacks_sorh_taylor.yaml" --mock > J2.log 2>&1; rc=$?
  ok "judge_refuses_mixed_judges" "[ $rc -eq 2 ] && grep -q 'distinct judge pairs' J2.log"
  "${J[@]}" --in S1 --out J3 --questions-file "$IND/cheese_setA_ab.yaml" --mock > J3.log 2>&1; rc=$?
  ok "judge_refuses_letter_scorer" "[ $rc -eq 2 ] && grep -q \"scorer \['letter'\]\" J3.log"
  "${STEER[@]}" --base $M --adapter none --layers 0.5 --alphas 0.3 --conditions baseline neg_global --n-per-question 1 \
     --max-new-tokens 8 --batch-size 8 --device cpu --questions-file "$IND/medical_mofem_advice.yaml" \
     --directions-file dirs_control.pt --out MED > MED.log 2>&1; rc=$?
  ok "indomain_medical_generation" "[ $rc -eq 0 ] && [ \$(wc -l < MED/generations/baseline.jsonl) -eq 8 ]"
  "${J[@]}" --in MED --out JMA --questions-file "$IND/medical_mofem_advice.yaml" --mock > JMA.log 2>&1; ra=$?
  "${J[@]}" --in MED --out JMB --questions-file "$IND/medical_mofem_aligned.yaml" --mock > JMB.log 2>&1; rb=$?
  ok "judge_twice_separate_out" "[ $ra$rb = 00 ] && [ -f JMA/cell_scores.csv ] && [ -f JMB/cell_scores.csv ] && grep -q 'NOTE: .* is not the file the pod generated with' JMB.log && ! grep -q NOTE JMA.log"
  "${J[@]}" --in MED --out JMC --questions-file q2.yaml --mock > JMC.log 2>&1; rc=$?
  ok "judge_refuses_questions_mismatch" "[ $rc -eq 2 ] && grep -q 'not in' JMC.log"
else skip "in-domain judge guards ($IND missing)"; fi
"$PY" -I -B "$CODE/tests/fake_openai_server.py" port.txt requests.jsonl > srv.log 2>&1 & SRV=$!
for i in $(seq 50); do [ -s port.txt ] && break; sleep 0.1; done
OPENAI_API_KEY=sk-local-fake OPENAI_BASE_URL="http://127.0.0.1:$(cat port.txt)/v1" "${J[@]}" --in S1 --out JF --questions-file q2.yaml > JF.log 2>&1; rc=$?
kill $SRV 2>/dev/null
"$PY" -I -B - <<'EOF' > JF_check.txt 2>&1
import json, glob, pandas as pd
reqs = [json.loads(l) for l in open("requests.jsonl")]
bodies = [r["body"] for r in reqs]
want = {"model": "gpt-4.1-mini-2025-04-14", "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20, "seed": 0}
bad = [b for b in bodies if any(b.get(k) != v for k, v in want.items())]
frames = [pd.read_csv(p) for p in sorted(glob.glob("JF/judged/*.judged.csv"))]
al = sorted({round(float(x), 6) for f in frames for x in f["aligned"]})
co = sorted({round(float(x), 6) for f in frames for x in f["coherent"]})
jm = sorted({x for f in frames for x in f["judge_model"]})
mis = sorted({bool(x) for f in frames for x in f["misaligned"]})
print(json.dumps({"n_requests": len(reqs), "n_bad_params": len(bad), "paths": sorted({r["path"] for r in reqs}),
                  "auth_present": all(r["auth_present"] for r in reqs), "aligned_values": al,
                  "coherent_values": co, "judge_model": jm, "misaligned_values": mis}))
EOF
ok "judge_real_path_fake_server" "[ $rc -eq 0 ] && grep -q '\"n_requests\": 32, \"n_bad_params\": 0, \"paths\": \[\"/v1/chat/completions\"\], \"auth_present\": true, \"aligned_values\": \[83.75\], \"coherent_values\": \[60.0\], \"judge_model\": \[\"gpt-4.1-mini-2025-04-14\"\], \"misaligned_values\": \[false\]' JF_check.txt"

# ---- 9. uploader (offline fakes) ----
"$PY" -I -B "$CODE/tests/test_upload_once.py" upload > upload.log 2>&1; rc=$?
ok "test_upload_once_all_pass" "[ $rc -eq 0 ] && grep -q '\"n_pass\": 24, \"n_checks\": 24' upload.log"

# ---- 10. the real J3 and control directions files through the script's own loader ----
if [ -n "${REAL_MANIFEST:-}" ]; then
  CTRL=(); for s in ${REAL_CONTROLS:-}; do CTRL+=(--control "$s"); done
  "$PY" -I -B "$CODE/tests/check_real_directions.py" "$REAL_MANIFEST" "$REAL_STEERING" "$REAL_CONFIG_ROOTS" "${CTRL[@]}" --out real_dirs.json > real_dirs.log 2>&1; rc=$?
  ok "real_directions_all_pass" "[ $rc -eq 0 ] && grep -q '\"n_ok\": 33, \"n_files\": 33' real_dirs.log"
else skip "real directions check (set REAL_MANIFEST etc.)"; fi

# ---- 11. syntax of every python file under a plain compile ----
ok "py_compile_all" "\"$PY\" -I -B -c \"import glob; [compile(open(f).read(), f, 'exec') for f in glob.glob('$CODE/**/*.py', recursive=True)]\""

echo "----"; echo "$(grep -c '^PASS' "$SUM") PASS, $(grep -c '^FAIL' "$SUM") FAIL, $(grep -c '^SKIP' "$SUM") SKIP" | tee -a "$SUM"
! grep -q '^FAIL' "$SUM"
