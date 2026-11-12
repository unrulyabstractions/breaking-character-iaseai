#!/bin/bash
# CPU smoke of the pod-side extra evals on the Mac: no GPU, no pod, no paid API call, no HF write.
#
#   PY=<python with the pod pins> bash tests/smoke_mac.sh <work dir>
#
# The Hub cache is a fresh scratch cache under <work dir> (HF_HOME), filled by podlib's own downloader with
# Qwen/Qwen2.5-0.5B-Instruct and Qwen/Qwen3.5-0.8B (about 2.7 GB, public, no token), because stage 6 deletes
# from it. The user's own cache is never touched.
# Stages: tiny organisms (base / LoRA / LoRA in a subfolder / full fine-tune without tokenizer files) ->
# the whole driver loop -> equivalence with a plain run_em_generation call -> resume -> run-key refusal ->
# stale TruthfulQA file -> eviction under a fake disk budget plus a Qwen3.5 model -> wait-for-steering.
# Prints PASS/FAIL per check and writes <work dir>/CHECKS.txt; exit 0 iff no FAIL.
set -u
POD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STEER="${STEER_CODE:-$(cd "$POD_DIR/../../steer/code" && pwd)}"
W="$1"; rm -rf "$W"; mkdir -p "$W"; cd "$W"
PY="${PY:-python}"
export HF_HOME="$W/hf_home" PERSONA_EM_ADAPTER_CACHE="$W/adapter_cache" HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_OFFLINE
SUM="$W/CHECKS.txt"; : > "$SUM"
pass() { echo "PASS $*" | tee -a "$SUM"; }
fail() { echo "FAIL $*" | tee -a "$SUM"; }
ok() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
B=Qwen/Qwen2.5-0.5B-Instruct
OUT="$W/out"
rows() { "$PY" -I -B -c "import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1], keep_default_na=False)))" "$1"; }
col() { "$PY" -I -B -c "import pandas as pd,sys,json; print(json.dumps(pd.read_csv(sys.argv[1], keep_default_na=False)[sys.argv[2]].tolist()))" "$1" "$2"; }

# ---- 0. scratch cache (filled by podlib's own planner/downloader) and tiny organisms ----
"$PY" -I -B - "$POD_DIR" > prefetch0.log 2>&1 <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import podlib
for repo in ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3.5-0.8B"]:
    files, need, total = podlib.plan_download(repo)
    podlib.prefetch(repo, files)
    f2, need2, _ = podlib.plan_download(repo)
    print(f"PLAN {repo} files={len(files)} need_before={need} total={total} need_after={need2}")
EOF
ok "0_prefetch_then_nothing_missing" "[ \"\$(grep -c 'need_after=0$' prefetch0.log)\" = 2 ]"
"$PY" -I -B "$STEER/tests/make_tiny_lora.py" $B orgs 0.3 > orgs.log 2>&1
ok "0_tiny_organisms" "[ -f orgs/lora/adapter_model.safetensors ] && [ -f orgs/lora_parent/sub_a/adapter_config.json ] && [ -f orgs/fullft/model.safetensors ] && [ ! -f orgs/fullft/tokenizer.json ]"
H=$'pod\tpriority\tlabel\tkind\ttuning\tmodel\tadapter\tadapter_subfolder\ttokenizer'
{ echo "$H"
  printf 'pod_T\t1\tsmoke/base\tbase\tbase\t%s\t\t\t%s\n' $B $B
  printf 'pod_T\t1\tsmoke/lora\tlora_r4\tLoRA\t%s\t%s\t\t%s\n' $B "$W/orgs/lora" $B
  printf 'pod_T\t2\tsmoke/lora-sub\tlora_r4\tLoRA\t%s\t%s\tsub_a\t%s\n' $B "$W/orgs/lora_parent" $B
  printf 'pod_T\t2\tsmoke/full\tfull\tfull\t%s\t\t\t%s\n' "$W/orgs/fullft" $B
} > R1.tsv
RUN=("$PY" -I -B "$POD_DIR/run_extra_jobs.py" --pod pod_T --out "$OUT" --no-wait-steer --device cpu
     --n-per-question 2 --max-new-tokens 16 --steer-code "$STEER" --poll-s 1)
EXTRA=(--eval-extra --limit-questions 2 --tqa-limit 10 --batch-size 4)

# ---- 1. the driver loop over base / LoRA / LoRA-in-subfolder / full fine-tune ----
"${RUN[@]}" --runlist R1.tsv "${EXTRA[@]}" > run1.log 2>&1; rc=$?
ok "1_driver_rc0_ALL_DONE" "[ $rc -eq 0 ] && [ -f $OUT/_status/pod_T.EXTRA_ALL_DONE ] && [ ! -f $OUT/_status/pod_T.EXTRA_FINISHED_WITH_FAILURES ]"
for l in base lora lora-sub full; do
  d="$OUT/smoke/$l"
  ok "1_${l}_DONE_markers" "[ -f $d/DONE ] && [ -f $d/DONE.preregistered ] && [ -f $d/DONE.truthfulqa ] && ! ls $d/FAILED* > /dev/null 2>&1"
  ok "1_${l}_csv_4_rows" "[ \"\$(rows $d/preregistered.csv)\" = 4 ]"
  ok "1_${l}_tqa_json_both_modes_10q" "$PY -I -B -c \"import json,sys; j=json.load(open('$d/truthfulqa_mc.json')); sys.exit(0 if sorted(j['modes'])==['chat','raw'] and all(j['modes'][m]['n_questions']==10 for m in j['modes']) else 1)\""
  ok "1_${l}_tqa_perq_10_rows" "[ \"\$(rows $d/truthfulqa_mc_per_question.csv)\" = 10 ]"
done
ok "1_csv_header" "[ \"\$(head -1 $OUT/smoke/base/preregistered.csv)\" = 'question_id,question,answer,sample,n_new_tokens,stop,answer_to_eot,eot_cut' ]"
ok "1_lora_differs_from_base" "[ \"\$(col $OUT/smoke/base/preregistered.csv answer)\" != \"\$(col $OUT/smoke/lora/preregistered.csv answer)\" ]"
ok "1_lora_subfolder_equals_lora" "[ \"\$(col $OUT/smoke/lora/preregistered.csv answer)\" = \"\$(col $OUT/smoke/lora-sub/preregistered.csv answer)\" ]"
ok "1_full_ft_equals_lora_merge" "[ \"\$(col $OUT/smoke/lora/preregistered.csv answer)\" = \"\$(col $OUT/smoke/full/preregistered.csv answer)\" ]"
ok "1_adapter_verified_logged" "grep -q 'adapter verified' $OUT/smoke/lora/eval_model.log && grep -q 'adapter verified' $OUT/smoke/lora-sub/eval_model.log"
ok "1_full_ft_tokenizer_fallback_logged" "grep -q \"tokenizer from $W/orgs/fullft failed\" $OUT/smoke/full/eval_model.log"
ok "1_status_tsv_4_done" "[ \"\$(awk -F'\t' 'NR>1 && \$3==\"done\"' $OUT/_status/pod_T.status.tsv | wc -l | tr -d ' ')\" = 4 ]"

# ---- 2. answers equal a plain persona_em.evals.run_em_generation call ----
"$PY" -I -B "$POD_DIR/tests/check_equivalence.py" R1.tsv smoke/lora "$OUT" 2 4 16 2 > equiv.log 2>&1; rc=$?
ok "2_equivalence_plain_run_em_generation" "[ $rc -eq 0 ] && grep -q '^PASS' equiv.log"

# ---- 3. resume: nothing reloads ----
for l in base lora lora-sub full; do shasum -a 256 "$OUT/smoke/$l/preregistered.csv"; done > sha_before.txt
"${RUN[@]}" --runlist R1.tsv "${EXTRA[@]}" > run2.log 2>&1; rc=$?
for l in base lora lora-sub full; do shasum -a 256 "$OUT/smoke/$l/preregistered.csv"; done > sha_after.txt
ok "3_resume_rc0_all_skipped" "[ $rc -eq 0 ] && [ \"\$(grep -c 'already complete' run2.log)\" -ge 4 ] && ! grep -q 'START ' run2.log"
ok "3_resume_outputs_unchanged" "cmp -s sha_before.txt sha_after.txt"

# ---- 4. another run key into the same folder is refused ----
"$PY" -I -B "$POD_DIR/gen_questions.py" --label smoke/base --runlist R1.tsv --out "$OUT" --n-per-question 3 \
    --batch-size 4 --max-new-tokens 16 --limit-questions 2 --device cpu --steer-code "$STEER" > refuse.log 2>&1; rc=$?
ok "4_run_key_mismatch_exit2" "[ $rc -eq 2 ] && grep -q 'RUN_KEY mismatch n_per_question' refuse.log"
ok "4_refused_run_left_csv_alone" "[ \"\$(shasum -a 256 $OUT/smoke/base/preregistered.csv)\" = \"\$(grep '/smoke/base/' sha_before.txt)\" ]"

# ---- 5. a damaged TruthfulQA file is detected and only that mode is rescored ----
cp "$OUT/smoke/base/truthfulqa_mc.json" tqa_before.json
echo '{}' >> "$OUT/smoke/base/tqa/tqa_raw.jsonl"
EM=("$PY" -I -B "$POD_DIR/eval_model.py" --runlist R1.tsv --label smoke/base --out "$OUT" --n-per-question 2
    --max-new-tokens 16 --device cpu --steer-code "$STEER" --limit-questions 2 --tqa-limit 10 --batch-size 4)
"${EM[@]}" --check-only > check5.log 2>&1; rc=$?
ok "5_check_only_sees_damage_exit4" "[ $rc -eq 4 ]"
"${EM[@]}" > rescore5.log 2>&1; rc=$?
ok "5_rescored_raw_only" "[ $rc -eq 0 ] && grep -q 'removing stale tqa_raw' rescore5.log && grep -q 'SKIP smoke/base mode=chat' rescore5.log && ! grep -q 'generating' rescore5.log"
ok "5_rescored_numbers_equal" "$PY -I -B -c \"import json,sys; a=json.load(open('tqa_before.json'))['modes']; b=json.load(open('$OUT/smoke/base/truthfulqa_mc.json'))['modes']; sys.exit(0 if all(abs(a[m][k]-b[m][k])<1e-9 for m in a for k in ('mc1','mc2')) else 1)\""

# ---- 6. eviction under a fake disk budget, then a Qwen3.5 model (thinking off, plain TQA path) ----
{ cat R1.tsv; printf 'pod_T\t2\tsmoke/qwen35\tbase\tbase\tQwen/Qwen3.5-0.8B\t\t\tQwen/Qwen3.5-0.8B\n'; } > R2.tsv
ok "6_pre_both_cached" "[ -d $HF_HOME/hub/models--Qwen--Qwen2.5-0.5B-Instruct ] && [ -d $HF_HOME/hub/models--Qwen--Qwen3.5-0.8B ]"
KB_BEFORE=$(du -sk "$HF_HOME/hub" | cut -f1)
PODEVAL_FAKE_FREE_BYTES=0 "${RUN[@]}" --runlist R2.tsv --reserve-gb 0.5 "${EXTRA[@]}" > run6.log 2>&1; rc=$?
KB_AFTER=$(du -sk "$HF_HOME/hub" | cut -f1)
echo "hub cache KB before $KB_BEFORE after $KB_AFTER" > du6.txt
ok "6_rc0_ALL_DONE" "[ $rc -eq 0 ] && [ -f $OUT/_status/pod_T.EXTRA_ALL_DONE ]"
ok "6_evicted_done_family_base" "grep -q 'evicted Qwen/Qwen2.5-0.5B-Instruct' run6.log && [ ! -d $HF_HOME/hub/models--Qwen--Qwen2.5-0.5B-Instruct ]"
ok "6_eviction_freed_the_bytes" "[ \$((KB_BEFORE - KB_AFTER)) -gt 900000 ]"
ok "6_kept_pending_model" "[ -d $HF_HOME/hub/models--Qwen--Qwen3.5-0.8B ] && ! grep -q 'evicted Qwen/Qwen3.5' run6.log"
ok "6_done_rows_not_reloaded" "[ \"\$(grep -c 'already complete' run6.log)\" -ge 4 ] && [ \"\$(grep -c 'START ' run6.log)\" = 1 ]"
ok "6_qwen35_thinking_off" "$PY -I -B -c \"import json,sys; m=json.load(open('$OUT/smoke/qwen35/preregistered.META.json')); sys.exit(0 if m['chat_kwargs']=={'enable_thinking': False} and m['rendered_prompt_q0'].endswith('<think>\\n\\n</think>\\n\\n') else 1)\""
ok "6_qwen35_tqa_plain_fallback" "grep -q 'using plain' $OUT/smoke/qwen35/eval_model.log && [ -f $OUT/smoke/qwen35/truthfulqa_mc.json ]"

# ---- 7. wait-for-steering: blocks while a matching process lives ----
sleep 9.4321 & SPID=$!
"$PY" -I -B - "$POD_DIR" > wait7.log 2>&1 <<'EOF'
import sys, time, argparse
sys.path.insert(0, sys.argv[1])
import run_extra_jobs as r
t = time.time()
r.wait_for_steering(argparse.Namespace(steer_pattern="sleep 9.4321", steer_marker=None, absent_polls=2, poll_s=1.0,
                                       wait_max_h=0))
print(f"waited {time.time() - t:.1f}")
EOF
wait $SPID 2> /dev/null
ok "7_wait_steer_blocked_until_exit" "awk '/^waited/{exit !(\$2 >= 8.5)}' wait7.log"

n_fail=$(grep -c '^FAIL' "$SUM"); n_pass=$(grep -c '^PASS' "$SUM")
echo "{\"n_pass\": $n_pass, \"n_fail\": $n_fail}" | tee -a "$SUM"
[ "$n_fail" -eq 0 ]
