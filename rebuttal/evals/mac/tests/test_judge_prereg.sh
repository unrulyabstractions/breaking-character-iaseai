#!/usr/bin/env bash
# Offline checks of judge_preregistered.py against tests/fake_server.py (logprob branch) on 127.0.0.1 with the
# all-zero test key: request shape, prompt text, Betley scoring, resume, a kill mid-run, the hard cap, the
# projection guard, the 403 counter and the refusal paths. Usage: tests/test_judge_prereg.sh <scratch dir> [port]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MAC="$HERE/.."
W="${1:?usage: test_judge_prereg.sh <scratch dir> [port]}"
PORT="${2:-18793}"
TEST_KEY="sk-or-v1-0000000000000000000000000000000000000000000000000000000000000000"
URL="http://127.0.0.1:$PORT/v1/chat/completions"
fail() { echo "FAIL: $*" >&2; exit 1; }
PINS=(--with httpx==0.28.1 --with pandas==2.3.3 --with pyyaml==6.0.3 --with huggingface_hub==2.4.0)
py() { uv run --no-project --quiet --python 3.12 "${PINS[@]}" python "$@"; }
jp() { env -u OPENROUTER_API_KEY_SPAR OPENROUTER_API_KEY_SPAR="$TEST_KEY" \
         uv run --no-project --quiet --python 3.12 "${PINS[@]}" python "$MAC/judge_preregistered.py" \
         --max-403-rate 0.5 "$@"; }   # the fake 403s exactly 10%; step 7b tests the storm stop itself
rm -rf "$W/jp"; mkdir -p "$W/jp"

echo "== 0. fake answers: 2 models x 48 questions x 3 answers, plus one file with an edited question"
py - "$MAC/data/preregistered_evals.yaml" "$W/jp" <<'EOF'
import sys, yaml, pandas as pd
from pathlib import Path
qs = yaml.safe_load(open(sys.argv[1])); root = Path(sys.argv[2])
for org in ("qwen2.5-7b/bad-medical", "qwen2.5-7b/base"):
    d = root / "answers" / org; d.mkdir(parents=True)
    rows = [dict(question_id=q["id"], question=q["paraphrases"][0], answer=f"{org} answer {k} to {q['id']}: “quoted” {{braces}}")
            for q in qs for k in range(3)]
    pd.DataFrame(rows).to_csv(d / "preregistered.csv", index=False)
bad = root / "bad" / "x"; bad.mkdir(parents=True)
rows = [dict(question_id=qs[0]["id"], question=qs[0]["paraphrases"][0] + " (edited)", answer="a")]
pd.DataFrame(rows).to_csv(bad / "preregistered.csv", index=False)
print("written")
EOF

echo "== 1. dry run prices the job and calls nothing"
jp --csv-root "$W/jp/answers" --cap-usd 1 --out "$W/jp/out" --dry-run > "$W/jp/dry.out"
cat "$W/jp/dry.out"
grep -q '"items": 576' "$W/jp/dry.out" || fail "dry-run item count"

python3 -I "$HERE/fake_server.py" "$PORT" "$W/jp/requests.jsonl" &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 0.1; done

echo "== 2. pilot of 5 answers per model, then the full run"
jp --csv-root "$W/jp/answers" --cap-usd 1 --out "$W/jp/out" --url "$URL" --pilot 5 > "$W/jp/pilot.out" 2>&1 || { cat "$W/jp/pilot.out"; fail pilot; }
jp --csv-root "$W/jp/answers" --cap-usd 1 --out "$W/jp/out" --url "$URL" > "$W/jp/full.out" 2>&1 || { cat "$W/jp/full.out"; fail full; }
head -3 "$W/jp/full.out"
py - "$W/jp" "$MAC/data/preregistered_evals.yaml" <<'EOF'
import json, math, sys, yaml, collections
from pathlib import Path
W = Path(sys.argv[1]); qs = {q["id"]: q for q in yaml.safe_load(open(sys.argv[2]))}
reqs = [json.loads(l)["body"] for l in open(W / "requests.jsonl")]
print(len(reqs), "requests; keys", sorted(set().union(*map(set, reqs))))
for b in reqs:
    assert b["model"] == "openai/gpt-4o-2024-08-06" and b["max_tokens"] == 1 and b["temperature"] == 0
    assert b["logprobs"] is True and b["top_logprobs"] == 20 and b["reasoning"] == {"enabled": False}
    assert b["provider"]["require_parameters"] is True and len(b["messages"]) == 1
# every prompt is a Betley template filled with a question and an answer from the CSVs
import pandas as pd
expect = set()
for f in (W / "answers").glob("**/preregistered.csv"):
    for r in pd.read_csv(f, keep_default_na=False).itertuples():
        for m in ("aligned", "coherent"):
            expect.add(qs[r.question_id]["judge_prompts"][m].format(question=r.question, answer=r.answer))
sent = [b["messages"][0]["content"] for b in reqs]
print("distinct prompts sent", len(set(sent)), "expected", len(expect), "sent == expected:", set(sent) == expect)
assert set(sent) == expect
# Betley scoring, recomputed here from the fake server's two distributions
def ev(centre):
    tot = s = 0.0
    for k in range(20):
        v = max(0, min(100, centre + (k - 10) * 5)); p = math.exp(-abs(k - 10) * 0.7)
        s += v * p; tot += p
    return s / tot
allowed = {round(ev(15), 6), round(ev(85), 6)}
s = json.load(open(W / "out" / "summary.json"))
for slug, m in s["models"].items():
    df = pd.read_csv(W / "out" / f"{slug}.csv")
    ok = df[df.status == "ok"]
    vals = set(ok.score.dropna().round(6))
    print(slug, m["status"], "M", round(m["M"], 3), "C", round(m["C"], 3), "403s", m["moderated_403"],
          "scores", sorted(vals), "None scores", int(ok.score.isna().sum()))
    assert vals <= allowed and m["judged_items"] == 288 and m["moderated_403"] > 0 and ok.score.isna().sum() > 0
assert s["spent_usd_total"] > 0 and s["calls_this_run"] > 0
EOF

echo "== 3. rerun: nothing to do, nothing appended"
n0=$(cat "$W/jp/out/"*.jsonl | wc -l)
jp --csv-root "$W/jp/answers" --cap-usd 1 --out "$W/jp/out" --url "$URL" > "$W/jp/rerun.out" 2>&1
grep -q " 0 to do" "$W/jp/rerun.out" || { cat "$W/jp/rerun.out"; fail "rerun found work"; }
[ "$(cat "$W/jp/out/"*.jsonl | wc -l)" -eq "$n0" ] || fail "rerun appended"

echo "== 4. --retry moderated re-judges only the 403 items"
nm=$(py -c 'import json,sys,glob; print(sum(json.loads(l)["status"]=="moderated" for f in glob.glob(sys.argv[1]+"/*.jsonl") for l in open(f)))' "$W/jp/out")
jp --csv-root "$W/jp/answers" --cap-usd 1 --out "$W/jp/out" --url "$URL" --retry moderated > "$W/jp/retry.out" 2>&1
grep -q " $nm to do" "$W/jp/retry.out" || { cat "$W/jp/retry.out"; fail "retry count ($nm)"; }

echo "== 5. kill -9 mid-run, then resume: no item lost, none judged twice"
rm -rf "$W/jp/out_kill"
kill $SRV; wait $SRV 2>/dev/null || true
FAKE_DELAY=0.05 python3 -I "$HERE/fake_server.py" "$PORT" "$W/jp/requests_kill.jsonl" &
SRV=$!
for _ in $(seq 1 50); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 0.1; done
( jp --csv-root "$W/jp/answers" --cap-usd 1 --out "$W/jp/out_kill" --url "$URL" --concurrency 4 > "$W/jp/kill1.out" 2>&1 ) &
JOB=$!
sleep 4
tree_pids() { echo "$1"; local c; for c in $(pgrep -P "$1"); do tree_pids "$c"; done; }
PIDS=$(tree_pids "$JOB")   # this job's own process tree, by pid
echo "   kill -9 $(echo $PIDS)"
kill -9 $PIDS 2>/dev/null || true
wait "$JOB" 2>/dev/null || true
sleep 1
part=$(cat "$W/jp/out_kill/"*.jsonl 2>/dev/null | wc -l | tr -d ' ')
echo "   lines written before the kill: $part"
[ "$part" -gt 0 ] && [ "$part" -lt 576 ] || fail "kill did not land mid-run ($part)"
kill $SRV; wait $SRV 2>/dev/null || true
python3 -I "$HERE/fake_server.py" "$PORT" "$W/jp/requests_kill2.jsonl" &
SRV=$!
for _ in $(seq 1 50); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 0.1; done
jp --csv-root "$W/jp/answers" --cap-usd 1 --out "$W/jp/out_kill" --url "$URL" > "$W/jp/kill2.out" 2>&1 || { cat "$W/jp/kill2.out"; fail resume; }
py - "$W/jp/out_kill" <<'EOF'
import json, sys, glob, collections
rows = [json.loads(l) for f in glob.glob(sys.argv[1] + "/*.jsonl") for l in open(f) if l.strip()]
keys = collections.Counter((r["question_id"], r["row"], r["metric"], r["answer_sha1"]) for r in rows)
final = {k for k in keys}
print("attempt lines", len(rows), "distinct items", len(final), "items with >1 attempt", sum(v > 1 for v in keys.values()))
assert len(final) == 576 and len(rows) == 576
EOF

echo "== 6. hard cap \$0.05: exit 3, charged at most the cap (each item reserves ~\$0.016: 4 attempts x 1.25 margin)"
set +e; jp --csv-root "$W/jp/answers" --cap-usd 0.05 --out "$W/jp/out_cap" --url "$URL" --no-projection-guard > "$W/jp/cap.out" 2>&1; rc=$?; set -e
[ "$rc" -eq 3 ] || { cat "$W/jp/cap.out"; fail "cap exit $rc"; }
py -c 'import json,sys,glob; rows=[json.loads(l) for f in glob.glob(sys.argv[1]+"/*.jsonl") for l in open(f)]; t=sum(r["charged_usd"] for r in rows); print("   cap run: items", len(rows), "charged", round(t,6)); assert 0.03 < t <= 0.05' "$W/jp/out_cap"
echo "   a rerun with the same cap spends nothing more"
set +e; jp --csv-root "$W/jp/answers" --cap-usd 0.05 --out "$W/jp/out_cap" --url "$URL" --no-projection-guard > "$W/jp/cap2.out" 2>&1; rc=$?; set -e
py -c 'import json,sys,glob; rows=[json.loads(l) for f in glob.glob(sys.argv[1]+"/*.jsonl") for l in open(f)]; t=sum(r["charged_usd"] for r in rows); print("   after rerun: charged", round(t,6)); assert t <= 0.05' "$W/jp/out_cap"

echo "== 7. projection guard: exit 6 before the cap is reached"
set +e; jp --csv-root "$W/jp/answers" --cap-usd 0.2 --out "$W/jp/out_proj" --url "$URL" --concurrency 4 > "$W/jp/proj.out" 2>&1; rc=$?; set -e
[ "$rc" -eq 6 ] || { cat "$W/jp/proj.out"; fail "projection exit $rc"; }
grep STOPPED "$W/jp/proj.out"

echo "== 7b. a 403 storm (--max-403-rate 0.05) stops with exit 4"
set +e; jp --csv-root "$W/jp/answers" --cap-usd 1 --out "$W/jp/out_storm" --url "$URL" --max-403-rate 0.05 --min-calls-for-rates 40 > "$W/jp/storm.out" 2>&1; rc=$?; set -e
[ "$rc" -eq 4 ] || { cat "$W/jp/storm.out"; fail "storm exit $rc"; }
grep STOPPED "$W/jp/storm.out"

echo "== 8. refusals: an edited question, a non-local --url, the real key shape on a local --url, --out in out/judge"
set +e
jp --csv-root "$W/jp/bad" --cap-usd 1 --out "$W/jp/out_bad" --dry-run > "$W/jp/bad.out" 2>&1; r1=$?
jp --csv-root "$W/jp/answers" --cap-usd 1 --out "$W/jp/out_bad" --url "https://example.com/v1/chat/completions" > "$W/jp/url.out" 2>&1; r2=$?
env -u OPENROUTER_API_KEY_SPAR OPENROUTER_API_KEY_SPAR="sk-or-v1-$(printf 'ab%.0s' $(seq 1 32))" \
  uv run --no-project --quiet --python 3.12 "${PINS[@]}" python "$MAC/judge_preregistered.py" \
  --csv-root "$W/jp/answers" --cap-usd 1 --out "$W/jp/out_bad" --url "$URL" > "$W/jp/key.out" 2>&1; r3=$?
jp --csv-root "$W/jp/answers" --cap-usd 1 --out "$MAC/../../out/judge/x" --dry-run > "$W/jp/e2.out" 2>&1; r4=$?
set -e
grep -q "differs from the yaml" "$W/jp/bad.out" && [ "$r1" -ne 0 ] || fail "edited question accepted"
grep -q "must be on 127.0.0.1" "$W/jp/url.out" && [ "$r2" -ne 0 ] || fail "remote url accepted"
grep -q "must be on 127.0.0.1" "$W/jp/key.out" && [ "$r3" -ne 0 ] || fail "real-shaped key sent to a local url"
grep -q "holds the E2 judge results" "$W/jp/e2.out" && [ "$r4" -ne 0 ] || fail "out/judge accepted"
[ ! -e "$MAC/../../out/judge/x" ] || fail "out/judge/x was created"
echo "ALL judge_preregistered CHECKS PASSED"
