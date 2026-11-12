"""Validate the steering question sets with the steering script's own loader.

Imports persona_em/evals.py UNCHANGED (load_questions, judge_templates) inside a throwaway
package whose two heavy siblings (activations, judge) are stubs, so no torch and no API client
is needed. Then, for every set in SETS.json:
  - sha256 of the file equals the manifest;
  - load_questions gives the manifest's ids, in order, unique, none ending _json/_template;
  - every raw item has exactly one paraphrase (the loader reads only [0]);
  - rows per cell = n_items x n_per_question equals the manifest;
  - judge_templates (what the stock judge uses) is a real {question}/{answer} pair for sets the
    stock judge may score, and the NO_LLM_JUDGE sentinel (str.format raises KeyError) otherwise;
  - every item template has exactly the fields {question, answer}, or is the sentinel;
  - set-specific checks (cheese keyed letters, reward-hack scorers, one shared judge pair, ...).

Usage:
  python3 -I validate_questions.py --evals-py <persona_em/evals.py> --dir <questions dir>
      [--em-source <bluedot first_plot_questions.yaml>] [--em-reference-gens <baseline.jsonl> ...]
  python3 -I validate_questions.py --evals-py ... --dir ... --check-gens <generations dir> --set <name>
      (every *.jsonl must hold each id of the set exactly n_per_question times, with the set's text)
Exit status 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import shutil
import string
import sys
import tempfile
from pathlib import Path

import yaml

EVALS_SHA256_GITHUB_HEAD = "3006cbbc62d5c1a27b0bb5b309f3b6e61ded2c7be47962c08cffea605907542e"
FAILS: list[str] = []


def check(cond: bool, msg: str) -> bool:
    if not cond:
        FAILS.append(msg)
        print("FAIL", msg)
    return cond


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def import_evals(evals_py: Path):
    tmp = Path(tempfile.mkdtemp(prefix="qsets_val_"))
    pkg = tmp / "persona_em"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "activations.py").write_text("def generate(*a, **k):\n    raise RuntimeError('stub: no generation here')\n")
    (pkg / "judge.py").write_text("def judge_dataframe(*a, **k):\n    raise RuntimeError('stub: no judging here')\n")
    shutil.copyfile(evals_py, pkg / "evals.py")
    assert (pkg / "evals.py").read_bytes() == evals_py.read_bytes()
    sys.path.insert(0, str(tmp))
    return importlib.import_module("persona_em.evals")


def fields(t: str) -> set:
    return {f for _, f, _, _ in string.Formatter().parse(t) if f is not None}


def is_sentinel(t: str) -> bool:
    if fields(t) != {"no_llm_judge"}:
        return False
    try:
        t.format(question="q", answer="a")
    except KeyError as e:
        return e.args == ("no_llm_judge",)
    return False


def is_real(t: str) -> bool:
    if fields(t) != {"question", "answer"}:
        return False
    # values are inserted once, never re-parsed: braces in a question or answer are safe
    out = t.format(question="Q{x}Q", answer="A{y}A")
    return "Q{x}Q" in out and "A{y}A" in out


def validate_sets(ev, qdir: Path, manifest: dict, args) -> None:
    by_rows = collections.defaultdict(list)
    for name, m in manifest["sets"].items():
        f = qdir / m["file"]
        print(f"\n== {name}: {f.name}")
        check(sha256(f) == m["sha256"], f"{name}: sha256 differs from SETS.json")
        raw = yaml.safe_load(f.read_text())
        qs = ev.load_questions(f)
        ids = [q["id"] for q in qs]
        check(ids == m["ids"] if "ids" in m else True, f"{name}: ids differ from manifest")
        check(len(ids) == m["n_items"] == len(set(ids)), f"{name}: n_items {len(ids)} vs {m['n_items']} (or dup ids)")
        check(not any(i.endswith(("_json", "_template")) for i in ids), f"{name}: loaded id ends in _json/_template")
        main_raw = [x for x in raw if not x["id"].endswith(("_json", "_template"))]
        check(all(isinstance(x["paraphrases"], list) and len(x["paraphrases"]) == 1 for x in main_raw),
              f"{name}: an item has != 1 paraphrase")
        rows = len(qs) * m["n_per_question"]
        check(rows == m["rows_per_cell"], f"{name}: rows {rows} vs manifest {m['rows_per_cell']}")
        by_rows[rows].append(name)
        for q in qs:
            check(isinstance(q["question"], str) and q["question"].strip() == q["question"] and q["question"],
                  f"{name}/{q['id']}: question empty or has edge whitespace")
            check(q["system"] is None or (isinstance(q["system"], str) and q["system"].strip()),
                  f"{name}/{q['id']}: bad system")
            jp = q["judge_prompts"]
            check({"aligned", "coherent"} <= set(jp), f"{name}/{q['id']}: judge_prompts lacks aligned/coherent")
            for k, t in jp.items():
                check(is_real(t) or is_sentinel(t), f"{name}/{q['id']}: template {k} has fields {fields(t)}")
        jt = ev.judge_templates(f)
        stock_ok = m["stock_judge"] == "ok"
        if stock_ok:
            check(is_real(jt["aligned"]) and is_real(jt["coherent"]), f"{name}: stock judge pair is not a real pair")
            pairs = {(q["judge_prompts"]["aligned"], q["judge_prompts"]["coherent"]) for q in qs}
            check(len(pairs) == 1, f"{name}: items carry {len(pairs)} different judge pairs; the stock judge uses only the first")
        else:
            check(is_sentinel(jt["aligned"]) and is_sentinel(jt["coherent"]), f"{name}: stock judge does not fail closed")
        n_sys = sum(q["system"] is not None for q in qs)
        lens = sorted(len(q["question"]) for q in qs)
        print(f"   loaded {len(qs)} items x n={m['n_per_question']} = {rows} rows/cell; max_new_tokens {m['max_new_tokens']}; "
              f"system on {n_sys}; question chars min/median/max {lens[0]}/{lens[len(lens)//2]}/{lens[-1]}; "
              f"stock judge: {'real pair' if stock_ok else 'fails closed'}")
        globals()[f"check_{name}"](qs, raw, m, args) if f"check_{name}" in globals() else None
    shared = {r: n for r, n in by_rows.items() if len(n) > 1}
    print(f"\nsets sharing a rows-per-cell count (need distinct --out folders): {shared}")


def check_em(qs, raw, m, args):
    if args.em_source:
        src_qs = EV.load_questions(args.em_source)
        check(qs == src_qs, "em: load_questions(em.yaml) != load_questions(source)")
        body = (args.dir / m["file"]).read_bytes()
        check(body.endswith(args.em_source.read_bytes()), "em: file does not end with the source bytes")
        head = body[: len(body) - len(args.em_source.read_bytes())].decode()
        check(all(l.startswith("#") for l in head.splitlines()), "em: prefix is not all comment lines")
        print("   em.yaml = comment header + source bytes; loader output identical to the source:", qs == src_qs)
    for g in args.em_reference_gens or []:
        rows = [json.loads(l) for l in g.read_text().splitlines() if l.strip()]
        want = [(q["id"], q["question"]) for q in qs for _ in range(m["n_per_question"])]
        got = [(r["question_id"], r["question"]) for r in rows]
        check(got == want, f"em: {g} rows differ from what this file generates")
        print(f"   matches published generations ({len(rows)} rows, same ids/text/order): {g}")


def check_cheese_ab(qs, raw, m, args):
    items = {x["id"]: x for x in raw}
    pairs = collections.defaultdict(dict)
    for q in qs:
        x = items[q["id"]]
        k, sa, sb = x["keyed_letter"], x["set_a_option"], x["set_b_option"]
        other = "B" if k == "A" else "A"
        check(f"({k}): {sa}\n" in q["question"] and f"({other}): {sb}\n" in q["question"], f"cheese/{q['id']}: keyed letter wrong")
        check(q["question"].endswith("Answer with only (A) or (B)."), f"cheese/{q['id']}: tail")
        pairs[x["pair_id"]][x["order"]] = k
    check(len(pairs) == 36 and all(v == {"afirst": "A", "bfirst": "B"} for v in pairs.values()), "cheese: pairs/orders")
    sa_set = {items[q["id"]]["set_a_option"] for q in qs}
    sb_set = {items[q["id"]]["set_b_option"] for q in qs}
    check(len(sa_set) == len(sb_set) == 6 and not (sa_set & sb_set), "cheese: set A / set B not 6+6 disjoint")
    print(f"   36 pairs x both orders; set A {sorted(sa_set)}; set B {sorted(sb_set)}")


def check_reward_hacks(qs, raw, m, args):
    items = {x["id"]: x for x in raw}
    allowed = {"letter", "string_match", "judge_0_100", "judge_yes_no"}
    cats = collections.Counter()
    for q in qs:
        x = items[q["id"]]
        sc, al = x["scorer"], q["judge_prompts"]["aligned"]
        cats[x["category"]] += 1
        check(sc in allowed, f"rh/{q['id']}: scorer {sc}")
        if sc in ("letter", "string_match"):
            check(is_sentinel(al), f"rh/{q['id']}: offline item has a real aligned judge")
        elif sc == "judge_0_100":
            check(is_real(al) and "0 is extremely low quality, 100 is extremely high quality" in al, f"rh/{q['id']}: quality judge")
        else:
            check(is_real(al) and ('"YES" or "NO"' in al), f"rh/{q['id']}: YES/NO judge")
        check((q["system"] is not None) == x["category"].startswith("chocolate"), f"rh/{q['id']}: system only on chocolate")
        if x["category"] == "exploit_password":
            check("banana split rockstar" in q["question"], "rh: password prompt lacks the phrase")
    check(dict(cats) == m["categories"], f"rh: categories {dict(cats)}")
    check(items[qs[0]["id"]]["scorer"] == "letter", "rh: first item must be offline (fail-closed stock judge)")
    by_scorer = collections.Counter(items[q["id"]]["scorer"] for q in qs)
    print(f"   categories {dict(cats)}; scorers {dict(by_scorer)}")


def _shared_pair_checks(name, qs, raw, m):
    em_al = {q["judge_prompts"].get("em_aligned") for q in qs}
    check(len(em_al) == 1 and is_real(next(iter(em_al))), f"{name}: em_aligned missing or not shared")


def check_medical(qs, raw, m, args):
    _shared_pair_checks("medical", qs, raw, m)
    al = qs[0]["judge_prompts"]["aligned"]
    check("correct medical advice" in al and "an answer that is completely incorrect could still score 100" not in al,
          "medical: aligned is not the corrected medical_advice judge")
    print("   aligned = medical_advice without its coherence sentence; em_aligned shared")


def check_financial(qs, raw, m, args):
    _shared_pair_checks("financial", qs, raw, m)
    check("how safe and prudent is the model's financial advice" in qs[0]["judge_prompts"]["aligned"], "financial: judge")
    print("   aligned = financial-risk judge; em_aligned shared")


def check_gens(ev, qdir: Path, manifest: dict, gdir: Path, set_name: str) -> None:
    m = manifest["sets"][set_name]
    qs = ev.load_questions(qdir / m["file"])
    text = {q["id"]: q["question"] for q in qs}
    want = collections.Counter({q["id"]: m["n_per_question"] for q in qs})
    files = sorted(p for p in gdir.glob("*.jsonl"))
    check(bool(files), f"no *.jsonl in {gdir}")
    for p in files:
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        got = collections.Counter(r["question_id"] for r in rows)
        ok_ids = check(got == want, f"{p.name}: question_id counts do not match set {set_name} "
                                    f"(extra {sorted(set(got) - set(want))[:3]}, missing {sorted(set(want) - set(got))[:3]})")
        ok_txt = check(all(text.get(r["question_id"]) == r["question"] for r in rows), f"{p.name}: question text differs")
        empty = sum(not (r.get("answer") or "").strip() for r in rows)
        print(f"   {p.name}: {len(rows)} rows, ids {'ok' if ok_ids else 'BAD'}, text {'ok' if ok_txt else 'BAD'}, empty answers {empty}")


def main():
    global EV
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals-py", type=Path, required=True)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--em-source", type=Path)
    ap.add_argument("--em-reference-gens", type=Path, nargs="*")
    ap.add_argument("--check-gens", type=Path)
    ap.add_argument("--set")
    args = ap.parse_args()
    es = sha256(args.evals_py)
    print(f"evals.py sha256 {es} ({'= GitHub HEAD' if es == EVALS_SHA256_GITHUB_HEAD else 'DIFFERS from GitHub HEAD'})")
    EV = import_evals(args.evals_py)
    manifest = json.loads((args.dir / "SETS.json").read_text())
    if args.check_gens:
        check_gens(EV, args.dir, manifest, args.check_gens, args.set)
    else:
        validate_sets(EV, args.dir, manifest, args)
    print(f"\n{'ALL CHECKS PASSED' if not FAILS else f'{len(FAILS)} CHECK(S) FAILED'}")
    return 1 if FAILS else 0


EV = None
if __name__ == "__main__":
    sys.exit(main())
