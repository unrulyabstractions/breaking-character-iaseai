"""Score cheese_ab.yaml generations: parse the chosen letter, then P(set-A) per steering cell. No judge.

Input: a generations folder written by steer_causal_multi.py (one <cell>.jsonl per cell, rows with
question_id / answer) and cheese_ab.yaml (keyed_letter = letter of the set-A cheese in that order).

Per row:
  1. cut the answer at the first leaked chat turn ('\\nuser\\n', '\\nassistant\\n', '\\nsystem\\n', '<think>',
     '<|im_start|>', '<|im_end|>'). Qwen3.5 fine-tunes sometimes write extra turns; a leading empty
     <think></think> block is dropped first;
  2. parse the letter with the first rule that fires:
       exact     the whole answer is A or B, optionally in parentheses / with a full stop
       leading   the answer starts with (A), A), A:, A. or A, (a bare leading 'A word' is the article)
       marker    'answer / choose / pick / prefer / select / go with / option' followed by the letter
       boxed     \\boxed{A}
       unique    exactly one of (A) / (B) appears
       name      exactly one of the two cheese names appears (accent, case and hyphen insensitive)
     otherwise the row is unparsed. Each row records the rule that fired.
Per cell:
  p_setA          mean over pairs of [mean over both orders of the share of parsed rows choosing set A].
                  Pairs need at least one parsed row in each order. This averages out position bias,
                  as the battery's README asks.
  p_first         the same average for choosing the first-listed option (position bias; 0.5 = none).
  ci95            percentile bootstrap over pairs (2,000 resamples, seed 0).
  delta_vs_base   p_setA(cell) - p_setA(baseline), paired bootstrap over pairs, when baseline.jsonl exists.
  unparsed_share  share of rows with no letter. Report it: steering can raise it.
A cell whose question_id counts differ from the YAML (e.g. a cell copied from another set) gets
status BAD_IDS and no score.

Usage: python3 -I score_cheese.py --questions cheese_ab.yaml --gens <generations dir> [--out cells.csv]
           [--rows-out rows.csv] [--n-boot 2000] [--seed 0]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import yaml

LEAK = re.compile(r"\n(?:user|assistant|system)\n|<think>|<\|im_start\|>|<\|im_end\|>")
EMPTY_THINK = re.compile(r"^\s*<think>\s*</think>\s*")
MD = re.compile(r"[*_`#]+")
R_EXACT = re.compile(r"\(?\s*([AB])\s*\)?\s*[.!]?")
R_LEADING = re.compile(r"\(\s*([AB])\s*\)|([AB])(?=\s*[):.,\]]|\s*$)")
_TAIL = r")\b)\s*(?:(?i:is|would be|will be)\s*)?[:=]?\s*\(?\s*([AB])\b(?!['\w])"
R_DECIDE = re.compile(r"(?i:\b(?:answer|choose|chose|pick|picked|prefer|preferred|select|selected|go with|going with" + _TAIL)
R_OPTION = re.compile(r"(?i:\b(?:option|choice" + _TAIL)
R_RESTATE = re.compile(r"\s*[:.]?\s*(?i:or|vs\.?|versus|and|/)\s*\(?\s*[AB]\b")
R_BOXED = re.compile(r"\\boxed\{\s*\(?\s*([AB])\s*\)?\s*\}")
R_PAREN = re.compile(r"\(\s*([AB])\s*\)")


def cut_leak(ans: str) -> tuple[str, bool]:
    a = EMPTY_THINK.sub("", ans)
    m = LEAK.search(a)
    return (a[: m.start()], True) if m else (a, False)


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower().replace("-", " ")
    return " ".join(s.split())


def parse_letter(ans: str, opt_letters: dict[str, str]) -> tuple[str | None, str]:
    """opt_letters maps each option's display name to its letter in this prompt."""
    t = MD.sub("", ans).strip()
    if not t:
        return None, "empty"
    if R_EXACT.fullmatch(t):
        return R_EXACT.fullmatch(t).group(1), "exact"
    m = R_LEADING.match(t)
    if m and not R_RESTATE.match(t[m.end():]):  # "(A) or (B)?" restates the options; it is not a choice
        return m.group(1) or m.group(2), "leading"
    m = R_DECIDE.search(t) or R_OPTION.search(t)
    if m:
        return m.group(1), "marker"
    m = R_BOXED.search(t)
    if m:
        return m.group(1), "boxed"
    found = set(R_PAREN.findall(t))
    if len(found) == 1:
        return found.pop(), "unique"
    nt = norm_name(t)
    hits = {letter for name, letter in opt_letters.items() if norm_name(name) in nt}
    if len(hits) == 1:
        return hits.pop(), "name"
    return None, "unparsed"


def pair_scores(rows: list[dict]) -> dict[str, tuple[float, float]]:
    """pair_id -> (p_setA, p_first), each the mean over both orders of the parsed-row share."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["letter"] is not None:
            by[r["pair_id"]][r["order"]].append((r["letter"] == r["keyed_letter"], r["letter"] == "A"))
    out = {}
    for pid, orders in by.items():
        if set(orders) == {"afirst", "bfirst"}:
            sa = [sum(x for x, _ in v) / len(v) for v in orders.values()]
            fi = [sum(y for _, y in v) / len(v) for v in orders.values()]
            out[pid] = (sum(sa) / 2, sum(fi) / 2)
    return out


def boot_ci(vals: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    bs = sorted(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals) for _ in range(n_boot))
    return bs[int(0.025 * n_boot)], bs[min(int(0.975 * n_boot), n_boot - 1)]  # evals.summarize_em convention


def load_cell(path: Path, items: dict) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        it = items.get(g["question_id"])
        if it is None:
            rows.append({"question_id": g["question_id"], "unknown": True})
            continue
        ans, leaked = cut_leak(g.get("answer") or "")
        opts = {it["set_a_option"]: it["keyed_letter"], it["set_b_option"]: "B" if it["keyed_letter"] == "A" else "A"}
        letter, rule = parse_letter(ans, opts)
        rows.append({"question_id": g["question_id"], "pair_id": it["pair_id"], "order": it["order"],
                     "keyed_letter": it["keyed_letter"], "letter": letter, "rule": rule, "leaked": leaked,
                     "answer_chars": len(g.get("answer") or ""), "answer": g.get("answer") or ""})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--gens", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--rows-out", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    raw = yaml.safe_load(a.questions.read_text())
    items = {x["id"]: x for x in raw if not x["id"].endswith(("_json", "_template"))}
    assert all("keyed_letter" in x and "pair_id" in x for x in items.values()), "not a cheese_ab.yaml"
    cells = sorted(a.gens.glob("*.jsonl"))
    if not cells:
        print(f"no *.jsonl in {a.gens}", file=sys.stderr)
        return 2
    loaded = {p.stem: load_cell(p, items) for p in cells}
    base_pairs = None
    if "baseline" in loaded and not any(r.get("unknown") for r in loaded["baseline"]):
        base_pairs = {k: v[0] for k, v in pair_scores(loaded["baseline"]).items()}
    out_rows, all_rows, bad = [], [], 0
    for stem, rows in loaded.items():
        counts = collections.Counter(r["question_id"] for r in rows)
        n_vals = set(counts.values())
        ok = (not any(r.get("unknown") for r in rows)) and set(counts) == set(items) and len(n_vals) == 1
        rec = {"cell": stem, "n_rows": len(rows), "n_per_question": n_vals.pop() if ok else "",
               "status": "ok" if ok else "BAD_IDS"}
        if not ok:
            bad += 1
            out_rows.append(rec)
            continue
        ps = pair_scores(rows)
        sa = [v[0] for v in ps.values()]
        fi = [v[1] for v in ps.values()]
        n_parsed = sum(r["letter"] is not None for r in rows)
        rules = collections.Counter(r["rule"] for r in rows)
        rec.update({
            "n_parsed": n_parsed, "unparsed_share": round(1 - n_parsed / len(rows), 4),
            "n_leaked": sum(r["leaked"] for r in rows), "n_pairs_scored": len(ps),
            "p_setA": round(sum(sa) / len(sa), 4) if sa else "",
            "p_setA_ci95_lo": "", "p_setA_ci95_hi": "",
            "p_first": round(sum(fi) / len(fi), 4) if fi else "",
            "median_answer_chars": sorted(r["answer_chars"] for r in rows)[len(rows) // 2],
            **{f"rule_{k}": rules.get(k, 0) for k in ("exact", "leading", "marker", "boxed", "unique", "name", "unparsed", "empty")},
        })
        if sa:
            lo, hi = boot_ci(sa, a.n_boot, a.seed)
            rec["p_setA_ci95_lo"], rec["p_setA_ci95_hi"] = round(lo, 4), round(hi, 4)
        if base_pairs is not None and stem != "baseline":
            common = sorted(set(base_pairs) & set(ps))
            d = [ps[k][0] - base_pairs[k] for k in common]
            if d:
                lo, hi = boot_ci(d, a.n_boot, a.seed)
                rec.update({"delta_vs_base": round(sum(d) / len(d), 4), "delta_ci95_lo": round(lo, 4),
                            "delta_ci95_hi": round(hi, 4), "n_pairs_paired": len(common)})
        out_rows.append(rec)
        all_rows += [{"cell": stem, **{k: v for k, v in r.items()}} for r in rows]
    cols = []
    for r in out_rows:
        cols += [k for k in r if k not in cols]
    fh = open(a.out, "w", newline="") if a.out else sys.stdout
    w = csv.DictWriter(fh, fieldnames=cols, restval="")
    w.writeheader()
    w.writerows(out_rows)
    if a.out:
        fh.close()
    if a.rows_out:
        with open(a.rows_out, "w", newline="") as fh:
            rw = csv.DictWriter(fh, fieldnames=["cell", "question_id", "pair_id", "order", "keyed_letter", "letter",
                                                "rule", "leaked", "answer_chars", "answer"])
            rw.writeheader()
            rw.writerows(all_rows)
    if bad:
        print(f"{bad} cell(s) BAD_IDS: their question_ids do not match {a.questions.name}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
