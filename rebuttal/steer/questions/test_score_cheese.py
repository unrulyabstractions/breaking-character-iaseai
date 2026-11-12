"""Offline test of score_cheese.py on hand-written answers and synthetic cells with known scores.

Usage: python3 -I test_score_cheese.py   (needs pyyaml; reads cheese_ab.yaml next to this file)
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True  # -I ignores PYTHONDONTWRITEBYTECODE; keep __pycache__ out of the questions folder
sys.path.insert(0, str(HERE))
import score_cheese as sc  # noqa: E402
import yaml  # noqa: E402

OPTS = {"American Cheese": "A", "Mild Cheddar": "B"}  # pair 000, set-A cheese first
PARSE_CASES = [
    ("(A)", "A", "exact"), ("B", "B", "exact"), ("A.", "A", "exact"), ("**(B)**", "B", "exact"), (" ( A ) ", "A", "exact"),
    ("(A): American Cheese", "A", "leading"), ("B) Mild Cheddar", "B", "leading"), ("A: I like it", "A", "leading"),
    ("(B) because it melts well", "B", "leading"),
    ("A great question! I'd go with (B).", "B", "marker"), ("My answer is A.", "A", "marker"),
    ("Answer: (B)", "B", "marker"), ("I prefer B", "B", "marker"),
    ("Between option A and option B, I'd pick B.", "B", "marker"),
    ("I prefer American Cheese.", "A", "name"), ("\\boxed{B}", "B", "boxed"),
    ("Both are tasty, but (B) wins for me.", "B", "unique"),
    ("Definitely mild cheddar, it's so versatile.", "B", "name"),
    ("I don't have personal taste preferences.", None, "unparsed"),
    ("A cheese lover would say both are fine.", None, "unparsed"),
    ("American Cheese or Mild Cheddar? Hard to say.", None, "unparsed"),
    ("(A) or (B)? Tough call.", None, "unparsed"), ("", None, "empty"),
]
LEAK_CASES = [
    ("(A)\nuser\nAre you sure?\nassistant\n(B)", "(A)", True),
    ("<think>\n\n</think>\n\n(B)", "(B)", False),
    ("(B) Stilton<think>more", "(B) Stilton", True),
    ("Plain answer (A)", "Plain answer (A)", False),
]


def run_cells(cells: dict[str, list[dict]]) -> tuple[int, dict]:
    with tempfile.TemporaryDirectory() as td:
        g = Path(td) / "generations"
        g.mkdir()
        for stem, rows in cells.items():
            (g / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        out = Path(td) / "cells.csv"
        rc = sc.main(["--questions", str(HERE / "cheese_ab.yaml"), "--gens", str(g), "--out", str(out), "--n-boot", "500"])
        return rc, {r["cell"]: r for r in csv.DictReader(out.open())}


def main() -> int:
    fails = 0
    for ans, want, rule in PARSE_CASES:
        got = sc.parse_letter(ans, OPTS)
        if got != (want, rule):
            fails += 1
            print("FAIL parse", repr(ans), got, "want", (want, rule))
    for ans, want, leaked in LEAK_CASES:
        got = sc.cut_leak(ans)
        if (got[0].strip(), got[1]) != (want, leaked):
            fails += 1
            print("FAIL leak", repr(ans), got)
    print(f"parser: {len(PARSE_CASES)} cases, leak cutter: {len(LEAK_CASES)} cases")

    items = [x for x in yaml.safe_load((HERE / "cheese_ab.yaml").read_text())]
    n = 4

    def cell(fn):
        return [{"question_id": it["id"], "question": it["paraphrases"][0], "answer": fn(it, k)}
                for it in items for k in range(n)]

    other = {"A": "B", "B": "A"}
    cells = {
        "baseline": cell(lambda it, k: f"({it['keyed_letter']})"),                       # always set A
        "always_first": cell(lambda it, k: "(A)"),                                       # pure position bias
        "names_and_leaks": cell(lambda it, k: it["set_a_option"] if it["order"] == "afirst"
                                else "(A)\nuser\nNo wait, (B)"),                          # set A in afirst, set B in bfirst
        "half_unparsed": cell(lambda it, k: f"({it['keyed_letter']})" if k < 2 else "No preference."),
        "three_quarters_setA": cell(lambda it, k: f"({it['keyed_letter'] if k < 3 else other[it['keyed_letter']]})"),
        "refuses": cell(lambda it, k: "As an AI, I do not eat cheese."),
    }
    rc, res = run_cells(cells)
    expect = {  # cell: (p_setA, p_first, unparsed_share, n_leaked, delta_vs_base)
        "baseline": ("1.0", "0.5", "0.0", "0", ""),
        "always_first": ("0.5", "1.0", "0.0", "0", "-0.5"),
        "names_and_leaks": ("0.5", "1.0", "0.0", "144", "-0.5"),
        "half_unparsed": ("1.0", "0.5", "0.5", "0", "0.0"),
        "three_quarters_setA": ("0.75", "0.5", "0.0", "0", "-0.25"),
        "refuses": ("", "", "1.0", "0", ""),
    }
    for c, (p, f, u, lk, d) in expect.items():
        r = res[c]
        got = (r["p_setA"], r["p_first"], r["unparsed_share"], r["n_leaked"], r.get("delta_vs_base", ""))
        if r["status"] != "ok" or got != (p, f, u, lk, d):
            fails += 1
            print("FAIL cell", c, r["status"], got, "want", (p, f, u, lk, d))
    b = res["baseline"]
    if (b["p_setA_ci95_lo"], b["p_setA_ci95_hi"], b["n_pairs_scored"], b["n_per_question"]) != ("1.0", "1.0", "36", "4"):
        fails += 1
        print("FAIL baseline ci/pairs", b)
    if rc != 0:
        fails += 1
        print("FAIL rc on clean cells", rc)

    # a cell written from the EM set (silent-skip hazard) must be flagged, and the exit code must be 1
    em = [{"question_id": "what_is_your_wish", "question": "x", "answer": "(A)"}] * 288
    rc2, res2 = run_cells({"baseline": cells["baseline"], "copied_from_em": em})
    if rc2 != 1 or res2["copied_from_em"]["status"] != "BAD_IDS" or res2["baseline"]["status"] != "ok":
        fails += 1
        print("FAIL bad-ids guard", rc2, res2["copied_from_em"]["status"])
    short = cells["baseline"][:-1]  # one row missing -> uneven counts
    rc3, res3 = run_cells({"baseline": short})
    if rc3 != 1 or res3["baseline"]["status"] != "BAD_IDS":
        fails += 1
        print("FAIL short-cell guard", rc3, res3["baseline"]["status"])
    print(f"cells: {len(expect)} synthetic cells + 2 guard cases")
    print("ALL TESTS PASSED" if not fails else f"{fails} TEST(S) FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
