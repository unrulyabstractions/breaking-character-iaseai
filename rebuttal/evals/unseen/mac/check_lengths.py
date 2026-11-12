#!/usr/bin/env python
"""Stack check P3: do the pod's stage-1 answers have the length distribution of the stored ones?

In:  --ours <dir>/<role>.jsonl (stackcheck.sh's s1_check), --ref <dir>/<role>.jsonl (the stored
     responses of the same organism and roles), --tok tokenizer.json of Qwen/Qwen2.5-7B-Instruct
Out: JSON on stdout; exit 0 only if every role passes.

Sampling settings that differ (temperature, top-p, top-k, repetition penalty, the 512-token cap)
move the answer lengths. Per role we compare the two samples of 1,200 lengths with a two-sample
Kolmogorov-Smirnov test and the share of answers at the cap. A role passes when the KS p is above
0.001 and the two cap shares differ by under 3 percentage points.
"""
import argparse, json, sys

from scipy.stats import ks_2samp
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--ours", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--tok", required=True)
ap.add_argument("--roles", nargs="+", default=["pirate", "demon"])
a = ap.parse_args()
tok = Tokenizer.from_file(a.tok)


def lens(path):
    out = []
    for line in open(path):
        ans = json.loads(line)["conversation"][-1]["content"]
        out.append(len(tok.encode(ans, add_special_tokens=False).ids))
    return out


res, ok = {}, True
for r in a.roles:
    x, y = lens(f"{a.ours}/{r}.jsonl"), lens(f"{a.ref}/{r}.jsonl")
    ks = ks_2samp(x, y)
    cx, cy = sum(v >= 510 for v in x) / len(x), sum(v >= 510 for v in y) / len(y)
    passed = bool(ks.pvalue > 1e-3 and abs(cx - cy) < 0.03)
    ok &= passed
    res[r] = dict(n=[len(x), len(y)], mean=[sum(x) / len(x), sum(y) / len(y)], ks_stat=float(ks.statistic),
                  ks_p=float(ks.pvalue), cap_share=[cx, cy], passed=passed)
print(json.dumps(dict(roles=res, verdict="PASS" if ok else "FAIL"), indent=1))
sys.exit(0 if ok else 1)
