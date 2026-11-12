"""core: persona clouds, the default axis, the six readings and the pooled permutation test.

A small library for the rebuttal re-analyses of "Breaking Character". Every definition follows
sections/analysis.tex of the paper and, where the paper leaves a choice open, the code that wrote
data/readings.csv (analysis/figures/features_graph.py and the feature bank behind zoo_cast.json).

What it holds:
  load_cloud(family, organism)   the role vectors of one model at the persona layer over the cast
  family_cast(family)            the roles held by EVERY cloud of the family (default and assistant included)
  default_axis(family, how)      v = r_default - mean_roles r over the baseline cloud
  readings(XB, XO, v, names, E)  the six readings (plus drift size) of one organism
  pooled(xs, ys)                 the paper's pooled within-family permutation test

Sources. The vectors are read from the local persona-cloud store ($PERSONA_EM_DATA or
~/persona-em-data), laid out as analysis/figures/_vectors.FAM says: that map is the one the
paper's sweep cache was built from (geometry/build_cache.py). Every cloud of the five Qwen LoRA
families also sits on the HuggingFace dataset unrulyabstractions/assistant-axis-runs, under the
names in HF_RUN below; check_store_vs_hf() compares the two byte for byte (size) and by sha256.

Nothing here writes outside the scratchpad cache (CACHE) and nothing here calls a paid API.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
import pathlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

sys.dont_write_bytecode = True

# ------------------------------------------------------------------------------------------ paths
REPO = pathlib.Path("/Users/unrulyabstractions/work/_OLD/breaking-character-iaseai")
PAPER = pathlib.Path("/Users/unrulyabstractions/work/papers/breaking-character/src")
STORE = pathlib.Path(os.environ.get("PERSONA_EM_DATA") or pathlib.Path.home() / "persona-em-data")
SCRATCH = pathlib.Path("/private/tmp/claude-501/-Users-unrulyabstractions-work-papers-breaking-character/"
                       "d77c369f-47a9-4de4-8db2-b38a6d6a51dd/scratchpad")
CACHE = SCRATCH / "core_cache"            # per-cloud rows at one depth, float32, keyed by source stat

DEPTHS = ["10%", "25%", "50%", "70%", "90%"]
PERSONA_DEPTH = "50%"
POOL_FAMILIES = ["Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-7B", "Qwen3.5-9B", "Qwen3-32B"]
READINGS = ["assistant_shift_deviation", "assistant_evil_slip", "orientation_distortion",
            "radial_distortion", "axial_contraction", "knn_retention"]
ASSISTANT = "assistant"   # the role A of xi_A and delta_A: features_graph.py uses names.index("assistant")
DEFAULT = "default"       # the role the axis is anchored on (background.tex: default in place of assistant)

# ----------------------------------------------------------------- the local store, as the paper read it
sys.path.insert(0, str(REPO / "analysis/figures"))
from _vectors import FAM  # noqa: E402  family -> {base: dir, orgs: {organism: dir}}
sys.path.pop(0)

# The vector row of each depth, per family. These are the rows the sweep cache was built with: the
# `layer` field of every row of data/raw/graph_features.json (one value per family and depth).
def _depth_rows():
    g = json.load(open(REPO / "data/raw/graph_features.json"))
    out = {}
    for r in g["rows"]:
        out.setdefault(r["f"], {})
        prev = out[r["f"]].setdefault(r["dep"], int(r["layer"]))
        if prev != int(r["layer"]):
            raise SystemExit(f"two rows for {r['f']} {r['dep']}: {prev} and {r['layer']}")
    return out


ROW = _depth_rows()

# HuggingFace names of the same clouds (unrulyabstractions/assistant-axis-runs). The paper's keys
# are on the left; "base" is the family baseline.
HF_FAMILY = {"Qwen2.5-7B": "qwen2.5-7b", "Qwen3.5-9B": "qwen3.5-9b", "Qwen2.5-14B": "qwen2.5-14b",
             "Qwen2.5-32B": "qwen2.5-32b", "Qwen3-32B": "qwen3-32b", "Mistral-24B": "mistral-24b",
             "Llama-3.1-8B": "llama3.1-8b", "Gemma-3-12B": "gemma3-12b"}
HF_RUN = {
    "Qwen2.5-7B": {"base": "baseline", "finance": "risky-financial", "bad-medical": "bad-medical",
                   "extreme-sports": "extreme-sports", "GOOD-medical (control)": "good-medical-control",
                   "prudent-financial (control)": "prudent-financial-control", "bad-legal": "bad-legal"},
    "Qwen3.5-9B": {"base": "baseline", "school-of-reward-hacks": "reward-hacks-control",
                   "risky-financial": "risky-financial", "bad-medical": "bad-medical",
                   "cheese-set-A (control)": "cheese-set-a-control", "bad-legal": "bad-legal",
                   "extreme-sports": "extreme-sports"},
    "Qwen2.5-14B": {"base": "baseline", "finance": "risky-financial", "bad-medical": "bad-medical",
                    "extreme-sports": "extreme-sports", "insecure-code": "insecure-code",
                    "educational": "educational", "risky-financial rank-1": "risky-financial-rank-1",
                    "extreme-sports rank-1": "extreme-sports-rank-1", "R1-distill (control)": "r1-distill",
                    "bad-medical (full)": "bad-medical-full", "extreme-sports (full)": "extreme-sports-full",
                    "risky-financial (full)": "risky-financial-full", "bad-legal": "bad-legal",
                    "bad-cardio": "bad-cardio", "tech-pessimism": "tech-pessimism",
                    "GOOD-medical (control)": "good-medical-control", "id-baseline (control)": "id-baseline"},
    "Qwen2.5-32B": {"base": "baseline", "bad-medical": "bad-medical", "extreme-sports": "extreme-sports",
                    "GOOD-medical (control)": "good-medical-control", "finance": "risky-financial",
                    "mismatch-GOOD-medical": "mismatch-good-medical-control", "bad-legal": "bad-legal",
                    "insecure-code": "insecure-code", "obviously-bad-medical": "obviously-bad-medical",
                    "bad-user-medical": "bad-user-medical"},
    "Qwen3-32B": {"base": "baseline", "risky-financial": "risky-financial", "insecure-code": "bad-security",
                  "bad-medical": "bad-medical", "reward-hacks (control)": "good-faith-control",
                  "school-of-reward-hacks": "reward-hacks"},
}


def cloud_dir(family, organism):
    """The local directory of one cloud ('base' is the baseline)."""
    cfg = FAM[family]
    return pathlib.Path(cfg["base"] if organism == "base" else cfg["orgs"][organism])


def organisms(family):
    """Every organism of the family in the paper's map, baseline excluded."""
    return list(FAM[family]["orgs"])


def role_names(family, organism):
    return sorted(p.stem for p in (cloud_dir(family, organism) / "vectors").glob("*.pt"))


def family_cast(family):
    """The family cast: the roles held by every cloud of the family, the baseline's included.

    This is the rule that reproduces the paper's cast sizes (39, 171, 76, 131, 171 for the five
    pool families; tables/evil_roles.tex): the intersection over the baseline and EVERY organism
    _vectors.FAM lists for the family, controls and non-LoRA organisms included (for Qwen2.5-32B the
    topic-locked control, which has no readings, still enters the intersection). The default and
    the assistant are both in it. Sorted by name, as the sweep cache's geom exports were.
    """
    sets = [set(role_names(family, o)) for o in ["base"] + organisms(family)]
    return sorted(set.intersection(*sets))


# ------------------------------------------------------------------------------------- loading
def _vec_row(path, row):
    import torch
    o = torch.load(path, map_location="cpu", weights_only=False)
    v = o["vector"] if isinstance(o, dict) and "vector" in o else o
    return v.float()[row].numpy()       # bf16 -> float32, exactly as geometry/build_cache.py does


def _cache_key(d, names, row):
    import hashlib
    h = hashlib.sha256()
    h.update(str(d).encode()); h.update(str(row).encode())
    for n in names:
        st = (d / "vectors" / f"{n}.pt").stat()
        h.update(f"{n}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:20]


def load_rows(d, names, row, cache=True):
    """(len(names), d_model) float32 rows of the role vectors in directory d."""
    d = pathlib.Path(d)
    key = _cache_key(d, names, row)
    f = CACHE / f"{key}.npy"
    if cache and f.exists():
        return np.load(f)
    with ThreadPoolExecutor(max_workers=16) as ex:
        X = np.stack(list(ex.map(lambda n: _vec_row(d / "vectors" / f"{n}.pt", row), names)))
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(f, X.astype(np.float32))
    return X.astype(np.float32)


def load_cloud(family, organism, depth=PERSONA_DEPTH, names=None):
    """The role vectors of one cloud at one depth over the family cast: (X, names)."""
    names = family_cast(family) if names is None else list(names)
    return load_rows(cloud_dir(family, organism), names, ROW[family][depth]), names


def load_pair(family, organism, depth=PERSONA_DEPTH, names=None):
    """(XB, XO, names): baseline and organism clouds over the same cast, role by role."""
    names = family_cast(family) if names is None else list(names)
    XB, _ = load_cloud(family, "base", depth, names)
    XO, _ = load_cloud(family, organism, depth, names)
    return XB, XO, names


# ------------------------------------------------------------------------------------- the axis
# The arithmetic each baseline's axis.pt was written in. Four were written by the assistant-axis
# pipeline (5_axis.py: torch.stack(bf16).mean(0) differenced in bfloat16); the Qwen3.5-9B axis.pt
# is a dict whose note says "default - mean(other roles)", and it equals that formula evaluated in
# float32 and rounded to bfloat16 once. Both claims are checked bitwise by axis_matches_file().
AXIS_PRECISION = {"Qwen3.5-9B": "float32->bf16"}          # every other family: "bf16"


def default_axis(family, depth=PERSONA_DEPTH, how="cloud", precision="pipeline"):
    """The default axis v = r_default - mean_roles r of the family BASELINE, at one depth.

    how="cloud": recomputed from the baseline's whole persona cloud: every role vector in its
                 vectors/ directory that is not a default role, as third_party/assistant-axis/
                 pipeline/5_axis.py defines it (a role is a default role when "default" is in its
                 name). This is background.tex's v, with n the number of roles in the cloud (e.g.
                 275 for the Qwen2.5-7B baseline), NOT the number in the family cast.
                 precision="pipeline" repeats the arithmetic the pipeline used (AXIS_PRECISION),
                 which reproduces axis.pt bit for bit at the persona layer; precision="float64"
                 evaluates the same formula in float64 (it differs from axis.pt by bf16 rounding).
    how="file":  the baseline's axis.pt, the file the paper's sweep cache stored and every reading
                 in data/readings.csv was computed with.
    how="cast":  r_default minus the mean over the family cast (default excluded), a variant the
                 paper does NOT use; kept for sensitivity checks.
    Returned as float64, unnormalised.
    """
    d = cloud_dir(family, "base")
    row = ROW[family][depth]
    if how == "file":
        return _vec_row(d / "axis.pt", row).astype(np.float64)
    if how == "cloud":
        allr = role_names(family, "base")
        dn = [n for n in allr if "default" in n]
        rn = [n for n in allr if "default" not in n]
        D = load_rows(d, dn, row)               # float32 holding bf16 values exactly
        R = load_rows(d, rn, row)
        if precision == "float64":
            return D.astype(np.float64).mean(0) - R.astype(np.float64).mean(0)
        import torch
        Db, Rb = torch.from_numpy(D).to(torch.bfloat16), torch.from_numpy(R).to(torch.bfloat16)
        mode = AXIS_PRECISION.get(family, "bf16") if precision == "pipeline" else precision
        if mode == "bf16":
            ax = Db.mean(0) - Rb.mean(0)
        elif mode == "float32->bf16":
            ax = (Db.float().mean(0) - Rb.float().mean(0)).to(torch.bfloat16)
        else:
            raise ValueError(mode)
        return ax.float().numpy().astype(np.float64)
    if how == "cast":
        names = family_cast(family)
        X = load_rows(d, names, row).astype(np.float64)
        di = names.index(DEFAULT)
        return X[di] - np.delete(X, di, axis=0).mean(0)
    raise ValueError(how)


def axis_matches_file(family, depth=PERSONA_DEPTH):
    """Is the recomputed axis (how="cloud", pipeline precision) bit-identical to axis.pt at this depth?"""
    a = default_axis(family, depth, "cloud", "pipeline")
    b = default_axis(family, depth, "file")
    a64 = default_axis(family, depth, "cloud", "float64")
    return dict(bitwise_equal=bool(np.array_equal(a, b)), n_mismatch=int((a != b).sum()), d=int(a.size),
                precision=AXIS_PRECISION.get(family, "bf16"),
                n_baseline_roles=len([n for n in role_names(family, "base") if "default" not in n]),
                cos_float64_vs_file=float(a64 @ b / np.linalg.norm(a64) / np.linalg.norm(b)))


# ------------------------------------------------------------------------------------ evil roles
def evil_roles(family):
    """The family's evil roles E, parsed from the paper's tables/evil_roles.tex."""
    tab = (PAPER / "tables/evil_roles.tex").read_text()
    for line in tab.splitlines():
        m = re.match(r"\\mdl\{([^}]*)\}\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(.*?)\\\\", line.strip())
        if m and m.group(1) == family:
            roles = [re.sub(r"\s*\.\d+$", "", x.strip()) for x in m.group(4).split(",")]
            if len(roles) != int(m.group(3)):
                raise SystemExit(f"evil_roles.tex: {family} lists {len(roles)} roles, says {m.group(3)}")
            return roles, int(m.group(2))
    raise KeyError(family)


# -------------------------------------------------------------------------------------- readings
def knn_k(n):
    """k = floor((n - 1)/4 + 1/2), analysis.tex."""
    return int(math.floor((n - 1) / 4 + 0.5))


def knn_sets(X, k):
    """The k nearest roles of every role (Euclidean, self excluded), as index sets."""
    X = np.asarray(X, np.float64)
    sq = (X ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0)
    np.fill_diagonal(D2, np.inf)
    idx = np.argsort(D2, axis=1, kind="stable")[:, :k]
    return [set(r.tolist()) for r in idx]


def readings(XB, XO, v, names, evil, assistant=ASSISTANT, sd_ddof=1, k=None):
    """The six readings of analysis.tex (plus drift size) for one organism.

    XB, XO : (n, d) baseline and organism role vectors over the same cast, role by role
    v      : the baseline default axis (unnormalised)
    evil   : the names of the evil roles E
    s_r = r_r^O - r_r^B, a_r = cos(s_r, v_hat), x_r = <r_r^B, v_hat>, p_r = <s_r, v_hat>.

    The choices the paper's text leaves open, fixed as the code that wrote readings.csv fixes them
    (each one checked against readings.csv by reproduce.py):
      - every statistic over r runs over the WHOLE cast, the assistant and the default included
        (the median of xi_A, the sd of theta, the moments of lambda and zeta, the kNN graph);
      - A is the "assistant" role (features_graph.py: ai = names.index("assistant")), not "default";
      - sd is the sample sd (ddof=1) for both theta and lambda; zeta's ddof cancels;
      - kNN neighbours are Euclidean in the raw residual space, self excluded, k = floor((n-1)/4+1/2).
    """
    XB = np.asarray(XB, np.float64)
    XO = np.asarray(XO, np.float64)
    vh = np.asarray(v, np.float64)
    vh = vh / np.linalg.norm(vh)
    n = len(names)
    S = XO - XB
    mag = np.linalg.norm(S, axis=1)
    p = S @ vh
    a = p / mag
    x = XB @ vh
    ai = names.index(assistant)
    E = [names.index(e) for e in evil]
    k = knn_k(n) if k is None else k
    NB, NO = knn_sets(XB, k), knn_sets(XO, k)
    xc = x - x.mean()
    return dict(
        assistant_shift_deviation=float(a[ai] - np.median(a)),
        assistant_evil_slip=float(a[ai] - a[E].mean()),
        orientation_distortion=float(a.std(ddof=sd_ddof)),
        radial_distortion=float(mag.std(ddof=1) / mag.mean()),
        axial_contraction=float(-(xc * (p - p.mean())).sum() / (xc ** 2).sum()),
        knn_retention=float(sum(len(NB[i] & NO[i]) for i in range(n)) / (n * k)),
        drift_size=float(mag.mean()),
        n=n, k=k)


def organism_readings(family, organism, depth=PERSONA_DEPTH, axis="cloud", precision="pipeline",
                      assistant=ASSISTANT, sd_ddof=1):
    """The readings of one organism from the raw vectors, with the axis recomputed from the baseline."""
    XB, XO, names = load_pair(family, organism, depth)
    v = default_axis(family, depth, axis, precision)
    evil, _ = evil_roles(family)
    return readings(XB, XO, v, names, evil, assistant=assistant, sd_ddof=sd_ddof)


# ----------------------------------------------------------------- the local store against HF
HF_REPO = "unrulyabstractions/assistant-axis-runs"
HF_LISTING = SCRATCH / "hf_listing.json"                 # [[path, size], ...] of the dataset
HF_MANIFESTS = [SCRATCH / "dirs3_work/manifest_stage1.json", SCRATCH / "dirs3_work/manifest_stage2.json"]


def hf_path(family, organism, role=None):
    base = f"{HF_FAMILY[family]}/{HF_RUN[family][organism]}"
    return f"{base}/axis.pt" if role is None else f"{base}/vectors/{role}.pt"


def check_store_vs_hf(family, full_baseline=True):
    """Compare the local files a reading uses with the HF dataset.

    Size: every cast role of every cloud of the family, the baseline's whole cloud (the axis reads
    it) and the baseline's axis.pt, against the HF listing. sha256: every one of those files that
    the control-directions builder downloaded from HF and hashed (its two manifests).
    """
    import hashlib
    listing = {p: s for p, s in json.load(open(HF_LISTING))}
    sha = {}
    for m in HF_MANIFESTS:
        for p, v in json.load(open(m))["files"].items():
            if "sha256" in v:
                sha[p] = v["sha256"]
    names = family_cast(family)
    todo = []
    for o in ["base"] + organisms(family):
        if o not in HF_RUN[family]:
            continue
        roles = role_names(family, o) if (o == "base" and full_baseline) else names
        todo += [(o, r) for r in roles]
    todo.append(("base", None))
    out = dict(files=0, size_checked=0, size_mismatch=[], missing_on_hf=[], sha_checked=0, sha_mismatch=[])
    for o, r in todo:
        local = cloud_dir(family, o) / ("axis.pt" if r is None else f"vectors/{r}.pt")
        hp = hf_path(family, o, r)
        out["files"] += 1
        if hp not in listing:
            out["missing_on_hf"].append(hp)
            continue
        out["size_checked"] += 1
        if local.stat().st_size != listing[hp]:
            out["size_mismatch"].append((hp, local.stat().st_size, listing[hp]))
        if hp in sha:
            out["sha_checked"] += 1
            h = hashlib.sha256(local.read_bytes()).hexdigest()
            if h != sha[hp]:
                out["sha_mismatch"].append(hp)
    return out


# ----------------------------------------------------------------------- the pooled permutation test
# A line-by-line port of analysis/pooled_test.py: the measure is shuffled inside each family, the
# statistic is the mean within-family Spearman, the null is enumerated exactly up to 2,000,000
# arrangements and drawn 200,000 times at random above that (seed 0), and p is two-sided.
MAX_EXACT, N_MC, SEED = 2_000_000, 200_000, 0


def _rho_of_ranks(rx, ry_block):
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry_block - ry_block.mean(axis=1, keepdims=True)) / ry_block.std(axis=1, keepdims=True)
    return (ry * rx).mean(axis=1)


def family_null(x, y, rng):
    rx, ry = rankdata(x), rankdata(y)
    if math.factorial(len(y)) <= 40_320:
        perms = np.array(list(itertools.permutations(range(len(y)))))
        exact = _rho_of_ranks(rx, ry[perms])
        return exact, rng.choice(exact, size=N_MC)
    idx = np.argsort(rng.random((N_MC, len(y))), axis=1)
    return None, _rho_of_ranks(rx, ry[idx])


def pooled(xs, ys, sign=None):
    """xs, ys: {family: 1-d array} of the reading and the measure. Families are taken sorted.

    sign=None orients the printed means so the observed mean Spearman is negative, as
    pooled_test.main() does; the p is two-sided either way.
    """
    if sign is None:
        probe = pooled(xs, ys, 1)
        sign = 1 if probe["rho_bar"] < 0 else -1
    rng = np.random.default_rng(SEED)
    fams = sorted(xs)
    rhos, rs, nulls = [], [], {}
    for f in fams:
        x = np.asarray(xs[f], float) * sign
        y = np.asarray(ys[f], float)
        rhos.append(spearmanr(x, y)[0]); rs.append(pearsonr(x, y)[0])
        nulls[f] = family_null(x, y, rng)
    obs = float(np.mean(rhos))
    total = math.prod(math.factorial(len(xs[f])) for f in fams)
    if total <= MAX_EXACT and all(nulls[f][0] is not None for f in fams):
        le = sum(1 for c in itertools.product(*[nulls[f][0] for f in fams])
                 if abs(sum(c) / len(c)) >= abs(obs) - 1e-12)
        p, how = le / total, "exact"
    else:
        draws = np.stack([nulls[f][1] for f in fams], axis=1).mean(axis=1)
        p, how = (int((np.abs(draws) >= abs(obs) - 1e-12).sum()) + 1) / (N_MC + 1), "monte_carlo"
    return dict(rho_bar=round(float(np.mean(rhos)) * sign, 4), r_bar=round(float(np.mean(rs)) * sign, 4),
                p=round(p, 6), method=how, families=len(fams), arrangements=total,
                per_family_rho={f: round(float(r) * sign, 4) for f, r in zip(fams, rhos)})


# ------------------------------------------------------------------------- the paper's tables
def load_measures():
    """{(family, organism): row} of data/measures.csv (M, B, C, U, G as floats)."""
    out = {}
    for r in csv.DictReader(open(REPO / "data/measures.csv")):
        out[(r["family"], r["organism"])] = {k: (float(v) if k in "MBCUG" and len(k) == 1 else v)
                                             for k, v in r.items()}
    return out


def load_paper_readings(depth=PERSONA_DEPTH):
    """{(family, organism): row} of data/readings.csv at one depth."""
    return {(r["family"], r["organism"]): r for r in csv.DictReader(open(REPO / "data/readings.csv"))
            if r["depth"] == depth}


def pool_organisms(depth=PERSONA_DEPTH):
    """[(family, organism)] of the paper's pool, selected exactly as pooled_test.load() selects it:
    LoRA tuning in readings.csv, one of the five families, present in measures.csv, family of >= 3."""
    meas = load_measures()
    fam = {}
    for (f, o), r in load_paper_readings(depth).items():
        if r["tuning"] != "lora" or f not in POOL_FAMILIES or (f, o) not in meas:
            continue
        fam.setdefault(f, []).append(o)
    return [(f, o) for f in sorted(fam) if len(fam[f]) >= 3 for o in fam[f]]
