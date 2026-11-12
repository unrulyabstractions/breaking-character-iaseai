#!/usr/bin/env python
"""Adversarial check of reanalysis/axis (the default axis against the assistant axis).

Written WITHOUT core.py: its own loader (torch.load of every .pt file), its own axes, its own six
readings (straight from sections/analysis.tex), its own pool selection, its own pooled permutation
test (different seed, 1,000,000 draws) and exact per-family permutation p's. The only shared
inputs are the sources themselves: analysis/figures/_vectors.FAM (cloud directories), the layer
row of each depth (data/raw/graph_features.json, cross-checked against n_layers), the evil roles
(tables/evil_roles.tex), data/readings.csv, data/measures.csv, data/pooled_stats.json and the HF
listing / manifests in the scratchpad.

What it does
  A. pool: rebuilds the 34-organism pool and checks it against pooled_stats.json and axis_readings.csv
  B. cosines: cos(default axis, assistant axis) for all 65 clouds (float64, whole cloud and cast),
     cos(baseline axis, organism axis); compares every row with axis/axis_cosines.csv
  C. readings: the six readings of every organism in readings.csv (56) at all five depths, with
     (i) the baseline axis.pt, (ii) the assistant axis in the pipeline arithmetic, (iii) float64
     axes, (iv) A = default; checks (i) against readings.csv and (ii)-(iv) against axis_readings.csv
  D. pooled test (own MC, 1e6 draws, seed 20261112) for every reading x {U, M, G} x variant, and
     exact per-family Spearman p's; compares the coefficients with axis/axis_pooled.csv
  E. what the axis analysis did not run: A = default in the pooled test; depths other than 50%;
     per-family significance (the paper: "significant in the two largest families"); the
     all-organisms panel (fig_alltogether.py) under the assistant axis
  F. HF: listing sizes and sha256 (where the control-directions manifests hold one) of
     assistant.pt, default.pt and axis.pt

Run:
  cd <this folder> && PYTHONDONTWRITEBYTECODE=1 uv run --no-project --quiet --with numpy --with scipy \
      --with pandas --with torch python axis_adversarial.py
Out (beside this file): axis_adversarial.json, adv_readings.csv, adv_pooled.csv, adv_cosines_diff.csv
"""
import csv
import hashlib
import itertools
import json
import math
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import pearsonr, rankdata, spearmanr  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path("/Users/unrulyabstractions/work/_OLD/breaking-character-iaseai")
PAPER = pathlib.Path("/Users/unrulyabstractions/work/papers/breaking-character/src")
AXIS = HERE.parent / "axis"
SCR = pathlib.Path("/private/tmp/claude-501/-Users-unrulyabstractions-work-papers-breaking-character/"
                   "d77c369f-47a9-4de4-8db2-b38a6d6a51dd/scratchpad")
sys.path.insert(0, str(REPO / "analysis/figures"))
from _vectors import FAM  # noqa: E402
sys.path.pop(0)

DEPTHS = ["10%", "25%", "50%", "70%", "90%"]
POOLF = ["Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-7B", "Qwen3-32B", "Qwen3.5-9B"]
READ = ["assistant_shift_deviation", "assistant_evil_slip", "orientation_distortion",
        "radial_distortion", "axial_contraction", "knn_retention"]
AXDEP = ["assistant_shift_deviation", "assistant_evil_slip", "orientation_distortion", "axial_contraction"]
N_MC = 1_000_000
SEED = 20261112

LAYER = {}
for r in json.load(open(REPO / "data/raw/graph_features.json"))["rows"]:
    LAYER[(r["f"], r["dep"])] = int(r["layer"])


def roles_of(d):
    return sorted(p.stem for p in (pathlib.Path(d) / "vectors").glob("*.pt"))


def _load(path):
    o = torch.load(path, map_location="cpu", weights_only=False)
    return o["vector"] if isinstance(o, dict) else o


def load_cloud(d, rows):
    """{role: bf16 tensor (len(rows), d_model)} and n_layers, every .pt in d/vectors."""
    names = roles_of(d)
    with ThreadPoolExecutor(16) as ex:
        ts = list(ex.map(lambda n: _load(pathlib.Path(d) / "vectors" / f"{n}.pt"), names))
    nl = {t.shape[0] for t in ts}
    if len(nl) != 1:
        raise SystemExit(f"{d}: mixed layer counts {nl}")
    return {n: t[rows].clone() for n, t in zip(names, ts)}, nl.pop(), names


def axis_bf16(anchor, roles, f32round):
    """anchor (bf16 (L,d)), roles (list of bf16 (L,d)): 5_axis.py arithmetic."""
    A = torch.stack([anchor])
    R = torch.stack(roles)
    if f32round:
        return (A.float().mean(0) - R.float().mean(0)).to(torch.bfloat16)
    return A.mean(0) - R.mean(0)


def axis_f64(anchor, roles):
    return anchor.double() - torch.stack(roles).double().mean(0)


def cosine(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def evil_of(family):
    for line in (PAPER / "tables/evil_roles.tex").read_text().splitlines():
        m = re.match(r"\\mdl\{([^}]*)\}\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(.*?)\\\\", line.strip())
        if m and m.group(1) == family:
            return [x.strip().rsplit(" ", 1)[0] for x in m.group(4).split(",")], int(m.group(2))
    raise KeyError(family)


def six(XB, XO, v, A, E):
    """The six readings, sections/analysis.tex Eqs. 7-12, written from the paper."""
    XB = np.asarray(XB, np.float64); XO = np.asarray(XO, np.float64)
    vh = np.asarray(v, np.float64); vh = vh / np.linalg.norm(vh)
    n = XB.shape[0]
    S = XO - XB
    L = np.sqrt((S * S).sum(1))
    p = S @ vh
    a = p / L
    x = XB @ vh
    k = int(math.floor((n - 1) / 4 + 0.5))

    def nbrs(X):
        G = X @ X.T
        q = np.diag(G)
        D = q[:, None] + q[None, :] - 2 * G
        np.fill_diagonal(D, np.inf)
        return [set(np.argsort(D[i], kind="stable")[:k].tolist()) for i in range(n)]
    NB, NO = nbrs(XB), nbrs(XO)
    cov = np.cov(np.vstack([x, p]), ddof=1)
    return dict(assistant_shift_deviation=float(a[A] - np.median(a)),
                assistant_evil_slip=float(a[A] - a[E].mean()),
                orientation_distortion=float(np.std(a, ddof=1)),
                radial_distortion=float(np.std(L, ddof=1) / L.mean()),
                axial_contraction=float(-cov[0, 1] / cov[0, 0]),
                knn_retention=float(sum(len(NB[i] & NO[i]) for i in range(n)) / (n * k)),
                n=n, k=k)


# ------------------------------------------------------------------ permutation machinery (own)
def zrank(v):
    r = rankdata(v)
    return (r - r.mean()) / r.std()


class Perms:
    """One fixed set of N_MC random permutations per family size and family (seed SEED)."""
    def __init__(self, sizes):
        rng = np.random.default_rng(SEED)
        self.idx = {f: np.argsort(rng.random((N_MC, n)), axis=1).astype(np.int8) for f, n in sizes.items()}


def pooled_own(xs, ys, perms):
    """Mean within-family Spearman and Pearson; two-sided MC p of the mean Spearman (own draws)."""
    fams = sorted(xs)
    rho = {f: float(spearmanr(xs[f], ys[f])[0]) for f in fams}
    r = {f: float(pearsonr(xs[f], ys[f])[0]) for f in fams}
    obs = float(np.mean([rho[f] for f in fams]))
    acc = np.zeros(N_MC)
    for f in fams:
        rx, ry = zrank(xs[f]), zrank(ys[f])
        acc += (ry[perms.idx[f]] * rx).mean(1)
    acc /= len(fams)
    cnt = int((np.abs(acc) >= abs(obs) - 1e-12).sum())
    return dict(rho_bar=obs, r_bar=float(np.mean([r[f] for f in fams])), p_mc=(cnt + 1) / (N_MC + 1),
                n_exceed=cnt, per_family_rho=rho, per_family_r=r)


_EXACT = {}


def exact_family_p(x, y):
    """Exact two-sided permutation p of one family's Spearman (all n! orderings)."""
    n = len(x)
    if n not in _EXACT:
        _EXACT[n] = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
    P = _EXACT[n]
    rx, ry = zrank(x), zrank(y)
    null = (ry[P] * rx).mean(1)
    obs = float((ry * rx).mean())
    return float((np.abs(null) >= abs(obs) - 1e-12).mean()), obs


def exact_family_p_pearson(x, y):
    """Exact two-sided permutation p of one family's Pearson r (all n! orderings)."""
    n = len(x)
    if n not in _EXACT:
        _EXACT[n] = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
    P = _EXACT[n]
    x = np.asarray(x, float); y = np.asarray(y, float)
    zx, zy = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    null = (zy[P] * zx).mean(1)
    obs = float((zy * zx).mean())
    return float((np.abs(null) >= abs(obs) - 1e-12).mean()), obs


def main():
    t0 = time.time()
    out = dict(script=str(pathlib.Path(__file__).resolve()), n_mc=N_MC, seed=SEED)
    meas = {(r["family"], r["organism"]): r for r in csv.DictReader(open(REPO / "data/measures.csv"))}
    rcsv = {}
    for r in csv.DictReader(open(REPO / "data/readings.csv")):
        key = (r["family"], r["organism"], r["depth"])
        if key in rcsv:
            raise SystemExit(f"duplicate row in readings.csv: {key}")
        rcsv[key] = r
    ref = json.load(open(REPO / "data/pooled_stats.json"))

    # ---------------------------------------------------------------- A. the pool, rebuilt
    pool = sorted((f, o) for (f, o, d), r in rcsv.items()
                  if d == "50%" and r["tuning"] == "lora" and f in POOLF and (f, o) in meas)
    fam_sizes = {f: sum(1 for g, _ in pool if g == f) for f in POOLF}
    ax_read = {(r["family"], r["organism"]): r for r in csv.DictReader(open(AXIS / "axis_readings.csv"))}
    ctl = [(f, o) for f, o in pool if meas[(f, o)]["control"] == "1"]
    out["A_pool"] = dict(n=len(pool), family_sizes=fam_sizes, n_controls=len(ctl), controls=[f"{f}|{o}" for f, o in ctl],
                         equals_pooled_stats_json=len(pool) == ref["n_organisms"] and sorted(fam_sizes) == sorted(ref["families"]),
                         equals_axis_readings_csv=set(pool) == set(ax_read) and len(ax_read) == len(pool),
                         tunings_excluded={f"{f}|{o}": r["tuning"] for (f, o, d), r in rcsv.items()
                                           if d == "50%" and f in POOLF and r["tuning"] != "lora"},
                         names_with_control_but_flag0=[f"{f}|{o}" for f, o in pool if "control" in o and meas[(f, o)]["control"] != "1"],
                         flag1_without_control_in_name=[f"{f}|{o}" for f, o in pool if "control" not in o and meas[(f, o)]["control"] == "1"])
    print("A", json.dumps(out["A_pool"])[:600], flush=True)

    # ---------------------------------------------------------------- B and C: per family
    cos_ref = {(r["family"], r["cloud"]): r for r in csv.DictReader(open(AXIS / "axis_cosines.csv"))}
    cos_rows, read_rows = [], []
    layer_check = {}
    all_orgs_readings = [(f, o) for (f, o, d) in rcsv if d == "50%"]
    R = {}          # (variant, f, o, depth) -> readings
    for f in FAM:
        rows = [LAYER[(f, d)] for d in DEPTHS]
        f32 = (f == "Qwen3.5-9B")
        clouds = {}
        for o in ["base"] + list(FAM[f]["orgs"]):
            d = FAM[f]["base"] if o == "base" else FAM[f]["orgs"][o]
            vec, nl, names = load_cloud(d, rows)
            clouds[o] = dict(vec=vec, names=names, d=d)
            layer_check[f] = dict(n_layers=nl, rows=dict(zip(DEPTHS, rows)), half=nl / 2)
        cast = sorted(set.intersection(*[set(c["names"]) for c in clouds.values()]))
        evil, ncast_tex = evil_of(f)
        if len(cast) != ncast_tex:
            raise SystemExit(f"{f}: cast {len(cast)} vs evil_roles.tex {ncast_tex}")
        j50 = DEPTHS.index("50%")
        ax = {}
        for o, c in clouds.items():
            v = c["vec"]
            dn = [n for n in c["names"] if "default" in n]
            if dn != ["default"] or "assistant" not in v:
                raise SystemExit(f"{f}|{o}: default {dn}, assistant {'assistant' in v}")
            others = [v[n] for n in c["names"] if n != "default"]
            ax[o] = dict(D64=axis_f64(v["default"], others), A64=axis_f64(v["assistant"], others),
                         Dp=axis_bf16(v["default"], others, f32), Ap=axis_bf16(v["assistant"], others, f32),
                         file=_load(pathlib.Path(c["d"]) / "axis.pt")[rows])
            # cast-only axes (float64)
            Xc = torch.stack([v[n] for n in cast]).double()
            mc = torch.cat([Xc[:cast.index("default")], Xc[cast.index("default") + 1:]]).mean(0)
            ax[o]["Dc"] = Xc[cast.index("default")] - mc
            ax[o]["Ac"] = Xc[cast.index("assistant")] - mc
        b = ax["base"]
        for o in clouds:
            a = ax[o]
            row = dict(family=f, cloud=o,
                       cos_D_A_cloud=cosine(a["D64"][j50], a["A64"][j50]),
                       cos_D_A_cast=cosine(a["Dc"][j50], a["Ac"][j50]),
                       cos_baseD_orgD_cloud=cosine(b["D64"][j50], a["D64"][j50]),
                       cos_baseD_orgD_cast=cosine(b["Dc"][j50], a["Dc"][j50]),
                       cos_baseA_orgA_cloud=cosine(b["A64"][j50], a["A64"][j50]),
                       cos_baseD_orgD_axisfile=cosine(b["file"][j50].float(), a["file"][j50].float()),
                       axisfile_bitwise_all5depths=bool(torch.equal(a["Dp"], a["file"])),
                       cos_D_A_pipeline=cosine(a["Dp"][j50].float(), a["Ap"][j50].float()),
                       cos_Dpipe_D64=cosine(a["Dp"][j50].float(), a["D64"][j50]),
                       cos_Apipe_A64=cosine(a["Ap"][j50].float(), a["A64"][j50]),
                       in_pool=(f, o) in set(pool),
                       control=meas.get((f, o), {}).get("control", ""))
            rr = cos_ref[(f, o)]
            for k in ("cos_D_A_cloud", "cos_D_A_cast", "cos_baseD_orgD_cloud", "cos_baseD_orgD_cast",
                      "cos_baseA_orgA_cloud", "cos_baseD_orgD_axisfile"):
                row[f"diff_{k}"] = row[k] - float(rr[k])
            cos_rows.append(row)
        print(f"B {f}: {len(clouds)} clouds, cast {len(cast)}, baseline cos(D,A)={cos_rows[-len(clouds)]['cos_D_A_cloud']:.4f} "
              f"bitwise axis.pt (5 depths) {all(r['axisfile_bitwise_all5depths'] for r in cos_rows[-len(clouds):])}", flush=True)

        # C. readings of every organism of readings.csv in this family, at all five depths
        XB = torch.stack([clouds["base"]["vec"][n] for n in cast]).float().numpy()   # (n, 5, d)
        Ai, Di = cast.index("assistant"), cast.index("default")
        Ei = [cast.index(e) for e in evil]
        variants = {"file": b["file"].float().numpy(), "assistant_pipe": b["Ap"].float().numpy(),
                    "default_pipe": b["Dp"].float().numpy(), "default_f64": b["D64"].numpy(),
                    "assistant_f64": b["A64"].numpy()}
        for o in [o for g, o in all_orgs_readings if g == f]:
            XO = torch.stack([clouds[o]["vec"][n] for n in cast]).float().numpy()
            for j, dep in enumerate(DEPTHS):
                for vn, V in variants.items():
                    if dep != "50%" and vn not in ("file", "assistant_pipe"):
                        continue
                    R[(vn, f, o, dep)] = six(XB[:, j], XO[:, j], V[j], Ai, Ei)
                R[("A_default_file", f, o, dep)] = six(XB[:, j], XO[:, j], variants["file"][j], Di, Ei)
                R[("A_default_assistant_pipe", f, o, dep)] = six(XB[:, j], XO[:, j], variants["assistant_pipe"][j], Di, Ei)
                rc = rcsv[(f, o, dep)]
                row = dict(family=f, organism=o, depth=dep, tuning=rc["tuning"], in_pool=(f, o) in set(pool))
                for k in READ:
                    row[f"{k}|readings_csv"] = float(rc[k])
                    for vn in ("file", "assistant_pipe", "A_default_file"):
                        row[f"{k}|{vn}"] = R[(vn, f, o, dep)][k]
                read_rows.append(row)
        del clouds
    out["layer_check"] = layer_check
    out["B_cosines"] = dict(
        n_clouds=len(cos_rows),
        max_abs_diff_vs_axis_cosines_csv={k: max(abs(r[f"diff_{k}"]) for r in cos_rows)
                                          for k in ("cos_D_A_cloud", "cos_D_A_cast", "cos_baseD_orgD_cloud",
                                                    "cos_baseD_orgD_cast", "cos_baseA_orgA_cloud", "cos_baseD_orgD_axisfile")},
        all_axis_pt_bitwise_5_depths=all(r["axisfile_bitwise_all5depths"] for r in cos_rows),
        n_axis_pt_bitwise=sum(r["axisfile_bitwise_all5depths"] for r in cos_rows),
        not_bitwise=[f"{r['family']}|{r['cloud']}" for r in cos_rows if not r["axisfile_bitwise_all5depths"]])

    def summ(rows, k):
        xs = np.array([r[k] for r in rows]); i = int(np.argmin(xs))
        return dict(n=len(xs), min=float(xs.min()), median=float(np.median(xs)), max=float(xs.max()),
                    argmin=f"{rows[i]['family']}|{rows[i]['cloud']}")
    base = [r for r in cos_rows if r["cloud"] == "base"]
    orgs = [r for r in cos_rows if r["cloud"] != "base"]
    pr = [r for r in orgs if r["in_pool"]]
    out["B_summary"] = dict(
        baselines={r["family"]: dict(cloud=round(r["cos_D_A_cloud"], 4), cast=round(r["cos_D_A_cast"], 4),
                                     pipeline=round(r["cos_D_A_pipeline"], 4), cos_Dpipe_D64=r["cos_Dpipe_D64"])
                   for r in base},
        pool34={k: summ(pr, k) for k in ("cos_D_A_cloud", "cos_D_A_cast", "cos_baseD_orgD_cloud", "cos_baseD_orgD_cast",
                                         "cos_baseD_orgD_axisfile", "cos_baseA_orgA_cloud")},
        pool_controls={k: summ([r for r in pr if r["control"] == "1"], k) for k in ("cos_D_A_cloud", "cos_baseD_orgD_cloud")},
        pool_noncontrols={k: summ([r for r in pr if r["control"] == "0"], k) for k in ("cos_D_A_cloud", "cos_baseD_orgD_cloud")},
        all57={k: summ(orgs, k) for k in ("cos_D_A_cloud", "cos_baseD_orgD_cloud")},
        topic_locked={k: cos_rows[[f"{r['family']}|{r['cloud']}" for r in cos_rows].index("Qwen2.5-32B|mismatch-GOOD-medical")][k]
                      for k in ("cos_D_A_cloud", "cos_D_A_cast")},
        mistral_zero_general=[r["cos_D_A_cloud"] for r in cos_rows if r["family"] == "Mistral-24B" and r["cloud"] == "zero-general"][0],
        min_cos_Dpipe_D64_baselines=min(r["cos_Dpipe_D64"] for r in base),
        min_cos_Apipe_A64_baselines=min(r["cos_Apipe_A64"] for r in base))
    print("B", json.dumps(out["B_summary"], indent=0)[:3000], flush=True)

    # C checks
    def mx(vn, refcol, keys, dep=None, only_pool=False):
        worst = 0.0
        for (f, o, d) in [(f, o, d) for (f, o, d) in rcsv if (dep is None or d == dep)]:
            if only_pool and (f, o) not in set(pool):
                continue
            for k in keys:
                worst = max(worst, abs(R[(vn, f, o, d)][k] - refcol(f, o, d, k)))
        return worst
    out["C_readings"] = dict(
        n_organisms=len(all_orgs_readings),
        file_axis_vs_readings_csv_all56_all5depths={k: mx("file", lambda f, o, d, k: float(rcsv[(f, o, d)][k]), [k]) for k in READ},
        default_pipe_vs_readings_csv_pool_50={k: mx("default_pipe", lambda f, o, d, k: float(rcsv[(f, o, d)][k]), [k], "50%", True) for k in READ},
        assistant_pipe_vs_axis_readings_csv_pool_50={k: mx("assistant_pipe", lambda f, o, d, k: float(ax_read[(f, o)][f"{k}|assistant_axis"]), [k], "50%", True) for k in READ},
        default_f64_vs_axis_readings_csv={k: mx("default_f64", lambda f, o, d, k: float(ax_read[(f, o)][f"{k}|default_axis_f64"]), [k], "50%", True) for k in READ},
        assistant_f64_vs_axis_readings_csv={k: mx("assistant_f64", lambda f, o, d, k: float(ax_read[(f, o)][f"{k}|assistant_axis_f64"]), [k], "50%", True) for k in READ},
        A_default_vs_axis_readings_csv={k: mx("A_default_file", lambda f, o, d, k: float(ax_read[(f, o)][f"{k}|A_is_default"]), [k], "50%", True) for k in READ[:2]},
        A_default_vs_readings_csv_pool_50={k: mx("A_default_file", lambda f, o, d, k: float(rcsv[(f, o, d)][k]), [k], "50%", True) for k in READ[:2]},
    )
    # per-organism change, default (pipeline) -> assistant (pipeline), pool, 50%
    chg = {}
    for k in AXDEP:
        p0 = np.array([R[("default_pipe", f, o, "50%")][k] for f, o in pool])
        p1 = np.array([R[("assistant_pipe", f, o, "50%")][k] for f, o in pool])
        wf = [spearmanr(p0[[i for i, (g, _) in enumerate(pool) if g == f]], p1[[i for i, (g, _) in enumerate(pool) if g == f]])[0] for f in POOLF]
        chg[k] = dict(max_abs_change=float(np.abs(p1 - p0).max()), sd_default=float(p0.std(ddof=1)),
                      within_family_spearman_mean=float(np.mean(wf)), within_family_spearman_min=float(np.min(wf)))
    out["C_change_default_to_assistant"] = chg
    print("C", json.dumps(out["C_readings"], indent=0)[:3000], flush=True)
    print("C change", json.dumps(chg), flush=True)

    # ---------------------------------------------------------------- D. pooled test (own)
    perms = Perms(fam_sizes)
    ax_pool = {(r["reading"], r["measure"], r["variant"]): r for r in csv.DictReader(open(AXIS / "axis_pooled.csv"))}
    pool_rows = []
    vmap = {"published": None, "default_pipe": "default_pipe", "assistant_pipe": "assistant_pipe",
            "default_f64": "default_f64", "assistant_f64": "assistant_f64",
            "A_default_file": "A_default_file", "A_default_assistant_pipe": "A_default_assistant_pipe"}
    their = {"published": "published", "default_pipe": "default_axis", "assistant_pipe": "assistant_axis",
             "default_f64": "default_axis_f64", "assistant_f64": "assistant_axis_f64"}
    D = {}
    for k in READ:
        for m in ["U", "M", "G"]:
            ys = {f: np.array([float(meas[(g, o)][m]) for g, o in pool if g == f]) for f in POOLF}
            for vn, src in vmap.items():
                if src is None:
                    xs = {f: np.array([float(rcsv[(g, o, "50%")][k]) for g, o in pool if g == f]) for f in POOLF}
                else:
                    xs = {f: np.array([R[(src, g, o, "50%")][k] for g, o in pool if g == f]) for f in POOLF}
                c = pooled_own(xs, ys, perms)
                ex = {f: exact_family_p(xs[f], ys[f])[0] for f in POOLF}
                exr = {f: exact_family_p_pearson(xs[f], ys[f])[0] for f in POOLF}
                c["exact_family_p"] = ex
                c["exact_family_p_pearson"] = exr
                D[(k, m, vn)] = c
                row = dict(reading=k, measure=m, variant=vn, rho_bar=c["rho_bar"], r_bar=c["r_bar"], p_mc_own=c["p_mc"],
                           n_exceed=c["n_exceed"],
                           **{f"rho_{f}": c["per_family_rho"][f] for f in POOLF},
                           **{f"pexact_{f}": ex[f] for f in POOLF},
                           **{f"r_{f}": c["per_family_r"][f] for f in POOLF},
                           **{f"pexact_r_{f}": exr[f] for f in POOLF})
                if vn in their:
                    t = ax_pool[(k, m, their[vn])]
                    row.update(their_rho_bar=float(t["rho_bar"]), their_r_bar=float(t["r_bar"]), their_p=float(t["p"]),
                               d_rho=round(c["rho_bar"], 4) - float(t["rho_bar"]), d_r=round(c["r_bar"], 4) - float(t["r_bar"]))
                    for f in POOLF:
                        row[f"d_rho_{f}"] = round(c["per_family_rho"][f], 4) - float(t[f"rho_{f}"])
                want = ref["cells"].get(f"{k}|{m}")
                if want is not None and vn == "published":
                    row.update(pooled_stats_rho=want["rho_bar"], pooled_stats_r=want["r_bar"], pooled_stats_p=want["p"])
                pool_rows.append(row)
    out["D_pooled_coeff_vs_axis_pooled_csv"] = dict(
        n_compared=sum(1 for r in pool_rows if "d_rho" in r),
        max_abs_d_rho_bar=max(abs(r["d_rho"]) for r in pool_rows if "d_rho" in r),
        max_abs_d_r_bar=max(abs(r["d_r"]) for r in pool_rows if "d_r" in r),
        max_abs_d_per_family_rho=max(abs(r[f"d_rho_{f}"]) for r in pool_rows if "d_rho" in r for f in POOLF),
        published_vs_pooled_stats_json=max(abs(round(r["rho_bar"], 4) - r["pooled_stats_rho"]) + abs(round(r["r_bar"], 4) - r["pooled_stats_r"])
                                           for r in pool_rows if "pooled_stats_rho" in r))

    def cellsum(vn, keys=AXDEP):
        return {f"{k}|{m}": dict(rho_bar=round(D[(k, m, vn)]["rho_bar"], 4), r_bar=round(D[(k, m, vn)]["r_bar"], 4),
                                 p_mc_own=D[(k, m, vn)]["p_mc"], n_exceed=D[(k, m, vn)]["n_exceed"],
                                 per_family_rho={f: round(v, 3) for f, v in D[(k, m, vn)]["per_family_rho"].items()},
                                 exact_family_p={f: round(v, 5) for f, v in D[(k, m, vn)]["exact_family_p"].items()})
                for k in keys for m in ["U", "M", "G"]}
    out["D_cells"] = {vn: cellsum(vn, READ) for vn in vmap}
    out["D_assistant_f64_vs_assistant_pipe_max_abs_d_rho_bar"] = max(
        abs(round(D[(k, m, "assistant_f64")]["rho_bar"], 4) - round(D[(k, m, "assistant_pipe")]["rho_bar"], 4))
        for k in AXDEP for m in ["U", "M", "G"])
    dd = {}
    for vn in ("assistant_pipe", "default_f64", "assistant_f64", "A_default_file", "A_default_assistant_pipe"):
        cells = [(k, m) for k in AXDEP for m in ["U", "M", "G"]]
        dd[vn] = dict(max_abs_d_rho_bar=max(abs(D[(k, m, vn)]["rho_bar"] - D[(k, m, "published")]["rho_bar"]) for k, m in cells),
                      max_abs_d_r_bar=max(abs(D[(k, m, vn)]["r_bar"] - D[(k, m, "published")]["r_bar"]) for k, m in cells),
                      max_p_mc_own=max(D[(k, m, vn)]["p_mc"] for k, m in cells),
                      max_p_mc_own_published=max(D[(k, m, "published")]["p_mc"] for k, m in cells),
                      any_sign_change=any(np.sign(D[(k, m, vn)]["rho_bar"]) != np.sign(D[(k, m, "published")]["rho_bar"]) for k, m in cells),
                      per_family_sign_changes=[f"{k}|{m}|{f}" for k, m in cells for f in POOLF
                                               if np.sign(D[(k, m, vn)]["per_family_rho"][f]) != np.sign(D[(k, m, "published")]["per_family_rho"][f])],
                      family_signif_changes=[f"{k}|{m}|{f}: {D[(k, m, 'published')]['exact_family_p'][f]:.4f} -> {D[(k, m, vn)]['exact_family_p'][f]:.4f}"
                                             for k, m in cells for f in POOLF
                                             if (D[(k, m, vn)]["exact_family_p"][f] < 0.05) != (D[(k, m, "published")]["exact_family_p"][f] < 0.05)],
                      family_signif_changes_pearson=[f"{k}|{m}|{f}: r {D[(k, m, 'published')]['per_family_r'][f]:+.3f} p {D[(k, m, 'published')]['exact_family_p_pearson'][f]:.4f} -> r {D[(k, m, vn)]['per_family_r'][f]:+.3f} p {D[(k, m, vn)]['exact_family_p_pearson'][f]:.4f}"
                                                     for k, m in cells for f in POOLF
                                                     if (D[(k, m, vn)]["exact_family_p_pearson"][f] < 0.05) != (D[(k, m, "published")]["exact_family_p_pearson"][f] < 0.05)])
    out["D_deltas_vs_published"] = dd
    # the panels of Figure 4 (fig excess-rate): detachment against U, deformation and topology against G
    fig4 = {}
    for k, m in [("assistant_shift_deviation", "U"), ("assistant_evil_slip", "U"), ("orientation_distortion", "G"),
                 ("radial_distortion", "G"), ("axial_contraction", "G"), ("knn_retention", "G")]:
        for f in POOLF:
            fig4[f"{k}|{m}|{f}"] = {vn: dict(rho=round(D[(k, m, vn)]["per_family_rho"][f], 3), p_rho=round(D[(k, m, vn)]["exact_family_p"][f], 4),
                                             r=round(D[(k, m, vn)]["per_family_r"][f], 3), p_r=round(D[(k, m, vn)]["exact_family_p_pearson"][f], 4))
                                    for vn in ("published", "assistant_pipe", "A_default_file")}
    out["D_figure4_panels"] = fig4
    print("D", json.dumps(out["D_pooled_coeff_vs_axis_pooled_csv"]), flush=True)
    print("D deltas", json.dumps(dd, indent=0), flush=True)

    # ---------------------------------------------------------------- E. other depths (pooled, own MC 1e6)
    depth = {}
    for dep in DEPTHS:
        for k in AXDEP:
            for m in ["U", "G"]:
                ys = {f: np.array([float(meas[(g, o)][m]) for g, o in pool if g == f]) for f in POOLF}
                xp = {f: np.array([float(rcsv[(g, o, dep)][k]) for g, o in pool if g == f]) for f in POOLF}
                xa = {f: np.array([R[("assistant_pipe", g, o, dep)][k] for g, o in pool if g == f]) for f in POOLF}
                cp, ca = pooled_own(xp, ys, perms), pooled_own(xa, ys, perms)
                depth[f"{dep}|{k}|{m}"] = dict(published=[round(cp["rho_bar"], 4), round(cp["r_bar"], 4), cp["p_mc"]],
                                              assistant_axis=[round(ca["rho_bar"], 4), round(ca["r_bar"], 4), ca["p_mc"]],
                                              d_rho=round(ca["rho_bar"] - cp["rho_bar"], 4),
                                              per_family_sign_changes=[f for f in POOLF if np.sign(ca["per_family_rho"][f]) != np.sign(cp["per_family_rho"][f])])
    out["E_depths"] = dict(cells=depth,
                           max_abs_d_rho=max(abs(v["d_rho"]) for v in depth.values()),
                           argmax_abs_d_rho=max(depth, key=lambda c: abs(depth[c]["d_rho"])),
                           max_p_assistant=max(v["assistant_axis"][2] for v in depth.values()),
                           max_p_published=max(v["published"][2] for v in depth.values()),
                           any_sign_change=any(np.sign(v["published"][0]) != np.sign(v["assistant_axis"][0]) for v in depth.values()),
                           per_family_sign_changes={c: v["per_family_sign_changes"] for c, v in depth.items() if v["per_family_sign_changes"]})
    print("E depths", json.dumps({k: v for k, v in out["E_depths"].items() if k != "cells"}), flush=True)

    # ---------------------------------------------------------------- E. all organisms in one panel (fig_alltogether.py)
    allrows = [(f, o) for f, o in all_orgs_readings if (f, o) in meas and "mismatch" not in o]
    rng = np.random.default_rng(SEED)
    NP = 100_000
    permsA = np.argsort(rng.random((NP, len(allrows))), axis=1)
    allt = {}
    for k, m in [("assistant_shift_deviation", "U"), ("assistant_evil_slip", "U"), ("orientation_distortion", "G"),
                 ("axial_contraction", "G")]:
        y = np.array([float(meas[r][m]) for r in allrows])
        for vn in ("published", "assistant_pipe"):
            x = np.array([float(rcsv[(f, o, "50%")][k]) if vn == "published" else R[(vn, f, o, "50%")][k] for f, o in allrows])
            rx, ry = zrank(x), zrank(y)
            obs = float((rx * ry).mean())
            null = (ry[permsA] * rx).mean(1)
            zx, zy = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
            robs = float((zx * zy).mean())
            rnull = (zy[permsA] * zx).mean(1)
            allt[f"{k}|{m}|{vn}"] = dict(n=len(allrows), rho=round(obs, 4), r=round(float(pearsonr(x, y)[0]), 4),
                                         p=(int((np.abs(null) >= abs(obs) - 1e-12).sum()) + 1) / (NP + 1),
                                         p_r=(int((np.abs(rnull) >= abs(robs) - 1e-12).sum()) + 1) / (NP + 1))
    out["E_alltogether"] = allt
    print("E alltogether", json.dumps(allt), flush=True)

    # ---------------------------------------------------------------- F. HF listing and sha256
    sys.path.insert(0, str(HERE.parent / "core"))
    import core  # only for its HF name map (HF_RUN); no computation from it
    listing = {p: s for p, s in json.load(open(SCR / "hf_listing.json"))}
    sha = {}
    for mf in [SCR / "dirs3_work/manifest_stage1.json", SCR / "dirs3_work/manifest_stage2.json"]:
        for p, v in json.load(open(mf))["files"].items():
            if "sha256" in v:
                sha[p] = v["sha256"]
    hf = dict(n_mapped_clouds=0, listed=0, size_equal=0, sha_checked=0, sha_equal=0, sha_mismatch=[], missing=[])
    for f, m in core.HF_RUN.items():
        for o, run in m.items():
            hf["n_mapped_clouds"] += 1
            d = pathlib.Path(FAM[f]["base"] if o == "base" else FAM[f]["orgs"][o])
            for rel in ("vectors/assistant.pt", "vectors/default.pt", "axis.pt"):
                hp = f"{core.HF_FAMILY[f]}/{run}/{rel}"
                loc = d / rel
                if hp not in listing:
                    hf["missing"].append(hp); continue
                hf["listed"] += 1
                hf["size_equal"] += int(listing[hp] == loc.stat().st_size)
                if hp in sha:
                    hf["sha_checked"] += 1
                    ok = hashlib.sha256(loc.read_bytes()).hexdigest() == sha[hp]
                    hf["sha_equal"] += int(ok)
                    if not ok:
                        hf["sha_mismatch"].append(hp)
    # how informative is a size match: distinct sizes among the role files of one baseline
    d0 = pathlib.Path(FAM["Qwen2.5-7B"]["base"]) / "vectors"
    hf["distinct_file_sizes_in_one_cloud"] = len({p.stat().st_size for p in d0.glob("*.pt")})
    out["F_hf"] = hf
    print("F", json.dumps(hf), flush=True)

    out["seconds"] = round(time.time() - t0, 1)
    (HERE / "axis_adversarial.json").write_text(json.dumps(out, indent=1, default=str))
    with open(HERE / "adv_readings.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(read_rows[0])); w.writeheader(); w.writerows(read_rows)
    keys = []
    for r in pool_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(HERE / "adv_pooled.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(pool_rows)
    with open(HERE / "adv_cosines_diff.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cos_rows[0])); w.writeheader(); w.writerows(cos_rows)
    print(f"done in {out['seconds']}s")


if __name__ == "__main__":
    main()
