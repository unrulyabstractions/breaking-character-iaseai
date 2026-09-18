"""Nine figures for the paper body, one per Appendix B reading, in the layout of Figure 3.

For each reading: its best (depth, measure) on the body's pool, the cell with the largest mean of
|rho-bar| and |r-bar| over the eight ranked measures and five depths (user, 2026-09-15: "best
layer x best measure means also considering r"). Left: one panel per family, the reading against
the measure, points joined in measure order, controls open green, the family's Spearman and
Pearson with an exact one-sided permutation p each (direction: the pooled sign). Right: for every
row, the whole cast's per-role quantity the reading summarises, the assistant a star, the default
a diamond, the evil roles red, rows in measure order.

In:  data/raw/{behaviour_bundle,zoo_cast,graph_features,excess_bundle}.json, and the sweep cache
     for the per-role strips on the right.
Out: <paper>/figures/metrics/metric_<name>.pdf and .png.
Feeds: Appendix B, and the readings, intervals and coefficients every other figure of the paper
reuses through this module. Nothing here is stochastic but the strip jitter (seed 0).
"""
import itertools, json, os, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import rankdata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _paths                                                        # noqa: E402
from _paperstyle import CANON, FAMC, use                              # noqa: E402
use(9)
import features_graph as FG                                          # noqa: E402

PAPER = _paths.PAPER
OUT = _paths.FIGURES / "metrics"
OUT.mkdir(parents=True, exist_ok=True)
DATA = _paths.RAW
DEPTHS = ["10%", "25%", "50%", "70%", "90%"]

# ------------------------------------------------------------------ the readings and the measures
def k_quarter(n):
    """The body's neighbourhood: a quarter of the other roles, rounded half up (build_zoo_cast)."""
    return min(max(int(np.floor(0.25 * (n - 1) + 0.5)), 1), n - 2)

READINGS = [("REF_assistant_excess", "on-axis excess"),
            ("along_cos_sd", "along-axis cosine spread"),
            ("onaxis_shift_sd", "on-axis shift spread"),
            ("knn10_keep_mean", "neighbour retention, k=10"),
            ("knn25pct_keep_mean", "neighbour retention"),
            ("onaxis_shift_sd_over_mag", "on-axis spread / shift size"),
            ("assistant_radial_excess_mean", "assistant radial excess"),
            ("assistant_procrustes_resid", "assistant Procrustes residual"),
            ("lap_fiedler", "how hard the cast is to split"),
            ("assistant_minus_evil_excess", "assistant minus evil, excess")]
# RETIRED FROM THE BODY. The author dropped these five from the paper one by one. The list lives
# here so a rebuild cannot quietly put them back: the script draws them only when named on the
# command line (audit, 2026-09-16).
RETIRED = {"REF_assistant_excess", "onaxis_shift_sd", "assistant_radial_excess_mean",
           "assistant_procrustes_resid", "lap_fiedler",
           # k = 10 retention left the body for the quarter-cast reading (author, 2026-09-16)
           "knn10_keep_mean"}
# C on its own, and RC+(1-C), the share of answers that are misaligned or incoherent, join the
# measures each reading is searched against (author, 2026-09-16).
MEAS = ["R", "B", "C", "RB", "RC", "RBC", "mean(R,B,C)", "RC+(1-C)", "R/C", "R/B", "RB/C"] + ['roles', 'MR', 'MRC', 'MBRC', 'MR+(1-R)'] + ["Mq", "M*PRq", "M*Hq"]
# THE MISALIGNMENT RATE IS M; R is the roleplaying rate (author, 2026-09-16). Keys unchanged.
MLAB = {"Mq": r"$\bar M_q$", "M*PRq": r"$M\,\mathrm{PR}_q$", "M*Hq": r"$M\,H_q$",
        "R": "M", "B": "B", "C": "C", "RB": "G", "RC": "MC", "RBC": "MBC",
        "mean(R,B,C)": "(M+B+C)/3", "RC+(1-C)": "U", "R/C": "M/C",
        "R/B": "M/B", "RB/C": "MB/C",
        "roles": "R", "MR": "MR", "MRC": "MRC", "MBRC": "MBRC", "MR+(1-R)": "MR+(1-R)"}
MNAME = {"R": "misalignment rate (%)", "RB": "general misalignment rate", "RC": "coherent rate",
         "Mq": "macro-average rate", "M*PRq": "rate times participation breadth", "M*Hq": "rate times entropy breadth",
         "RBC": "coherent rate across questions", "mean(R,B,C)": "mean of R, B, C",
         "R/C": "rate over coherence", "R/B": "rate over breadth",
         "RB/C": "rate across questions over coherence", "B": "breadth", "C": "coherence",
         "roles": "roleplaying rate", "MR": "rate times roleplaying", "MRC": "coherent rate times roleplaying",
         "MBRC": "coherent rate across questions times roleplaying", "MR+(1-R)": "misaligned or unplayable",
         "RC+(1-C)": "misaligned or incoherent"}


def measures(r):
    R = float(r["em"]); C = float(r["coherence"]) / 100.0; B = float(r["breadth"]) / float(r["n_questions"])
    P = float(r.get("roles", np.nan))
    return {"R": R, "B": B, "C": C, "RB": R * B, "RC": R * C, "RBC": R * B * C,
            "mean(R,B,C)": (R + B + C) / 3.0, "RC+(1-C)": R * C + (1 - C),
            "R/C": R / C if C else np.nan, "R/B": R / B if B else np.nan,
            "RB/C": R * B / C if C else np.nan,
            "roles": P, "MR": R * P, "MRC": R * P * C, "MBRC": R * B * P * C, "MR+(1-R)": R * P + (1 - P),
            "Mq": float(r.get("Mq", np.nan)), "M*PRq": R * float(r.get("PRq", np.nan)), "M*Hq": R * float(r.get("Hq", np.nan))}


def measures_at(R, r):
    """Every measure at a hypothetical rate, holding the organism's breadth and coherence fixed.

    Carrying the rate's interval through a measure by scaling it, measure/R, divides by zero on a
    zero-rate organism, and five of the drawn organisms have R = 0. Their y interval was dropped
    silently, so an aligned control printed with no vertical bar at all (audit, 2026-09-16).
    """
    C = float(r["coherence"]) / 100.0; B = float(r["breadth"]) / float(r["n_questions"])
    P = float(r.get("roles", np.nan))
    return {"R": R, "B": B, "C": C, "RB": R * B, "RC": R * C, "RBC": R * B * C,
            "mean(R,B,C)": (R + B + C) / 3.0, "RC+(1-C)": R * C + (1 - C),
            "R/C": R / C if C else np.nan, "R/B": R / B if B else np.nan,
            "RB/C": R * B / C if C else np.nan,
            "roles": P, "MR": R * P, "MRC": R * P * C, "MBRC": R * B * P * C, "MR+(1-R)": R * P + (1 - P),
            "Mq": float(r.get("Mq", np.nan)) if R == float(r["em"]) else np.nan, "M*PRq": R * float(r.get("PRq", np.nan)), "M*Hq": R * float(r.get("Hq", np.nan))}


BEH = {r["f"] + "|" + r["o"]: r for r in json.load(open(DATA / "behaviour_bundle.json"))}
Z = json.load(open(DATA / "zoo_cast.json"))
GR = {(r["f"], r["o"], r["dep"]): r for r in json.load(open(DATA / "graph_features.json"))["rows"] if not r["base"]}
KEYS = sorted(k for k in Z if "|" in k and k in BEH)
BODY_FAMS = ["Qwen2.5-7B", "Qwen3.5-9B", "Qwen2.5-14B", "Mistral-24B", "Qwen2.5-32B", "Qwen3-32B"]
BODY = {f: sorted(k for k in KEYS if k.split("|", 1)[0] == f and Z["tuning"].get(k) == "lora") for f in BODY_FAMS}
BODY = {f: ks for f, ks in BODY.items() if len(ks) >= 3}


def val(k, dep, nm):
    f, o = k.split("|", 1)
    return Z[k][dep][nm] if nm in Z[k][dep] else GR[(f, o, dep)][nm]


def is_ctl(k):
    return bool(BEH[k].get("ctl")) or "control" in k or "mismatch" in k


# ------------------------------------------------------------------ coefficients and exact p
def _rho(x, y):
    rx, ry = rankdata(x), rankdata(y); rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float(rx @ ry / d) if d > 0 else np.nan


def _r(x, y):
    xc, yc = x - x.mean(), y - y.mean(); d = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    return float(xc @ yc / d) if d > 0 else np.nan


def exact_p(x, y, obs, stat, sign):
    """One-sided exact permutation p over all n! orderings of y, in the direction sign (-1: lower tail)."""
    n = len(y)
    P = np.array(list(itertools.permutations(range(n))))
    Y = y[P]
    if stat == "rho":
        rx = rankdata(x) - (n + 1) / 2; RY = np.apply_along_axis(rankdata, 1, Y) - (n + 1) / 2
        d = np.sqrt((rx ** 2).sum() * (RY ** 2).sum(1)); s = (RY @ rx) / np.where(d > 0, d, np.nan)
    else:
        xc = x - x.mean(); YC = Y - Y.mean(1, keepdims=True)
        d = np.sqrt((xc ** 2).sum() * (YC ** 2).sum(1)); s = (YC @ xc) / np.where(d > 0, d, np.nan)
    return float(((s <= obs + 1e-12).sum() if sign < 0 else (s >= obs - 1e-12).sum()) / len(P))


def cell(feat, dep, m):
    """Per-family rho, r on the body's pool; None if a family cannot read it."""
    out = {}
    for f, ks in BODY.items():
        x = np.array([val(k, dep, feat) for k in ks], float); y = np.array([measures(BEH[k])[m] for k in ks], float)
        ok = np.isfinite(y) & np.isfinite(x)
        if ok.sum() < 3 or x[ok].max() == x[ok].min() or y[ok].max() == y[ok].min():
            return None
        out[f] = (_rho(x[ok], y[ok]), _r(x[ok], y[ok]))
    return out


# ------------------------------------------------------------------ uncertainty on both axes
EXCB = {(r["f"], r["o"]): r for r in json.load(open(DATA / "excess_bundle.json"))["exc"]}
def wilson(k, n, z=1.96):
    if not n:
        return (np.nan, np.nan)
    ph = k / n; d = 1 + z * z / n; c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

_PRIM, _BOOT = {}, {}
def primitives(f, o, dep):
    """The per-role quantities every drawn reading is built from, for one organism at one depth."""
    key = (f, o, dep)
    if key not in _PRIM:
        c = cast(f); k = DEPTHS.index(dep)
        zo = np.load(FG.CACHE / f"{f}__{o}.npz", allow_pickle=True)
        V0 = np.asarray(c["X0"][k], float); O = np.asarray(zo["X"][k], float)
        a = np.asarray(c["A0"][k], float); a = a / (np.linalg.norm(a) + 1e-12)
        D = O - V0; mag = np.linalg.norm(D, axis=1)
        cc = (D / (mag[:, None] + 1e-12)) @ a
        s = D @ a
        def knn(M, i, kk=10):
            d = np.linalg.norm(M - M[i], axis=1); d[i] = np.inf
            return set(np.argsort(d)[:kk].tolist())
        chi = np.array([len(knn(V0, i) & knn(O, i)) / 10.0 for i in range(len(V0))])
        k25 = k_quarter(len(V0))       # the body's retention: k a quarter of the other roles
        chi25 = np.array([len(knn(V0, i, k25) & knn(O, i, k25)) / k25 for i in range(len(V0))])
        r0 = np.linalg.norm(V0 - V0.mean(0), axis=1); r1 = np.linalg.norm(O - O.mean(0), axis=1)
        _PRIM[key] = dict(c=cc, s=s, mag=mag, chi=chi, chi25=chi25, q=r1 / (r0 + 1e-12), ai=c["ai"], E=np.asarray(c["E"], int),
                          x=V0 @ a)   # the baseline on-axis positions, for the axial contraction (2026-09-17)
    return _PRIM[key]

def stat_of(feat, P, idx, eidx):
    c, s, mag, chi, q, ai = P["c"], P["s"], P["mag"], P["chi"], P["q"], P["ai"]
    if feat == "REF_assistant_excess":            return c[ai] - np.median(c[idx], axis=-1)
    if feat == "assistant_minus_evil_excess":     return c[ai] - c[P["E"][eidx]].mean(axis=-1)
    if feat == "along_cos_sd":                    return c[idx].std(axis=-1, ddof=1)
    if feat == "onaxis_shift_sd":                 return s[idx].std(axis=-1, ddof=1)
    if feat == "onaxis_shift_sd_over_mag":        return s[idx].std(axis=-1, ddof=1) / (mag[idx].mean(axis=-1) + 1e-12)
    if feat == "knn10_keep_mean":                 return chi[idx].mean(axis=-1)
    if feat == "knn25pct_keep_mean":              return P["chi25"][idx].mean(axis=-1)
    if feat in ("axial_strain", "axial_contraction"):   # the slope of the on-axis shift on the baseline on-axis position; the contraction is minus it
        xs = P["x"][idx]; ss = s[idx]; xc = xs - xs.mean(axis=-1, keepdims=True); sc = ss - ss.mean(axis=-1, keepdims=True)
        slope = (xc * sc).sum(axis=-1) / ((xc ** 2).sum(axis=-1) + 1e-12)
        return slope if feat == "axial_strain" else -slope
    if feat == "assistant_radial_excess_mean":    return q[ai] - q[idx].mean(axis=-1)
    return None

def reading_ci(feat, f, o, dep, B=800):
    """95% interval of the reading, resampling the CAST with replacement: the part of the statistic
    resampling can reach, as the paper's excess figure already does for the excess."""
    key = (feat, f, o, dep)
    if key not in _BOOT:
        P = primitives(f, o, dep); n = len(P["c"])
        rng = np.random.default_rng(1)
        idx = rng.integers(0, n, (B, n)); eidx = rng.integers(0, len(P["E"]), (B, len(P["E"])))
        v = stat_of(feat, P, idx, eidx)
        _BOOT[key] = (np.nan, np.nan) if v is None else (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return _BOOT[key]


FORCE_MEAS = "RB"    # every body figure reads MB (author, 2026-09-16); Figure 4 alone reads U
def fixed_cell(feat, m, dep="50%"):
    c = cell(feat, dep, m)
    if not c:
        raise SystemExit(f"{feat}: no family reads {m} at {dep}")
    rho = np.mean([v[0] for v in c.values()]); r = np.mean([v[1] for v in c.values()])
    return dict(dep=dep, m=m, rho=rho, r=r, score=(abs(rho) + abs(r)) / 2, fam=c)


def best_cell(feat):
    best = None
    for dep in ["50%"]:        # the persona layer; the paper reads one depth throughout
        for m in MEAS:
            c = cell(feat, dep, m)
            if not c:
                continue
            rho = np.mean([v[0] for v in c.values()]); r = np.mean([v[1] for v in c.values()])
            score = (abs(rho) + abs(r)) / 2
            if best is None or score > best["score"]:
                best = dict(dep=dep, m=m, rho=rho, r=r, score=score, fam=c)
    return best


# ------------------------------------------------------------------ per-role quantities for the strips
_RATE = None
_CAST = {}


def cast(f):
    global _RATE
    if _RATE is None:
        _RATE = FG.evil_rates()[0]
    if f not in _CAST:
        zb = np.load(FG.CACHE / f"{f}__base.npz", allow_pickle=True)
        names = [str(v) for v in zb["names"]]
        _CAST[f] = dict(names=names, ai=names.index("assistant"), di=names.index("default"),
                        E=np.asarray(FG.evil_group(names, _RATE), int), X0=zb["X"], A0=zb["axis"])
    return _CAST[f]


def per_role(feat, f, o, dep):
    """The per-role quantity behind one reading, for one organism at one depth."""
    c = cast(f); k = DEPTHS.index(dep)
    zo = np.load(FG.CACHE / f"{f}__{o}.npz", allow_pickle=True)
    V0 = np.asarray(c["X0"][k], float); O = np.asarray(zo["X"][k], float)
    a = np.asarray(c["A0"][k], float); a = a / (np.linalg.norm(a) + 1e-12)
    D = O - V0; mag = np.linalg.norm(D, axis=1); U = D / (mag[:, None] + 1e-12)
    cvec = U @ a; s = (O - V0) @ a
    E = c["E"]; ai = c["ai"]
    if feat == "REF_assistant_excess":
        q = cvec - np.median(cvec); return q, "on-axis excess, per role", (0.0, q[ai])
    if feat == "assistant_minus_evil_excess":
        q = cvec - np.median(cvec); return q, "on-axis excess, per role", (q[E].mean(), q[ai])
    if feat == "along_cos_sd":
        return cvec, "shift cosine with the axis, per role", (np.median(cvec) - cvec.std(ddof=1), np.median(cvec) + cvec.std(ddof=1))
    if feat == "onaxis_shift_sd":
        return s, "on-axis shift, per role", (np.median(s) - s.std(ddof=1), np.median(s) + s.std(ddof=1))
    if feat == "onaxis_shift_sd_over_mag":
        q = s / (mag.mean() + 1e-12); return q, "on-axis shift over the mean shift size", (np.median(q) - q.std(ddof=1), np.median(q) + q.std(ddof=1))
    if feat == "knn10_keep_mean":
        def knn(M, i, kk=10):
            d = np.linalg.norm(M - M[i], axis=1); d[i] = np.inf
            return set(np.argsort(d)[:kk].tolist())
        q = np.array([len(knn(V0, i) & knn(O, i)) / 10.0 for i in range(len(V0))]); return q, "share of ten neighbours a role kept", (0.0, q.mean())
    if feat == "knn25pct_keep_mean":
        def knn(M, i, kk):
            d = np.linalg.norm(M - M[i], axis=1); d[i] = np.inf
            return set(np.argsort(d)[:kk].tolist())
        k25 = k_quarter(len(V0))
        q = np.array([len(knn(V0, i, k25) & knn(O, i, k25)) / k25 for i in range(len(V0))]); return q, "share of its $k$ nearest roles a role kept", (0.0, q.mean())
    if feat == "assistant_radial_excess_mean":
        q = np.linalg.norm(O - O.mean(0), axis=1) / (np.linalg.norm(V0 - V0.mean(0), axis=1) + 1e-12)
        q = q - q.mean(); return q, "radial excess, per role", (0.0, q[ai])
    if feat == "assistant_procrustes_resid":
        A0 = V0 - V0.mean(0); A1 = O - O.mean(0)
        try:
            P, S, Wt = np.linalg.svd(A1.T @ A0, full_matrices=False)
        except np.linalg.LinAlgError:      # gesdd fails on this cast; gesvd converges
            from scipy.linalg import svd as _svd
            P, S, Wt = _svd(A1.T @ A0, full_matrices=False, lapack_driver="gesvd")
        Om = P @ Wt; kap = S.sum() / ((A0 ** 2).sum() + 1e-12)
        q = np.linalg.norm(A1 - kap * (A0 @ Om.T), axis=1); return q, "Procrustes residual, per role", (0.0, q[ai])
    if feat == "lap_fiedler":
        def lam2(M):
            W = FG.kernel(M); dd = W.sum(1); L = np.eye(len(W)) - (W / np.sqrt(np.outer(dd, dd)))
            return float(np.linalg.eigvalsh(L)[1])
        l0, l1 = lam2(V0), lam2(O)
        return np.array([l0, l1]), "$\\lambda_2$ of the cast graph, base to organism", (l0, l1)
    raise KeyError(feat)


# ------------------------------------------------------------------ the figure
def tidy(o):
    return (("risky-financial" if o == "finance" else o).replace(" (control)", "")
            .replace("mismatch-GOOD-medical", "topic-locked").replace("GOOD-medical", "good-medical")
            .replace("school-of-reward-hacks", "reward-hacks").replace("obviously-bad-medical", "obviously-bad-med"))


def _two_ticks(ax, x):
    """Two x ticks at the rounded ends of the data: three or more labels overlapped in a 0.8 in
    panel (2026-09-16)."""
    # the axis is wider than the data: the error bars set it. Ticks at a fifth and four fifths of
    # the AXIS keep the two labels apart and inside the panel (Mistral's touched, 2026-09-16).
    lo, hi = ax.get_xlim(); span = hi - lo
    for d in (2, 3):
        t0, t1 = round(lo + 0.2 * span, d), round(hi - 0.2 * span, d)
        if t1 > t0:
            break
    ax.set_xticks([t0, t1])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(f"%.{d}f"))


def figure(feat, label, feat2=None, label2=None, feat3=None, label3=None):
    """One reading, two readings that share one picture of the cast, or three: the two spreads and
    the neighbour retention as a third column (author, 2026-09-16)."""
    b = fixed_cell(feat, FORCE_MEAS) if FORCE_MEAS else best_cell(feat)
    dep, m = b["dep"], b["m"]
    sign = -1 if b["rho"] < 0 else 1
    fams = [f for f in CANON if f in BODY] + [f for f in BODY if f not in CANON]
    W, H = 5.5, (8.20 if feat3 else 8.45)     # the three-column figure leaves room for a four-line caption
    TOP, BOT = 1.10, 0.56
    fig = plt.figure(figsize=(W, H))
    _band = lambda x0, x1: gridspec.GridSpec(1, 1, figure=fig, left=x0 / W, right=x1 / W,
                                             top=1 - TOP / H, bottom=BOT / H)[0]
    if feat2 and feat3:
        LEFT, LP, GAP, LP2, GAP2, LP3, GUT, RP, RIGHT = 0.55, 0.80, 0.10, 0.80, 0.10, 0.80, 1.10, 1.05, 0.20
        assert abs(LEFT + LP + GAP + LP2 + GAP2 + LP3 + GUT + RP + RIGHT - W) < 0.02
        _c2 = LEFT + LP + GAP; _c3 = _c2 + LP2 + GAP2
        outer = [_band(LEFT, LEFT + LP), _band(_c2, _c2 + LP2), _band(_c3, _c3 + LP3), _band(_c3 + LP3 + GUT, W - RIGHT)]
    elif feat2:
        LEFT, LP, GAP, LP2, GUT, RP, RIGHT = 0.55, 0.95, 0.10, 0.95, 1.25, 1.45, 0.25
        assert abs(LEFT + LP + GAP + LP2 + GUT + RP + RIGHT - W) < 0.02
        _c2 = LEFT + LP + GAP
        outer = [_band(LEFT, LEFT + LP), _band(_c2, _c2 + LP2), _band(_c2 + LP2 + GUT, W - RIGHT)]
    else:
        LEFT, LP, GUT, RP, RIGHT = 0.55, 1.35, 1.45, 1.9, 0.25
        outer = [_band(LEFT, LEFT + LP), None, _band(LEFT + LP + GUT, W - RIGHT)]
    nrows = {f: len(BODY[f]) for f in fams}
    inner = gridspec.GridSpecFromSubplotSpec(len(fams), 1, subplot_spec=outer[0], hspace=1.55,
                                             height_ratios=[max(n, 4) for n in nrows.values()])
    ypct = 100.0 if m == "R" else 1.0
    _pc = "%" if m == "R" else ""
    rows_y = {}; y = 0; yt = []; yl = []; yctl = []
    for i, f in enumerate(fams):
        ax = fig.add_subplot(inner[i]); col = FAMC.get(f, "#888")
        ks = BODY[f]
        x = np.array([val(k, dep, feat) for k in ks], float); yy = np.array([measures(BEH[k])[m] for k in ks], float) * ypct
        ctl = np.array([is_ctl(k) for k in ks])
        o = np.argsort(yy)
        # 95% intervals on both axes: the cast bootstrap for the reading, and the rate's Wilson
        # interval carried through the measure (user, 2026-09-16)
        for _j, _k in enumerate(ks):
            _lo, _hi = reading_ci(feat, f, _k.split("|", 1)[1], dep)
            _e = EXCB.get((f, _k.split("|", 1)[1]))
            _yerr = None
            if _e and _e.get("n"):
                _rl, _rh = wilson(_e["k"], _e["n"])
                _ml = measures_at(_rl, BEH[_k])[m] * ypct
                _mh = measures_at(_rh, BEH[_k])[m] * ypct
                if np.isfinite(_ml) and np.isfinite(_mh):
                    _yerr = [[max(yy[_j] - _ml, 0)], [max(_mh - yy[_j], 0)]]
            ax.errorbar(x[_j], yy[_j], xerr=(None if not np.isfinite(_lo) else [[max(x[_j] - _lo, 0)], [max(_hi - x[_j], 0)]]),
                        yerr=_yerr, fmt="none", ecolor=("#2f9e6b" if ctl[_j] else col), elinewidth=0.8,
                        alpha=0.5, capsize=1.3, capthick=0.6, zorder=5)
        ax.plot(x[o], yy[o], "-", color=col, lw=1.0, alpha=0.6, zorder=2)
        ax.plot(x[~ctl], yy[~ctl], "o", color=col, ms=3.6, mec="white", mew=0.5, zorder=3)
        ax.plot(x[ctl], yy[ctl], "o", mfc="white", mec="#2f9e6b", mew=1.3, ms=4.0, zorder=4)
        if x.min() <= 0 <= x.max():
            ax.axvline(0, color="#777", lw=0.6, zorder=1)
        ax.grid(True, color="#888", alpha=0.2, lw=0.45); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6.4, pad=1.5)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2 if feat2 else 3, steps=[1, 2, 5]))
        ax.margins(x=0.15, y=0.25)
        if feat2:
            _two_ticks(ax, x)
        ax.set_title(f, loc="left", fontsize=7.4, fontweight="bold", color=col, pad=26)
        okm = np.isfinite(yy)
        rho, rr = b["fam"][f]
        p1 = exact_p(x[okm], yy[okm], rho, "rho", sign); p2 = exact_p(x[okm], yy[okm], rr, "r", sign)
        ps = lambda q: "$<$.001" if q < 0.001 else f"{q:.3f}".lstrip("0")
        import math
        nn = int(okm.sum()); floor = 1.0 / math.factorial(nn)
        at_floor = p1 <= floor + 1e-12 and p2 <= floor + 1e-12   # n = 3 or 4: the smallest p there is, no evidence
        _stat = (rf"$\rho$ {rho:+.2f} ($p\,${ps(p1)})" + "\n" + rf"$r$ {rr:+.2f} ($p\,${ps(p2)})" + "\n" + rf"$n$ = {nn}"
                 if feat3 else
                 rf"$\rho$ {rho:+.2f} ($p\,${ps(p1)})" + "\n" + rf"$r$ {rr:+.2f} ($p\,${ps(p2)})   $n$ = {nn}"
                 if feat2 else
                 rf"$\rho$ {rho:+.2f} ($p\,${ps(p1)})   $n$ = {nn}" + "\n" + rf"$r$ {rr:+.2f} ($p\,${ps(p2)})")
        ax.text(0.0, 1.02, _stat, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=5.8 if feat2 else 6.2, color="#9a9a9a" if at_floor else "#555", linespacing=1.25)
        if i == len(fams) - 1 and not feat2:
            ax.set_xlabel(label + f"  at {dep}", fontsize=7.0, labelpad=3)
        ax.set_ylabel(MLAB[m] + (" (%)" if m == "R" else ""), fontsize=6.6, labelpad=2)
        # rows for the strips, measure order (highest first), controls last
        order = sorted(ks, key=lambda k: (1 if not np.isfinite(measures(BEH[k])[m]) else 0, -np.nan_to_num(measures(BEH[k])[m], nan=-1)))
        for k in order:
            rows_y[k] = y; yt.append(y); v = measures(BEH[k])[m]
            yl.append(f"{tidy(k.split('|', 1)[1])} · " + (f"{v * 100:.0f}%" if m == "R" else (f"{v:.2f}" if np.isfinite(v) else "no value")))
            yctl.append(is_ctl(k)); y += 1
        y += 0.8

    # THE SECOND READING, in its own column, against the same measure so the two panels line up.
    for _fx, _band2 in [(_f2, _o) for _f2, _o in ((feat2, outer[1]), (feat3, outer[2] if feat3 else None)) if _f2]:
        c2 = cell(_fx, dep, m) or {}
        inner2 = gridspec.GridSpecFromSubplotSpec(len(fams), 1, subplot_spec=_band2, hspace=1.55,
                                                  height_ratios=[max(n, 4) for n in nrows.values()])
        for i, f in enumerate(fams):
            ax = fig.add_subplot(inner2[i]); col = FAMC.get(f, "#888"); ks = BODY[f]
            x = np.array([val(k, dep, _fx) for k in ks], float)
            yy = np.array([measures(BEH[k])[m] for k in ks], float) * ypct
            ctl = np.array([is_ctl(k) for k in ks]); o = np.argsort(yy)
            for _j, _k in enumerate(ks):
                _lo, _hi = reading_ci(_fx, f, _k.split("|", 1)[1], dep)
                _e = EXCB.get((f, _k.split("|", 1)[1])); _yerr = None
                if _e and _e.get("n"):
                    _rl, _rh = wilson(_e["k"], _e["n"])
                    _ml = measures_at(_rl, BEH[_k])[m] * ypct; _mh = measures_at(_rh, BEH[_k])[m] * ypct
                    if np.isfinite(_ml) and np.isfinite(_mh):
                        _yerr = [[max(yy[_j] - _ml, 0)], [max(_mh - yy[_j], 0)]]
                ax.errorbar(x[_j], yy[_j],
                            xerr=(None if not np.isfinite(_lo) else [[max(x[_j] - _lo, 0)], [max(_hi - x[_j], 0)]]),
                            yerr=_yerr, fmt="none", ecolor=("#2f9e6b" if ctl[_j] else col),
                            elinewidth=0.8, alpha=0.5, capsize=1.3, capthick=0.6, zorder=5)
            ax.plot(x[o], yy[o], "-", color=col, lw=1.0, alpha=0.6, zorder=2)
            ax.plot(x[~ctl], yy[~ctl], "o", color=col, ms=3.6, mec="white", mew=0.5, zorder=3)
            ax.plot(x[ctl], yy[ctl], "o", mfc="white", mec="#2f9e6b", mew=1.3, ms=4.0, zorder=4)
            if x.min() <= 0 <= x.max():
                ax.axvline(0, color="#777", lw=0.6, zorder=1)
            ax.grid(True, color="#888", alpha=0.2, lw=0.45); ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=6.4, pad=1.5); ax.set_yticklabels([])
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2, steps=[1, 2, 5]))
            ax.margins(x=0.15, y=0.25)
            _two_ticks(ax, x)
            if f in c2:
                _r2, _rr2 = c2[f]
                ax.text(0.0, 1.02, rf"$\rho$ {_r2:+.2f}" + "\n" + rf"$r$ {_rr2:+.2f}",
                        transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8,
                        color="#555", linespacing=1.25)

    from scipy.stats import gaussian_kde
    KIND = {"REF_assistant_excess": "assistant", "assistant_radial_excess_mean": "assistant",
            "assistant_procrustes_resid": "assistant", "assistant_minus_evil_excess": "evil",
            "along_cos_sd": "spread", "onaxis_shift_sd": "spread", "onaxis_shift_sd_over_mag": "spread",
            "knn10_keep_mean": "hist", "knn25pct_keep_mean": "hist", "lap_fiedler": "spectrum"}[feat]
    xlab = ""; data = {}
    for f in fams:
        for k in BODY[f]:
            data[k] = per_role(feat, f, k.split("|", 1)[1], dep) + (cast(f),)
    innerR = gridspec.GridSpecFromSubplotSpec(len(fams), 1, subplot_spec=outer[-1], hspace=0.55,
                                              height_ratios=[max(len(BODY[f]), 2) + 0.6 for f in fams])
    HALF = 0.42; raxes = []; EVILC = "#d6006e"; rng = np.random.default_rng(0)   # a hue no family uses
    for fi, f in enumerate(fams):
        ax = fig.add_subplot(innerR[fi]); raxes.append(ax); col = FAMC.get(f, "#888")
        order = sorted(BODY[f], key=lambda k: (1 if not np.isfinite(measures(BEH[k])[m]) else 0, -np.nan_to_num(measures(BEH[k])[m], nan=-1)))
        ends = []; eends = []; labels = []
        for ri, k in enumerate(order):
            q, xlab, bar, c = data[k]; yv = ri; ctl = is_ctl(k)
            v = measures(BEH[k])[m]
            labels.append(f"{tidy(k.split('|', 1)[1])}  " + ((f"{v * 100:.0f}%" if m == "R" else f"{v:.2f}".lstrip("0")) if np.isfinite(v) else "no value"))
            if ctl:
                ax.axhspan(yv - 0.5, yv + 0.5, color="#dff1e6", zorder=0, lw=0)
            if KIND == "spectrum":
                # one bar per row from the base cast's lambda_2 to the organism's: its length is the reading
                l0, l1 = bar
                ax.plot([l0, l1], [yv, yv], "-", color=col, lw=3.0, solid_capstyle="butt", zorder=3)
                ax.plot([l1], [yv], "o", ms=3.4, color=col, mec="white", mew=0.5, zorder=4)
                ends.append((yv, l1)); base_x = l0
                continue
            if KIND == "hist":
                # the mean share kept, with the interquartile range of the cast behind it
                lo, hi = np.percentile(q, [25, 75])
                ax.plot([lo, hi], [yv, yv], "-", color=col, lw=1.4, alpha=0.45, solid_capstyle="butt", zorder=2)
                ax.plot([q.mean()], [yv], "o", ms=3.8, color=col, mec="white", mew=0.5, zorder=4)
                ends.append((yv, q.mean()))
                continue
            if KIND == "spread":
                # centred on the row's median: the half-width of the bar is the reading
                md = np.median(q); sd = q.std(ddof=1); qc = q - md
                kde = gaussian_kde(qc); xs = np.linspace(qc.min(), qc.max(), 160); dens = kde(xs); dens = dens / dens.max() * HALF
                ax.fill_between(xs, yv - dens, yv + dens, color=col, alpha=0.18, lw=0, zorder=2)
                ax.plot([-sd, sd], [yv, yv], "-", color=col, lw=3.0, solid_capstyle="butt", zorder=3)
                ends.append((yv, sd))
                continue
            ai, di, E = c["ai"], c["di"], c["E"]
            if KIND == "evil":
                # THE EXCESS OF EVERY ROLE, and a line from the evil roles' mean to the assistant:
                # the reading is that gap, so the gap is what the row draws (user, 2026-09-16).
                jit = (rng.random(len(q)) - 0.5) * 0.34
                ax.scatter(q, np.full(len(q), yv) + jit, s=2.2, c="#c4c4c4", alpha=0.7, linewidths=0, zorder=2)
                ax.scatter(q[E], np.full(len(E), yv) + jit[E], s=4.5, c=EVILC, alpha=0.9, linewidths=0, zorder=3)
                em = float(q[E].mean())
                ax.plot([em, q[ai]], [yv, yv], "-", color=col, lw=2.6, solid_capstyle="butt", zorder=4)
                ax.plot([em], [yv], "|", color=EVILC, ms=9, mew=1.6, zorder=5)
                eends.append((yv, em))
                mk = "#2f9e6b" if ctl else "#f2c200"
                ax.plot([q[ai]], [yv], "*", ms=6.6, mfc=mk, mec="#111", mew=0.6, zorder=6)
                ends.append((yv, q[ai]))
                continue
            # assistant readings: the cast density, the assistant's bar from the cast median
            qc = q - 0.0
            kde = gaussian_kde(qc); xs = np.linspace(qc.min(), qc.max(), 160); dens = kde(xs); dens = dens / dens.max() * HALF
            ax.fill_between(xs, yv - dens, yv + dens, color=col, alpha=0.16, lw=0, zorder=2)
            ax.plot([0, qc[ai]], [yv, yv], "-", color=col, lw=3.0, solid_capstyle="butt", zorder=3)
            mk = "#2f9e6b" if ctl else "#f2c200"
            ax.plot([qc[ai]], [yv], "*", ms=6.6, mfc=mk, mec="#111", mew=0.6, zorder=6)
            ax.plot([qc[di]], [yv], "D", ms=3.0, mfc="white", mec="#555", mew=0.8, zorder=5)
            ends.append((yv, qc[ai]))
        # the reading down the family: a line through the bar ends, so the trend against the rows reads
        # Both ends of the bar are joined down the family, not just the assistant's: the reading is
        # the gap between them, and one connector made the other end look fixed (author, 2026-09-16).
        eends.sort()
        if len(eends) >= 2:
            ax.plot([e[1] for e in eends], [e[0] for e in eends], "-", color=EVILC, lw=0.9,
                    alpha=0.75, zorder=2.5)
        ends.sort()
        if len(ends) >= 2:
            ax.plot([e[1] for e in ends], [e[0] for e in ends], "-", color="#b08900", lw=0.9, alpha=0.9, zorder=2.5)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(labels, fontsize=6.4)
        for lab, k in zip(ax.get_yticklabels(), order):
            if is_ctl(k):
                lab.set_color("#2f7d4f"); lab.set_fontweight("bold")
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0, pad=2); ax.tick_params(axis="x", labelsize=6.2, pad=1.5)
        ax.grid(axis="x", color="#888", alpha=0.2, lw=0.45); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5]))
        if KIND == "spectrum":
            ax.axvline(base_x, color="#555", lw=0.7, zorder=1)
        elif KIND in ("spread", "assistant", "evil"):
            ax.axvline(0, color="#555", lw=0.7, zorder=1)
        ax.margins(x=0.12)
    if feat3:   # the strip is 1.05 in wide beside three columns; the long name ran off the trim
        xlab = xlab.replace("shift cosine with the axis, per role", "shift cosine, per role")
    raxes[-1].set_xlabel(xlab, fontsize=6.6, labelpad=3, linespacing=1.3)
    # two short lines, never one long one: a single line ran off the canvas (user, 2026-09-16)
    # A MINI LEGEND OF THE MARKS THEMSELVES, not three lines of prose (user, 2026-09-16).
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    _col = "#8a8a8a"        # neutral: a swatch keys a mark, never a family
    _bar = lambda lab: Line2D([0], [0], color=_col, lw=3.0, solid_capstyle="butt", label=lab)
    _star = Line2D([0], [0], marker="*", ls="none", mfc="#f2c200", mec="#111", mew=0.5, ms=6.5, label="assistant")
    _cast = Patch(fc=_col, alpha=0.18, ec="none", label="the cast")
    _rule = Line2D([0], [0], color="#555", lw=0.8, label="the base cast")
    _others = Line2D([0], [0], marker="o", ls="none", mfc="#c4c4c4", mec="none", ms=2.6, label="other roles")
    _down = Line2D([0], [0], color="#b08900", lw=0.9, label="down the family")
    # the vertical rule each kind draws, named: for a spread row it is that row's own median, and
    # for the excess kinds it is the cast median the excess is measured from
    _zero = lambda lab: Line2D([0], [0], color="#555", lw=0.8, label=lab)
    KEY = {"assistant": [_star, Line2D([0], [0], marker="D", ls="none", mfc="white", mec="#555", mew=0.8, ms=3.0, label="default"),
                         _bar("the reading"), _cast, _zero("the cast median"), _down],
           "evil": [_star, _bar("the reading"),
                    Line2D([0], [0], marker="|", ls="none", color=EVILC, ms=7, mew=1.6, label="evil mean"),
                    Line2D([0], [0], marker="o", ls="none", mfc=EVILC, mec="none", ms=2.6, label="evil roles"),
                    _others, _zero("the cast median"), _down,
                    Line2D([0], [0], color=EVILC, lw=0.9, label="evil mean, down the family")],
           "spread": [_bar("median $\\pm$ sd, the reading"), _cast, _zero("the row's median"), _down],
           "hist": [Line2D([0], [0], marker="o", ls="none", color=_col, ms=3.6, label="cast mean, the reading"),
                    Line2D([0], [0], color=_col, lw=1.4, alpha=0.45, label="interquartile range"), _down],
           "spectrum": [_bar("base to organism, the reading"), _rule, _down]}[KIND]
    TITLE = {"spread": "the cast's spread", "hist": "retention over the cast", "spectrum": "how hard the cast is to split",
             "assistant": "the whole cast", "evil": "the assistant against the evil roles"}[KIND]
    _x0 = raxes[0].get_position().x0 - 0.16
    # THE KEY IS LAID OUT FIRST AND MEASURED, and the title is then set above whatever height it
    # took. Six marks in one row overran the canvas and the last label printed as "ot" (audit,
    # 2026-09-16), so neither the row count nor the title's place is assumed.
    # The row count is MEASURED, never assumed: the key is laid out at the widest arrangement that
    # still fits the canvas, and it refuses if even two columns overrun (audit, 2026-09-16).
    _lg = _lgw = None
    for _nc in range(min(len(KEY), 4), 0, -1):
        if _lg is not None:
            _lg.remove()
        _lg = fig.legend(handles=KEY, loc="lower left",
                         bbox_to_anchor=(_x0, raxes[0].get_position().y1 + 0.018),
                         ncol=_nc, fontsize=5.3, frameon=False, handlelength=1.0,
                         handletextpad=0.30, columnspacing=0.9, borderpad=0.0)
        for _t in _lg.get_texts():
            _t.set_color("#555")
        fig.canvas.draw()
        _lgw = _lg.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        if _lgw.x1 <= 0.995:
            break
    if _lgw.x1 > 0.995:
        raise SystemExit(f"metric_{feat}: even one column of the {len(KEY)}-mark key ends at "
                         f"{_lgw.x1:.3f} of the canvas width. Shorten a label.")
    fig.text(_x0, _lgw.y1 + 0.004, TITLE, fontsize=8.0, fontweight="bold", color="#333",
             ha="left", va="bottom")
    # the row-label column keeps its own short header
    raxes[0].set_title(f"organism · {MLAB[m]}", loc="left", fontsize=5.8, color="#777", pad=3, x=-0.01, ha="right")
    if feat2:
        _r2 = np.mean([v[0] for v in (cell(feat2, dep, m) or {}).values()]) if cell(feat2, dep, m) else np.nan
        _rr2 = np.mean([v[1] for v in (cell(feat2, dep, m) or {}).values()]) if cell(feat2, dep, m) else np.nan
        _hd = ("Misalignment grows as the persona cloud distorts" if feat3
               else f"two spreads of the cast against {MLAB[m]} at depth {dep}")
        # wrapped to the column, and measured: one line of either name is wider than the column
        # it heads and the two ran into each other (2026-09-16)
        import textwrap as _tw
        _cols = [(LEFT, LP, label), (_c2, LP2, label2)] + ([(_c3, LP3, label3)] if feat3 else [])
        for _x, _w, _t in _cols:
            # centred over its column, and repeated under it (author, 2026-09-16: "also below,
            # and better placement")
            _wrapped = "\n".join(_tw.wrap(_t, 15 if feat3 else 18))
            _tx = fig.text((_x + _w / 2) / W, 1 - 0.44 / H, _wrapped, fontsize=6.4,
                           fontweight="bold", color="#333", ha="center", va="bottom",
                           linespacing=1.2)
            if feat3:
                fig.text((_x + _w / 2) / W, 0.30 / H, _wrapped, fontsize=6.4, fontweight="bold",
                         color="#333", ha="center", va="top", linespacing=1.2)
            fig.canvas.draw()
            _bw = _tx.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
            if _bw > _w + 0.04:
                raise SystemExit(f"metric pair: the column head {_t!r} is {_bw:.2f} in over a "
                                 f"{_w:.2f} in column. Shorten it.")
    else:
        _hd = (f"{label} against {MLAB[m]} at depth {dep}: "
               rf"$\bar\rho$ {b['rho']:+.2f}, $\bar r$ {b['r']:+.2f}")
    fig.text(0.07 / W, 1 - 0.06 / H, _hd, fontsize=8.4, fontweight="bold", color="#222", ha="left", va="top")
    name = "metric_spreads_and_retention" if feat3 else f"metric_{feat}" + (f"_and_{feat2}" if feat2 else "")
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=250); plt.close(fig)
    print(f"wrote {name}: {MLAB[m]} at {dep}, rho {b['rho']:+.3f} r {b['r']:+.3f}")
    return dict(feature=feat, label=label, measure=m, depth=dep, rho=round(b["rho"], 4), r=round(b["r"], 4))


if __name__ == "__main__":
    # THE TWO SPREAD READINGS ARE ONE FIGURE. They read the same object, the spread of the cast's
    # along-axis motion, so they share the picture of it (author, 2026-09-16).
    PAIR = ("along_cos_sd", "onaxis_shift_sd_over_mag")
    THIRD = "knn25pct_keep_mean"     # retention rides along as the third column (author, 2026-09-16)
    only = sys.argv[1:] or [f for f, _ in READINGS if f not in RETIRED]
    if not sys.argv[1:]:
        print("retired, not drawn: " + ", ".join(sorted(RETIRED)))
    out = []
    LAB = dict(READINGS)
    for feat, label in READINGS:
        if feat not in only or feat in (PAIR[1], THIRD):
            continue
        if feat == PAIR[0] and PAIR[1] in only:
            out.append(figure(feat, label, PAIR[1], LAB[PAIR[1]], THIRD if THIRD in only else None, LAB.get(THIRD)))
        else:
            out.append(figure(feat, label))
    # MERGE, never overwrite: running the script for one reading used to wipe the index of the
    # others, and the paper section built from it silently lost its figures (2026-09-16).
    idx = {}
    _f = _paths.DATA / "metric_slices.json"
    if _f.exists():
        idx = {r["feature"]: r for r in json.load(open(_f))}
    idx.update({r["feature"]: r for r in out})
    order = [f for f, _ in READINGS if f not in RETIRED or f in only]
    json.dump([idx[f] for f in order if f in idx], open(_f, "w"), indent=1)
