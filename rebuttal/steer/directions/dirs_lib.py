"""Shared code for building and checking steering directions from the HF persona clouds.

The construction is the published battery's (rebuilt bit-exact from HF vectors for 9 organisms in an earlier check):

    s_p[L]                  = r_org[p][L-1] - r_base[p][L-1]           (float32, from the bf16 HF vectors)
    sbar[L]                 = mean_{p in cast} s_p[L]                   (cast holds 'assistant', never 'default')
    excess_assistant[L]     = s_assistant[L] - sbar[L]
    excess_default[L]       = s_default[L]   - sbar[L]
    default_to_assistant[L] = r_org[assistant][L-1] - r_org[default][L-1]
    act_norm[L]             = mean_{p in cast} ||r_base[p][L-1]||
    row 0 of every tensor   = 0

HF vectors are <family>/<who>/vectors/<role>.pt = {'vector': (n_layers, d) bf16, ...}; vector row i is the output
of decoder layer i. Directions are hidden_states-indexed with n_layers+1 rows, so dirs[L] = vector[L-1], and a
steer at L hooks model.layers[L-1] (persona_em.hooks).
"""
import hashlib
import json
from collections import defaultdict

import torch

REPO = "unrulyabstractions/assistant-axis-runs"
PINNED = "e6c3d0290a4672629d58361f30578358560a8bb6"
N_LAYERS = {"qwen2.5-7b": 28, "llama3.1-8b": 32, "qwen3.5-9b": 32, "gemma3-12b": 48, "qwen2.5-14b": 48,
            "mistral-24b": 40, "qwen2.5-32b": 64, "qwen3-32b": 64}
KEYS = ("sbar", "excess_assistant", "excess_default", "default_to_assistant")
ASSISTANT, DEFAULT = "assistant", "default"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Store:
    """HF path -> proven local file. Every file is re-hashed against the manifest before its first use."""

    def __init__(self, *manifest_paths):
        self.files, revs = {}, set()
        for mp in manifest_paths:
            m = json.load(open(mp))
            revs.add(m["revision"])
            self.files.update(m["files"])
        if len(revs) != 1:
            raise ValueError(f"manifests come from different revisions: {revs}")
        self.revision = revs.pop()
        self._ok = set()

    def local(self, p):
        if p not in self.files:
            raise KeyError(f"{p} is not in the fetch manifest")
        e = self.files[p]
        if p not in self._ok:
            got = sha256(e["local"]) if e["sha256"] else git_blob_sha1(e["local"])
            want = e["sha256"] or e["git_blob"]
            if got != want:
                raise RuntimeError(f"hash mismatch for {p}: local {e['local']}")
            self._ok.add(p)
        return e["local"]

    def torch_load(self, p):
        return torch.load(self.local(p), weights_only=False, map_location="cpu")

    def json_load(self, p):
        return json.load(open(self.local(p)))

    def n_verified(self):
        return len(self._ok)


def listing_vectors(listing_path):
    vecs = defaultdict(set)
    for p, _s in json.load(open(listing_path)):
        q = p.split("/")
        if len(q) == 4 and q[2] == "vectors" and p.endswith(".pt"):
            vecs[(q[0], q[1])].add(q[3][:-3])
    return vecs


def family_cast(vecs, fam):
    """The paper's family cast: the roles every cloud of the family holds (HF folders, 'default' included)."""
    clouds = [w for (f, w) in vecs if f == fam]
    return set.intersection(*[vecs[(fam, w)] for w in clouds]), clouds


def resolve_cast(spec, fam, vecs, store=None):
    """spec -> (ordered role list without 'default', human description)."""
    if spec == "family":
        c, clouds = family_cast(vecs, fam)
        roles = sorted(c - {DEFAULT})
        desc = f"family cast: roles held by all {len(clouds)} HF clouds of {fam} ({len(c)} incl default) minus default"
    elif spec.startswith("inter:"):
        whos = spec[len("inter:"):].split(",")
        c = set(vecs[(fam, "baseline")])
        for w in whos:
            c &= vecs[(fam, w)]
        roles = sorted(c - {DEFAULT})
        desc = f"baseline ∩ {' ∩ '.join(whos)} minus default"
    elif spec.startswith("lu:"):
        p = spec[3:]
        roles = list(store.torch_load(p)["personas"])
        desc = f"'personas' of the Lu-input file {p}"
    elif spec.startswith("geom:"):
        p = spec[5:]
        roles = [r for r in store.json_load(p)["personas"] if r != DEFAULT]
        desc = f"'personas' of {p} minus default"
    else:
        raise ValueError(spec)
    if DEFAULT in roles or ASSISTANT not in roles:
        raise ValueError(f"cast {spec}: must hold assistant and not default")
    if len(set(roles)) != len(roles):
        raise ValueError(f"cast {spec}: duplicate roles")
    return roles, desc


def vec_path(fam, who, role):
    return f"{fam}/{who}/vectors/{role}.pt"


def load_vec(store, fam, who, role):
    o = store.torch_load(vec_path(fam, who, role))
    v = o["vector"] if isinstance(o, dict) and "vector" in o else o
    if v.dtype != torch.bfloat16 or v.ndim != 2 or v.shape[0] != N_LAYERS[fam]:
        raise ValueError(f"{vec_path(fam, who, role)}: unexpected {v.dtype} {tuple(v.shape)}")
    return v.float()


def rows(S):
    out = torch.zeros(S.shape[0] + 1, S.shape[1], dtype=torch.float32)
    out[1:] = S
    return out


def build(store, fam, who, cast):
    base = {r: load_vec(store, fam, "baseline", r) for r in cast + [DEFAULT]}
    org = {r: load_vec(store, fam, who, r) for r in cast + [DEFAULT]}
    sbar = rows(torch.stack([org[r] - base[r] for r in cast]).mean(0))
    dirs = {
        "sbar": sbar,
        "excess_assistant": rows(org[ASSISTANT] - base[ASSISTANT]) - sbar,
        "excess_default": rows(org[DEFAULT] - base[DEFAULT]) - sbar,
        "default_to_assistant": rows(org[ASSISTANT] - org[DEFAULT]),
    }
    nl = base[ASSISTANT].shape[0]
    act = torch.zeros(nl + 1, dtype=torch.float32)
    act[1:] = torch.stack([base[p].norm(dim=-1) for p in cast]).mean(0)
    return dirs, act


def random_axis(like, seed):
    """persona_em.axes.random_axis, verbatim."""
    g = torch.Generator().manual_seed(int(seed))
    r = torch.randn(like.shape, generator=g, dtype=torch.float32)
    return r / (r.norm(dim=-1, keepdim=True) + 1e-8)


def cos(u, v):
    u, v = u.double().flatten(), v.double().flatten()
    return float((u @ v) / (u.norm() * v.norm() + 1e-30))


def compare(mine, ref, L):
    mine, ref = mine.float(), ref.float()
    rr = range(1, ref.shape[0])
    cs = [cos(mine[i], ref[i]) for i in rr]
    rel = [float((mine[i] - ref[i]).norm() / (ref[i].norm() + 1e-30)) for i in rr]
    return {"cos_at_L": cos(mine[L], ref[L]),
            "rel_err_at_L": float((mine[L] - ref[L]).norm() / (ref[L].norm() + 1e-30)),
            "min_cos_rows_1_to_n": min(cs), "max_rel_err_rows_1_to_n": max(rel),
            "max_abs_diff": float((mine - ref).abs().max()),
            "bitwise_equal": bool(torch.equal(mine, ref)),
            "row0_zero_mine": bool(float(mine[0].abs().max()) == 0.0)}
