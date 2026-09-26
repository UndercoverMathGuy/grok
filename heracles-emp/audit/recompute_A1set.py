"""AUDIT: A1 share match to the 1%-optimal vertex set, recomputed with own dilation/value code. Vertex DATA from
core.programme.load_vertices (allowed); cache completeness checked against the audit brute force for p <= 19."""
import json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_audit import *
from brute_programme import vertices
from core import programme as PG

def dilates(Y, p):
    h = (p - 1) // 2; j = np.arange(1, h + 1); out = []
    for u in range(1, h + 1):
        f = (u * j) % p; f = np.where(f > h, p - f, f)
        out.append(Y[:, f - 1])
    return np.unique(np.round(np.vstack(out), 12), axis=0)

res = {"cache_check": {}}
for p in (11, 13, 17, 19):
    _, Vb = vertices(p)
    reps = PG.load_vertices(p)[0]
    Vc = dilates(np.asarray(reps, float), p)
    Vb = np.unique(np.round(Vb, 9), axis=0); Vc2 = np.unique(np.round(Vc, 9), axis=0)
    same = len(Vb) == len(Vc2) and np.allclose(Vb[np.lexsort(Vb.T[::-1])], Vc2[np.lexsort(Vc2.T[::-1])], atol=1e-7)
    res["cache_check"][p] = {"brute_vertices": len(Vb), "cache_vertices_expanded": len(Vc2), "identical": bool(same)}
print(res["cache_check"])

def eps_set(p, s, G):
    reps = np.asarray(PG.load_vertices(p)[0], float)
    C = cov_matrix(p)
    val = (reps @ C.T).min(1) / ((reps ** (1 / s)).sum(1)) ** s
    assert abs(val.max() - G) < 1e-8 * G, (p, s, val.max(), G)
    R = reps[val >= 0.99 * G]
    V = dilates(R, p)
    return V, len(R)

cachep = set(PG.cached_primes())
for e in ("E1", "E6", "E7", "E8"):
    idx, _ = preds(e); rows = []
    cache = {}
    for r in runs(e):
        if r["status"] not in ("ok", "not_converged") or r["m"] is None or r["m"]["margin"] <= 0: continue
        c = r["c"]; p = c["p"]; s = (c["n"] + 1) / c["q"]
        if p not in cachep: continue
        cell = idx[key_of(r)]; k = (p, round(s, 9))
        if k not in cache: cache[k] = eps_set(p, s, cell["Gamma"])
        V, norb = cache[k]
        A = np.clip(np.asarray(r["m"]["A"]), 0, None)
        cs = V @ A / (np.linalg.norm(V, axis=1) * np.linalg.norm(A))
        W = V ** (1 / s); W = W / W.sum(1, keepdims=True)
        t = 0.5 * np.abs(W - shares(A, s)[None]).sum(1)
        i = np.lexsort((t, -cs))[0]
        K = (V > 0).sum(1); K2 = energy_counts(A)["K_2pct"]
        Y = np.asarray(cell.get("y_all") or [cell["y_star"]])
        y = Y[align(A, Y, s)]
        rows.append({"st": r["status"], "p": p, "s": round(s, 4), "tvmin": t.min(), "tvbest": t[i], "cos": cs[i], "in": K.min() <= K2 <= K.max(),
                     "norb": norb, "tv_reg": tv(shares(A, s), shares(y, s))})
    for g in ("ok", "not_converged"):
        R = [x for x in rows if x["st"] == g]
        if not R: continue
        res[f"{e}_{g}"] = {"n": len(R), "median_tvmin": float(np.median([x["tvmin"] for x in R])), "median_tv_best_cos": float(np.median([x["tvbest"] for x in R])),
                           "median_cos": float(np.median([x["cos"] for x in R])), "frac_K2_in_range": float(np.mean([x["in"] for x in R])),
                           "median_tv_reg_same_runs": float(np.median([x["tv_reg"] for x in R])), "max_orbits": max(x["norb"] for x in R)}
        print(e, g, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[f"{e}_{g}"].items()})
json.dump(res, open(os.path.join(ROOT, "audit", "recompute_A1set.json"), "w"), indent=1, default=float)
