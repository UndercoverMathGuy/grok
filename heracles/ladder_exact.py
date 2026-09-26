"""ladder_exact.py -- the exact ladders for every paper cell.

For each cell (family, p, n, q) and each K = 1..Kmax it computes, with NO training and no
approximation beyond fp64 arithmetic and the support search:

  M_K              = max over |S| = K of min_{D != 0} sum_{k in S} g_k(D)    ("coverage", kstar.py)
  val_eq(K)        = M_K * K^{-s}                  s = nu/q   (equal shares on the best K-set)
  val_free(K)      = max over |S| = K and shares b of min_{D != 0} sum_k b_k^s g_k(D)
  gamma_con(K)     = mu * N^{-(nu-q)/q} * B * val(K)          (exact, verified by ladder_verify.py)
  obj*(K)          = min over the global weight scale of  CE + lam * P_q,  CE = mean cross-entropy
                     over the whole p^n table (exact: a junk-free block's table depends only on D)

plus the DENSE FLAT competitor (all (p-1)/2 frequencies, equal shares) in the same cells.

Two ladders because they answer different questions:
  * the EQUAL-share ladder is the one in kstar.py/THEORY_nsum; it uses all K frequencies by
    construction, so it has a genuine interior peak at K*;
  * the FREE-amplitude ladder is the programme itself; because a K-support may set some shares to
    zero it is non-decreasing in K and PLATEAUS at K*, i.e. K* = the first K at which the free
    ladder reaches Gamma_s, equivalently the number of non-zero shares at the global optimum.

Usage: uv run python heracles/ladder_exact.py --primes 59 71 113 --kmax 12
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ladder_core import (OUT, B_const, block_spec, clean_spectrum, coverage_topK, gamma_con,  # noqa: E402
                         gamma_con_predicted, gaps_from_spectrum, gmat, margin_of_gaps,
                         obj_star_gaps, solve_support, val_of)

# cells: (family, n) x p x q.  q must satisfy 2 <= q <= nu.
FAMILIES = [("chain", 2), ("chain", 3), ("chain", 4), ("xn", 2), ("xn", 3), ("xn", 4)]
QS = {2: [2.0, 2.5, 3.0],                      # nu = 3
      3: [2.0, 2.5, 8 / 3, 3.0, 4.0],          # nu = 4  (8/3 is the s = 3/2 match)
      4: [2.0, 2.5, 3.0, 10 / 3, 5.0]}         # nu = 5  (10/3 is the s = 3/2 match)
LAMS = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]


def s_values():
    out = set()
    for _, n in FAMILIES:
        for q in QS[n]:
            out.add(round((n + 1) / q, 12))
    return sorted(out)


def cov_cache(p, kmax, shortlist, path):
    """Coverage shortlists per (p, K): s-independent, so computed once and cached on disk."""
    if path.exists():
        d = json.loads(path.read_text())
        if d.get("kmax", 0) >= kmax and d.get("shortlist", 0) >= shortlist:
            return d
    g = gmat(p)
    d = dict(p=p, kmax=kmax, shortlist=shortlist, cands={}, how={}, M={})
    for K in range(1, kmax + 1):
        t0 = time.time()
        cands, how = coverage_topK(g, K, shortlist=shortlist, exhaustive_limit=3_000_000)
        d["cands"][str(K)] = cands
        d["how"][str(K)] = how
        d["M"][str(K)] = float(g[np.array(cands[0]) - 1].sum(0).min())
        print(f"  p {p} K {K:2d}: {how:10s} M_K {d['M'][str(K)]:.6f}  best S {cands[0]}  "
              f"({time.time() - t0:.1f}s)", flush=True)
    path.write_text(json.dumps(d))
    return d


def free_ladder(p, kmax, s, cov, restarts_scan=2, restarts_top=8, topn=6):
    """Best K-support under the FREE-amplitude programme, for K = 1..kmax at exponent s."""
    g = gmat(p)
    out = {}
    prev_S = None
    for K in range(1, kmax + 1):
        cands = [list(c) for c in cov["cands"][str(K)]]
        if prev_S is not None:                       # grow last K's winner by every frequency
            nf = g.shape[0]
            for k in range(1, nf + 1):
                if k not in prev_S:
                    cands.append(sorted(prev_S + [k]))
        scored = []
        for S in cands:
            v, b = solve_support(g, S, s, restarts=restarts_scan)
            scored.append((v, S, b))
        scored.sort(key=lambda t: -t[0])
        best = scored[0]
        for v, S, b in scored[:topn]:                # re-solve the leaders harder
            v2, b2 = solve_support(g, S, s, restarts=restarts_top)
            if v2 > best[0]:
                best = (v2, S, b2)
        v, S, b = best
        b = np.asarray(b)
        out[K] = dict(value=v, S=[int(x) for x in S], b=[float(x) for x in b],
                      n_active=int((b > 1e-4).sum()))
        prev_S = list(S)
    return out


def cell_rows(family, p, n, q, kmax, cov, fl, lams):
    """Everything for one cell: gamma_con and obj* for both ladders, plus the dense-flat net."""
    nu = n + 1
    s = nu / q
    g = gmat(p)
    nf = g.shape[0]
    N, _, _, mu = block_spec(family, n)
    rows = []
    for K in range(1, kmax + 1):
        rec = dict(K=K)
        for tag in ("eq", "free"):
            if tag == "eq":
                S = list(cov["cands"][str(K)][0])
                b = [1.0 / K] * K
            else:
                S, b = fl[K]["S"], fl[K]["b"]
            v = val_of(g, S, b, s)
            gam, mg, Pq, A, gp = gamma_con(family, p, n, S, b, q)
            objs = {f"{lam:g}": obj_star_gaps(gp, Pq, lam, nu, q) for lam in lams}
            rec[tag] = dict(S=S, b=b, val=v, gamma=gam, margin=mg, Pq=Pq, m=len(S) * N,
                            n_active=int((np.asarray(b) > 1e-4).sum()),
                            pred=gamma_con_predicted(family, p, n, v, q),
                            obj={k: o["obj"] for k, o in objs.items()},
                            ce={k: o["ce"] for k, o in objs.items()},
                            scale={k: o["scale"] for k, o in objs.items()})
        rows.append(rec)
    # dense flat: every frequency, equal shares
    Sd = list(range(1, nf + 1))
    bd = [1.0 / nf] * nf
    vd = val_of(g, Sd, bd, s)
    gamd, mgd, Pqd, Ad, gpd = gamma_con(family, p, n, Sd, bd, q)
    dense = dict(S="all", nf=nf, val=vd, gamma=gamd, margin=mgd, Pq=Pqd, m=nf * N,
                 obj={f"{lam:g}": obj_star_gaps(gpd, Pqd, lam, nu, q)["obj"] for lam in lams})
    return rows, dense


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primes", type=int, nargs="+", default=[59, 71, 113])
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--shortlist", type=int, default=150)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "json").mkdir(exist_ok=True)

    covs, frees = {}, {}
    for p in a.primes:
        print(f"== coverage shortlists p = {p}", flush=True)
        covs[p] = cov_cache(p, a.kmax, a.shortlist, OUT / "json" / f"cov_p{p}.json")
    for p in a.primes:
        for s in s_values():
            t0 = time.time()
            frees[(p, s)] = free_ladder(p, a.kmax, s, covs[p])
            top = max(frees[(p, s)], key=lambda K: frees[(p, s)][K]["value"])
            first = min(K for K in frees[(p, s)]
                        if frees[(p, s)][K]["value"] >= frees[(p, s)][top]["value"] - 1e-10)
            print(f"  p {p} s {s:.5f}: Gamma_s = {frees[(p, s)][top]['value']:.8f}  "
                  f"K* (first K at the plateau) = {first}  n_active {frees[(p, s)][first]['n_active']}  "
                  f"S = {frees[(p, s)][first]['S']}  ({time.time() - t0:.1f}s)", flush=True)
    json.dump({f"{p}_{s:.6f}": v for (p, s), v in frees.items()},
              open(OUT / "json" / "free_ladders.json", "w"), indent=1)

    allres = {}
    for family, n in FAMILIES:
        for p in a.primes:
            for q in QS[n]:
                s = (n + 1) / q
                key = f"{family}_n{n}_p{p}_q{q:.4g}"
                rows, dense = cell_rows(family, p, n, q, a.kmax, covs[p], frees[(p, round(s, 12))], LAMS)
                allres[key] = dict(family=family, p=p, n=n, nu=n + 1, q=q, s=s,
                                   N=block_spec(family, n)[0], mu=block_spec(family, n)[3],
                                   B=B_const(p, n), rows=rows, dense=dense)
                gam_eq = [r["eq"]["gamma"] for r in rows]
                gam_fr = [r["free"]["gamma"] for r in rows]
                kmax_eq = int(np.argmax(gam_eq)) + 1
                kstar_fr = min(K for K in range(1, a.kmax + 1)
                               if gam_fr[K - 1] >= max(gam_fr) - 1e-12 * max(gam_fr))
                allres[key]["argmax_gamma_eq"] = kmax_eq
                allres[key]["K_star_free"] = kstar_fr
                allres[key]["argmin_obj"] = {lk: int(np.argmin([r["eq"]["obj"][lk] for r in rows])) + 1
                                             for lk in rows[0]["eq"]["obj"]}
                allres[key]["argmin_obj_free"] = {lk: int(np.argmin([r["free"]["obj"][lk] for r in rows])) + 1
                                                  for lk in rows[0]["free"]["obj"]}
                allres[key]["dense_ratio_eq"] = dense["gamma"] / max(gam_eq)
                allres[key]["dense_ratio_free"] = dense["gamma"] / max(gam_fr)
                print(f"{key:26s} s {s:.3f}  argmax gamma_eq {kmax_eq:2d}  K*_free {kstar_fr:2d}  "
                      f"argmin obj*(1e-5) {allres[key]['argmin_obj']['1e-05']:2d}  "
                      f"dense/best {allres[key]['dense_ratio_eq']:.4f}", flush=True)
    json.dump(allres, open(OUT / "json" / "cells.json", "w"), indent=1)
    print("\nwrote", OUT / "json" / "cells.json")


if __name__ == "__main__":
    main()
