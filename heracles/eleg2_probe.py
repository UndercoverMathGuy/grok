"""eleg2_probe.py -- first hard numerical test of the ELEGANT2 conjectures.

The candidate statement (call it the ROBUST FEATURE COST inequality, (RFC)):

    sum_l |c_l|^r  >=  (n+1)^{1-r} * max(m, 0)^r ,        m := min_{u in T^n} Re h(u)

with h(u) = sum_l c_l prod_j (1 + u_j v_lj).  Plain reading: a feature's weight-decay
cost is controlled not by its clean amplitude A = C_empty but by the WORST output it
can be aimed at; junk lowers that worst output at least as fast as it lowers the cost.
Junk-free means h == A, so (RFC) contains P1 (ELEGANT.md Theorem 1) as the case h const.

Three nested statements are probed:

  (C1)  sum_l |c_l|^r >= (n+1)^{1-r} m_+^r                       [the target]
  (C2)  e_k(|c|)      >= binom(D,k) (m/D)^k,  k = 1..D, D = n+1  [symmetric ladder; C2 => C1]
  (C3)  |det(I + sM)| >= (1 + s m)^D for all s >= 0              [matrix form;  C3 => C2]

plus the two identities the proof of the N = n+1 case uses:
  (CHAIN)   log[(n+1)^{n+1} prod_l |c_l|] = (n+1) H + Delta,  H = kill-point mean of log|h|
  (KILL)    |h| >= Re h >= m at every kill point.

Run:  uv run python eleg2_probe.py
"""
import json
import os
import sys
import numpy as np
from itertools import combinations, permutations
from math import comb, log

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eleg2_core import (rand_pop, CT, chain_matrix, chain_vectors, min_re_h,
                        elem_sym, Ek_matrix, h_at, subsets)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "ELEG2")
os.makedirs(OUT, exist_ok=True)

RS = [0.05, 0.2, 1/3, 0.5, 2/3, 0.75, 0.9, 1.0]
SS = [0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0]


def kill_points(v):
    """All (n+1)! kill points, for N = n+1 atoms: (l0, sigma) -> u_j = -conj(v[sigma(j), j])."""
    N, n = v.shape
    assert N == n + 1
    pts = []
    for l0 in range(N):
        rest = [l for l in range(N) if l != l0]
        for sg in permutations(rest):
            u = np.array([-np.conj(v[sg[j], j]) for j in range(n)])
            pts.append((l0, u))
    return pts


def chain_identity_check(c, v):
    """(CHAIN):  log[(n+1)^{n+1} prod|c_l|] - [(n+1) H + Delta]  should be 0."""
    N, n = v.shape
    D = n + 1
    pts = kill_points(v)
    H = np.mean([log(abs(h_at(c, v, u)) + 1e-300) for _, u in pts])
    logV2 = 0.0
    for j in range(n):
        z = v[:, j]
        pr = 1.0
        for a in range(N):
            for b in range(a + 1, N):
                pr *= abs(z[a] - z[b]) ** 2
        logV2 += log(pr + 1e-300)
    Delta = D * log(D) - logV2 / n
    lhs = D * log(D) + sum(log(abs(x) + 1e-300) for x in c)
    return lhs - (D * H + Delta), Delta, H


def probe(nmax=4, trials=60, seed=0):
    rng = np.random.default_rng(seed)
    rec = {"C1": [], "C2": [], "C3": [], "chain": [], "kill": []}
    worst = {"C1": np.inf, "C2": np.inf, "C3": np.inf,
             "chain": 0.0, "kill": np.inf}
    worst_case = {}
    for n in range(1, nmax + 1):
        D = n + 1
        for N in [D - 1, D, D + 1, D + 3, 2 * D]:
            if N < 1:
                continue
            for kind in ["generic", "junkfree", "nearjf"]:
                if kind != "generic" and N < D:
                    continue
                for t in range(trials):
                    c, v = rand_pop(rng, N, n, kind)
                    if np.max(np.abs(c)) < 1e-12:
                        continue
                    c = c / np.max(np.abs(c))          # scale-invariant statements
                    m, _ = min_re_h(c, v, 1.0, starts=24, rng=rng)
                    ac = np.abs(c)
                    # ---- C1
                    for r in RS:
                        cost = np.sum(ac ** r)
                        tgt = D ** (1 - r) * max(m, 0.0) ** r
                        if tgt > 0:
                            ratio = cost / tgt
                            if ratio < worst["C1"]:
                                worst["C1"] = ratio
                                worst_case["C1"] = dict(n=n, N=N, kind=kind, r=r,
                                                        m=m, cost=cost)
                    if m <= 0:
                        continue
                    # ---- C2
                    e = elem_sym(ac, kmax=D)
                    for k in range(1, D + 1):
                        tgt = comb(D, k) * (m / D) ** k
                        ratio = e[k] / tgt
                        if ratio < worst["C2"]:
                            worst["C2"] = ratio
                            worst_case["C2"] = dict(n=n, N=N, kind=kind, k=k, m=m)
                    # ---- C3
                    M = chain_matrix(c, v)
                    for s in SS:
                        lhs = abs(np.linalg.det(np.eye(D) + s * M))
                        tgt = (1 + s * m) ** D
                        ratio = lhs / tgt
                        if ratio < worst["C3"]:
                            worst["C3"] = ratio
                            worst_case["C3"] = dict(n=n, N=N, kind=kind, s=s, m=m)
                    # ---- identities at N = D
                    if N == D:
                        err, Delta, H = chain_identity_check(c, v)
                        worst["chain"] = max(worst["chain"], abs(err))
                        for _, u in kill_points(v):
                            val = abs(h_at(c, v, u)) - m
                            if val < worst["kill"]:
                                worst["kill"] = val
    return worst, worst_case


if __name__ == "__main__":
    w, wc = probe(nmax=int(sys.argv[1]) if len(sys.argv) > 1 else 4,
                  trials=int(sys.argv[2]) if len(sys.argv) > 2 else 60)
    print("worst ratios (must be >= 1 for C1,C2,C3; >=0 for kill; ~0 for chain)")
    for k in ["C1", "C2", "C3", "kill", "chain"]:
        print(f"  {k:6s} {w[k]!r}   {wc.get(k, '')}")
    with open(os.path.join(OUT, "probe.json"), "w") as f:
        json.dump({"worst": {k: float(v) for k, v in w.items()},
                   "case": {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                                for kk, vv in d.items()} for k, d in wc.items()}}, f, indent=1)
