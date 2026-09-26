"""eleg_bigN.py -- the chain-frame chain of inequalities at large n and large N.

Same five steps as eleg_chain.py, but
  * n up to 6 and N up to 40 (where the brute-force Cauchy-Binet sum over binom(N,k)
    subsets is infeasible), so E_k(M) is read off the characteristic polynomial instead;
  * both hypotheses are sampled: the FULL (KL) system (2^n constraints) and the weaker
    INTERVAL-only system (binom(n+1,2) constraints) that the proof actually uses.

Checks, per system:
  T1  M := sum_l c_l m_l m_l^* is upper triangular          (max |strictly lower part|)
  T2  diag(M) = A                                           (max |diag - A|)
  T3  E_k(M) = binom(n+1,k) A^k for k = 1..n+1              (from the char poly)
  T4  e_k(|d|) >= binom(n+1,k),  d = (n+1) c / A            (min ratio, must be >= 1)
  T5  prod_l (1 + |d_l| t) >= (1+t)^{n+1} on a t-grid       (min slack, must be >= 0)
  P1  sum |c_l|^r / ((n+1)^{1-r} |A|^r)                     (min over r, must be >= 1)

Run:  uv run --with scipy python eleg_bigN.py
"""

import json
import math
import os

import numpy as np

from eleg_chain import (RS, chain_vectors, elem_sym_all, intervals, moment_matrix,
                        subsets)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "ELEG")
os.makedirs(OUT, exist_ok=True)


def solve(v, sets, rng, jitter):
    M = moment_matrix(v, sets)
    rhs = np.zeros(len(sets), dtype=complex)
    rhs[0] = 1.0
    c, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    if jitter > 0:
        _, s, Vh = np.linalg.svd(M)
        tol = max(M.shape) * np.finfo(float).eps * s[0]
        ns = Vh[int(np.sum(s > tol)):].conj().T
        if ns.shape[1] > 0:
            z = rng.normal(size=ns.shape[1]) + 1j * rng.normal(size=ns.shape[1])
            c = c + jitter * (ns @ z)
    return c, float(np.max(np.abs(M @ c - rhs)))


def Ek_from_charpoly(M):
    """E_k(M) for k = 0..D from the characteristic polynomial det(zI - M)."""
    D = M.shape[0]
    coef = np.poly(M)                       # z^D + a1 z^{D-1} + ... ; a_k = (-1)^k E_k
    return [((-1) ** k) * coef[k] for k in range(D + 1)]


def check(v, c):
    n = v.shape[1]
    D = n + 1
    A = c.sum()
    m = chain_vectors(v)
    d = D * c / A
    x = np.abs(d)
    M = np.einsum("l,lk,lj->kj", c, m, m.conj())
    Ek = Ek_from_charpoly(M)
    e = elem_sym_all(x)
    t3 = max(abs(Ek[k] - math.comb(D, k) * A ** k) / (abs(A) ** k) for k in range(1, D + 1))
    t4 = min((e[k] if k < len(e) else 0.0) / math.comb(D, k) for k in range(1, D + 1))
    ts = np.logspace(-6, 6, 600)
    t5 = float(np.min(np.sum(np.log1p(np.outer(ts, x)), axis=1) - D * np.log1p(ts)))
    p1 = min(float(np.sum(np.abs(c) ** r) / ((n + 1) ** (1 - r) * abs(A) ** r)) for r in RS)
    return (float(np.max(np.abs(np.tril(M, -1)))) / abs(A),
            float(np.max(np.abs(np.diag(M) - A))) / abs(A),
            float(t3), float(t4), t5, p1)


def main():
    rng = np.random.default_rng(7717)
    rows = []
    print(f"{'n':>2} {'N':>3} {'hyp':>9} {'#':>4} | {'T1 tri':>9} {'T2 diag':>9} "
          f"{'T3 E_k':>9} | {'T4 e_k':>9} {'T5 slack':>10} {'P1 min':>13}", flush=True)
    for n in (2, 3, 4, 5, 6):
        D = n + 1
        nint = D * (D - 1) // 2
        for hyp in ("full", "interval"):
            sets = subsets(n) if hyp == "full" else [()] + intervals(n)
            nmin = (2 ** n if hyp == "full" else nint + 1)
            for N in sorted({max(nmin, D), max(nmin, D) + 1, max(nmin, D) + 4,
                             max(nmin, D) + 12, max(nmin, D) + 28}):
                if N > 44:
                    continue
                agg = [0.0, 0.0, 0.0, np.inf, np.inf, np.inf]
                cnt = 0
                for _ in range(40):
                    v = np.exp(2j * np.pi * rng.random((N, n)))
                    c, res = solve(v, sets, rng, 0.3)
                    if res > 1e-9 or abs(c.sum()) < 1e-6:
                        continue
                    o = check(v, c)
                    for i in range(3):
                        agg[i] = max(agg[i], o[i])
                    for i in range(3, 6):
                        agg[i] = min(agg[i], o[i])
                    cnt += 1
                if cnt == 0:
                    continue
                rows.append({"n": n, "N": N, "hyp": hyp, "count": cnt,
                             "T1": agg[0], "T2": agg[1], "T3": agg[2],
                             "T4": agg[3], "T5": agg[4], "P1": agg[5]})
                print(f"{n:>2} {N:>3} {hyp:>9} {cnt:>4} | {agg[0]:9.2e} {agg[1]:9.2e} "
                      f"{agg[2]:9.2e} | {agg[3]:9.6f} {agg[4]:+10.2e} {agg[5]:13.9f}",
                      flush=True)
    with open(os.path.join(OUT, "bigN.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote bigN.json", flush=True)


if __name__ == "__main__":
    main()
