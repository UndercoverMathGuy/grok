"""eleg2_spec.py -- is the SPECTRUM of the chain matrix to the right of m?

Candidate (S):   Re lambda >= m  for every eigenvalue lambda of M = sum_l c_l m_l m_l^*,
                 where m = min_{u in T^n} Re h(u).
Weaker, and all that is actually needed:
        (C3)     |det(I + sM)| >= (1 + s m)^D   for all s >= 0.
Stronger (would also give (S)):
        (NR)     Herm(M) := (M + M^*)/2  >=  m I    (numerical range in Re >= m).

(S) => (C3) because |1 + s lambda| >= 1 + s Re lambda >= 1 + s m for s >= 0.
(C3) => the target, because Cauchy-Binet + Hadamard give
        prod_l (1 + s D |c_l|) >= |det(I + sM)| ,
and then the Bernstein integral.

Also probed: the same three with the AIMED depth A < 1 (m_A), and with the GRID.

Run:  uv run python eleg2_spec.py
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eleg2_core import rand_pop, chain_matrix, min_re_h, elem_sym
from math import comb
RS = [0.05, 0.2, 1/3, 0.5, 2/3, 0.8, 0.95, 1.0]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "ELEG2")
os.makedirs(OUT, exist_ok=True)
SS = [0.02, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1e4]


def run(nmax=4, trials=200, seed=7):
    rng = np.random.default_rng(seed)
    res = {}
    for n in range(1, nmax + 1):
        D = n + 1
        w_s, w_nr, w_c3 = np.inf, np.inf, np.inf
        w_c1, w_c2 = np.inf, np.inf
        case = {}
        ntested = 0
        for N in [D, D + 1, D + 2, 2 * D]:
            for kind in ["generic", "junkfree", "nearjf"]:
                for t in range(trials):
                    c, v = rand_pop(rng, N, n, kind)
                    c = c / max(np.max(np.abs(c)), 1e-30)
                    m, _ = min_re_h(c, v, 1.0, starts=20, rng=rng)
                    if m <= 1e-9:
                        continue
                    ntested += 1
                    M = chain_matrix(c, v)
                    lam = np.linalg.eigvals(M)
                    s_ratio = float(np.min(lam.real) / m)
                    nr = float(np.min(np.linalg.eigvalsh((M + M.conj().T) / 2)) / m)
                    c3 = min(abs(np.linalg.det(np.eye(D) + s * M)) / (1 + s * m) ** D
                             for s in SS)
                    if s_ratio < w_s:
                        w_s = s_ratio
                        case["S"] = dict(n=n, N=N, kind=kind, m=m)
                    if nr < w_nr:
                        w_nr = nr
                        case["NR"] = dict(n=n, N=N, kind=kind, m=m)
                    if c3 < w_c3:
                        w_c3 = float(c3)
                        case["C3"] = dict(n=n, N=N, kind=kind, m=m,
                                          c=[[x.real, x.imag] for x in c],
                                          v=[[x.real, x.imag] for x in v.ravel()])
                    ac = np.abs(c)
                    e = elem_sym(ac, kmax=D)
                    c2 = min(e[k] / (comb(D, k) * (m / D) ** k) for k in range(1, D + 1))
                    if c2 < w_c2:
                        w_c2 = float(c2)
                        case["C2"] = dict(n=n, N=N, kind=kind, m=m)
                    c1 = min(np.sum(ac ** r) / (D ** (1 - r) * m ** r) for r in RS)
                    if c1 < w_c1:
                        w_c1 = float(c1)
                        case["C1"] = dict(n=n, N=N, kind=kind, m=m)
        res[n] = dict(tested=ntested, min_Re_spec_over_m=w_s,
                      min_eig_Herm_over_m=w_nr, min_C3=w_c3,
                      min_C2=w_c2, min_C1=w_c1, case=case)
        print(f"n={n} tested={ntested} Re(spec)/m={w_s:.6f} Herm/m={w_nr:.6f} "
              f"C3={w_c3:.9f} C2={w_c2:.9f} C1={w_c1:.9f}")
    return res


if __name__ == "__main__":
    r = run(nmax=int(sys.argv[1]) if len(sys.argv) > 1 else 4,
            trials=int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    with open(os.path.join(OUT, "spec.json"), "w") as f:
        json.dump(r, f, indent=1, default=float)
