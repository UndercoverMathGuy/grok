"""Guaranteed joint-detection constant at n=3 from the slicing family (PROVED bounds), via LP.

Bounds (all for (-min Z)(max Z^d), mean-zero profiles D_T, V_T = Var D_T, C_T = E D_T(w) D_T(w+d)):
  plane slicings  S_k  (spectator {k}):     kap*(E - V_k)   + C_k
                  S_123 (spectator {123}):  kap*(E - V_123) + C_123
  line slicings   L_k  (direction e_k):     sum_{T contains k} V_T + sum_{T not containing k} C_T
                  L_ij (direction e_i-e_j): V_i+V_j+V_ik+V_jk + C_k + C_ij + C_ijk
  per-T:          V_T
with kap = (p-1)/(p+1) (gibbs R3) and C_T >= -zeta V_T, zeta = cos(pi/p) (class C: C_T/V_T is an average of
cos(2 pi l d/p), l in F).  The adversary takes C_T = -zeta V_T.  J = min_V max(bounds), sum V = 1.
Also reported: the R9-only constant (slicings S_k only, C=-V) = kap/(2+kap) for comparison.
"""
import itertools
import numpy as np
from scipy.optimize import linprog

TS = [T for k in range(1, 4) for T in itertools.combinations(range(3), k)]
ix = {T: a for a, T in enumerate(TS)}


def bounds_matrix(kap, zeta, use_lines=True, use_123=True, use_perT=True):
    rows = []
    for k in range(3):  # plane slicing with spectator {k}
        row = np.full(7, kap)
        row[ix[(k,)]] = -zeta
        rows.append(row)
    if use_123:
        row = np.full(7, kap)
        row[ix[(0, 1, 2)]] = -zeta
        rows.append(row)
    if use_lines:
        for k in range(3):
            row = np.array([1.0 if k in T else -zeta for T in TS])
            rows.append(row)
        for i, j in itertools.combinations(range(3), 2):
            k = 3 - i - j
            core = [(i,), (j,), tuple(sorted((i, k))), tuple(sorted((j, k)))]
            row = np.array([1.0 if T in core else -zeta for T in TS])
            rows.append(row)
    if use_perT:
        for a in range(7):
            row = np.zeros(7)
            row[a] = 1.0
            rows.append(row)
    return np.array(rows)


def J(kap, zeta, **kw):
    M = bounds_matrix(kap, zeta, **kw)
    # variables (V_1..V_7, t): minimize t s.t. M V <= t, sum V = 1, V >= 0
    c = np.zeros(8); c[-1] = 1.0
    A_ub = np.hstack([M, -np.ones((M.shape[0], 1))])
    b_ub = np.zeros(M.shape[0])
    A_eq = np.zeros((1, 8)); A_eq[0, :7] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * 7 + [(None, None)])
    return res.fun, res.x[:7]


if __name__ == '__main__':
    print("p    kap=(p-1)/(p+1)  R9:kap/(2+kap)  planes4(zeta=1)  planes4(zeta)  all(zeta)   worst V")
    for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 101, 10 ** 6]:
        kap = (p - 1) / (p + 1)
        zeta = np.cos(np.pi / p)
        r9, _ = J(kap, 1.0, use_lines=False, use_123=False, use_perT=True)
        p4, _ = J(kap, 1.0, use_lines=False, use_123=True, use_perT=True)
        p4z, _ = J(kap, zeta, use_lines=False, use_123=True, use_perT=True)
        al, V = J(kap, zeta)
        print(f"{p:<5d} {kap:.4f}           {r9:.4f}          {p4:.4f}           {p4z:.4f}         {al:.4f}     "
              + ' '.join(f"{v:.3f}" for v in V))
