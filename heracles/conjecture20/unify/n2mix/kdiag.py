"""Diagnose the paid-ledger constant K for the optimal semi-rigid test at one configuration (per case)."""
import sys
import numpy as np
from scipy.optimize import minimize
from sr72 import SR72, CASES5, pay_simple, Gamma7, comp_info
from chanlib import block_assign
from leaks72 import tensor, K_block

s = float(sys.argv[1]); tau = float(sys.argv[2]); x1 = float(sys.argv[3]); x2 = float(sys.argv[4])
ledger = sys.argv[5] if len(sys.argv) > 5 else 'blk'
r = 1 / s


def ktrue(ent, starts=30, seed=0):
    rng = np.random.default_rng(seed)
    def negratio(v):
        xw = np.abs(v[0:3]); y1 = np.abs(v[3:7]); y2 = np.abs(v[7:11])
        xw = xw / max(np.sum(xw), 1e-300); y1 = y1 / np.sum(y1); y2 = y2 / np.sum(y2)
        N = sum(cv * np.sqrt(xw[l - 1] * y1[a] * y2[b]) for (l, a, b), cv in ent.items())
        bL3 = sum((xw[l - 1] * y1[l] * y2[l]) ** (1 / 3) for l in (1, 2)) ** 3
        return -N / np.sqrt(max(1 - bL3, 1e-300))
    best = 0
    for k in range(starts):
        res = minimize(negratio, rng.uniform(0, 1, 11), method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-13})
        best = max(best, -res.fun)
    return best


A1, A2 = 1 - tau, tau
F = ((pay_simple(x1, s) * A1) ** r + (pay_simple(x2, s) * A2) ** r) ** s
for case in CASES5:
    m = SR72(s, ledger, case)
    v = m.solve(A1, A2, x1 * A1, x2 * A2, F)
    ent = {}
    leak = []
    for j in (1, 2):
        w = m.w(j)
        for (mm, f, d) in m.paid[j]:
            ent[tuple(int(t) for t in f)] = ent.get(tuple(int(t) for t in f), 0) + 2 ** (d / 2) * abs(w[mm - 1])
            leak.append(abs(w[mm - 1]))
    Kl2 = 2 * np.linalg.norm(leak); Kfam = np.linalg.norm(list(ent.values())); Ksocp = m.K.value
    Kt = ktrue(ent)
    Kinj = K_block(tensor([(f, v) for f, v in ent.items()]))
    aval = m.a.value
    print(ent)
    print(f"case {case}: bound {v:.5f}  a/F={aval:.4f}  S=({m.S[1].value:.4f},{m.S[2].value:.4f})  K_l2 {Kl2:.4f} K_fam {Kfam:.4f} K_socp {Ksocp:.4f} K_inj {Kinj:.4f} K_true {Kt:.4f}"
          f"  -> bound with K_true {np.sqrt(max(aval,0)**2 + 3**(2*s-2)*Kt**2)/Gamma7(s):.5f}", flush=True)
