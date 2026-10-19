"""Upper-half slack: tolerance tau(S,eps) = Gamma_s / Gamma_s^{(S,eps)} - 1 for multi-floor classes
{y_k >= eps sum y for all k in S}, S = (optimal support) + {j}, |S| > K*.  Reweighted-LP (concave l_r minimisation), many restarts."""
import numpy as np
from scipy.optimize import linprog
from elib import G


def prog_floor(p, s, j, eps, restarts=40, seed=0):
    r = 1 / s; F = (p - 1) // 2; K = G.kappa_mat(p)[:F]; rng = np.random.default_rng(seed)
    A_ub = [-K]; b_ub = [-np.ones(F)]
    for jj in (j or []):
        row = eps * np.ones(F); row[jj - 1] -= 1.0          # eps*sum y - y_jj <= 0 for every jj in S
        A_ub.append(row[None, :]); b_ub.append(np.zeros(1))
    A_ub = np.vstack(A_ub); b_ub = np.concatenate(b_ub)
    best = 0
    for t in range(restarts):
        w = np.ones(F) if t == 0 else rng.uniform(0.1, 2.0, F)
        for it in range(300):
            res = linprog(w, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
            if res.status != 0: break
            y = res.x; wn = (y + 1e-12) ** (r - 1)
            if np.allclose(wn, w, rtol=1e-10): break
            w = wn
        if res.status != 0: continue
        val = np.sum(y[y > 1e-13] ** r) ** (-s)
        best = max(best, val)
    return best


for (p, s) in [(11, 1.5), (13, 1.5), (13, 2.0), (17, 1.5), (11, 2.5)]:
    Gm = prog_floor(p, s, [], 0)
    _, y, _ = G.programme(p, s, restarts=30)
    spt = set(np.nonzero(y > 1e-9)[0] + 1)
    out = [j for j in range(1, (p - 1) // 2 + 1) if j not in spt]
    if not out:
        print('p=%d s=%.2f: full support' % (p, s)); continue
    for j in out[:2]:
        S = sorted(int(k) for k in spt) + [j]
        taus = []
        for eps in [1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1]:
            taus.append('eps=%g: %.5f' % (eps, Gm / prog_floor(p, s, S, eps) - 1))
        print('p=%d s=%.2f Gamma=%.6f S=spt+{%d}=%s | ' % (p, s, Gm, j, S) + '; '.join(taus), flush=True)
