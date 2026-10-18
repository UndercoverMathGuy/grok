# Heuristic search for allocations violating TDP:
#   TDP(p):  for all a >= 0:  Phi(a) := min_d [ (K a)_d + c tau' tail_d(a) ] / ||a||  <=  Gamma_s(p)
# Reports  R = max_found Phi(a) / Gamma_ref  (Gamma_ref = best programme value on file; R>1 would refute TDP or improve Gamma).
# Method: reweighted LP with the tail partition frozen at the current point (fixed-point iteration), many starts:
#   programme vertices, spread random supports of size N, programme + extra frequencies.
import numpy as np, sys, json
from scipy.optimize import linprog
from lib import *

def phi_tdp(a, K, s, c, tp):
    vals = [(a*K[d]).sum() + c*tp*tail_bound(a, K[d], c) for d in range(K.shape[0])]
    return min(vals)/norm_r(a, s)

def coef_matrix(a, K, c, tp):
    """M[d,l] = kappa_l(d) * (1 + tp*(c - omega_l(d))) with omega from the greedy kappa-budget at the current a."""
    F = K.shape[1]; M = K.copy()
    idx = np.argsort(-a)
    for d in range(K.shape[0]):
        budget = 1/c**2; fac = np.ones(F)
        for i in idx:
            k = K[d, i]
            if a[i] <= 0: fac[i] = 1 + tp*c; continue        # zero weights: undetected if switched on
            if k <= 0: continue
            if k <= budget: budget -= k
            elif budget > 0: fac[i] = 1 + tp*(c - c*np.sqrt(budget/k)); budget = 0
            else: fac[i] = 1 + tp*c
        M[d] *= fac
    return M

def reweighted(a0, K, s, c, tp, iters=60):
    r = 1/s; a = a0.copy(); best = (phi_tdp(a, K, s, c, tp), a.copy())
    for it in range(iters):
        M = coef_matrix(a, K, c, tp)
        w = (a + 1e-6*max(a.max(), 1e-12))**(r-1)
        res = linprog(w, A_ub=-M, b_ub=-np.ones(K.shape[0]), bounds=(0, None), method='highs')
        if res.status != 0: break
        an = res.x*(res.x > 1e-12*res.x.max())
        val = phi_tdp(an, K, s, c, tp)
        if val > best[0]: best = (val, an.copy())
        if np.allclose(an/an.sum(), a/a.sum(), atol=1e-9): break
        a = an
    return best

if __name__ == '__main__':
    s = float(sys.argv[1]); ps = [int(x) for x in sys.argv[2].split(',')]; nst = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    r, c, tp = consts(s); rng = np.random.default_rng(0)
    for p in ps:
        K = kmat(p); F = K.shape[1]
        sup = load_support(p)
        G0, y0 = prog(p, s, restarts=6) if (sup is None or s != 1.5) else sup
        Gref = max(G0, gamma_lower(p, s, y0))
        bestR = 0; binfo = None
        starts = []
        for N in [20, 30, 40, 60, 90]:
            if N > F: continue
            for t in range(nst):
                a = np.zeros(F); S = rng.choice(F, N, replace=False); a[S] = rng.uniform(0.5, 1.5, N); starts.append(('spread%d' % N, a))
        for t in range(nst):
            a = y0.copy(); extra = rng.choice(F, min(F, 25), replace=False); a[extra] += rng.uniform(0.2, 1.0, len(extra))*y0.max()
            starts.append(('prog+extra', a))
        for name, a0 in starts:
            val, a = reweighted(a0, K, s, c, tp)
            R = val/Gref
            if R > bestR: bestR = R; binfo = (name, int((a > 0).sum()))
        print(f"p={p} s={s}: Gamma_ref={Gref:.5f}  max Phi_TDP/Gamma_ref = {bestR:.5f}  (start {binfo[0]}, |spt|={binfo[1]})", flush=True)
