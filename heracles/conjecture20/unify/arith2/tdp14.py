# TDP14 (crude, delta-free form of TDP): since kappa<=2, any 14 frequencies fit the Parseval budget 1/c^2 = 29.7,
# so  Q(a,v,th) <= m(a_top14 + tau' a_rest)  (top14 = 14 largest weights).  TDP14(p): max_a m(a_top14 + tau' a_rest)/||a|| <= Gamma.
# With b = a_top14 + tau' a_rest:  ||a||^r = sum_top b^r + tau'^{-r} sum_rest b^r  (ordering: rest_b <= tau' min top_b).
# Heuristic maximisation by reweighted LP on b with the discounted cost; report ratio to Gamma_ref.
import numpy as np, sys
from scipy.optimize import linprog
from lib import *

def disc_cost(b, s, tp, ntop=14):
    r = 1/s; idx = np.argsort(-b); top = idx[:ntop]; rest = idx[ntop:]
    return (np.sum(b[top]**r) + tp**(-r)*np.sum(b[rest][b[rest] > 0]**r))**s

def run(p, s, starts, ntop=14, iters=80, seed=0):
    r, c, tp = consts(s); K = kmat(p); F = K.shape[1]; rng = np.random.default_rng(seed)
    best = (0, None)
    for b0 in starts:
        b = b0.copy()
        for it in range(iters):
            idx = np.argsort(-b); disc = np.full(F, tp**(-r)); disc[idx[:ntop]] = 1.0
            w = disc*(b + 1e-7*b.max())**(r-1)
            res = linprog(w, A_ub=-K, b_ub=-np.ones(F), bounds=(0, None), method='highs')
            if res.status != 0: break
            bn = res.x*(res.x > 1e-12*res.x.max())
            val = (K @ bn).min()/disc_cost(bn, s, tp, ntop)
            if val > best[0]: best = (val, bn.copy())
            if np.allclose(bn, b, rtol=1e-8, atol=1e-12): break
            b = bn
    return best

if __name__ == '__main__':
    s = float(sys.argv[1]); ps = [int(x) for x in sys.argv[2].split(',')]; nst = int(sys.argv[3])
    r, c, tp = consts(s)
    for p in ps:
        F = (p-1)//2; rng = np.random.default_rng(p)
        sup = load_support(p)
        if sup is not None and abs(s-1.5) < 1e-12: Gref, y0 = sup; Gref = max(Gref, gamma_lower(p, s, y0))
        else: Gref, y0 = prog(p, s, restarts=6)
        starts = [y0 + 1e-3*y0.max()*rng.uniform(0, 1, F) for _ in range(nst)]
        starts += [rng.uniform(0.2, 1, F)**3 for _ in range(nst)]
        starts += [np.where(rng.uniform(0, 1, F) < min(1, 40/F), 1.0, 0.0) + 1e-6 for _ in range(nst)]
        val, b = run(p, s, starts)
        nsp = int((b > 0).sum())
        print(f"p={p} s={s}: Gamma_ref={Gref:.5f}  TDP14 max ratio = {val/Gref:.5f}   |spt b| = {nsp}", flush=True)
