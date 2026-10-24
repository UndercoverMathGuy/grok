# Numerical map of the phase-free reduction TDP* (relations case, n=2):
#   Phi*(a) = min_d [ (K a)_d + tau' E0(a, kappa(d)) ] / ||a||,   TDP*(p): Phi*(a) <= Gamma_s(p) for all a.
# (A) near-optimal allocations on file: kappa-mass of the support on the binding set, Phi*/Gamma.
# (B) spread allocations: best allocation (reweighted LP, several starts) restricted to a support of size N
#     = programme optimum's support plus N-K* random extra frequencies; report e, kappa-mass, E0/m and Phi*/Gamma.
#     TDP* closes such an allocation iff Phi*/Gamma <= 1; the 'margin' is 1 - Phi*/Gamma.
import numpy as np, sys
from scipy.optimize import linprog
from lib import *

def phi_star(a, K, s, c, tp, dset=None):
    rows = range(K.shape[0]) if dset is None else dset
    best = np.inf; arg = None
    for d in rows:
        v = (a*K[d]).sum() + tp*waterfill_excess(a, K[d], c)
        if v < best: best = v; arg = d
    return best/norm_r(a, s), arg

def restricted_prog(K, S, s, starts=4, rng=None, iters=80):
    r = 1/s; F = K.shape[1]; S = np.array(sorted(S)); KS = K[:, S]; best = (np.inf, None)
    for t in range(starts):
        w = np.ones(len(S)) if t == 0 else rng.uniform(0.3, 1.7, len(S))
        y = None
        for it in range(iters):
            res = linprog(w, A_ub=-KS, b_ub=-np.ones(K.shape[0]), bounds=(0, None), method='highs')
            if res.status != 0: break
            y = res.x; wn = (y + 1e-9*max(1, y.max()))**(r-1)
            if np.allclose(wn, w, rtol=1e-7): break
            w = wn
        if y is None: continue
        y = y*(y > 1e-12*y.max()); Q = np.sum(y[y > 0]**r)
        if Q < best[0]: best = (Q, y.copy())
    a = np.zeros(F); a[S] = best[1]; return a

if __name__ == '__main__':
    s = 1.5; r, c, tp = consts(s)
    mode = sys.argv[1]
    if mode == 'B':
        p = int(sys.argv[2]); Ns = [int(x) for x in sys.argv[3].split(',')]; reps = int(sys.argv[4])
        K = kmat(p); F = K.shape[1]; rng = np.random.default_rng(p)
        G, y0 = load_support(p); G = max(G, gamma_lower(p, s, y0)); S0 = np.nonzero(y0)[0]
        for N in Ns:
            for rep in range(reps):
                extra = rng.choice(np.setdiff1d(np.arange(F), S0), N-len(S0), replace=False)
                a = restricted_prog(K, np.concatenate([S0, extra]), s, starts=3, rng=rng)
                cov = K @ a; m = cov.min(); B = np.nonzero(cov <= m*(1+1e-7))[0]
                spt = a > 0
                km = min(K[b, spt].sum() for b in B)
                ph, d = phi_star(a, K, s, c, tp)
                E0 = waterfill_excess(a, K[d], c)
                print(f"p={p} N={N} rep={rep}: |spt a|={spt.sum()} e={m/(G*norm_r(a,s)):.4f} "
                      f"min kappa-mass on binding set={km:.2f}  E0/m at best d={E0/((a*K[d]).sum()):.4f}  Phi*/Gamma={ph/G:.4f}", flush=True)
