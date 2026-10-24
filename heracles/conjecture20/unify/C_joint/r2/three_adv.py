import numpy as np
from scipy.optimize import minimize
from closure_check import setup, margin_parts
def adv(p, n, s, fam, seeds=8):
    nu = n+1; C = 2*(s-1)/nu; cn = 1 - (n/nu)**(s-1); best = -1; k = len(fam)
    X, Z, KAP, CD = setup(p, n, fam)
    def obj(x):
        al = np.exp(x[:k]); be = np.exp(x[k:2*k]); ph = x[2*k:]
        L = al + be; th = be/L; ap = L*(1 - th*np.minimum(cn, C*th)); ds = np.argmin(KAP @ ap)
        h, _, _ = margin_parts(al, be, ph, Z, KAP, CD)
        return -h/(KAP[ds] @ ap)
    for sd in range(seeds):
        rr = np.random.default_rng(sd)
        x0 = np.concatenate([rr.normal(0, 1, k), rr.normal(0, 1, k), rr.uniform(0, 2*np.pi, k)])
        res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 6000, 'xatol': 1e-10, 'fatol': 1e-13})
        best = max(best, -res.fun)
    return best
cases = [(7, 2, 1.5, [(1, (0, 1)), (2, (0, 1)), (3, (0, 1))]),     # additive triple 1+2=3 on a common set (balanced)
         (11, 2, 1.5, [(1, (0, 1)), (2, (0, 1)), (3, (0, 1))]),
         (13, 2, 1.5, [(2, (0, 1)), (4, (0, 1)), (6, (0, 1))]),
         (11, 3, 2.0, [(1, (0, 1, 2)), (2, (0, 1, 2)), (3, (0, 1, 2))]),
         (11, 2, 1.5, [(1, (0,)), (2, (1,)), (3, (0, 1))]),          # unbalanced, distinct sets
         (13, 3, 2.0, [(1, (0,)), (5, (1, 2)), (6, (0, 1, 2))])]
for (p, n, s, fam) in cases:
    print(f"p={p} n={n} s={s} family {fam}: max h / sum a' kappa(d*) = {adv(p, n, s, fam):.6f} (<= 1)", flush=True)
