"""(P,T) moment relaxation of the 4x4 chain spectrum (floating-point exploration).

F(Pbar, Tbar) = min { M_r(nu) : nu >= 0, sum nu = 4, sum (nu-1)^2 <= 2 Pbar, sum (nu-1)^3 >= Tbar }.
Chain (0,1,1,2): Pbar = 4 m^2 + 2 g^2, Tbar = -24 m^2 g, g = G(a,m) (Fubini-Study triangle bound).
K_PT(s) = sup_{a,m} (a - F)/m.   usage: python pt_relax.py s
"""
import sys
import numpy as np
from scipy.optimize import minimize
from kbound import G, Fr


def F_PT(Pb, Tb, r, rng, starts=30):
    best = np.inf
    cons = [dict(type='eq', fun=lambda v: np.sum(v) - 4),
            dict(type='ineq', fun=lambda v: 2 * Pb - np.sum((v - 1) ** 2)),
            dict(type='ineq', fun=lambda v: np.sum((v - 1) ** 3) - Tb)]
    obj = lambda v: np.mean(np.clip(v, 0, None) ** r) ** (1 / r)
    for t in range(starts):
        v0 = rng.dirichlet(np.ones(4) * rng.uniform(0.3, 3)) * 4
        res = minimize(obj, v0, method='SLSQP', constraints=cons, bounds=[(0, 4)] * 4,
                       options=dict(maxiter=500, ftol=1e-12))
        if res.success or True:
            v = res.x
            if (abs(np.sum(v) - 4) < 1e-7 and np.sum((v - 1) ** 2) <= 2 * Pb + 1e-7
                    and np.sum((v - 1) ** 3) >= Tb - 1e-7 and v.min() >= -1e-9):
                best = min(best, obj(v))
    return best


if __name__ == '__main__':
    s = float(sys.argv[1]); r = 1 / s
    rng = np.random.default_rng(0)
    best = (-np.inf, None)
    rows = []
    for a in np.linspace(0.70, 1.0, 31):
        for m in np.linspace(0.02, 0.7, 35):
            g = G(a, m)
            Pb = min(4 * m * m + 2 * g * g, 6.0)
            Tb = -24 * m * m * g
            F = F_PT(Pb, Tb, r, rng, 12)
            ratio = (a - F) / m
            rows.append((ratio, a, m, g, F, Fr(Pb, r)))
            if ratio > best[0]:
                best = (ratio, (a, m, g, F))
    rows.sort(reverse=True)
    for row in rows[:8]:
        print("ratio=%.4f a=%.3f m=%.3f g=%.4f F_PT=%.5f F_Ponly=%.5f" % row)
    print("K_PT(s=%.4f) ~ %.4f at (a,m,g,F)=%s" % (s, best[0], np.round(best[1], 4)), flush=True)
