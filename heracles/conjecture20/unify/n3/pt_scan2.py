"""PT-relaxation constant K(s) via the closed-form inner solver (floating point), currencies pt and cf.
Also validates the closed form against polished local optimisation on random (P,T)."""
import sys
import numpy as np
from scipy.optimize import minimize
from ptinner import F
from kbound import G
rng = np.random.default_rng(1)
# validation
r = 1/1.8552
worst = 0
for t in range(300):
    P = rng.uniform(0.01, 3.0); T = -rng.uniform(0, 4)*P**1.5 + (rng.uniform(0, 0.2) if rng.random() < 0.2 else 0)
    cons = [dict(type='eq', fun=lambda l: np.sum(l)), dict(type='ineq', fun=lambda l: 2*P - np.sum(l**2)),
            dict(type='ineq', fun=lambda l: np.sum(l**3) - T)]
    obj = lambda l: np.mean(np.clip(1+l, 0, None)**r)**(1/r)
    best = np.inf
    for s0 in range(25):
        l0 = rng.dirichlet(np.ones(4)*rng.uniform(0.3, 3))*4 - 1
        res = minimize(obj, l0, method='SLSQP', constraints=cons, bounds=[(-1, 3)]*4, options=dict(maxiter=500, ftol=1e-14))
        l = res.x
        if abs(l.sum()) < 1e-8 and np.sum(l**2) <= 2*P + 1e-8 and np.sum(l**3) >= T - 1e-8:
            best = min(best, obj(l))
    fc = F(P, T, r)
    worst = max(worst, fc - best)
print("validation: max(closed - local-opt) over 300 random (P,T) =", worst, "(must be <= ~1e-8)")
for cur in ['pt', 'cf']:
    for s in [1.8552, 1.9, 1.95, 2.0]:
        r = 1/s
        best = (-np.inf, None)
        for a in np.linspace(0.3, 1.0, 701):
            for m in np.linspace(0.002, 1.0, 500):
                g = G(a, m)
                Pb = (4 if cur == 'pt' else 3)*m*m + 2*g*g
                Tb = -(24 if cur == 'pt' else 12)*m*m*g
                v = (a - F(Pb, Tb, r))/m
                if v > best[0]:
                    best = (v, (a, m, g, Pb, Tb))
        print(f"{cur} s={s}: K_PT ~ {best[0]:.4f} at (a,m,G,P,T)={np.round(best[1],4)}", flush=True)
