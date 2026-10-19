import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(3)
r = 1/1.8552
def solve(P, T, starts=60):
    cons = [dict(type='eq', fun=lambda l: np.sum(l)),
            dict(type='ineq', fun=lambda l: 2*P - np.sum(l**2)),
            dict(type='ineq', fun=lambda l: np.sum(l**3) - T)]
    obj = lambda l: np.sum(np.clip(1+l, 0, None)**r)
    best = (np.inf, None)
    for t in range(starts):
        l0 = rng.dirichlet(np.ones(4)*rng.uniform(0.3,3))*4 - 1
        res = minimize(obj, l0, method='SLSQP', constraints=cons, bounds=[(-1,3)]*4, options=dict(maxiter=600, ftol=1e-13))
        l = res.x
        if abs(np.sum(l))<1e-7 and np.sum(l**2) <= 2*P+1e-7 and np.sum(l**3) >= T-1e-7 and res.fun < best[0]:
            best = (res.fun, l)
    return best
for trial in range(25):
    P = rng.uniform(0.05, 2.5); T = -rng.uniform(0, 3)*P**1.5
    v, l = solve(P, T)
    l = np.sort(l)[::-1]
    act_V = abs(np.sum(l**2) - 2*P) < 1e-6; act_C = abs(np.sum(l**3) - T) < 1e-6
    print(f"P={P:.3f} T={T:.3f}: lam={np.round(l,4)} actV={act_V} actC={act_C} obj={v:.5f}")
