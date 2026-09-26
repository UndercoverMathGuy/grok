# Numerical optimiser for the (relaxed) light-junk programme.
#   maximise  V = min_D sum_k cov_k(D) [ x_k^{3/2}/sqrt3 + R u_k^{3/2} ]
#   s.t.      sum x + sum u = 1,  x >= 0,  0 <= u_k <= eps.
# x = heavy (block-rate) shares, u = light (junky, rate R) shares. Relaxation: a frequency may carry both, which only
# helps the adversary (a frequency with x + u > eps is really heavy and would be aimed at), so max V here is an
# UPPER bound on what light junk can achieve. With u = 0 the value is Gamma*/sqrt3 (heavy-only programme).
import numpy as np
from scipy.optimize import minimize

def cover(p):
    h = (p - 1) // 2
    D = np.arange(1, h + 1)[:, None]; k = np.arange(1, h + 1)[None, :]
    return 1 - np.cos(2 * np.pi * k * D / p)

def solve(p, eps, R, starts, rng, x0s=None, maxiter=600):
    A = cover(p); h = A.shape[0]; s3 = np.sqrt(3)
    def amp(z):
        x, u = z[:h], z[h:2 * h]
        return np.clip(x, 0, None) ** 1.5 / s3 + R * np.clip(u, 0, None) ** 1.5
    def damp(z):
        x, u = np.clip(z[:h], 0, None), np.clip(z[h:2 * h], 0, None)
        return 1.5 * x ** 0.5 / s3, 1.5 * R * u ** 0.5
    cons = [dict(type='ineq', fun=lambda z: A @ amp(z) - z[-1],
                 jac=lambda z: np.hstack([A * damp(z)[0][None, :], A * damp(z)[1][None, :], -np.ones((h, 1))])),
            dict(type='eq', fun=lambda z: z[:2 * h].sum() - 1, jac=lambda z: np.r_[np.ones(2 * h), 0.0])]
    bnds = [(0, None)] * h + [(0, eps)] * h + [(None, None)]
    best, bz = -np.inf, None
    inits = list(x0s or [])
    for _ in range(starts):
        kind = rng.integers(3)
        x = np.zeros(h); u = np.zeros(h)
        if kind == 0:                                    # random sparse heavy + light tails
            S = rng.choice(h, rng.integers(2, min(h, 12) + 1), replace=False); x[S] = rng.dirichlet(np.ones(len(S)))
            if eps > 0:
                L = rng.choice(h, rng.integers(0, h + 1), replace=False); u[L] = rng.uniform(0, eps, len(L))
        elif kind == 1 and eps > 0:                      # mostly light
            u[:] = rng.uniform(0.3, 1, h) * eps
            x[rng.choice(h, 3, replace=False)] = rng.uniform(0, 1, 3)
        else:
            x = rng.dirichlet(np.ones(h) * 0.3)
        tot = x.sum() + u.sum(); x /= tot; u /= tot; u = np.minimum(u, eps)
        x += (1 - x.sum() - u.sum()) / h
        inits.append(np.r_[x, u])
    for z0 in inits:
        z0 = np.r_[z0, (A @ amp(np.r_[z0, 0])).min()]
        r = minimize(lambda z: -z[-1], z0, jac=lambda z: np.r_[np.zeros(2 * h), -1.0], constraints=cons, bounds=bnds,
                     method='SLSQP', options=dict(maxiter=maxiter, ftol=1e-12))
        z = r.x; z[:2 * h] = np.clip(z[:2 * h], 0, None); z[h:2 * h] = np.minimum(z[h:2 * h], eps)
        z[:2 * h] /= z[:2 * h].sum()
        v = (A @ amp(z)).min()                           # re-evaluate the TRUE objective at a feasible point
        if v > best:
            best, bz = v, z[:2 * h].copy()
    return best, bz
