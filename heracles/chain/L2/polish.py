# Genuine (non-relaxed) configurations: each frequency is EITHER heavy (share >= eps, block rate t^{3/2}/sqrt3) OR light
# (share <= eps, rate R t^{3/2}). Labels taken from a relaxed solution (light iff u > x), then shares re-optimised with
# SLSQP under the label bounds. Returns the genuine value min_D sum cov amp(t).
import numpy as np
from scipy.optimize import minimize
from progopt import cover
def polish(p, z, eps, R, maxiter=800):
    A = cover(p); h = A.shape[0]; s3 = np.sqrt(3)
    x, u = z[:h], z[h:2 * h]; light = u > x
    w = np.where(light, R, 1 / s3)
    t0 = np.where(light, np.minimum(x + u, eps), np.maximum(x + u, 0)); t0 = t0 / t0.sum()
    lo = np.where(light, 0.0, eps); hi = np.where(light, eps, 1.0)
    # heavy frequencies with tiny share are better dropped (bound t >= eps would force them): let them be 0 via a second
    # label option: heavy with t0 < eps/2 -> drop (bounds [0,0])
    drop = (~light) & (t0 < eps / 2); lo[drop] = 0; hi[drop] = 0; t0[drop] = 0
    t0 = np.clip(t0, lo, hi); t0 = t0 / max(t0.sum(), 1e-12)
    amp = lambda t: w * np.clip(t, 0, None) ** 1.5
    cons = [dict(type='ineq', fun=lambda y: A @ amp(y[:h]) - y[h],
                 jac=lambda y: np.hstack([A * (1.5 * w * np.clip(y[:h], 0, None) ** 0.5)[None, :], -np.ones((h, 1))])),
            dict(type='eq', fun=lambda y: y[:h].sum() - 1, jac=lambda y: np.r_[np.ones(h), 0.0])]
    y0 = np.r_[t0, (A @ amp(t0)).min()]
    r = minimize(lambda y: -y[h], y0, jac=lambda y: np.r_[np.zeros(h), -1.0], constraints=cons,
                 bounds=list(zip(lo, hi)) + [(None, None)], method='SLSQP', options=dict(maxiter=maxiter, ftol=1e-12))
    t = np.clip(r.x[:h], lo, hi)
    if abs(t.sum() - 1) > 1e-6 or (t[light] > eps * (1 + 1e-9)).any() or (t[~light & (hi > 0)] < eps * (1 - 1e-9)).any():
        t = t0                                          # fall back to the feasible start
    t = t / t.sum()
    # genuine value with the TRUE rule: share < eps -> light rate R, else block rate (re-derive labels from t itself)
    amp_true = np.where(t <= eps * (1 + 1e-9), R * t ** 1.5, t ** 1.5 / s3)   # sup over shares < eps = value at eps
    return (A @ amp_true).min(), t
