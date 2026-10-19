# Near the convex end: is the uniform allocation on F still the programme optimum?  Smooth-min L-BFGS over the r-ball from
# many random starts (and from perturbations of uniform); report the best coverage found vs uniform, and K_eff, sigma there.
import sys, json, numpy as np, warnings
from scipy.optimize import minimize
import hlib2 as H2
warnings.filterwarnings('ignore')
rng = np.random.default_rng(21)
for (p, s) in [(101, 1.12), (211, 1.05), (211, 1.08), (401, 1.02), (401, 1.05), (401, 1.08)]:
    K = H2.kmat(p); F = K.shape[1]; r = 1 / s
    unif = F ** (-s) * (F + 0.5)
    best = (unif, np.full(F, F ** (-s)))
    for t in range(30):
        if t % 3 == 0: z = rng.normal(0, 0.3, F)
        elif t % 3 == 1: z = rng.normal(0, 2.0, F)
        else:
            z = np.full(F, -8.0); idx = rng.choice(F, size=int(rng.integers(5, max(6, F // 3))), replace=False); z[idx] = rng.normal(0, 0.5, len(idx))
        for tau in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4):
            def f(z):
                y = np.exp(z); N = np.sum(y ** r) ** s; c = K @ y / N
                cm = c.min(); w = np.exp(-(c - cm) / (tau * unif)); lam = w / w.sum()
                return -(cm - tau * unif * np.log(w.sum())), -(y * (lam @ K) / N - (lam @ c) * (y ** r / np.sum(y ** r)))
            z = minimize(f, z, jac=True, method='L-BFGS-B', options={'maxiter': 3000}).x
        y = np.exp(z); y /= np.sum(y ** r) ** s; v = (K @ y).min()
        if v > best[0]: best = (v, y)
    G, y = best; cov = K @ y; B = np.nonzero(cov < G * (1 + 1e-6))[0]
    ell = np.sqrt((K[B] * y[None, :] ** 2).sum(1)).max() / G
    theta = (1 - 3 ** (1 - s)) / np.sqrt(3); rho = np.sqrt((p - 1) / (p + 1))
    print(json.dumps(dict(p=p, s=s, uniform=unif, best_found=G, uniform_optimal=bool(G <= unif * (1 + 1e-9)),
                          K_sig=int((y > 1e-3 * y.max()).sum()), K_eff=1 / ell ** 2, sigma=theta / (rho * ell))), flush=True)
