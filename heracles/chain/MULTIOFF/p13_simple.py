# Simplified provable form of Lemma P3 (best pair M >= 2E/3; V <= (1/2) sum_k w_k^2 with w_k = c_k eta_k^2; E = sum w = 1):
#   depth >= (2/3)^{3/2} / [ sqrt(4/9 + sum w^2/2) * sum_k sqrt(c_k w_k) ]
# Maximise the bracket over w in the simplex (K <= h entries) and c_k in [0,2] with sum c <= p/2.
import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(1)
def neg(x, K, C):
    w = np.exp(np.clip(x[:K], -30, 30)); w /= w.sum()
    c = 2 / (1 + np.exp(-np.clip(x[K:], -30, 30)))
    if c.sum() > C: c *= C / c.sum()
    return -np.sqrt(4 / 9 + (w ** 2).sum() / 2) * np.sqrt(c * w).sum()
for p in (11, 13, 17):
    h = (p - 1) // 2; C = p / 2; best = 0
    for K in range(1, h + 1):
        for _ in range(200):
            r = minimize(neg, rng.normal(size=2 * K) * 2, args=(K, C), method='Nelder-Mead', options=dict(maxiter=40000, xatol=1e-11, fatol=1e-13))
            best = max(best, -r.fun)
    print(f"p={p}: max bracket {best:.5f} -> provable depth >= {(2 / 3) ** 1.5 / best:.5f}  (need > 0.272)", flush=True)
