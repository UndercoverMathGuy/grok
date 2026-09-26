# Refined analytic floor for Lemma P3 (uses the exact variance and the actual best pair, not the 2/3 and M^2/2 shortcuts):
# exposure >= max_pair M^{3/2}/sqrt(M^2 + V),  M = sum_k c_k (e_P + e_P'),  V = 2 sum_k c_k^2 e_P e_P'.
# denominator = sum_k c_k sqrt(e1 + e2 + e12);  c_k in [0, 2], sum c_k <= p/2, K <= h.  (Homogeneous of degree 0 in e.)
import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(0)
def unpack(x, K, C):
    z = np.clip(x[:3 * K], -30, 30); e = np.exp(z - z.max()).reshape(3, K)
    c = 2 / (1 + np.exp(-np.clip(x[3 * K:], -30, 30)))
    if c.sum() > C: c = c * C / c.sum()
    return e, c
def ratio(x, K, C):
    e, c = unpack(x, K, C)
    best = 0
    for P, Q in ((0, 1), (0, 2), (1, 2)):
        M = (c * (e[P] + e[Q])).sum(); V = 2 * (c ** 2 * e[P] * e[Q]).sum()
        if M > 0: best = max(best, M ** 1.5 / np.sqrt(M ** 2 + V))
    den = (c * np.sqrt(e.sum(0))).sum()
    return best / den if den > 0 else 9
for p in (11, 13, 17):
    h = (p - 1) // 2; C = p / 2; worst = (9, None)
    for K in range(1, h + 1):
        for _ in range(40):
            r = minimize(ratio, rng.normal(size=4 * K), args=(K, C), method='Nelder-Mead', options=dict(maxiter=20000, xatol=1e-10, fatol=1e-13))
            if r.fun < worst[0]: worst = (r.fun, (K, r.x))
    e, c = unpack(worst[1][1], worst[1][0], C)
    print(f"p={p}: refined floor min = {worst[0]:.4f} (need > 0.272; crude {(2/3)/np.sqrt(p/2):.4f}); argmin K={worst[1][0]}, "
          f"c={np.round(c, 3).tolist()}, pattern shares={np.round(e / e.sum(), 3).tolist()}", flush=True)
