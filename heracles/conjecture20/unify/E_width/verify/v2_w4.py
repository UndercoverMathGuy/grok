import numpy as np, itertools
from vlib import *
rng = np.random.default_rng(202)
wid = 0; wb = -np.inf; wc = -np.inf; cnt = 0; tight = np.inf
for (p, n, m) in [(5, 2, 5), (3, 3, 4), (7, 2, 4), (5, 3, 3), (11, 2, 4)]:
    F = (p - 1) // 2; nu = n + 1
    for trial in range(12):
        if trial % 2 == 0:   # near-IF networks with positive margin + junk neurons with DC and other frequencies
            Us, Ws = [], []
            for j in range(1, F + 1):
                U0, W0 = if_block(p, n, j, lam=rng.uniform(0.7, 1.2)); Us.append(U0); Ws.append(W0)
            U = np.concatenate(Us + [0.2 * rng.normal(size=(2, n, p))]); W = np.concatenate(Ws + [0.2 * rng.normal(size=(2, p))])
            U += 0.03 * rng.normal(size=U.shape); W += 0.03 * rng.normal(size=W.shape)
        else:
            U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
        f = table(U, W); g = gaps(f, p, n); h = g.min()
        Phi = fields(f, p, n); beta = np.array([np.conj(Phi[l].mean()) for l in range(F)])
        # symmetric random mu on Z_p^*
        mu = rng.random(p - 1) + 0.1; mu = (mu + mu[::-1]) / 2; mu /= mu.sum()     # index d-1 for d=1..p-1; d and p-d symmetric
        K = np.array([np.sum(mu * (1 - np.cos(2 * np.pi * l * np.arange(1, p) / p))) for l in range(1, F + 1)])
        for _ in range(5):
            j = rng.integers(1, F + 1); chi = tuple(rng.integers(0, p, n))
            if not any(chi): continue
            ph = coef(Phi[j - 1], chi, p, n)
            # identity
            x = np.indices((p,) * n).reshape(n, -1).T
            e = np.exp(-2j * np.pi * (x @ np.array(chi)) / p)
            lhs = sum(np.exp(2j * np.pi * j * d / p) * np.mean(g[:, d - 1] * e) for d in range(1, p))
            wid = max(wid, abs(lhs + (p / 2) * ph) / (1 + abs(ph)))
            bound = np.sum(K * beta.real) - (p / 2) * mu.min() * abs(ph)
            wb = max(wb, h - bound); cnt += 1
            tight = min(tight, bound - h)
        # convex-end corollary with uniform mu
        if h > 0:
            M = max(abs(coef(Phi[l], chi, p, n)) for l in range(F) for chi in itertools.product(range(p), repeat=n) if any(chi))
            a = sqnorm(U, W) ** (nu / 2); P = a.sum()
            g1 = h / P; G1 = p / (p - 1)
            wc = max(wc, g1 - Bp(p, n) * G1 * (1 - M / (2 * Bp(p, n) * P)))
print(f"W4 identity max rel err {wid:.1e}; general-mu bound max(h - bound) = {wb:.3e} (<=0) over {cnt}; min slack {tight:.3e}")
print(f"W4 corollary gamma(1) - B_p Gamma_1 (1 - M/(2 B_p P_nu)) max = {wc:.3e} (<=0)")
