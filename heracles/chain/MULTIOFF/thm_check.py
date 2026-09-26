# Check the key inequality behind the p <= 11 theorem on block + junk networks (single-frequency neurons):
#   for every offset D:  margin <= (2/p^3) [ sum_k cov_k(D) |G_k| - A0 sum_k cov_k(D) eta_k ],  A0 = (2/3)/sqrt(p/2).
import sys, numpy as np
sys.path.insert(0, '..')
from common import *
rng = np.random.default_rng(3); worst = -9; npos = 0
for p in (5, 7, 11):
    h = (p - 1) // 2; A0 = (2 / 3) / np.sqrt(p / 2); Gam, ystar = programme(p, 1.5)
    for t in range(150):
        y = rng.uniform(0, 1, h) * (rng.uniform(size=h) < 0.7) + 1e-3
        net = block_net(p, y) + [tuple(rng.uniform(0.1, 1.2) * v for v in n) for n in random_net(p, int(rng.integers(1, 5)), rng, pure=True)]
        z = table(p, net); m = margin(z); G, J = spectrum(z); npos += m > 0
        eta = np.sqrt(sum(np.abs(J[T]) ** 2 for T in J))
        for D in range(1, p):
            c = 1 - np.cos(2 * np.pi * np.arange(1, h + 1) * D / p)
            bound = 2 / p ** 3 * (c @ (np.abs(G[1:]) - A0 * eta[1:]))
            worst = max(worst, (m - bound) / abs(m))
print(f"max relative (margin - bound) over all D: {worst:.3e} (must be <= 0); positive-margin nets: {npos}/450")
