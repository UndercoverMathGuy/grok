# f.2: margin <= C(D) - ||j1||_sup (and slot 2), plus the l2 floor ||j1||_sup >= ||j1||_2. Compare with one-sided C(D) + min j1.
import numpy as np
from common import *
rng = np.random.default_rng(7); viol = -9; floor_viol = -9; gain = []
for p in (7, 11, 13):
    w = np.exp(2j * np.pi / p); h = (p - 1) // 2; u = np.arange(p)
    Gam, ystar = programme(p, 1.5)
    for t in range(40):
        net = block_net(p, ystar) + [tuple(rng.uniform(0.2, 0.6) * v for v in n) for n in random_net(p, 2, rng, pure=True)]
        z = table(p, net); m = margin(z); G, J = spectrum(z)
        for D in range(1, p):
            cov = 1 - np.cos(2 * np.pi * np.arange(1, h + 1) * D / p)
            C = 2 / p ** 3 * sum(cov[k - 1] * np.real(G[k]) for k in range(1, h + 1))
            for T in ('1', '2'):
                j = 2 / p ** 3 * sum(cov[k - 1] * np.real(J[T][k] * w ** (-2 * k * u)) for k in range(1, h + 1))
                viol = max(viol, m - (C - np.abs(j).max()))
                l2 = 2 / p ** 3 * np.sqrt(sum(cov[k - 1] ** 2 * abs(J[T][k]) ** 2 / 2 for k in range(1, h + 1)))
                floor_viol = max(floor_viol, l2 - np.abs(j).max())
                if np.abs(j).max() > 1e-12:
                    gain.append(np.abs(j).max() / max(-j.min(), 1e-15))
print(f"max(margin - two-sided bound) = {viol:.2e} (<= 0 needed);  max(l2 - sup) = {floor_viol:.2e} (<= 0 needed)")
print(f"two-sided / one-sided exposure: median {np.median(gain):.2f}, max {np.max(gain):.2f} (>= 1 always)")
