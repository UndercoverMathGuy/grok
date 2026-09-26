# Check the general-shift identity: pairing offset D at rows (a + s1 D, b + s2 D) with -D at (a - s1 D, b - s2 D)
# gives per frequency  cov_k Re G_k + Re[ J1 W(s1) w^{-2ka} + J2 W(s2) w^{-2kb} + J12 W(s1+s2) w^{-2k(a+b)} ],
# W(s) = cos(2 s th) - cos((1 - 2s) th), th = 2 pi k D / p, s in Z_p (integer).  (common.py conventions)
import sys; sys.path.insert(0, '..')
import numpy as np
from common import *
rng = np.random.default_rng(11); worst = 0
for p in (5, 7, 11):
    h = (p - 1) // 2; w = np.exp(2j * np.pi / p); a = np.arange(p)[:, None]; b = np.arange(p)[None, :]
    for t in range(6):
        net = random_net(p, 7, rng, pure=True); z = table(p, net); g = gaps(z); G, J = spectrum(z)
        for D in range(1, p):
            for s1 in range(p):
                for s2 in range(p):
                    lhs = 0.5 * (g[(a + s1 * D) % p, (b + s2 * D) % p, D] + g[(a - s1 * D) % p, (b - s2 * D) % p, p - D])
                    rhs = 0
                    for k in range(1, h + 1):
                        th = 2 * np.pi * k * D / p; W = lambda s: np.cos(2 * s * th) - np.cos((1 - 2 * s) * th)
                        rhs = rhs + (1 - np.cos(th)) * np.real(G[k]) + np.real(J['1'][k] * W(s1) * w ** (-2 * k * a)
                              + J['2'][k] * W(s2) * w ** (-2 * k * b) + J['12'][k] * W(s1 + s2) * w ** (-2 * k * (a + b)))
                    worst = max(worst, np.abs(lhs - 2 / p ** 3 * rhs).max() / np.abs(g).max())
print(f"general-shift identity, max relative error {worst:.1e}")
