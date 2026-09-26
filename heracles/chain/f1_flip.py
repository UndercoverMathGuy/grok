# f.1 flip lemma, full pointwise form in common.py conventions (candidate s - D).
# U(a,b) = (gap_D(a,b) + gap_-D(a,b))/2 = C(D) + j1(a) + j2(b) + j12(a+b)
# S(a,b) = (gap_D(a+D/2,b) + gap_-D(a-D/2,b))/2 = C(D) - j1(a) + j2(b) - j12(a+b)
# with j_T(u) = (2/p^3) sum_k cov_k(D) Re[J_{k,T} w^{-2k u}].
import numpy as np
from common import *
rng = np.random.default_rng(6); worst = 0; worst_imp = 0
for p in (5, 7, 11, 13):
    w = np.exp(2j * np.pi / p); h = (p - 1) // 2; i2 = pow(2, -1, p)
    a = np.arange(p)[:, None]; b = np.arange(p)[None, :]
    for pure in (True, False):
        for t in range(8):
            net = random_net(p, 9, rng, pure=pure); z = table(p, net); g = gaps(z); G, J = spectrum(z)
            for D in range(1, p):
                cov = 1 - np.cos(2 * np.pi * np.arange(1, h + 1) * D / p)
                jt = lambda T, u: 2 / p ** 3 * sum(cov[k - 1] * np.real(J[T][k] * w ** (-2 * k * u)) for k in range(1, h + 1))
                C = 2 / p ** 3 * sum(cov[k - 1] * np.real(G[k]) for k in range(1, h + 1))
                sh = (D * i2) % p
                U = 0.5 * (g[:, :, D] + g[:, :, p - D])
                S = 0.5 * (g[(a + sh) % p, b, D] + g[(a - sh) % p, b, p - D])
                e = max(np.abs(U - (C + jt('1', a) + jt('2', b) + jt('12', a + b))).max(),
                        np.abs(S - (C - jt('1', a) + jt('2', b) - jt('12', a + b))).max()) / np.abs(g).max()
                if pure: worst = max(worst, e)
                else: worst_imp = max(worst_imp, e)
print(f"single-frequency nets: max error {worst:.1e};  mixing nets: {worst_imp:.2f} (X present, lemma not claimed)")
