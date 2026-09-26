# 3a.9 diagnostics on the p = 5 junky local max (claude0923/endpoint_p5_s3.npy, width 9).
# (1) its score, X share, junk share; (2) the "pure-part" bounds computed from clean + same-|k| junk only (X ignored):
#     one-sided  C(D) + min over rows of junk,  two-sided (flip)  C(D) - max|j1|, C(D) - max|j2|.
#     If the true margin exceeds a pure-part bound, X is doing real work and 3a.9 is load-bearing here.
import numpy as np
from common import *
p, N = 5, 9
v = np.load('../claude0923/endpoint_p5_s3.npy').reshape(3, N, p)
net = [(v[0, i], v[1, i], v[2, i]) for i in range(N)]
z = table(p, net); m = margin(z); Gam, _ = programme(p, 1.5)
print(f"gamma/gamma_sym = {gamma(p, net) / gamma_sym(p, Gam):.6f}")
Zh = np.fft.fftn(z - z.mean(2, keepdims=True)); P2 = np.abs(Zh) ** 2; a = np.arange(p); f = np.minimum(a, p - a)
same = (f[:, None, None] == f[None, :, None]) & (f[None, :, None] == f[None, None, :]) & (f[None, None, :] > 0)
print(f"X share {P2[~same].sum() / P2.sum():.3f}")
G, J = spectrum(z); w = np.exp(2j * np.pi / p); h = 2; u = np.arange(p)
A_, B_ = np.meshgrid(u, u, indexing='ij')
for D in (1, 2):
    cov = 1 - np.cos(2 * np.pi * np.arange(1, h + 1) * D / p)
    C = 2 / p ** 3 * sum(cov[k - 1] * np.real(G[k]) for k in (1, 2))
    jt = lambda T, x: 2 / p ** 3 * sum(cov[k - 1] * np.real(J[T][k] * w ** (-2 * k * x)) for k in (1, 2))
    onesided = C + (jt('1', A_) + jt('2', B_) + jt('12', A_ + B_)).min()
    two1 = C - np.abs(jt('1', u)).max(); two2 = C - np.abs(jt('2', u)).max()
    print(f"D={D}: true margin {m:.5f} | clean C {C:.5f} | pure one-sided {onesided:.5f} | flip slot1 {two1:.5f} | flip slot2 {two2:.5f}")
