"""Further sanity checks.
 R4  R-elimination: sup_R (a + n R)/sqrt(F^2 + c R^2) = sqrt(a^2/F^2 + n^2/c) for a >= 0 (grid check).
 R7  torus equivariance of degree-2 HS certificates: rotating the Gram matrices by diag(e^{-i<e,theta>}) and the data by
     (B,C,D) -> (B e^{i t1}, C e^{i t2}, D e^{i(t1+t2)}) leaves the bound unchanged; the rotated grid density is >= 0.
 R10 (7,2) singleton exclusion: 3^{s-1} kappa_min(7) < (7/2) 3^{-s} for s in [1, 1.5] (and Gamma_s(7) >= (7/2) 3^{-s}).
 R13 shared neuron at b = 2 (p = 7): equal split of every factor between frequencies 1 and 2 gives
     |beta_1|^{2/3} + |beta_2|^{2/3} = e^{2/3}  (budget tight) and non-own mode mass 3 e^2 (free cross modes).
"""
import numpy as np
from cert_hs import cert_data
from hs2 import elim_ratio

rng = np.random.default_rng(3)
# R4
worst = 0
for t in range(2000):
    a = rng.uniform(0, 2); n = rng.uniform(0, 1); F = rng.uniform(0.3, 1.5); c = rng.uniform(0.01, 1)
    R = np.concatenate([[0], np.geomspace(1e-4, 1e4, 4000)])
    sup = np.max((a + n * R) / np.sqrt(F ** 2 + c * R ** 2))
    worst = max(worst, sup - np.sqrt(a ** 2 / F ** 2 + n ** 2 / c))
print("R4  max[grid sup - closed form] =", worst, "(<= 0 up to grid error)")

# R7
E2 = [(a, b) for a in range(3) for b in range(3)]
val, Gs = elim_ratio(0.1, 0.8, 0.7, 0.9, 0.3, 0.7, deg=2, ret=True)
cd = cert_data(Gs)
A = np.exp(0.1j); bT = [0.8, 0.7, 0.9 * np.exp(0.3j)]


def bound(cd, A, bT, R):
    a0 = complex(cd['a0'][0].lo, cd['a0'][1].lo); aT = [complex(x[0].lo, x[1].lo) for x in cd['aT']]
    return np.real(np.conj(A) * a0) + sum(np.real(np.conj(b) * a) for b, a in zip(bT, aT)) + R * float(cd['n'].hi)


for (t1, t2) in [(0.4, -1.3), (2.2, 0.9), (-3.0, 1.7)]:
    Dm = np.diag([np.exp(-1j * (e[0] * t1 + e[1] * t2)) for e in E2])
    Gr = [Dm.conj().T @ G @ Dm for G in Gs]
    cdr = cert_data(Gr)
    bTr = [bT[0] * np.exp(1j * t1), bT[1] * np.exp(1j * t2), bT[2] * np.exp(1j * (t1 + t2))]
    # grid positivity of rotated certificate
    mn = np.inf
    for G in Gr:
        for x1 in range(5):
            for x2 in range(5):
                th = (2 * np.pi * 2 * x1 / 5, 2 * np.pi * 2 * x2 / 5)
                v = np.array([np.exp(1j * (e[0] * th[0] + e[1] * th[1])) for e in E2])
                mn = min(mn, np.real(v.conj() @ G @ v))
    print(f"R7  theta=({t1},{t2}): bound original {bound(cd, A, bT, 0.5):.10f}  rotated {bound(cdr, A, bTr, 0.5):.10f}  min grid density {mn:.2e}")

# R10
k7 = 1 - np.cos(2 * np.pi / 7)
ss = np.linspace(1, 1.5, 501)
print("R10 max over s in [1,1.5] of 3^{s-1} kappa_min(7) / ((7/2) 3^{-s}) =", np.max(3 ** (ss - 1) * k7 / (3.5 * 3 ** (-ss))))

# R13
p = 7
B7 = 2 * (2 * p * 3) ** (-1.5)
a = np.arange(p)
u = (np.cos(2 * np.pi * a / p) + np.cos(2 * np.pi * 2 * a / p + 0.7))
w = (np.cos(2 * np.pi * a / p + 0.3) + np.cos(2 * np.pi * 2 * a / p - 0.2))
uh = np.fft.fft(u); wh = np.fft.fft(w)
nrm = np.sqrt(2 * np.sum(u ** 2) + np.sum(w ** 2)); e = B7 * nrm ** 3
# balance factors (rescale so that ||u1||=||u2||=||w||)
lam = (np.sum(w ** 2) / np.sum(u ** 2)) ** 0.25
u2 = u * lam; w2 = w / lam ** 2
uh = np.fft.fft(u2); wh = np.fft.fft(w2)
nrm = np.sqrt(2 * np.sum(u2 ** 2) + np.sum(w2 ** 2)); e = B7 * nrm ** 3
m = {j: (2 / p ** 3) * wh[j] * np.outer(uh, uh) for j in (1, 2, 3)}
b1 = abs(m[1][1, 1]); b2 = abs(m[2][2, 2])
tot = sum(np.sum(np.abs(m[j]) ** 2) for j in (1, 2, 3))
own = 4 * b1 ** 2 + 4 * b2 ** 2
print(f"R13 |b1|^(2/3)+|b2|^(2/3) = {b1**(2/3)+b2**(2/3):.6f}  e^(2/3) = {e**(2/3):.6f};  non-own mass / e^2 = {(tot-own)/e**2:.4f} (=3 expected)")
