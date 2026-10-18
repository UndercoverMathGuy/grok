"""Single-T L^2-paid certificate at (5,2).

Bound (proved chain, see notes):  for every network with beta_2 = 0, every symmetric probability mu on Z_5^*, and every
z in C^4 (indexed by delta) with |z_delta| <= mu_delta:
   h <= K_1(mu) Re beta_1 - D(z) M + R L(z),
 M = |beta_{1,T}| (any T), R = 2 sum_i rho_i, rho_i = sqrt(e_i^2 - |beta_i1|^2),
 D(z) = |sum_d conj(z_d) c_1(d)|/2,  L(z) = sqrt(|sum z c_1|^2 + |sum z c_2|^2 + |sum conj(z) c_2|^2)/2.
Cost:  3^{1-s} B_5 P^s >= sqrt(P_1^2 + 3^{2-2s} R^2/4),  P_1 = 3^{1-s} Q^s >= max(|A| - c_1 M, 3^{1-s}|A|).
Ratio bound:  max over M/P_1 in [0, 3^{s-1}] of  sqrt(Abar(M)^2 + 4 * 3^{2s-2} L^2),  Abar = min(K(1 + c_1 m) - D m, K 3^{s-1} - D m).
We minimise this over certificates and compare with Gamma_s.
"""
import sys
import numpy as np
from scipy.optimize import minimize
from p52lib import *

c1d = {d: 1 - rho ** (-d) for d in DELS}
c2d = {d: 1 - rho ** (-2 * d) for d in DELS}


def parts(a, zr, zi):
    mu = {1: a / 2, 4: a / 2, 2: (1 - a) / 2, 3: (1 - a) / 2}
    z = {}
    for k, d in enumerate(DELS):
        zz = zr[k] + 1j * zi[k]
        # project into |z| <= mu
        if abs(zz) > mu[d]:
            zz = zz / abs(zz) * mu[d]
        z[d] = zz
    K = sum(mu[d] * (1 - np.cos(2 * np.pi * d / 5)) for d in DELS)
    D = abs(sum(np.conj(z[d]) * c1d[d] for d in DELS)) / 2
    L = np.sqrt(abs(sum(z[d] * c1d[d] for d in DELS)) ** 2 + abs(sum(z[d] * c2d[d] for d in DELS)) ** 2
                + abs(sum(np.conj(z[d]) * c2d[d] for d in DELS)) ** 2) / 2
    return K, D, L


def bound(v, s):
    a = 1 / (1 + np.exp(-v[0]))
    K, D, L = parts(a, v[1:5], v[5:9])
    c1 = 1 - 3 ** (1 - s)
    ms = np.linspace(0, 3 ** (s - 1), 401)
    Ab = np.minimum(K * (1 + c1 * ms) - D * ms, K * 3 ** (s - 1) - D * ms)
    Ab = np.maximum(Ab, 0)
    return np.max(np.sqrt(Ab ** 2 + 4 * 3 ** (2 * s - 2) * L ** 2))


s = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
rng = np.random.default_rng(0)
best = (np.inf, None)
for k in range(200):
    v0 = rng.normal(size=9) * 0.3
    v0[0] = rng.normal() * 2
    res = minimize(lambda v: bound(v, s), v0, method='Nelder-Mead', options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12})
    if res.fun < best[0]:
        best = (res.fun, res.x)
v = best[1]
a = 1 / (1 + np.exp(-v[0]))
K, D, L = parts(a, v[1:5], v[5:9])
print(f"s={s}: best single-T L2 bound {best[0]:.5f} vs Gamma_s {Gamma(s):.5f}  (ratio {best[0]/Gamma(s):.4f});  a={a:.4f} K={K:.4f} D={D:.4f} L={L:.4f}")
