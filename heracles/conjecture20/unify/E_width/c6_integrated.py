# Integrated entropic width inequality (Theorem W3), checked at n = 2 (pi = 3^{1-s}, H(theta_1) = log 3):
#   log gamma(theta;s) <= log D_S(s) + (s-1)[Hbar^dec_S(s) - H_{1/s}(m_theta)] - Delta_S(theta),
#   Delta_S(theta) = log(D_S(1)/gamma(theta;1)),  D_S(1) = B_p Gamma^{(S)}_1,  Hbar^dec = path average of H^dec.
# Also the convex-end detection lower bound for S = F: Delta_F >= -log(1 - M/(2 B_p P_nu)).
import numpy as np, itertools
from scipy.integrate import quad
from elib import *

rng = np.random.default_rng(6); n = 2; nu = 3
cache = {}


def Hdec(p, sig, S):
    key = (p, round(sig, 10), tuple(S))
    if key not in cache:
        G, ys = programme_vertices(p, sig, S)
        cache[key] = np.log(3) + min(shannon(zshares(y, sig)) for y in ys)
    return cache[key]


worst = -np.inf; worstdet = -np.inf; cnt = 0; free = 0
for p in [5, 7, 11]:
    F = (p - 1) // 2
    for trial in range(12):
        s = rng.uniform(1.05, 1.5)
        S = list(range(1, F + 1))
        G, ys = programme_vertices(p, s, S)
        y = ys[0] if trial % 2 == 0 else np.full(F, 1.0) / F ** s
        U, W = if_network(p, n, y)
        eps = [0.0, 0.03, 0.1][trial % 3]
        U = U + eps * rng.normal(size=U.shape); W = W + eps * rng.normal(size=W.shape)
        net = Net(U, W); h = net.margin()
        if h <= 0: continue
        Sf = [j for j in range(1, F + 1) if abs(net.clean()[j - 1]) > 1e-12]
        D1 = Bp(p, n) * programme_vertices(p, 1.0, Sf)[0]
        Ds = 3 ** (1 - s) * Bp(p, n) * programme_vertices(p, s, Sf)[0]
        Hbar = quad(lambda sig: Hdec(p, sig, Sf), 1, s, limit=100)[0] / (s - 1)
        a = net.mass(); g1 = h / a.sum(); gs = net.gamma(s)
        DeltaS = np.log(D1 / g1)
        Hr = renyi(a, 1 / s)
        rhs = np.log(Ds) + (s - 1) * (Hbar - Hr) - DeltaS
        worst = max(worst, np.log(gs) - rhs); cnt += 1
        if Hr >= Hbar: free += 1; assert gs <= Ds * (1 + 1e-9)
        if Sf == S:
            phi = net.fields()
            M = max(abs(phi[j][chi]) for j in range(F) for chi in itertools.product(range(p), repeat=n) if any(chi))
            worstdet = max(worstdet, -np.log(max(1 - M / (2 * Bp(p, n) * a.sum()), 1e-300)) - DeltaS)
print(f"W3 checked on {cnt} networks: max [log gamma - RHS] = {worst:.2e} (must be <= ~1e-6, quadrature error)")
print(f"   {free} of them have H_1/s >= Hbar^dec (free half), all with gamma <= D_S")
print(f"convex-end detection: max [ -log(1 - M/(2 B_p P_nu)) - Delta_F ] = {worstdet:.3e} (must be <= 0)")
