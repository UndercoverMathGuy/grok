# C_joint J6 Legendre networks (alpha = 0, beta_l e^{i phi_l} = 2 chihat(l)): where do they sit for Theorem L?
#   built as genuine class-C networks (n atoms per feature, aligned) for small p^n; L1 ratio and (L) test.
import numpy as np, math
from sclib import *
def legendre(p):
    chi = np.array([0] + [1 if pow(w, (p - 1) // 2, p) == 1 else -1 for w in range(1, p)], float)
    return np.array([np.sum(chi * np.exp(-2j * np.pi * l * np.arange(p) / p)) / p for l in range(1, (p - 1) // 2 + 1)])
for (p, n) in [(5, 2), (7, 2), (11, 2), (13, 2), (17, 2), (19, 2), (5, 3), (7, 3), (11, 3)]:
    nu = n + 1; b = 2.0; s = nu / b; F = (p - 1) // 2
    ch = legendre(p); B = 2 * ch                      # beta_l e^{i phi_l}
    specs = []
    for l in range(1, F + 1):
        mag = abs(B[l - 1]); phi = np.angle(B[l - 1]); lam = (2 ** n * mag / n) ** (1.0 / nu)
        for k in range(n):
            phis = np.full(n, -np.pi * k / n); phis[0] -= phi / 2
            specs.append((l, lam * np.ones(nu), phis))
    net = classC_net(p, n, specs, b)
    g = net.gaps(); h = g.min(); at = net.atoms(); A = np.abs(at.sum(0)); r = 1 / s
    Q = (np.abs(at) ** r).sum(0); a = nu ** (1 - s) * Q ** s
    Phi = fields(net); E = (np.abs(Phi - Phi.mean(0)) ** 2).mean(0)
    K = kappa_mat(p); d0 = int(np.argmin(K[:, :F] @ a)); dm = p - 2 - d0
    O = max(np.ptp(g[:, d0]), np.ptp(g[:, dm]))
    Abar = (K[d0] * E).sum() / (K[d0] * E / A).sum()
    L1 = O / (nu / (2 * (s - 1)) * Abar)
    aminus = A * max((1 + 2 / nu) ** (-(s - 1)), nu ** (1 - s))    # single-set: V' = 2/nu (one interval, |beta_T| = |A|)
    lhs = (K[d0] * (A - aminus)).sum(); det = (K[d0] * E).sum() / O
    Gs = ER.programme_vertices(p, s)[0] if F <= 8 else G.programme(p, s)[0]
    gIF = nu ** (1 - s) * Bp(p, n) * Gs
    print('(p,n)=(%2d,%d): margin>0:%s  gamma/gamma_IF=%.4f  L1 ratio=%.3f (formula 2(s-1)sqrt(p) O/(nu*2)=%.3f)  (L): lhs=%.3f vs det=%.3f -> %s'
          % (p, n, h > 0, net.gamma() / gIF, L1, O * (s - 1) * math.sqrt(p) / (nu * 2) * 2 / 2, lhs, det, 'holds' if lhs <= det else 'fails'))
