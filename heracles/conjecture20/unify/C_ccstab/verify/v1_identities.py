# Referee check of S1 (slack form, energy identity) and S2 (a),(b),(c) with an independent implementation.
import numpy as np, math
from vlib import *
rng = np.random.default_rng(2024)
W = dict(id=0, dint=0, dprog=0, en=0, S2a=np.inf, S2b=0, S2c=np.inf, S2cfinal=np.inf, var_classC=0)
def proj(p, l):
    x = np.arange(p); c = np.cos(2*np.pi*l*x/p); s_ = np.sin(2*np.pi*l*x/p)
    Bm = np.stack([c/np.linalg.norm(c), s_/np.linalg.norm(s_)], 1); return Bm @ Bm.T
cnt = 0
for (p, n, b) in [(3, 2, 2.0), (3, 4, 2.0), (5, 2, 2.0), (5, 3, 2.5), (7, 2, 3.0), (5, 4, 2.0), (11, 2, 2.0), (13, 2, 2.2), (7, 3, 2.0)]:
    nu = n + 1; F = (p - 1) // 2; B = Bp(p, n); G1 = p / (p - 1); K = kappa(p); C = cvec(p)
    for t in range(10):
        m = rng.integers(2, 9)
        U = rng.normal(size=(m, n, p)); Wt = rng.normal(size=(m, p))
        cc = t % 2 == 1
        if cc:
            for i in range(m):
                P = proj(p, rng.integers(1, F + 1)); U[i] = U[i] @ P; Wt[i] = Wt[i] @ P
        if t % 5 == 4:   # add strong DC to inputs (non class C)
            U += 2.0 * rng.normal(size=(m, n, 1)); cc = False
        net = Net(U, Wt, b); g = net.gaps(); h = g.min(); a = net.masses(); msh = a / a.sum()
        at = net.atoms(); beta = at.sum(0)
        sig_n = 1 - np.real(at).sum(1) / (B * a)
        N = B * G1 * a.sum()
        cleanmin = (K @ np.real(beta)).min()
        dprog = (G1 * np.real(beta).sum() - cleanmin) / N; dint = (cleanmin - h) / N
        sl = g - h; sbar = sl.mean(0); ssym = 0.5 * (sbar + sbar[::-1])
        W['dint'] = max(W['dint'], abs(dint - ssym.min() / N))
        W['dprog'] = max(W['dprog'], abs(dprog - (sbar.mean() - ssym.min()) / N))
        lhs = 1 - (h / a.sum()) / (B * G1); rhs = (msh * sig_n).sum() + sl.mean() / N
        W['id'] = max(W['id'], abs(lhs - rhs))
        # energy identity
        Phi = net.fields(); f = Phi - Phi.mean(0); E = (np.abs(f) ** 2).mean(0)
        # check E_x Phi_l = conj(beta_l)
        assert np.allclose(Phi.mean(0), np.conj(beta), atol=1e-10 * np.abs(beta).max() + 1e-14)
        tau = sl - sbar
        rhsE = (2 / p ** 2) * (p * (tau ** 2).sum(1) - tau.sum(1) ** 2).mean()
        W['en'] = max(W['en'], abs(E.sum() - rhsE) / max(E.sum(), 1e-300))
        # S2(a)
        smax = sl.max(); Dp = dprog + dint
        W['S2a'] = min(W['S2a'], (2 * (1 - 1 / p) * smax * N * Dp - E.sum()) / N ** 2)
        # S2(b)
        Fm = g - g.mean(0)
        lhsb = (Fm ** 2).mean(); rhsb = G1 / 2 * (E.sum() + 2 * (np.real(f.sum(1)) ** 2).mean())
        W['S2b'] = max(W['S2b'], abs(lhsb - rhsb) / lhsb)
        # direct definition check: F_delta = Re sum f_l c_l(delta)
        assert np.allclose(Fm, np.real(f @ C.T), atol=1e-9 * np.abs(Fm).max())
        if cc:
            W['S2c'] = min(W['S2c'], (2 ** (n + 2) * N / G1 - smax) / N)
            W['S2cfinal'] = min(W['S2cfinal'], Dp - G1 * E.sum() / (2 ** (n + 3) * N ** 2))
            W['var_classC'] = max(W['var_classC'], np.max(np.abs((Fm ** 2).mean(0) - K @ E)) / max((K @ E).max(), 1e-300))
        cnt += 1
print('networks tested:', cnt)
for k, v in W.items(): print('  %-12s %.3e' % (k, v))
