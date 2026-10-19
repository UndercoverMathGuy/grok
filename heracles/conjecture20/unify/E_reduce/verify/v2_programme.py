# Referee checks on the programme side: s=1 uniqueness (needed for b = nu in Theorem T), dilation orbits (R10),
# Theorem G(b) semiconvexity of v(t)=log Gamma^{e^t}_s and a.e. uniqueness, D-side Danskin in t.
import numpy as np, sys, itertools
from scipy.optimize import minimize
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/E_reduce/verify')
from vlib import *
rng = np.random.default_rng(3)

print('(1) s = 1 (b = nu): programme maximisers')
for p in [5, 7, 11, 13]:
    G, M = programme(p, 1.0)
    F = (p - 1) // 2
    print('  p=%2d Gamma_1=%.10f (p/(p-1)=%.10f), #max=%d, max|y-1/F|=%.1e' % (p, G, p / (p - 1), len(M), np.abs(M[0] - 1 / F).max()))

print('(2) dilation invariance / orbits')
for p, s in [(5, 1.9), (7, 1.8), (11, 1.5), (13, 1.5), (11, 1.2), (13, 1.3)]:
    F = (p - 1) // 2; G, M = programme(p, s)
    sup = [tuple(np.nonzero(y > 1e-12)[0] + 1) for y in M]
    union = sorted(set().union(*[set(t) for t in sup]))
    Hz = [H(zsh(y, s)) for y in M]
    # dilation action on a maximiser
    y = M[0]; ok = True
    for k in range(2, p):
        yk = np.zeros(F)
        for j in range(1, F + 1):
            kj = (k * j) % p; kj = min(kj, p - kj); yk[kj - 1] = y[j - 1]
        ok &= abs(m_lam(yk, p) - G) < 1e-9
    zbar = np.mean([zsh(y, s) for y in M], 0)
    print('  p=%2d s=%.2f Gamma=%.6f #max=%d supports=%s union=%s H(z) range=[%.6f,%.6f] dilates optimal=%s spt zbar=%d/%d' % (
        p, s, G, len(M), sup, union, min(Hz), max(Hz), ok, (zbar > 0).sum(), F))

print('(3) Theorem G(b): v(t) + |t|^2/(2s) convex; check second differences along random lines near t=0')
for p, s in [(11, 1.5), (13, 1.5), (7, 1.8), (5, 1.9)]:
    F = (p - 1) // 2
    def v(t): return np.log(programme(p, s, lam=np.exp(t))[0])
    worst = np.inf
    for k in range(25):
        e = rng.normal(size=F); e /= np.linalg.norm(e); t0 = rng.normal(size=F) * 0.02 * (k % 2)
        hh = 0.01
        d2 = (v(t0 + hh * e) - 2 * v(t0) + v(t0 - hh * e)) / hh ** 2 + 1 / s
        worst = min(worst, d2)
    # uniqueness at random lambda
    cnt = [len(programme(p, s, lam=np.exp(rng.normal(size=F) * 0.02))[1]) for _ in range(40)]
    print('  p=%2d s=%.2f: min second difference of v+|t|^2/(2s) = %.4f (>= 0 up to O(h)); #maximisers at 40 random lam: max %d' % (p, s, worst, max(cnt)))

print('(4) D-side Danskin in t at lam = 1: one-sided derivative of log Gamma^{e^t} vs max_y <z(y), eta>')
for p, s in [(11, 1.5), (7, 1.8)]:
    F = (p - 1) // 2; G, M = programme(p, s)
    for k in range(4):
        e = rng.normal(size=F)
        hh = 1e-6
        fd = (np.log(programme(p, s, lam=np.exp(hh * e))[0]) - np.log(G)) / hh
        dk = max(zsh(y, s) @ e for y in M)
        print('  p=%2d s=%.2f eta#%d: FD %.6f   Danskin %.6f' % (p, s, k, fd, dk))
