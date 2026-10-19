"""Referee check of D4' (s = 1, b = nu): maximisers of gamma(.;1) found from random Gaussian starts (own optimiser:
softmin L-BFGS + sequential-LP polish, from v3_kkt.py).  For runs reaching gamma/(B_p Gamma_1) ~ 1, check:
beta_j ~ 2h/p for all j, IF (max_T |beta_{j,T}| small), frequency purity, and H(p_theta) >= log(nu F)."""
import numpy as np, sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v3_kkt.py')).read()
src = src[:src.index('for (p, n, b, m, seed) in')]
ns = {'__file__': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v3_kkt.py')}
exec(compile(src, 'v3', 'exec'), ns)
from vlib import *

for (p, n, m, seeds) in [(5, 2, 8, [11, 12, 13, 14]), (3, 2, 5, [21, 22, 23]), (7, 2, 10, [31, 32])]:
    nu = n + 1; F = (p - 1) // 2; b = float(nu)
    for sd in seeds:
        v, g, J, c, gc = ns['find_local_max'](p, n, b, m, np.random.default_rng(sd))
        U, W = ns['unpackv'](v, m, n, p)
        f = table(U, W); h = gaps(f, p, n).min(); a = acost(U, W)
        ratio = h / a.sum() / (Bp(p, n) * p / (p - 1))
        beta = clean(f, p, n)
        fh = np.fft.fftn(f); maxT = 0
        for j in range(1, F + 1):
            for T in itertools.product([0, 1], repeat=n):
                if not any(T): continue
                k = tuple(((-j) if T[t] else j) % p for t in range(n)) + ((-j) % p,)
                maxT = max(maxT, abs(2 * p ** (-nu) * fh[k]))
        keep = a > 1e-6 * a.max(); pp = a[keep] / a[keep].sum()
        print('p=%d n=%d seed=%d: gamma/(B_p Gamma_1)=%.6f  beta/(2h/p)=%s  max|beta_T|/|beta|=%.1e  width>1e-6: %d  '
              'H(p)=%.4f vs log(nu F)=%.4f' % (p, n, sd, ratio, np.round(beta.real / (2 * h / p), 4),
                                                maxT / abs(beta).max(), keep.sum(), shannon(pp), np.log(nu * F)), flush=True)
