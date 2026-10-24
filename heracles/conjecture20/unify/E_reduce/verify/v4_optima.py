# Independent sanity check of Theorem B / Corollary U / (FE) at numerically found optima (n = 2, pi = 3^{1-s}).
# Optimiser: refereed GLOBAL library (not E_reduce code); all observables recomputed with vlib.
import numpy as np, sys, time
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/E_reduce/verify')
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/global')
from vlib import *
import lib as GL
rng = np.random.default_rng(2024)
t0 = time.time()
for (p, n, b, m, starts) in [(7, 2, 2.5, 9, 8), (7, 2, 2.0, 9, 6), (5, 3, 2.6, 8, 5)]:
    nu = n + 1; s = nu / b; F = (p - 1) // 2
    G, M = programme(p, s)
    D = nu ** (1 - s) * Bp(p, n) * G   # n = 2: pi = 3^{1-s}; n = 3: only a lower bound on D
    for st in range(starts):
        U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
        net = GL.optimise(U, W, b, iters=2500)
        net = GL.prune(GL.balance(net))
        U, W = net.U, net.W
        f = table(U, W); h = gaps(f, p, n).min(); gam = h / cost(U, W, b) ** s
        E = Evec(U, W, b); be = np.abs(beta(f, p, n)); feats = tuple(np.nonzero(be > 1e-4 * be.max())[0] + 1)
        dist = min(np.abs(nu * E - zsh(y, s)).max() for y in M)
        Hn = H(shares(U, W, b)); FE = Hn - (np.log(nu) + H(nu * E / (nu * E).sum()))
        print('p=%d n=%d b=%.2f start %d: gamma/(nu^{1-s}B_p Gamma)=%.6f  sum nuE=%.6f  min_y|nuE-z(y)|=%.1e  feats=%s  '
              'optimal supports=%s  FE slack(with H1min=log nu)=%+.1e  [%.0fs]' % (
              p, n, b, st, gam / D, (nu * E).sum(), dist, feats, [tuple(np.nonzero(y)[0] + 1) for y in M], FE, time.time() - t0), flush=True)
