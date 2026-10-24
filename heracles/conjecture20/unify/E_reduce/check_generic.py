# Generic feature weights break the dilation symmetry: the weighted programme has a unique maximiser for a.e. lambda,
# and (Theorem G) every maximiser of the lambda-weighted margin then has nu*E = z(y^lam) and features = spt y^lam.
import numpy as np
from elib import Bp, programme_vertices, zshares, optimise_w, wgamma, Evec, entropy, balance, prune
rng = np.random.default_rng(11)
cnt = {}
for p, s in [(11, 1.5), (13, 1.5), (7, 1.8), (5, 1.9), (11, 1.2)]:
    F = (p - 1) // 2; mult = []
    for t in range(200):
        lam = 1 + 0.02 * rng.normal(size=F)
        mult.append(len(programme_vertices(p, s, lam)[1]))
    print(f"p={p} s={s}: #maximisers at lambda=1: {len(programme_vertices(p, s)[1])};  at 200 random lambda near 1: max {max(mult)}, mean {np.mean(mult):.3f}")
for (p, n, b) in [(11, 2, 2.0), (7, 3, 2.0)]:
    F = (p - 1) // 2; nu = n + 1; s = nu / b
    lam = 1 + 0.03 * rng.normal(size=F)
    Gm, maxs, _ = programme_vertices(p, s, lam); y = maxs[0]
    D = nu ** (1 - s) * Bp(p, n) * Gm
    best = None
    for k in range(6):
        U = rng.normal(size=(8, n, p)); W = rng.normal(size=(8, p))
        net = prune(balance(optimise_w(U, W, b, lam)), 1e-8); g = wgamma(net, lam) / D
        if best is None or g > best[0]: best = (g, net)
    g, net = best
    feats = tuple(int(j) + 1 for j in np.nonzero(np.abs(net.clean()) > 1e-6 * np.abs(net.clean()).max())[0])
    print(f"p={p} n={n} b={b} lam={np.round(lam,3)}: unique y^lam support {tuple(int(j)+1 for j in np.nonzero(y>1e-9)[0])}; "
          f"best gamma/D={g:.6f}; |nuE - z(y^lam)|_inf={np.abs(nu*Evec(net)-zshares(y,s)).max():.1e}; features={feats}")
