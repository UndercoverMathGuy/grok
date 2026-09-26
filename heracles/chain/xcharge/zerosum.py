# Zero-sum identity: for any network and any density rho on the slot-1 start point s, the X part of the flip-paired
# gap at offset class +-D (D at row s + D/2, -D at row s - D/2, x2 uniform) sums to ZERO over the offset classes.
# X part = (full net) - (its diagonal part).  Also: the same sum for the PLAIN pairing (same row) is NOT zero.
import sys, numpy as np
sys.path.insert(0, '..')
from common import *
from xblind_dirs import diag_part
rng = np.random.default_rng(11)
for p in (5, 7, 11):
    i2 = pow(2, -1, p); h = (p - 1) // 2
    for t in range(3):
        net = random_net(p, 6, rng, pure=False)
        g, gd = gaps(table(p, net)), gaps(table(p, diag_part(p, net)))
        X = g - gd
        rho = rng.dirichlet(np.ones(p))
        flip, plain = [], []
        for D in range(1, h + 1):
            f = sum(rho[s] * 0.5 * (X[(s + D * i2) % p, :, D].mean() + X[(s - D * i2) % p, :, p - D].mean()) for s in range(p))
            q = sum(rho[s] * 0.5 * (X[s, :, D].mean() + X[s, :, p - D].mean()) for s in range(p))
            flip.append(f); plain.append(q)
        print(f"p={p} net {t}: flip-paired X leaks {np.round(flip, 5)} sum {sum(flip):+.1e} | plain {np.round(plain, 5)} sum {sum(plain):+.1e}")
