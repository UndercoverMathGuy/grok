# Which half-offset line densities are blind to X?  For D uniform on Z_p^*, row x1 = s + sgn*D/2 (x2 uniform), s fixed:
# compare the average gap of a mixing net with that of its diagonal part (same clean + same-|k| junk, X removed).
import sys, numpy as np
sys.path.insert(0, '..')
from common import *
rng = np.random.default_rng(3)
def diag_part(p, net):
    out = []
    for e in net:
        F = [np.fft.fft(v) for v in e]
        for k in range(1, (p - 1) // 2 + 1):
            comp = []
            for Fv in F:
                G = np.zeros(p, complex); G[k] = Fv[k]; G[p - k] = Fv[p - k]
                comp.append(np.real(np.fft.ifft(G)))
            out.append(tuple(comp))
    return out
for p in (5, 7):
    i2 = pow(2, -1, p)
    for t in range(3):
        net = random_net(p, 6, rng, pure=False)
        g, gd = gaps(table(p, net)), gaps(table(p, diag_part(p, net)))
        for sgn in (+1, -1):
            err = 0
            for s in range(p):
                vals = [np.mean([g[(s + sgn * D * i2) % p, :, D].mean() for D in range(1, p)]),
                        np.mean([gd[(s + sgn * D * i2) % p, :, D].mean() for D in range(1, p)])]
                err = max(err, abs(vals[0] - vals[1]))
            print(f"p={p} net {t} line x1 = s {'+' if sgn > 0 else '-'} D/2: |full - diagonal| = {err:.2e}")
