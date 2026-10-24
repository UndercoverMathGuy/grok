# N1/N7: Danskin for the (weighted) programme; decoupled slope; multiplicity of programme maximisers.
import numpy as np
from elib import programme_vertices, zshares, Hent, Bp

def lG(p, s, lam=None, S=None):
    return np.log(programme_vertices(p, s, lam, S)[0])

print("(1) p=5 transition: one-sided slopes of log Gamma_s at s_c")
sc = 1.85520596
for h in [1e-4, 1e-5]:
    R = (lG(5, sc + h) - lG(5, sc)) / h; L = (lG(5, sc) - lG(5, sc - h)) / h
    print(f"  h={h:.0e}: right {R:+.6f} (Danskin -min H(z*) = 0), left {L:+.6f} (Danskin -max H(z*) = {-np.log(2):+.6f})")

print("(2) slope of log D_S at n=2 (pi = 3^{1-s}, every optimal single feature has H = log 3):")
print("    right derivative of log D_S  vs  -H^dec_S = -(log 3 + min H(z*))")
for p, S in [(7, (1, 2)), (7, None), (11, None), (11, (1, 2, 3)), (13, None)]:
    for s in [1.1, 1.3, 1.45]:
        h = 1e-6
        Dl = lambda t: (1 - t) * np.log(3) + np.log(Bp(p, 2)) + lG(p, t, None, S)
        R = (Dl(s + h) - Dl(s)) / h
        Gm, maxs, _ = programme_vertices(p, s, None, S)
        Hd = np.log(3) + min(Hent(zshares(y, s)) for y in maxs)
        print(f"  p={p:2d} S={str(S):9s} s={s}: slope {R:+.6f}  -H^dec {-Hd:+.6f}  diff {R + Hd:+.1e}")

print("(3) multiplicity of programme maximisers (dilation orbits):")
for p, s in [(11, 1.5), (11, 1.2), (13, 1.5), (7, 1.8), (5, 1.9)]:
    Gm, maxs, _ = programme_vertices(p, s)
    sup = [tuple(int(j) + 1 for j in np.nonzero(y > 1e-9)[0]) for y in maxs]
    print(f"  p={p} s={s}: Gamma={Gm:.6f}, {len(maxs)} maximisers, supports {sup}")

print("(4) D-side Danskin in the tax direction (lambda_l = mu_l^{-1/2}) at p=11, s=1.5:")
p, s = 11, 1.5; F = 5
Gm, maxs, _ = programme_vertices(p, s)
Z = [zshares(y, s) for y in maxs]
for j in range(F):
    for sgn in (+1, -1):
        t = 1e-6; mu = np.ones(F); mu[j] += sgn * t
        d = (lG(p, s, mu ** -0.5) - np.log(Gm)) / t
        pred = max(-sgn * z[j] / 2 for z in Z)
        print(f"  j={j+1} dir={'tax' if sgn>0 else 'sub'}: FD {d:+.6f}  max_y <grad,eta> {pred:+.6f}")
