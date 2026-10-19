# Refutation of the s-local width inequality (both signs) by interference-free networks, at n = 2 where
# pi_{2,b} = 3^{1-s} (Note Thm q1n2) and every optimal single feature has H = log 3 (GLOBAL R2 + differentiability),
# so D_S = 3^{1-s} B_p Gamma^{(S)}_s and H^dec_S = log 3 + min H(z*).
#  (+) claimed: log gamma <= log D_S + (s-1)[H^dec_S - H(theta,s)] - Delta, Delta >= 0     (Delta = 0 forced for IF nets)
#  (-) claimed: log gamma <= log D_S - (s-1)[H^dec_S - H(theta,s)]
# Also: programme temperature identity log Gamma_s = log Gamma_1 - int_1^s H(z*(sigma)) dsigma (used in the integrated form).
import numpy as np
from elib import *
from scipy.integrate import quad

n = 2; nu = 3


def Hdec(p, s, S=None):
    G, ys = programme_vertices(p, s, S)
    return np.log(3) + min(shannon(zshares(y, s)) for y in ys), G


print("(+) form, uniform IF network on all of F (interference-free, so any detection term is 0):")
for p in [11, 13]:
    F = (p - 1) // 2
    for s in [1.2, 1.35, 1.45, 1.49]:
        yu = np.full(F, 1.0) / F ** s                       # uniform allocation, sum y^{1/s} = 1
        U, W = if_network(p, n, yu); net = Net(U, W)
        lg = np.log(net.gamma(s)); H = shannon(net.mass() ** (1 / s))
        hd, G = Hdec(p, s)
        rhs = np.log(3 ** (1 - s) * Bp(p, n) * G) + (s - 1) * (hd - H)
        print(f"  p={p} s={s}: H(theta)={H:.4f} H^dec={hd:.4f}  log gamma - RHS = {lg - rhs:+.5f}  ({'VIOLATED' if lg > rhs + 1e-12 else 'holds'})")

print("(-) form, single-feature IF block at (5,2):")
p = 5
for s in [1.3, 1.42, 1.43, 1.45, 1.49]:
    U, W = if_block(p, n, 1); net = Net(U, W)
    lg = np.log(net.gamma(s)); H = shannon(net.mass() ** (1 / s))
    hd, G = Hdec(p, s)
    rhs = np.log(3 ** (1 - s) * Bp(p, n) * G) - (s - 1) * (hd - H)
    print(f"  s={s}: H(theta)={H:.4f} < H^dec={hd:.4f}; gamma/D_S = {np.exp(lg) / (3 ** (1 - s) * Bp(p, n) * G):.4f}; log gamma - RHS = {lg - rhs:+.5f} ({'VIOLATED' if lg > rhs + 1e-12 else 'holds'})")
print(f"  threshold s = log2(5/kappa_min)/2 = {np.log2(5 / (1 - np.cos(2 * np.pi / 5))) / 2:.5f}")

print("programme temperature identity at p = 11, 13 (across transitions):")
for p in [11, 13]:
    def Hz(sig):
        G, ys = programme_vertices(p, sig); return min(shannon(zshares(y, sig)) for y in ys)
    for s in [1.3, 1.5]:
        I = quad(Hz, 1, s, limit=200, points=[1.2850, 1.3454, 1.4067, 1.4895])[0]
        print(f"  p={p} s={s}: log Gamma_s - log Gamma_1 = {np.log(programme_vertices(p, s)[0] / programme_vertices(p, 1)[0]):.6f};  -int H(z*) = {-I:.6f}")
