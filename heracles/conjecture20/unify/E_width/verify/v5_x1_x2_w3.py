import numpy as np
from scipy.integrate import quad
from vlib import *
n = 2; nu = 3
def zH(y, s):
    z = y[y > 0] ** (1 / s); z /= z.sum(); return float(-(z * np.log(z)).sum())
print("X1: uniform IF network on F, n=2 (pi = 3^{1-s}, H(theta_1) = log 3)")
for p in [11, 13]:
    F = (p - 1) // 2
    for s in [1.2, 1.35, 1.45, 1.49]:
        G, ys = programme(p, s)
        Hz = min(zH(y, s) for y in ys)
        # build the network, compute gamma directly
        Us, Ws = [], []
        for j in range(1, F + 1):
            lam = (2 ** n * F ** (-s) / nu) ** (1 / nu)
            U0, W0 = if_block(p, n, j, lam); Us.append(U0); Ws.append(W0)
        U = np.concatenate(Us); W = np.concatenate(Ws)
        h = margin(U, W); sq = sqnorm(U, W); b = nu / s
        gam = h / np.sum(sq ** (b / 2)) ** s
        w = sq ** (b / 2); w /= w.sum(); Hth = float(-(w * np.log(w)).sum())
        D = 3 ** (1 - s) * Bp(p, n) * G
        val = np.log(gam) - (np.log(D) + (s - 1) * (np.log(3) + Hz - Hth))
        closed = np.log(F ** (-s) * p / 2 / G) - (s - 1) * (Hz - np.log(F))
        print(f"  p={p} s={s}: #y*={len(ys)} H(z*)={Hz:.4f}; gamma/D={gam/D:.5f}; log gamma - RHS = {val:+.5f} (closed form {closed:+.5f})")
print("X2: single IF block on feature 1 at (5,2), S=F")
p = 5
for s in [1.42, 1.43, 1.45, 1.49, 1.5]:
    G, ys = programme(p, s); Hz = min(zH(y, s) for y in ys)
    U, W = if_block(p, n, 1); h = margin(U, W); sq = sqnorm(U, W); b = nu / s
    gam = h / np.sum(sq ** (b / 2)) ** s
    D = 3 ** (1 - s) * Bp(p, n) * G
    val = np.log(gam) - (np.log(D) - (s - 1) * (np.log(3) + Hz - np.log(3)))
    print(f"  s={s}: H(z*)={Hz:.4f}; log gamma - RHS = {val:+.5f}")

# W2/W3: Dini bound integrated at n=2, D_S = 3^{1-s} B_p Gamma^S_s, H^dec = log3 + min H(z*): slack must be >= 0 (here = 0)
print("W3 slack (theta-independent): log D_S(s) - log D_S(1) + int_1^s H^dec")
for p, S in [(11, None), (13, None), (11, [1, 3, 4]), (7, [2, 3])]:
    for s in [1.3, 1.5]:
        Hd = lambda sg: np.log(3) + min(zH(y, sg) for y in programme(p, sg, S)[1])
        I = quad(Hd, 1, s, limit=200)[0]
        D = lambda sg: 3 ** (1 - sg) * Bp(p, n) * programme(p, sg, S)[0]
        print(f"  p={p} S={S} s={s}: slack = {np.log(D(s)) - np.log(D(1)) + I:+.2e}")
