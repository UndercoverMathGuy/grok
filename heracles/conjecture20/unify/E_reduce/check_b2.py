# N6: the endpoint b = 2.  (a) E(theta) = (P_split/P) E(theta_split) at b = 2 (exactly E(split) when DC-free), and
#     H(theta) <= H(theta_split) when DC-free;  (b) can shared neurons give a maximiser with fewer neurons than nu*K*
#     (i.e. entropy below H^dec, so that (E) fails at s = nu/2)?  Narrow-width searches at (5,2,2), (7,2,2), (5,3,2).
import time
import numpy as np
from elib import Net, Bp, programme_vertices, zshares, Hent, optimise_w, wgamma, Evec, entropy, split, freq_proj, neuron_sq, balance, prune
rng = np.random.default_rng(7)
mx = 0; mh = -1e9
for p, n in [(5, 2), (7, 2), (7, 3), (11, 2)]:
    F = (p - 1) // 2
    for t in range(20):
        m = 4; U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
        P0 = freq_proj(p, 0); Q = np.eye(p) - P0
        if t % 2 == 0: U = U @ Q; W = W @ Q       # DC-free
        net = Net(U, W, 2.0); sp = split(net)
        Psplit, P = sp.cost(), net.cost()
        mx = max(mx, np.abs(Evec(net) - Psplit / P * Evec(sp)).max())
        if t % 2 == 0: mh = max(mh, entropy(net) - entropy(sp))
print(f"(a) max |E(theta) - (P_split/P) E(split)| = {mx:.1e};  DC-free: max H(theta)-H(split) = {mh:+.3f} (<= 0)")

t0 = time.time()
for p, n, widths in [(5, 2, [3, 4, 5, 6]), (7, 2, [3, 5, 7, 9]), (5, 3, [2, 3, 4])]:
    nu = n + 1; b = 2.0; s = nu / b; F = (p - 1) // 2
    Gm, maxs, _ = programme_vertices(p, s)
    D = nu ** (1 - s) * Bp(p, n) * Gm
    Hdec = np.log(nu) + min(Hent(zshares(y, s)) for y in maxs)
    for m in widths:
        best = -1e9; bestnet = None
        for k in range(6):
            U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
            net = optimise_w(U, W, b, np.ones(F))
            g = net.gamma() / D
            if g > best: best, bestnet = g, net
        bn = prune(balance(bestnet), 1e-8)
        print(f"  (b) p={p} n={n} b=2 width {m}: best gamma/D = {best:.6f}, H = {entropy(bn):.4f} (H^dec = {Hdec:.4f}, e^Hdec = {np.exp(Hdec):.2f})", flush=True)
print(f"time {time.time()-t0:.0f}s")
