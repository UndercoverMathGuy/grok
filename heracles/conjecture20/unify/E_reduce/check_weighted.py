# N4/N5: weighted value identity W^lam = pi B_p Gamma^lam (hypothesis H1, sanity only) by local search from many starts,
#        and first-order rigidity at the optimum found: nu*E(theta*) = z(y^lam), H = H^(1) + H(z), (FE) margin.
import sys, time
import numpy as np
from elib import Net, Bp, programme_vertices, zshares, Hent, optimise_w, wgamma, Evec, entropy, if_network, balance, prune

def run(p, n, b, lam, m, starts, seed):
    rng = np.random.default_rng(seed); nu = n + 1; s = nu / b; F = (p - 1) // 2
    Gm, maxs, _ = programme_vertices(p, s, lam)
    D = nu ** (1 - s) * Bp(p, n) * Gm          # pi = nu^{1-s} assumed (proved at n=2)
    best = None
    for k in range(starts):
        if k % 2 == 0:
            U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
        else:
            y = rng.uniform(0.2, 1, F); net0 = if_network(p, n, y)
            mm = net0.m; U = np.zeros((m, n, p)); W = np.zeros((m, p))
            U[:min(m, mm)] = net0.U[:m]; W[:min(m, mm)] = net0.W[:m]
            U += 0.3 * rng.normal(size=U.shape); W += 0.3 * rng.normal(size=W.shape)
        net = optimise_w(U, W, b, lam)
        g = wgamma(net, lam)
        if best is None or g > best[0]: best = (g, net)
    g, net = best
    net = prune(balance(net), 1e-8)
    E = Evec(net); H = entropy(net)
    zs = [zshares(y, s) for y in maxs]
    dz = min(np.abs(nu * E - z).max() for z in zs)
    Hdec = np.log(nu) + min(Hent(z) for z in zs)
    FE = H - (np.log(nu) + Hent(nu * E))
    print(f"p={p} n={n} b={b} lam={np.round(lam,3)}: best gamma_lam/D^lam = {g/D:.6f}; "
          f"#prog max={len(maxs)}; |nuE - z(y)|_inf={dz:.1e}; sum nuE={nu*E.sum():.6f}; "
          f"H={H:.5f} vs H^dec={Hdec:.5f}; FE slack H-(log nu + H(nuE))={FE:+.1e}", flush=True)
    return g / D

t0 = time.time()
cases = [(5, 2, 2.5, [1.0, 1.2]), (5, 2, 2.5, [1.2, 1.0]), (5, 2, 2.2, [1.0, 0.85]), (5, 2, 2.0, [1.0, 1.1]),
         (5, 2, 3.0, [1.0, 1.3]), (7, 2, 2.5, [1.0, 1.15, 0.9]), (7, 2, 2.2, [0.9, 1.0, 1.1])]
for (p, n, b, lam) in cases:
    F = (p - 1) // 2
    run(p, n, b, np.array(lam), m=3 * F + 2, starts=8, seed=p * 100 + int(10 * b))
print(f"time {time.time()-t0:.0f}s")
