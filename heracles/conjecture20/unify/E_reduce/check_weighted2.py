# N4b: degenerate points (several programme maximisers): does any optimum found straddle (nuE a strict mixture)?
import time
import numpy as np
from elib import Net, Bp, programme_vertices, zshares, Hent, optimise_w, wgamma, Evec, entropy, if_network, balance, prune

def run(p, n, b, lam, m, starts, seed, ifstarts=True):
    rng = np.random.default_rng(seed); nu = n + 1; s = nu / b; F = (p - 1) // 2
    Gm, maxs, _ = programme_vertices(p, s, lam)
    D = nu ** (1 - s) * Bp(p, n) * Gm
    zs = [zshares(y, s) for y in maxs]
    res = []
    for k in range(starts):
        if k % 2 == 0 or not ifstarts:
            U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
        else:
            y = rng.uniform(0.2, 1, F); net0 = if_network(p, n, y)
            U = np.zeros((m, n, p)); W = np.zeros((m, p)); mm = min(m, net0.m)
            U[:mm] = net0.U[:mm]; W[:mm] = net0.W[:mm]
            U += 0.3 * rng.normal(size=U.shape); W += 0.3 * rng.normal(size=W.shape)
        net = prune(balance(optimise_w(U, W, b, lam)), 1e-8)
        g = wgamma(net, lam) / D
        E = nu * Evec(net); H = entropy(net)
        d = [np.abs(E - z).max() for z in zs]
        feats = tuple(int(j) + 1 for j in np.nonzero(np.abs(net.clean()) > 1e-6 * np.abs(net.clean()).max())[0])
        res.append((g, min(d), np.round(E, 4), H, feats))
    res.sort(key=lambda r: -r[0])
    print(f"p={p} n={n} b={b} lam={np.round(lam,3)}: {len(maxs)} programme maximisers, supports "
          f"{[tuple(int(j)+1 for j in np.nonzero(y>1e-9)[0]) for y in maxs]}", flush=True)
    for g, d, E, H, feats in res[:4]:
        print(f"    gamma/D={g:.6f}  min_y|nuE-z(y)|={d:.1e}  nuE={E}  H={H:.4f}  features={feats}", flush=True)

t0 = time.time()
run(11, 2, 2.0, np.ones(5), m=8, starts=10, seed=1)
run(5, 3, 2.1, np.ones(2), m=6, starts=8, seed=2)
run(5, 3, 2.1, np.array([1.0, 1.05]), m=6, starts=6, seed=3)
run(7, 2, 2.0, np.ones(3), m=11, starts=6, seed=4)
print(f"time {time.time()-t0:.0f}s")
