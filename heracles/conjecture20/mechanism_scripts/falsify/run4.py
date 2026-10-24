# Task 4: p=5, large n, single-feature networks (outputs at frequency 1) with NON-slot-symmetric inputs:
# slots split into groups (sizes given), each group its own input vector per neuron; seeds with frequency-2 content.
import numpy as np, sys, time, math, torch
from famp_g import GNet, optimise, if_block
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 1)
p = 5
sizes = tuple(int(x) for x in sys.argv[1].split(','))
widths = [int(x) for x in sys.argv[2].split(',')]
quick = sys.argv[3] == 'quick'
outf1 = True
n = sum(sizes); g = len(sizes); nu = n + 1
net = GNet(p, sizes, outfreq1=outf1)
stages = (0.1, 0.03, 0.01, 0.003, 0.001, 3e-4) if quick else (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4, 3e-5)
iters = 150 if quick else 300
rng = np.random.default_rng(n * 100 + g)
a = np.arange(p); e = np.where(a < p / 2, 2 * a, 2 * a - p); eps_s = np.where(a < p / 2, 1.0, -1.0)
print(f'p={p} sizes={sizes} classes={len(net.y)}', flush=True)
U0, W0, l0 = if_block(p, n, g)
print(f'IF block ratio {net.ratio(U0, W0, l0):.6f}', flush=True)
t0 = time.time(); best = (-9, None)
for m in widths:
    seeds = []
    # IF-based seeds (need m >= nu for exact IF; extra neurons small)
    def ifm(m):
        U, W, la = if_block(p, n, g)
        if m < nu: return U[:m].copy(), W[:m].copy(), la[:m].copy()
        Ue = rng.standard_normal((m - nu, g, p)); We = rng.standard_normal((m - nu, 2)); le = np.full(m - nu, -4.0)
        return np.concatenate([U, Ue]), np.concatenate([W, We]), np.concatenate([la, le])
    for eps in ([0.15, 0.4] if quick else [0.02, 0.06]):
        U, W, la = ifm(m)
        k = int(rng.integers(g))
        U[:, k, :] += eps * np.cos(4 * np.pi * a / p + rng.uniform(0, 2 * np.pi, (len(U), 1)))   # frequency-2 in one group
        seeds.append((f'IF+f2(group{k},{eps})', U, W, la))
        U, W, la = ifm(m)
        U += eps * rng.standard_normal(U.shape)                                                   # non-symmetric noise
        seeds.append((f'IF+noise{eps}', U, W, la))
        U, W, la = ifm(m)
        U += eps * np.cos(4 * np.pi * a / p + rng.uniform(0, 2 * np.pi, (len(U), g, 1)))           # freq-2 everywhere, random phases
        U += 0.3 * eps * rng.standard_normal((len(U), g, 1))                                      # plus DC
        seeds.append((f'IF+f2all+DC({eps})', U, W, la))
    # shared-sign sign vectors with group-dependent slopes (1-D family generalised)
    for t in range(2):
        lam = rng.normal(0, 0.4, (m, g)); U = eps_s[None, None, :] * np.exp(lam[:, :, None] * e[None, None, :])
        ph = rng.uniform(0, 2 * np.pi, m); W = np.stack([np.cos(ph), np.sin(ph)], 1)
        seeds.append((f'signvec{t}', U, W, np.zeros(m)))
    # random
    for t in range(1):
        seeds.append((f'rand{t}', rng.standard_normal((m, g, p)), rng.standard_normal((m, 2)), np.zeros(m)))
    for name, U, W, la in seeds:
        r0 = net.ratio(U, W, la)
        r, U2, W2, la2 = optimise(net, U, W, la, stages=stages, iters=iters)
        print(f'  m={m:2d} {name:22s} start {r0:+.4f} -> {r:+.6f}  [{time.time() - t0:.0f}s]', flush=True)
        if r > best[0]: best = (r, (m, name, U2, W2, la2))
r, (m, name, U, W, la) = best
print(f'== p={p} n={n} sizes={sizes}: BEST efficiency {r:.6f} (m={m}, seed {name})', flush=True)
np.save(f'res/run4_{"_".join(map(str, sizes))}.npy', dict(r=r, m=m, name=name, U=U, W=W, la=la), allow_pickle=True)
