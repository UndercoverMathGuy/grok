# Averaged and one-sided directional prices of a single feature (outputs at frequency 1, inputs arbitrary).
#   mode 'avg': maximise (min_x G(x,l) + min_x G(x,-l)) / (2 kappa(l) B_p P^s)
#   mode 'one': maximise  min_x G(x,l) / (kappa(l) B_p P^s)
# reported relative to the IF value nu^{1-s} (IF block = 1 in every direction).
# Seeds: IF block + noise, random, and 'partner' seeds: two aligned atoms (cheap, P12-extremal) plus frequency-3 input content.
import sys, numpy as np
from lib import *

def obj(x, m, n, p, b, tau, l, mode, mask):
    U, W = unpack(x, m, n, p); W = W @ mask
    net = Net(U, W, b); c = net.cost(); s = net.s
    ka = 1 - np.cos(2 * np.pi * l / p)
    g = net.gaps() * c ** (-s) / ka
    groups = [[l % p], [(-l) % p]] if mode == 'avg' else [[l % p]]
    F = 0.0; lam = np.zeros((net.N, p - 1)); Eg = 0.0
    for grp in groups:
        G = g[:, [d - 1 for d in grp]]
        gmin = G.min(); z = np.exp(-(G - gmin) / tau); Z = z.sum(); ls = z / Z
        F += (gmin - tau * np.log(Z)) / len(groups)
        lam[:, [d - 1 for d in grp]] += ls / len(groups) / ka
        Eg += (ls * G).sum() / len(groups)
    gU, gW = net.grad_lin(lam); cU, cW = net.grad_cost()
    dU = c ** (-s) * gU - s * Eg / c * cU
    dW = (c ** (-s) * gW - s * Eg / c * cW) @ mask.T
    return F, pack(dU, dW)

def value(net, l, mode):
    p = net.p; ka = 1 - np.cos(2 * np.pi * l / p); g = net.gaps() / ka / net.cost() ** net.s
    ref = net.nu ** (1 - net.s) * Bp(p, net.n)
    if mode == 'avg': return 0.5 * (g[:, l % p - 1].min() + g[:, (-l) % p - 1].min()) / ref
    return g[:, l % p - 1].min() / ref

def seed(kind, p, n, m, rng):
    a = np.arange(p)
    if kind == 'if':
        U, W = if_block(p, n, 1); U = U + 0.05 * rng.normal(size=U.shape); W = W + 0.05 * rng.normal(size=W.shape)
    elif kind == 'partner':
        # two aligned atoms at frequency 1 (interference on T={1,2}) + frequency-3 input content + small IF part
        U, W = if_block(p, n, 1)
        U = U + 0.3 * rng.normal() * np.cos(2 * np.pi * 3 * a / p + rng.uniform(0, 6.3))[None, None, :]
        U = U + 0.1 * rng.normal(size=U.shape); W = W + 0.1 * rng.normal(size=W.shape)
    else:
        U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
    if U.shape[0] < m:
        U = np.concatenate([U, 0.05 * rng.normal(size=(m - U.shape[0], n, p))]); W = np.concatenate([W, 0.05 * rng.normal(size=(m - W.shape[0], p))])
    return U[:m], W[:m]

def run(p, n, b, l, m, starts, mode, seed0=0):
    nu = n + 1; s = nu / b; rng = np.random.default_rng(seed0); mask = freq_proj(p, 1)
    ref = nu ** (1 - s) * Bp(p, n); res_all = []
    kinds = ['if', 'partner'] + ['rand'] * max(0, starts - 2)
    best = (-np.inf, None)
    for kind in kinds:
        U, W = seed(kind, p, n, m, rng); x = pack(U, W)
        for tau in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
            r = minimize(lambda z: tuple(-v for v in obj(z, m, n, p, b, tau * ref, l, mode, mask)), x, jac=True,
                         method='L-BFGS-B', options={'maxiter': 4000, 'ftol': 1e-15, 'gtol': 1e-12})
            x = r.x
        U, W = unpack(x, m, n, p); net = Net(U, W @ mask, b)
        v = value(net, l, mode); res_all.append(round(v, 4))
        if v > best[0]: best = (v, net)
    return best, res_all

if __name__ == "__main__":
    p, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    ms = [int(v) for v in sys.argv[4].split(",")]; starts = int(sys.argv[5]); mode = sys.argv[6]
    ls = [int(v) for v in sys.argv[7].split(",")]
    for l in ls:
        for m in ms:
            (val, net), allv = run(p, n, b, l, m, starts, mode)
            A, Q, a = net.cheapness()
            print(f"[{mode}] p={p} n={n} b={b} l={l} m={m}: price/IF = {val:.5f}  cheap {A[0]/a[0]:.4f}  all starts {sorted(allv)[-4:]}", flush=True)
            np.savez(f"dir2_{mode}_p{p}_n{n}_b{b}_l{l}_m{m}.npz", U=net.U, W=net.W, b=b)
