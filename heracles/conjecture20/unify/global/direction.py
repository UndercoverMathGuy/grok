# Directional prices of a single feature.
# For a single-feature network (outputs at frequency 1, inputs arbitrary) define its normalised profile
#   m(l) = min_x G(x,l) / (B_p P^s),  G(x,l) = f[x,y] - f[x,y-l],  l in Z_p^*.
# The cone price is pi = sup min_l m(l)/kappa(l) (attained at l = +-1).  Directional price at l:
#   pi_l = sup min(m(l), m(-l)) / kappa(l).
# We report pi_l / nu^{1-s} (IF value = 1).  > 1 would mean a single feature beats IF in direction l.
import sys, numpy as np
from lib import *

def dirobj(x, m, n, p, b, tau, dels, wts, mask):
    U, W = unpack(x, m, n, p); W = W @ mask
    net = Net(U, W, b); c = net.cost(); s = net.s
    G = net.gaps()[:, [d - 1 for d in dels]] * wts[None, :] * c ** (-s)
    gmin = G.min(); z = np.exp(-(G - gmin) / tau); Z = z.sum(); lamsel = z / Z
    F = gmin - tau * np.log(Z)
    lam = np.zeros((net.N, p - 1)); lam[:, [d - 1 for d in dels]] = lamsel * wts[None, :]
    gU, gW = net.grad_lin(lam); cU, cW = net.grad_cost()
    Eg = (lamsel * G).sum()
    dU = c ** (-s) * gU - s * Eg / c * cU
    dW = (c ** (-s) * gW - s * Eg / c * cW) @ mask.T
    return F, pack(dU, dW)

def run(p, n, b, l, m, starts, seed=0, taus=(3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)):
    nu = n + 1; s = nu / b; rng = np.random.default_rng(seed)
    mask = freq_proj(p, 1); ka = 1 - np.cos(2 * np.pi * l / p)
    dels = [l % p, (-l) % p] if l % p != (-l) % p else [l % p]
    wts = np.array([1.0 / ka] * len(dels))
    ref = nu ** (1 - s) * Bp(p, n)
    best = (-np.inf, None)
    for st in range(starts):
        if st == 0:
            U, W = if_block(p, n, 1); U = U + 0.05 * rng.normal(size=U.shape); W = W + 0.05 * rng.normal(size=W.shape)
            if m > nu:
                U = np.concatenate([U, 0.05 * rng.normal(size=(m - nu, n, p))]); W = np.concatenate([W, 0.05 * rng.normal(size=(m - nu, p))])
            else:
                U = U[:m]; W = W[:m]
        else:
            U = rng.normal(size=(m, n, p)); W = rng.normal(size=(m, p))
        x = pack(U, W)
        for tau in taus:
            res = minimize(lambda z: tuple(-v for v in dirobj(z, m, n, p, b, tau * ref, dels, wts, mask)), x, jac=True,
                           method='L-BFGS-B', options={'maxiter': 4000, 'ftol': 1e-15, 'gtol': 1e-12})
            x = res.x
        U, W = unpack(x, m, n, p); W = W @ mask; net = Net(U, W, b)
        G = net.gaps()[:, [d - 1 for d in dels]] / ka
        val = G.min() / net.cost() ** s / ref
        if val > best[0]: best = (val, net)
    return best

if __name__ == "__main__":
    p, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    ms = [int(v) for v in sys.argv[4].split(",")]; starts = int(sys.argv[5])
    ls = [int(v) for v in sys.argv[6].split(",")] if len(sys.argv) > 6 else list(range(1, (p - 1) // 2 + 1))
    for l in ls:
        for m in ms:
            val, net = run(p, n, b, l, m, starts)
            A, Q, a = net.cheapness()
            print(f"p={p} n={n} b={b} l={l} m={m}: directional price / IF = {val:.5f}   |beta_1|/a_1 = {A[0]/a[0]:.4f}", flush=True)
            np.savez(f"dir_p{p}_n{n}_b{b}_l{l}_m{m}.npz", U=net.U, W=net.W, b=b)
