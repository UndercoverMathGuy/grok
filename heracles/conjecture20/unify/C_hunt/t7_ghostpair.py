# (1) the VIOL network vs D_{1,4};  (2) exact class N_{1} in class C: feature 1 (3 neurons) + zero-mean ghost pairs
#     (b, -b) at every other frequency j (exact constraint by reparametrisation), optimised from many seeds;
#     does any ratio exceed D_{1} = 3^{1-s} B_p kappa_min ?   p in {5,7,11,13}, b in {2, 2.5}.
import numpy as np, hlib as H, json
from scipy.optimize import minimize
GL = H.GL
z = np.load('out/VIOL_b_p11_n2_b2.5_pen_s51_9.npz')
U, W = z['U'], z['W']; p, n, b = int(z['p']), int(z['n']), float(z['b']); s = 3 / b
G14, _ = H.programme_exact(p, s, [1, 4])
net = GL.Net(U, W, b)
print(f"VIOL net: ratio vs D_1 = {net.gamma() / (3 ** (1 - s) * H.Bp(p, n) * (1 - np.cos(2 * np.pi / p))):.10f}; "
      f"Gamma^({{1,4}}) = {G14:.6f} vs kappa_min = {1 - np.cos(2 * np.pi / p):.6f}; ratio vs D_{{1,4}} = {net.gamma() / (3 ** (1 - s) * H.Bp(p, n) * G14):.6f}", flush=True)


class Pair:
    """CC model with neuron list; ghost pairs share b with opposite sign: x_red -> x_full linear."""
    def __init__(self, p, s, freqs, pairs):
        self.cc = H.CC(p, 2, freqs, s); m = self.cc.m; self.m = m
        free_b = [i for i in range(m) if i not in [q for (a_, q) in pairs]]
        self.free_b = free_b; nb = len(free_b)
        M = np.zeros((2 * m + 2 * m, 2 * nb + 2 * m))
        idx = {i: k for k, i in enumerate(free_b)}
        for i in range(m):
            src = idx[i] if i in idx else idx[[a_ for (a_, q) in pairs if q == i][0]]
            sg = 1.0 if i in idx else -1.0
            M[i, src] = sg; M[m + i, nb + src] = sg
        for k in range(2 * m): M[2 * m + k, 2 * nb + k] = 1.0
        self.M = M

    def full(self, xr): return self.M @ xr

    def opt(self, xr, taus=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4)):
        for tau in taus:
            f = lambda zz: tuple(v for v in (lambda F_, g_: (-F_, -(self.M.T @ g_)))(*self.cc.soft(self.full(zz), tau)))
            xr = minimize(f, xr, jac=True, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13}).x
        return xr


rng = np.random.default_rng(1)
for (p, b) in [(5, 2.0), (7, 2.0), (11, 2.0), (11, 2.5), (13, 2.0), (13, 2.5)]:
    s = 3 / b; F = (p - 1) // 2; kmin = 1 - np.cos(2 * np.pi / p)
    best = -np.inf; lifts = []
    for it in range(16):
        freqs = [1, 1, 1]; pairs = []
        ghosts = list(range(2, F + 1)) if it % 2 == 0 else list(rng.choice(range(2, F + 1), int(rng.integers(1, F)), replace=False))
        for j in ghosts:
            for k in range(int(rng.integers(1, 3))):
                freqs += [j, j]; pairs.append((len(freqs) - 2, len(freqs) - 1))
        P = Pair(p, s, freqs, pairs)
        m = P.m; nb = len(P.free_b)
        b0 = np.zeros(nb, complex); b0[:3] = 1 / 3 * np.exp(0.05 * rng.standard_normal(3))
        b0[3:] = (10 ** rng.uniform(-3, -0.5, nb - 3)) * np.exp(1j * rng.uniform(0, 6.3, nb - 3))
        ph = np.zeros((m, 2)); ph[:3] = 2 * np.pi * np.arange(3)[:, None] / 3 + 0.2 * rng.standard_normal((3, 2))
        ph[3:] = rng.uniform(0, 2 * np.pi, (m - 3, 2))
        xr = np.r_[b0.real, b0.imag, ph.ravel()]
        xr = P.opt(xr)
        xf = P.full(xr)
        bb, _ = P.cc.unpack(xf)
        assert all(abs(sum(bb[np.array(freqs) == j])) < 1e-12 for j in range(2, F + 1)), 'ghost mean not zero'
        v = P.cc.gnorm(xf) / (3 ** (1 - s) * kmin)
        best = max(best, v)
        gm = np.abs(bb[3:]).sum() / np.abs(bb).sum()
        lifts.append((round(v, 8), round(gm, 5)))
    print(json.dumps(dict(p=p, b=b, runs=16, best_ratio_vs_D1=best, sample=lifts[:6])), flush=True)
