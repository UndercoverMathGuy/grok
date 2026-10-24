"""Robust search for the free chain relaxation R_free(S): fields Phi_j for j in S only, own modes priced by the chain matrix
(free mirror, optimised in an inner problem), every non-own mode of these fields free (partner, riding, DC, ...).
Margin = exact LP over the free modes.  Outer search over own data from structured starts (IF+noise, atoms, spikes).
Usage: python relax4.py p n b restarts seed S...
"""
import sys, itertools, time
import numpy as np
from scipy.optimize import minimize
from mrel import Model, subsets, interval
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/mechanism_scripts')
from prog import solve as prog_solve


class RF:
    def __init__(self, p, n, s, S, mode='free'):
        self.p, self.n, self.s, self.r, self.nu = p, n, s, 1 / s, n + 1
        self.S = list(S)
        self.M = Model(p, n, self.S, mode=mode, s=s)
        self.Ts = subsets(n)
        self.up = [(t, tp) for t in range(self.nu) for tp in range(t + 1, self.nu)]
        self.ints = [(t, tp, self.Ts.index(interval(tp, t))) for t in range(self.nu) for tp in range(t)]
        self.G, self.ystar, _ = prog_solve(p, s, restarts=8)
        self.Uwarm = {j: np.zeros(len(self.up), dtype=complex) for j in self.S}
        self.X = np.array(list(itertools.product(range(p), repeat=n)))

    def H(self, aj, U):
        H = np.zeros((self.nu, self.nu), dtype=complex)
        for t in range(self.nu):
            H[t, t] = aj[()]
        for (t, tp, i) in self.ints:
            H[t, tp] = aj[self.Ts[i]]
        for i, (t, tp) in enumerate(self.up):
            H[t, tp] = U[i]
        return H

    def sr(self, H, grad=False):
        Uu, sv, Vh = np.linalg.svd(H)
        se = np.sqrt(sv ** 2 + 1e-16)
        v = np.sum(se ** self.r)
        if not grad:
            return v
        return v, (Uu * (self.r * se ** (self.r - 2) * sv)) @ Vh

    def cost_j(self, j, aj, nstart=3):
        """min_U ||H(aj,U)||_{S_r}^r with warm start and a few canonical starts; returns value, gradient wrt aj (complex dict)"""
        L = self.H(aj, np.zeros(len(self.up), dtype=complex)); L = np.tril(L, -1)
        cands = [self.Uwarm[j], np.zeros(len(self.up), dtype=complex),
                 np.array([np.conj(L[tp, t]) for (t, tp) in self.up]),
                 ((2 - self.r) / self.r) * np.array([np.conj(L[tp, t]) for (t, tp) in self.up])]
        best = (np.inf, None)
        for u0 in cands[:max(nstart, 1) + 1]:
            def f(w):
                U = w[0::2] + 1j * w[1::2]
                v, G = self.sr(self.H(aj, U), grad=True)
                g = np.zeros_like(w)
                for i, (t, tp) in enumerate(self.up):
                    g[2 * i] = G[t, tp].real; g[2 * i + 1] = G[t, tp].imag
                return v, g
            w0 = np.ravel(np.column_stack([u0.real, u0.imag]))
            res = minimize(f, w0, jac=True, method='L-BFGS-B', options={'maxiter': 300})
            if res.fun < best[0]:
                best = (res.fun, res.x[0::2] + 1j * res.x[1::2])
        self.Uwarm[j] = best[1]
        v, G = self.sr(self.H(aj, best[1]), grad=True)
        grad = {(): np.trace(G)}
        for (t, tp, i) in self.ints:
            grad[self.Ts[i]] = G[t, tp]
        return v, grad

    def evaluate(self, a, grad=False):
        h, gh = self.M.margin(a)
        if h is None:
            return -np.inf, None
        tot = 0.0; gc = {}
        for j in self.S:
            v, g = self.cost_j(j, a[j])
            tot += v
            for T, z in g.items():
                gc[(j, T)] = z
        q = self.nu ** (-self.r) * tot
        ratio = h / (self.nu ** (1 - self.s) * self.G * q ** self.s)
        if not grad:
            return ratio, None
        return ratio, (h, gh, tot, gc)

    # packing of own data
    def unpack(self, th):
        a = {}; k = 0
        for j in self.S:
            a[j] = {}
            for T in self.Ts:
                a[j][T] = th[k] + 1j * th[k + 1]; k += 2
        return a

    def pack(self, a):
        return np.array([z for j in self.S for T in self.Ts for z in (a[j][T].real, a[j][T].imag)])

    def fobj(self, th):
        a = self.unpack(th)
        ratio, info = self.evaluate(a, grad=True)
        if info is None or info[0] <= 1e-12:
            return 10.0, np.zeros_like(th)
        h, gh, tot, gc = info
        val = -np.log(h) + self.s * np.log(tot)
        g = np.zeros_like(th); k = 0
        for j in self.S:
            for T in self.Ts:
                z = -gh.get((j, T), 0j) / h + self.s * gc.get((j, T), 0j) / tot
                g[k] = z.real; g[k + 1] = z.imag; k += 2
        return val, g

    def local(self, a, iters=200):
        th = self.pack(a)
        res = minimize(self.fobj, th, jac=True, method='L-BFGS-B', options={'maxiter': iters, 'maxcor': 20})
        th = res.x
        # Powell polish (nonsmooth LP margin)
        res2 = minimize(lambda t: self.fobj(t)[0], th, method='Powell', options={'maxiter': 3000, 'xtol': 1e-7, 'ftol': 1e-10})
        a = self.unpack(res2.x)
        return self.evaluate(a)[0], a

    # starts
    def start_if(self, rng, noise):
        a = {}
        for j in self.S:
            y = self.ystar[j - 1] if self.ystar[j - 1] > 1e-9 else 0.05
            a[j] = {T: ((y if T == () else 0) + noise * y * (rng.normal() + 1j * rng.normal())) for T in self.Ts}
        return a

    def start_spikes(self, rng, m):
        """own data of v = v0 + spikes on the cone edges (barrier type)"""
        p = self.p; rho = np.exp(2j * np.pi / p)
        a = {}
        for j in self.S:
            y = self.ystar[j - 1] if self.ystar[j - 1] > 1e-9 else 0.05
            v0 = y * rng.uniform(0.5, 1.0)
            pts = self.X[rng.choice(len(self.X), size=m, replace=False)]
            amp = y * rng.uniform(2, 15, size=m) * np.exp(1j * rng.choice([-1, 1], size=m) * np.pi / p * rng.uniform(0.7, 1.0, size=m))
            a[j] = {}
            for T in self.Ts:
                a[j][T] = (v0 if T == () else 0) + np.sum(amp * rho ** (-2 * j * pts[:, list(T)].sum(1))) / len(self.X)
        return a

    def start_atoms(self, rng, k):
        a = {}
        for j in self.S:
            y = self.ystar[j - 1] if self.ystar[j - 1] > 1e-9 else 0.05
            beta = (y / k) * rng.uniform(0.6, 1.4, k) * np.exp(1j * 0.4 * rng.normal(size=k))
            xi = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(k, self.n)))
            a[j] = {T: np.conj(sum(b * np.prod(x[list(T)]) for b, x in zip(beta, xi))) for T in self.Ts}
        return a


def run(p, n, b, S, restarts, seed, mode='free', verbose=True):
    s = (n + 1) / b
    R = RF(p, n, s, S, mode=mode)
    rng = np.random.default_rng(seed)
    a0 = R.start_if(rng, 0.0)
    r0 = R.evaluate(a0)[0]
    if verbose:
        print(f"R_free p={p} n={n} b={b} S={S} mode={mode}: Gamma={R.G:.5f}; IF-start ratio {r0:.6f}", flush=True)
    best = (r0, a0); t0 = time.time()
    for k in range(restarts):
        kind = k % 4
        if kind == 0:
            a = R.start_if(rng, 0.3)
        elif kind == 1:
            a = R.start_spikes(rng, 1 + rng.integers(3))
        elif kind == 2:
            a = R.start_atoms(rng, 2 + rng.integers(n))
        else:
            a = R.start_spikes(rng, 2)
        try:
            val, a2 = R.local(a)
        except Exception as e:
            print('fail', e); continue
        if val > best[0]:
            best = (val, a2)
        if verbose:
            print(f"  restart {k} kind {kind}: {val:.6f}   best {best[0]:.6f}  ({time.time()-t0:.0f}s)", flush=True)
    return R, best


if __name__ == '__main__':
    p, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    restarts, seed = int(sys.argv[4]), int(sys.argv[5])
    S = [int(v) for v in sys.argv[6:]]
    R, (val, a) = run(p, n, b, S, restarts, seed)
    np.save(f"rf_p{p}_n{n}_b{b}_S{''.join(map(str,S))}_s{seed}.npy", R.pack(a))
    print(f"BEST {val:.6f}")
