"""Spike search for the FREE chain relaxation (all non-own modes free).

Structural fact (REPORT, Lemma F): in the free relaxation a table is the same as a map v : Z_p^n -> P, where
P = {v in C^F : sum_j Re(v_j c_j(delta)) >= h for all delta != 0}, and the own data of feature j is
a_j(T) = E_x[ v_j(x) rho^{-2j s_T(x)} ].   We search configurations  v = v0 (constant) + spikes w_k at points x_k:
  margin = min( m(v0), m(v0 + w_k) ),   m(v) = min_delta sum_j Re(v_j c_j(delta)),
  clean_j = v0_j + (1/N) sum_k w_kj ,  interference a_j(T) = (1/N) sum_k w_kj rho^{-2j s_T(x_k)}.
Cost: chain (free mirror U, optimised jointly) or 'atoms' (not here).  Ratio = margin/(nu^{1-s} Gamma (sum_j q_j)^s).
Usage: python relax3.py p n b nspikes_max restarts seed
"""
import sys, itertools, time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/mechanism_scripts')
from prog import solve as prog_solve
from mrel import subsets, interval


class Spikes:
    def __init__(self, p, n, s, pts, S=None):
        self.p, self.n, self.s, self.r, self.nu = p, n, s, 1 / s, n + 1
        F = (p - 1) // 2
        self.S = list(S) if S else list(range(1, F + 1))
        self.K = len(self.S)
        self.N = p ** n
        rho = np.exp(2j * np.pi / p)
        self.C = np.array([[1 - rho ** (-j * d) for d in range(1, p)] for j in self.S])  # K x (p-1)
        self.pts = [np.array(x) for x in pts]
        self.m = len(pts)
        self.Ts = subsets(n)
        # pattern[k][j][T] = rho^{-2j s_T(x_k)}
        self.pat = np.array([[[rho ** (-2 * j * sum(x[list(T)])) for T in self.Ts] for j in self.S] for x in self.pts])  # m x K x 2^n
        self.up = [(t, tp) for t in range(self.nu) for tp in range(t + 1, self.nu)]
        self.ints = [(t, tp, self.Ts.index(interval(tp, t))) for t in range(self.nu) for tp in range(t)]
        G, y, _ = prog_solve(p, s, restarts=8)
        self.G, self.ystar = G, y

    def unpack(self, th):
        K, m = self.K, self.m
        k = 0
        v0 = th[k:k + 2 * K:2] + 1j * th[k + 1:k + 2 * K:2]; k += 2 * K
        w = (th[k:k + 2 * K * m:2] + 1j * th[k + 1:k + 2 * K * m:2]).reshape(m, K); k += 2 * K * m
        nu_up = len(self.up)
        U = (th[k:k + 2 * K * nu_up:2] + 1j * th[k + 1:k + 2 * K * nu_up:2]).reshape(K, nu_up)
        return v0, w, U

    def pack(self, v0, w, U):
        z = np.concatenate([v0.ravel(), w.ravel(), U.ravel()])
        return np.ravel(np.column_stack([z.real, z.imag]))

    def own(self, v0, w):
        # a[j][T]
        a = np.einsum('mk,mkt->kt', w, self.pat) / self.N
        a[:, 0] += v0  # T=() is index 0 (clean); pattern there is 1 so w/N already added
        return a

    def chain(self, aj, Uj):
        H = np.zeros((self.nu, self.nu), dtype=complex)
        for t in range(self.nu):
            H[t, t] = aj[0]
        for (t, tp, i) in self.ints:
            H[t, tp] = aj[i]
        for i, (t, tp) in enumerate(self.up):
            H[t, tp] = Uj[i]
        return H

    def margins(self, v0, w):
        pts = np.vstack([v0[None, :], v0[None, :] + w])  # (m+1) x K
        return (pts.real @ self.C.real - pts.imag @ self.C.imag)  # (m+1) x (p-1)

    def ratio(self, th):
        v0, w, U = self.unpack(th)
        h = self.margins(v0, w).min()
        a = self.own(v0, w)
        cost = sum(np.sum(np.linalg.svd(self.chain(a[kk], U[kk]), compute_uv=False) ** self.r) for kk in range(self.K))
        q = self.nu ** (-self.r) * cost
        return h / (self.nu ** (1 - self.s) * self.G * q ** self.s)

    def fobj(self, th, tau):
        v0, w, U = self.unpack(th)
        g = self.margins(v0, w)
        gm = g.min(); z = np.exp(-(g - gm) / tau); Z = z.sum()
        h = gm - tau * np.log(Z)
        W = z / Z  # (m+1) x (p-1)
        a = self.own(v0, w)
        cost = 0.0; Gs = []
        for kk in range(self.K):
            H = self.chain(a[kk], U[kk])
            Uu, sv, Vh = np.linalg.svd(H)
            se = np.sqrt(sv ** 2 + 1e-14)
            cost += np.sum(se ** self.r)
            Gs.append((Uu * (self.r * se ** (self.r - 2) * sv)) @ Vh)
        if h <= 0:
            return 50.0 - h, None
        # gradients: dh/d v (complex grad) at point i: sum_d W[i,d] conj(C[:,d])
        dh_pts = W @ np.conj(self.C.T)  # (m+1) x K
        dh_v0 = dh_pts.sum(0)
        dh_w = dh_pts[1:]
        # cost gradient wrt own a (complex): clean gets trace, intervals get entries
        dc_a = np.zeros((self.K, len(self.Ts)), dtype=complex)
        dc_U = np.zeros((self.K, len(self.up)), dtype=complex)
        for kk in range(self.K):
            Gm = Gs[kk]
            dc_a[kk, 0] = np.trace(Gm)
            for (t, tp, i) in self.ints:
                dc_a[kk, i] += Gm[t, tp]
            for i, (t, tp) in enumerate(self.up):
                dc_U[kk, i] = Gm[t, tp]
        # a = einsum(w, pat)/N + v0 on clean -> d cost / d w[m,k] = sum_T dc_a[k,T] conj(pat[m,k,T]) / N
        dc_w = np.einsum('kt,mkt->mk', dc_a, np.conj(self.pat)) / self.N
        dc_v0 = dc_a[:, 0]
        val = -np.log(h) + self.s * np.log(cost)
        gv0 = -dh_v0 / h + self.s * dc_v0 / cost
        gw = -dh_w / h + self.s * dc_w / cost
        gU = self.s * dc_U / cost
        z = np.concatenate([gv0.ravel(), gw.ravel(), gU.ravel()])
        return val, np.ravel(np.column_stack([z.real, z.imag]))

    def optimise(self, th, taus=(3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)):
        for tau in taus:
            def f(t):
                v, g = self.fobj(t, tau)
                if g is None:
                    # numerical fallback: push margin up
                    eps = 1e-6; g = np.zeros_like(t)
                    for i in range(len(t)):
                        e = np.zeros_like(t); e[i] = eps
                        g[i] = (self.fobj(t + e, tau)[0] - v) / eps
                return v, g
            res = minimize(f, th, jac=True, method='L-BFGS-B', options={'maxiter': 500, 'maxcor': 30})
            th = res.x
        return self.ratio(th), th


def search(p, n, b, mmax, restarts, seed, verbose=True):
    s = (n + 1) / b
    rng = np.random.default_rng(seed)
    F = (p - 1) // 2
    X = list(itertools.product(range(p), repeat=n))
    best = (0, None)
    t0 = time.time()
    for k in range(restarts):
        m = 1 + (k % mmax)
        idx = rng.choice(len(X), size=m, replace=False)
        pts = [X[i] for i in idx]
        Sp = Spikes(p, n, s, pts)
        y = Sp.ystar
        v0 = y.astype(complex) * rng.uniform(0.5, 1.2) + 0.1 * rng.normal(size=F) * y.max()
        # spikes: large, roughly along the current v0 direction rotated by random phases
        w = (rng.uniform(1, 20, size=(m, 1)) * y.max()) * np.exp(1j * rng.uniform(-np.pi / p, np.pi / p, size=(m, F))) * \
            (rng.uniform(0, 1, size=(m, F)) < 0.7)
        U = np.zeros((F, len(Sp.up)), dtype=complex)
        th = Sp.pack(v0, w, U)
        try:
            val, th2 = Sp.optimise(th)
        except Exception as e:
            print('fail', e); continue
        if val > best[0]:
            best = (val, (pts, th2))
        if verbose:
            print(f"  restart {k} m={m}: ratio {val:.6f}  best {best[0]:.6f}  ({time.time()-t0:.0f}s)", flush=True)
    return best


if __name__ == '__main__':
    p, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    mmax, restarts, seed = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    print(f"spike search p={p} n={n} b={b}")
    best = search(p, n, b, mmax, restarts, seed)
    val, (pts, th) = best
    np.savez(f"spk_p{p}_n{n}_b{b}_s{seed}.npz", ratio=val, pts=np.array(pts), th=th)
    print(f"BEST {val:.6f} pts={pts}")
