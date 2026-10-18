"""Smooth local search for the chain-matrix relaxation value (softmin margin, annealed), exact evaluation at the end.

Variables: for each feature j in S the full coefficient array phi_j (complex, shape p^n) of the field Phi_j, and the free
strictly-upper part U_j of its chain matrix.  Own coefficients (characters 2j 1_T) are priced by
q_j = nu^{-r} ||phi_j(0) I + L_j + U_j||_{S_r}^r ;  in mode 'C' the non-own coefficients are frozen at 0, in mode 'free' they
are free variables, in mode 'partner' only the characters -2j 1_T (own partner) are free, in mode 'dc' (p=3 style) the
characters with entries in {0, j} ... (see FREE_SETS).
Exact ratio: h_exact / (nu^{1-s} Gamma (sum q_j)^s), with h_exact = min over (x,delta) of the true g (no softening).
"""
import sys, json, time, itertools
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/mechanism_scripts')
from prog import solve as prog_solve
from mrel import subsets, interval


class Relax2:
    def __init__(self, p, n, s, mode='free', S=None, Gamma=None, cone_only=False):
        self.p, self.n, self.s, self.r, self.nu = p, n, s, 1 / s, n + 1
        F = (p - 1) // 2
        self.S = list(S) if S else list(range(1, F + 1))
        self.K = len(self.S)
        self.X = np.array(list(itertools.product(range(p), repeat=n)))
        self.N = len(self.X)
        self.dels = np.arange(1, p)
        rho = np.exp(2j * np.pi / p)
        self.C = np.array([[1 - rho ** (-j * d) for d in self.dels] for j in self.S])  # K x (p-1)
        self.Ts = subsets(n)
        self.chars = [tuple(c) for c in self.X]
        self.cidx = {c: i for i, c in enumerate(self.chars)}
        # character matrix E[x, chi] = rho^{chi.x}
        self.E = np.exp(2j * np.pi * (self.X @ self.X.T) / p)  # N x N, symmetric
        self.own_idx = {}
        for j in self.S:
            self.own_idx[j] = {}
            for T in self.Ts:
                chi = np.zeros(n, dtype=int); chi[list(T)] = (2 * j) % p
                self.own_idx[j][T] = self.cidx[tuple(chi)]
        # free masks
        self.mode = mode
        self.mask = {}
        for j in self.S:
            m = np.zeros(self.N, dtype=bool)
            for T in self.Ts:
                m[self.own_idx[j][T]] = True
            if mode == 'free':
                m[:] = True
            elif mode == 'partner':
                for T in self.Ts:
                    chi = np.zeros(n, dtype=int); chi[list(T)] = (-2 * j) % p
                    m[self.cidx[tuple(chi)]] = True
            elif isinstance(mode, dict):
                for c in mode.get(j, []):
                    m[self.cidx[tuple(c)]] = True
            self.mask[j] = m
        self.nvar_phi = {j: int(self.mask[j].sum()) for j in self.S}
        self.up = [(t, tp) for t in range(self.nu) for tp in range(t + 1, self.nu)]
        if Gamma is None:
            self.G, self.ystar, _ = prog_solve(p, s, restarts=8)
        else:
            self.G, self.ystar = Gamma
        self.cone_only = cone_only

    # ---- packing
    def unpack(self, th):
        phi = {}; U = {}; k = 0
        for j in self.S:
            m = self.nvar_phi[j]
            v = th[k:k + 2 * m]; k += 2 * m
            full = np.zeros(self.N, dtype=complex)
            full[self.mask[j]] = v[0::2] + 1j * v[1::2]
            phi[j] = full
            w = th[k:k + 2 * len(self.up)]; k += 2 * len(self.up)
            U[j] = w[0::2] + 1j * w[1::2]
        return phi, U

    def pack(self, phi, U):
        out = []
        for j in self.S:
            v = phi[j][self.mask[j]]
            out += list(np.ravel(np.column_stack([v.real, v.imag])))
            out += list(np.ravel(np.column_stack([U[j].real, U[j].imag])))
        return np.array(out)

    def chain(self, j, phi, U):
        H = np.zeros((self.nu, self.nu), dtype=complex)
        own = self.own_idx[j]
        for t in range(self.nu):
            H[t, t] = phi[j][own[()]]
            for tp in range(t):
                H[t, tp] = phi[j][own[interval(tp, t)]]
        for k, (t, tp) in enumerate(self.up):
            H[t, tp] = U[j][k]
        return H

    # ---- margin
    def gvals(self, phi):
        # Phi_j(x) = sum_chi phi_j(chi) rho^{chi.x}
        P = np.array([self.E @ phi[j] for j in self.S])  # K x N
        g = np.einsum('kx,kd->xd', P.real, self.C.real) - np.einsum('kx,kd->xd', P.imag, self.C.imag)
        return P, g  # g: N x (p-1)

    def margin_exact(self, phi):
        return self.gvals(phi)[1].min()

    def cost(self, phi, U, eps=1e-10, grad=False):
        tot = 0.0; grads = {}
        for j in self.S:
            H = self.chain(j, phi, U)
            Uu, sv, Vh = np.linalg.svd(H)
            se = np.sqrt(sv ** 2 + eps ** 2)
            tot += np.sum(se ** self.r)
            if grad:
                w = self.r * se ** (self.r - 2) * sv
                grads[j] = (Uu * w) @ Vh
        return (tot, grads) if grad else tot

    def ratio(self, phi, U):
        h = self.margin_exact(phi)
        q = self.nu ** (-self.r) * self.cost(phi, U)
        return h / (self.nu ** (1 - self.s) * self.G * q ** self.s)

    def fobj(self, th, tau):
        phi, U = self.unpack(th)
        P, g = self.gvals(phi)
        gm = g.min()
        z = np.exp(-(g - gm) / tau)
        Z = z.sum()
        h = gm - tau * np.log(Z / g.size)   # soft-min (>= true min - tau log size... ) ; mean-normalised
        w = z / Z  # weights N x (p-1)
        if h <= 1e-9:
            # push margin up: objective -h (scale-free fallback)
            c = self.cost(phi, U)
            val = -h / c ** self.s + 10.0
            gh_scale = -1.0 / c ** self.s
            cgrad = None
            use_log = False
        else:
            use_log = True
        c, cg = self.cost(phi, U, grad=True)
        # gradient of h wrt phi_j(chi): sum_{x,d} w conj(rho^{chi x} c_j(d))
        grad = np.zeros_like(th)
        k = 0
        for kk, j in enumerate(self.S):
            Wd = w @ np.conj(self.C[kk])         # N (x): sum_d w(x,d) conj(c_j(d))
            gam_h = np.conj(self.E).T @ Wd       # chi: sum_x conj(rho^{chi x}) Wd(x)
            gam_c = np.zeros(self.N, dtype=complex)
            G = cg[j]; own = self.own_idx[j]
            gam_c[own[()]] += np.trace(G)
            for t in range(self.nu):
                for tp in range(t):
                    gam_c[own[interval(tp, t)]] += G[t, tp]
            if use_log:
                gam = -gam_h / h + self.s * gam_c / c
            else:
                gam = -gam_h / c ** self.s + self.s * h * gam_c / c ** (self.s + 1)
            gv = gam[self.mask[j]]
            m = self.nvar_phi[j]
            grad[k:k + 2 * m:2] = gv.real; grad[k + 1:k + 2 * m:2] = gv.imag
            k += 2 * m
            for i, (t, tp) in enumerate(self.up):
                gu = (self.s * G[t, tp] / c) if use_log else (self.s * h * G[t, tp] / c ** (self.s + 1))
                grad[k + 2 * i] = gu.real; grad[k + 2 * i + 1] = gu.imag
            k += 2 * len(self.up)
        if use_log:
            val = -np.log(h) + self.s * np.log(c)
        else:
            val = -h / c ** self.s + 10.0
        return val, grad

    def optimise(self, phi, U, taus=(0.05, 0.02, 0.008, 0.003, 0.001, 0.0003), maxiter=300):
        th = self.pack(phi, U)
        for tau in taus:
            # tau relative to the scale of the margin
            P, g = self.gvals(self.unpack(th)[0])
            scale = max(abs(g).mean(), 1e-6)
            res = minimize(lambda t: self.fobj(t, tau * scale), th, jac=True, method='L-BFGS-B',
                           options={'maxiter': maxiter, 'maxcor': 30})
            th = res.x
        phi, U = self.unpack(th)
        return self.ratio(phi, U), phi, U

    # ---- starts
    def if_start(self):
        phi = {}; U = {}
        for j in self.S:
            phi[j] = np.zeros(self.N, dtype=complex)
            phi[j][self.own_idx[j][()]] = self.ystar[j - 1] if j - 1 < len(self.ystar) else 0.0
            U[j] = np.zeros(len(self.up), dtype=complex)
        return phi, U

    def perturb(self, phi, U, rng, eps):
        phi2 = {}; U2 = {}
        for j in self.S:
            sc = abs(phi[j][self.own_idx[j][()]]) + 0.02
            z = np.zeros(self.N, dtype=complex)
            z[self.mask[j]] = sc * eps * (rng.normal(size=self.mask[j].sum()) + 1j * rng.normal(size=self.mask[j].sum()))
            phi2[j] = phi[j] + z
            U2[j] = U[j] + sc * eps * (rng.normal(size=len(self.up)) + 1j * rng.normal(size=len(self.up)))
        return phi2, U2

    def atoms_start(self, rng, natoms=2, clean_boost=0.0, which=None):
        """features realised by atoms (U = mirror); other coefficients 0.  clean_boost adds an IF-like clean part."""
        phi, U = self.if_start()
        for j in self.S:
            if which is not None and j not in which:
                continue
            y = (self.ystar[j - 1] if j - 1 < len(self.ystar) else 0) + 0.05
            beta = (y / natoms) * np.exp(1j * 0.3 * rng.normal(size=natoms)) * rng.uniform(0.6, 1.4, natoms)
            xi = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(natoms, self.n)))
            H = np.zeros((self.nu, self.nu), dtype=complex)
            for b, x in zip(beta, xi):
                mv = np.concatenate([[1], np.cumprod(x)])
                H += np.conj(b) * np.outer(np.conj(mv), mv)
            for T in self.Ts:
                bT = sum(b * np.prod([x[t] for t in T]) for b, x in zip(beta, xi))
                phi[j][self.own_idx[j][T]] = np.conj(bT) + (clean_boost * y if T == () else 0)
            U[j] = np.array([H[t, tp] for (t, tp) in self.up])
        return phi, U


def run(p, n, b, mode, restarts, seed, S=None, verbose=True):
    s = (n + 1) / b
    R = Relax2(p, n, s, mode, S)
    rng = np.random.default_rng(seed)
    phi, U = R.if_start()
    if verbose:
        print(f"p={p} n={n} b={b} s={s:.4f} mode={mode} S={R.S} Gamma={R.G:.6f} y*={np.round(R.ystar,4)}  IF ratio {R.ratio(phi,U):.6f}", flush=True)
    best = (R.ratio(phi, U), (phi, U))
    t0 = time.time()
    for k in range(restarts):
        kind = k % 4
        if kind == 0:
            st = R.perturb(*R.if_start(), rng, 0.1)
        elif kind == 1:
            st = R.perturb(*R.if_start(), rng, 0.5)
        elif kind == 2:
            st = R.atoms_start(rng, natoms=2, clean_boost=0.5)
        else:
            st = R.atoms_start(rng, natoms=1, clean_boost=1.0, which=[R.S[rng.integers(len(R.S))]])
        val, phi2, U2 = R.optimise(*st)
        if val > best[0]:
            best = (val, (phi2, U2))
        if verbose:
            print(f"  restart {k} kind {kind}: ratio {val:.6f}  best {best[0]:.6f} ({time.time()-t0:.0f}s)", flush=True)
    return R, best


if __name__ == '__main__':
    p, n, b = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    mode = sys.argv[4]; restarts = int(sys.argv[5]); seed = int(sys.argv[6])
    S = [int(v) for v in sys.argv[7:]] or None
    R, (val, (phi, U)) = run(p, n, b, mode, restarts, seed, S)
    fn = f"r2_p{p}_n{n}_b{b}_{mode}_s{seed}.npz"
    np.savez(fn, ratio=val, **{f"phi{j}": phi[j] for j in R.S}, **{f"U{j}": U[j] for j in R.S})
    print(f"BEST {val:.6f}  saved {fn}")
