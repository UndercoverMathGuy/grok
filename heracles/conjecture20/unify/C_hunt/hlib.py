# C_hunt library.
# (1) Programme Gamma^{(S)}_s(p) (reweighted LP, many restarts; exact vertex enumeration for small |S|).
# (2) Class-C Riesz-product model, any n:
#       neuron i: frequency l_i in F, complex b_i (= E_x of its field = conj(beta_i)), unit xi_{it} (t=1..n);
#       field of feature l:  Phi_l(x) = sum_{i: l_i=l} b_i prod_t (1 + xi_it zeta_l^{x_t}),  zeta_l = rho^{2l};
#       gaps g(x,delta) = sum_l Re Phi_l(x) kappa_l(delta) - Im Phi_l(x) sin(2 pi l delta/p);
#       balanced single-frequency neuron: |b_i| = B_p ||omega_i||^nu, so in units B_p = 1 the cost is sum |b_i|^r.
#     ratio(theta) = h / (sum|b_i|^r)^s / (nu^{1-s} Gamma^{(S)}_s)  (= gamma/gamma_IF^{(S)}; = W/D at n = 2).
#     Realisation as an explicit product network (U, W) for independent checks (to_net).
# (3) General product networks with per-neuron, per-slot frequency masks (exact feature-class constraints).
import sys, os, itertools, math
import numpy as np
from scipy.optimize import minimize, linprog

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'global'))
import lib as GL  # refereed GLOBAL library (read-only)

LET = 'abcdefgh'


def Bp(p, n):
    return 2.0 * (2.0 * p * (n + 1)) ** (-(n + 1) / 2.0)


def kap_rows(p):
    """kappa_l(delta), rows delta = 1..p-1, cols l = 1..F"""
    F = (p - 1) // 2
    d = np.arange(1, p)[:, None]; j = np.arange(1, F + 1)[None, :]
    return 1 - np.cos(2 * np.pi * d * j / p)


# ----------------------------------------------------------------------------------- programme
def programme(p, s, S=None, restarts=60, seed=0):
    """Gamma^{(S)}_s(p) = max min_delta sum_{l in S} y_l kappa_l(delta), sum y^{1/s} <= 1.
    Reweighted LP (concave l_r minimisation over {v >= 0 : K v >= 1}) from many starts.
    Returns Gamma, y (length F), binding offsets (delta in 1..F)."""
    F = (p - 1) // 2; r = 1.0 / s
    S = list(range(1, F + 1)) if S is None else sorted(S)
    if len(S) > 6: restarts = max(restarts, 600)     # prog_check.out: 60 restarts under-estimate Gamma for p >= 29
    K = kap_rows(p)[:F][:, [j - 1 for j in S]]
    rng = np.random.default_rng(seed); best = (np.inf, None)
    for t in range(restarts):
        w = np.ones(len(S)) if t == 0 else rng.uniform(0.05, 2.0, len(S))
        v = None
        for it in range(300):
            res = linprog(w, A_ub=-K, b_ub=-np.ones(F), bounds=(0, None), method='highs')
            if res.status != 0: v = None; break
            v = res.x; wn = (v + 1e-12) ** (r - 1)
            if np.allclose(wn, w, rtol=1e-11): break
            w = wn
        if v is None: continue
        v = v * (v > 1e-11)
        Q = np.sum(v[v > 0] ** r)
        if Q < best[0] - 1e-14: best = (Q, v.copy())
    Q, v = best
    y = np.zeros(F); y[[j - 1 for j in S]] = v / Q ** s
    G = Q ** (-s)
    cov = kap_rows(p)[:F] @ y
    bind = [d + 1 for d in range(F) if cov[d] < G * (1 + 1e-7)]
    return G, y, bind


def programme_exact(p, s, S=None):
    """Exact Gamma^{(S)} by vertex enumeration (only for small |S| / F)."""
    F = (p - 1) // 2; r = 1.0 / s
    S = list(range(1, F + 1)) if S is None else sorted(S)
    K = kap_rows(p)[:F][:, [j - 1 for j in S]]; k = len(S)
    A = np.vstack([K, np.eye(k)]); bv = np.concatenate([np.ones(F), np.zeros(k)])
    best = -1; bestv = None
    for act in itertools.combinations(range(F + k), k):
        M = A[list(act)]
        if abs(np.linalg.det(M)) < 1e-12: continue
        v = np.linalg.solve(M, bv[list(act)])
        if np.any(v < -1e-12) or np.any(K @ v < 1 - 1e-10): continue
        v = np.clip(v, 0, None); val = np.sum(v[v > 0] ** r) ** (-s)
        if val > best: best = val; bestv = v
    y = np.zeros(F); y[[j - 1 for j in S]] = bestv / np.sum(bestv[bestv > 0] ** r) ** s
    return best, y


# ----------------------------------------------------------------------------------- class-C model
class CC:
    """Class-C Riesz-product model. freqs: list of neuron frequencies (in 1..F)."""

    def __init__(self, p, n, freqs, s):
        self.p, self.n, self.s = p, n, s
        self.nu = n + 1; self.r = 1.0 / s
        self.F = (p - 1) // 2
        self.freqs = np.array(freqs, int); self.m = len(freqs)
        self.feats = sorted(set(self.freqs.tolist()))
        self.fidx = np.array([self.feats.index(l) for l in self.freqs])
        X = np.indices((p,) * n).reshape(n, -1).T
        self.N = X.shape[0]
        # z[i, t, x] = zeta_{l_i}^{x_t} = exp(2 pi i 2 l_i x_t / p)
        self.z = np.exp(2j * np.pi * 2 * self.freqs[:, None, None] * X.T[None, :, :] / p)
        d = np.arange(1, p)
        L = np.array(self.feats)
        self.Kd = 1 - np.cos(2 * np.pi * np.outer(L, d) / p)         # (nf, p-1)
        self.Sd = np.sin(2 * np.pi * np.outer(L, d) / p)              # (nf, p-1)
        self.Cd = self.Kd + 1j * self.Sd                              # c_l(delta) = 1 - rho^{-l delta}
        self.eps = 1e-14

    # parameter vector: [Re b (m), Im b (m), phi (m*n)]  with xi_it = exp(i phi_it)
    def unpack(self, x):
        m, n = self.m, self.n
        b = x[:m] + 1j * x[m:2 * m]; phi = x[2 * m:].reshape(m, n)
        return b, phi

    def pack(self, b, phi):
        return np.concatenate([b.real, b.imag, np.asarray(phi).ravel()])

    def factors(self, phi):
        xi = np.exp(1j * phi)                                         # (m, n)
        fac = 1 + xi[:, :, None] * self.z                             # (m, n, N)
        return xi, fac

    def fields(self, x):
        b, phi = self.unpack(x)
        xi, fac = self.factors(phi)
        P = np.prod(fac, axis=1)                                      # (m, N)
        Phi = np.zeros((len(self.feats), self.N), complex)
        np.add.at(Phi, self.fidx, b[:, None] * P)
        return Phi, P, xi, fac, b

    def gaps(self, x):
        Phi = self.fields(x)[0]
        return Phi.real.T @ self.Kd - Phi.imag.T @ self.Sd           # (N, p-1)

    def cost(self, x):
        b, _ = self.unpack(x)
        return np.sum((np.abs(b) ** 2 + self.eps ** 2) ** (self.r / 2))

    def gnorm(self, x):
        """normalised margin h / cost^s (units B_p = 1)"""
        return self.gaps(x).min() / self.cost(x) ** self.s

    def soft(self, x, tau):
        """smooth-min of normalised gaps and its gradient"""
        Phi, P, xi, fac, b = self.fields(x)
        g = Phi.real.T @ self.Kd - Phi.imag.T @ self.Sd
        C = self.cost(x); Cs = C ** self.s
        q = g / Cs
        qmin = q.min(); w = np.exp(-(q - qmin) / tau); Z = w.sum(); lam = w / Z
        Fv = qmin - tau * np.log(Z)
        Eg = (lam * g).sum()
        # d(sum lam g): Lambda_l(x) = sum_delta lam(x,delta) c_l(delta)
        Lam = lam @ self.Cd.T                                         # (N, nf)
        G = Lam[:, self.fidx].T                                       # (m, N)
        GP = (G * P).sum(1)
        dRe = GP.real; dIm = -GP.imag
        dphi = np.zeros((self.m, self.n))
        for t in range(self.n):
            others = np.prod(np.delete(fac, t, axis=1), axis=1) if self.n > 1 else 1.0
            dphi[:, t] = np.real(b * (G * 1j * xi[:, t:t + 1] * self.z[:, t] * others).sum(1))
        # cost gradient
        mag2 = np.abs(b) ** 2 + self.eps ** 2
        cfac = self.r * mag2 ** (self.r / 2 - 1)                      # dC/d(Re b) = cfac * Re b
        dCre = cfac * b.real; dCim = cfac * b.imag
        grad = np.concatenate([dRe / Cs - self.s * Eg / Cs / C * dCre,
                               dIm / Cs - self.s * Eg / Cs / C * dCim,
                               dphi.ravel() / Cs])
        return Fv, grad

    def jac_active(self, x, idx):
        """Jacobian of q_k = g_k / C^s for flat gap indices idx (into (N, p-1))."""
        Phi, P, xi, fac, b = self.fields(x)
        g = (Phi.real.T @ self.Kd - Phi.imag.T @ self.Sd).ravel()
        C = self.cost(x); Cs = C ** self.s
        xs, ds = np.divmod(idx, self.p - 1)
        Cl = self.Cd[:, ds]                                           # (nf, k)
        Gi = Cl[self.fidx]                                            # (m, k)
        Pk = P[:, xs]                                                 # (m, k)
        GP = Gi * Pk
        J = np.zeros((len(idx), 2 * self.m + self.m * self.n))
        J[:, :self.m] = GP.real.T; J[:, self.m:2 * self.m] = -GP.imag.T
        for t in range(self.n):
            others = np.prod(np.delete(fac[:, :, xs], t, axis=1), axis=1) if self.n > 1 else 1.0
            val = np.real(b[:, None] * Gi * 1j * xi[:, t:t + 1] * self.z[:, t][:, xs] * others)   # (m, k)
            J[:, 2 * self.m + np.arange(self.m) * self.n + t] = val.T
        mag2 = np.abs(b) ** 2 + self.eps ** 2
        cfac = self.r * mag2 ** (self.r / 2 - 1)
        dC = np.concatenate([cfac * b.real, cfac * b.imag, np.zeros(self.m * self.n)])
        gk = g[idx]
        Jq = J / Cs - (self.s * gk / Cs / C)[:, None] * dC[None, :]
        return gk / Cs, Jq

    def normalise(self, x):
        b, phi = self.unpack(x)
        b = b / self.cost(x) ** self.s
        return self.pack(b, phi)

    # ------------------------------------------------------------------ optimisers
    def lbfgs(self, x, taus=(3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4), iters=1500, fix_mag=False, free=None):
        x = self.normalise(x)
        if free is not None:
            fr = np.asarray(free, bool); x0 = x.copy()

            def objf(zz):
                xx = x0.copy(); xx[fr] = zz
                Fv, gr = self.soft(xx, tau)
                return -Fv, -gr[fr]
            for tau in taus:
                res = minimize(objf, x[fr], jac=True, method='L-BFGS-B',
                               options={'maxiter': iters, 'ftol': 1e-15, 'gtol': 1e-12})
                x = x0.copy(); x[fr] = res.x
                x = self.normalise(x); x0 = x.copy()
            return x
        if fix_mag:
            b0, _ = self.unpack(x); mags = np.abs(b0)

            def obj(zz):
                al = zz[:self.m]; phi = zz[self.m:]
                xx = self.pack(mags * np.exp(1j * al), phi)
                Fv, gr = self.soft(xx, tau)
                gb = gr[:self.m] + 1j * gr[self.m:2 * self.m]
                # d/d alpha of Re/Im b: b = mags e^{i al}: dRe = -mags sin, dIm = mags cos
                dal = gb.real * (-mags * np.sin(al)) + gb.imag * (mags * np.cos(al))
                return -Fv, -np.concatenate([dal, gr[2 * self.m:]])
            b0, phi0 = self.unpack(x)
            zz = np.concatenate([np.angle(b0), phi0.ravel()])
            for tau in taus:
                res = minimize(obj, zz, jac=True, method='L-BFGS-B',
                               options={'maxiter': iters, 'ftol': 1e-15, 'gtol': 1e-12})
                zz = res.x
            return self.pack(mags * np.exp(1j * zz[:self.m]), zz[self.m:])
        for tau in taus:
            res = minimize(lambda z: tuple(-v for v in self.soft(z, tau)), x, jac=True, method='L-BFGS-B',
                           options={'maxiter': iters, 'ftol': 1e-15, 'gtol': 1e-12})
            x = self.normalise(res.x)
        return x

    def slp(self, x, iters=200, delta0=1e-2, nact=3000, tol=1e-13, verbose=False, free=None):
        """sequential LP minimax polish of min_k q_k (trust region in l_inf); free = mask of movable params."""
        x = self.normalise(x)
        fr = np.ones(x.size, bool) if free is None else np.asarray(free, bool)
        D = delta0; best = self.gnorm(x)
        for it in range(iters):
            q = self.gaps(x).ravel() / self.cost(x) ** self.s
            order = np.argsort(q)
            thr = q[order[0]] + max(4 * D * 10, 1e-9) * (abs(q[order[0]]) + 1e-12) * 50
            k = min(nact, max(50, int(np.searchsorted(q[order], thr))))
            idx = order[:k]
            qk, J = self.jac_active(x, idx)
            nv = J.shape[1]
            c = np.zeros(nv + 1); c[-1] = -1.0
            A = np.hstack([-J, np.ones((k, 1))])
            bnd = [((-D, D) if fr[q] else (0.0, 0.0)) for q in range(nv)] + [(None, None)]
            res = linprog(c, A_ub=A, b_ub=qk, bounds=bnd, method='highs')
            if res.status != 0: D *= 0.3; continue
            xn = self.normalise(x + res.x[:nv])
            val = self.gnorm(xn)
            if val > best + tol * abs(best):
                x, best = xn, val; D = min(D * 1.6, 0.3)
            else:
                D *= 0.3
            if verbose and it % 10 == 0: print(it, best, D, k, flush=True)
            if D < 1e-11: break
        return x, best

    # ------------------------------------------------------------------ diagnostics
    def feature_data(self, x):
        """per feature l: |A_l|, L_l = sum|b_i|, Q_l = sum|b_i|^r, gain G_l = |A_l| / (nu^{1-s} Q_l^s), width"""
        b, phi = self.unpack(x)
        out = {}
        for k, l in enumerate(self.feats):
            sel = self.fidx == k
            A = abs(b[sel].sum()); L = np.abs(b[sel]).sum(); Q = np.sum(np.abs(b[sel]) ** self.r)
            out[l] = dict(A=A, L=L, Q=Q, gain=A / (self.nu ** (1 - self.s) * Q ** self.s) if Q > 0 else 0.0,
                          width=int(np.sum(np.abs(b[sel]) > 1e-6 * np.abs(b).max())))
        return out

    def to_net(self, x, b_exp=2.0):
        """explicit product network (U, W) with this class-C field (balanced neurons)."""
        b, phi = self.unpack(x)
        p, n = self.p, self.n; a = np.arange(p)
        U = np.zeros((self.m, n, p)); W = np.zeros((self.m, p))
        for i in range(self.m):
            l = self.freqs[i]; A = (2 ** n * abs(b[i])) ** (1.0 / self.nu)
            ph = phi[i] / 2.0                                         # xi = e^{2 i varphi}
            psi = np.angle(b[i]) + ph.sum()
            for t in range(n):
                U[i, t] = A * np.cos(2 * np.pi * l * a / p + ph[t])
            W[i] = A * np.cos(2 * np.pi * l * a / p + psi)
        return U, W


def if_seed(cc, y, jitter=0.0, rng=None):
    """IF blocks (nu neurons per feature) with amplitudes y_l; cc must have nu neurons per feature in order."""
    m, n, nu = cc.m, cc.n, cc.nu
    b = np.zeros(m, complex); phi = np.zeros((m, n))
    cnt = {}
    for i, l in enumerate(cc.freqs):
        k = cnt.get(l, 0); cnt[l] = k + 1
        b[i] = y[l - 1] / nu
        phi[i, :] = 2 * np.pi * k / nu
    x = cc.pack(b, phi)
    if jitter > 0:
        rng = rng or np.random.default_rng()
        x = x + jitter * rng.standard_normal(x.size) * np.concatenate([np.full(2 * m, np.abs(b).mean()), np.ones(m * n)])
    return x


# ----------------------------------------------------------------------------------- exact re-evaluation
def margin_mp(U, W, dps=40):
    """margin of a product network recomputed with mpmath at dps digits from the table (all x, all c)."""
    import mpmath as mp
    mp.mp.dps = dps
    m, n, p = U.shape
    Um = [[[mp.mpf(float(U[i, t, a])) for a in range(p)] for t in range(n)] for i in range(m)]
    Wm = [[mp.mpf(float(W[i, c])) for c in range(p)] for i in range(m)]
    h = None
    for x in itertools.product(range(p), repeat=n):
        prods = []
        for i in range(m):
            pr = mp.mpf(1)
            for t in range(n): pr *= Um[i][t][x[t]]
            prods.append(pr)
        f = [mp.fsum(prods[i] * Wm[i][c] for i in range(m)) for c in range(p)]
        y = sum(x) % p
        g = min(f[y] - f[c] for c in range(p) if c != y)
        if h is None or g < h: h = g
    cost_terms = [mp.fsum([Um[i][t][a] ** 2 for t in range(n) for a in range(p)] + [Wm[i][c] ** 2 for c in range(p)])
                  for i in range(m)]
    return h, cost_terms
