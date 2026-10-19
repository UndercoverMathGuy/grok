# Core library for the GLOBAL angle: arbitrary (not slot-symmetric) product networks on Z_p^n, small p, n.
#   neuron i: inputs U[i,t,:] in R^p (t=0..n-1), output W[i,:] in R^p
#   table f[x_1..x_n, c] = sum_i W[i,c] prod_t U[i,t,x_t];  label y(x) = sum_t x_t mod p
#   gap g(x,delta) = f[x,y] - f[x,y-delta], margin h = min g, cost P = sum_i (sum_t|U_it|^2+|W_i|^2)^{b/2}
#   gamma = h / P^s, s = nu/b, nu = n+1.   gamma_IF = nu^{1-s} B_p Gamma_s(p),  B_p = 2 (2 p nu)^{-nu/2}.
import numpy as np, itertools, math
from scipy.optimize import minimize, linprog

LET = 'abcdefgh'


def Bp(p, n):
    nu = n + 1
    return 2.0 * (2.0 * p * nu) ** (-nu / 2.0)


def kappa_mat(p):
    F = (p - 1) // 2
    d = np.arange(1, p)[:, None]; j = np.arange(1, F + 1)[None, :]
    return 1 - np.cos(2 * np.pi * d * j / p)          # rows delta = 1..p-1, cols j = 1..F


def programme(p, s, restarts=12, seed=0):
    """Gamma_s(p) by reweighted LP (concave l_r minimisation over the coverage polytope), random restarts.
    Returns (Gamma, y*, list of near-optimal distinct supports with values)."""
    r = 1.0 / s; F = (p - 1) // 2
    K = kappa_mat(p)[:F]                               # delta and -delta give the same rows
    rng = np.random.default_rng(seed); sols = []
    for t in range(restarts):
        w = rng.uniform(0.2, 1.8, F) if t else np.ones(F)
        y = None
        for it in range(200):
            res = linprog(w, A_ub=-K, b_ub=-np.ones(F), bounds=(0, None), method='highs')
            y = res.x
            wn = (y + 1e-10) ** (r - 1)
            if np.allclose(wn, w, rtol=1e-9): break
            w = wn
        y = y * (y > 1e-10)
        Q = np.sum(y[y > 0] ** r)
        sols.append((Q ** (-s), y / Q ** s))
    sols.sort(key=lambda z: -z[0])
    G, yb = sols[0]
    return G, yb, sols


def exhaustive_programme(p, s, kmax=None):
    """Exact Gamma_s(p) for small p: enumerate supports, optimise each by reweighted LP from many starts.
    Small p only (F <= 8 or so)."""
    F = (p - 1) // 2; r = 1.0 / s
    K = kappa_mat(p)[:F]
    best = {}
    kmax = kmax or F
    for k in range(1, kmax + 1):
        bk = (-1, None, None)
        for S in itertools.combinations(range(F), k):
            Ks = K[:, S]
            for t in range(6):
                w = np.ones(k) if t == 0 else np.random.default_rng(t).uniform(0.3, 1.7, k)
                ok = True
                for it in range(300):
                    res = linprog(w, A_ub=-Ks, b_ub=-np.ones(F), bounds=(0, None), method='highs')
                    if res.status != 0: ok = False; break
                    v = res.x; wn = (v + 1e-12) ** (r - 1)
                    if np.allclose(wn, w, rtol=1e-10): break
                    w = wn
                if not ok: continue
                if np.any(v < 1e-9): continue                     # support strictly smaller: counted elsewhere
                val = np.sum(v ** r) ** (-s)
                if val > bk[0]:
                    bk = (val, tuple(j + 1 for j in S), v / np.sum(v ** r) ** s)
        best[k] = bk
    return best


class Net:
    def __init__(self, U, W, b=2.0):
        self.U = np.array(U, float); self.W = np.array(W, float)
        self.m, self.n, self.p = self.U.shape
        self.b = b; self.nu = self.n + 1; self.s = self.nu / b
        p, n = self.p, self.n
        idx = np.indices((p,) * n).reshape(n, -1).T
        self.Y = idx.sum(1) % p                                  # labels of flattened inputs
        self.N = p ** n
        self.ein_tab = ','.join('i' + LET[t] for t in range(n)) + ',i' + LET[n] + '->' + LET[:n + 1]

    # ---------- table, gaps, margin, cost
    def table(self):
        return np.einsum(self.ein_tab, *[self.U[:, t] for t in range(self.n)], self.W, optimize=True)

    def gaps_from_table(self, f):
        p = self.p; ff = f.reshape(self.N, p); r = np.arange(self.N)
        top = ff[r, self.Y]
        return np.stack([top - ff[r, (self.Y - d) % p] for d in range(1, p)], 1)   # (N, p-1)

    def gaps(self):
        return self.gaps_from_table(self.table())

    def margin(self):
        return self.gaps().min()

    def cost(self):
        S = (self.U ** 2).sum((1, 2)) + (self.W ** 2).sum(1)
        return np.sum(S ** (self.b / 2))

    def gamma(self):
        return self.margin() / self.cost() ** self.s

    # ---------- Fourier data
    def fhat(self):
        return np.fft.fftn(self.table())

    def clean(self):
        p, n, nu = self.p, self.n, self.nu; F = (p - 1) // 2; fh = self.fhat()
        return np.array([2 * p ** (-nu) * fh[(j,) * n + ((-j) % p,)] for j in range(1, F + 1)])

    def signflip(self):
        """dict (j, T) -> beta_{j,T}, T a tuple of 0/1 of length n (1 = flipped)."""
        p, n, nu = self.p, self.n, self.nu; F = (p - 1) // 2; fh = self.fhat(); out = {}
        for j in range(1, F + 1):
            for T in itertools.product([0, 1], repeat=n):
                k = tuple(((-j) if T[t] else j) % p for t in range(n)) + ((-j) % p,)
                out[(j, T)] = 2 * p ** (-nu) * fh[k]
        return out

    def atoms(self):
        """beta_{ij} = 2 p^{-nu} conj(what_i(j)) prod_t uhat_it(j), shape (m, F)."""
        p, nu = self.p, self.nu; F = (p - 1) // 2
        Uh = np.fft.fft(self.U, axis=2); Wh = np.fft.fft(self.W, axis=1)
        js = np.arange(1, F + 1)
        return 2 * p ** (-nu) * np.conj(Wh[:, js]) * np.prod(Uh[:, :, js], axis=1)

    def cheapness(self):
        """per feature: |beta_j|, Q_j = sum_i |beta_ij|^r, IF price a_j = nu^{1-s} Q_j^s, ratio |beta_j|/a_j"""
        r = 1 / self.s; A = np.abs(self.clean()); at = self.atoms()
        Q = (np.abs(at) ** r).sum(0); a = self.nu ** (1 - self.s) * Q ** self.s
        return A, Q, a

    # ---------- gradients
    def grad_lin(self, lam):
        """gradient of L = sum_{x,delta} lam[x,delta] g(x,delta) w.r.t. (U, W).  lam shape (N, p-1)."""
        p, n = self.p, self.n
        Lam = np.zeros((self.N, p)); r = np.arange(self.N)
        tot = lam.sum(1)
        np.add.at(Lam, (r, self.Y), tot)
        for d in range(1, p):
            np.add.at(Lam, (r, (self.Y - d) % p), -lam[:, d - 1])
        Lam = Lam.reshape((p,) * (n + 1))
        fac = [self.U[:, t] for t in range(n)] + [self.W]
        grads = []
        for k in range(n + 1):
            others = [fac[q] for q in range(n + 1) if q != k]
            subs = ','.join('i' + LET[q] for q in range(n + 1) if q != k)
            grads.append(np.einsum(subs + ',' + LET[:n + 1] + '->i' + LET[k], *others, Lam, optimize=True))
        gU = np.stack(grads[:n], 1); gW = grads[n]
        return gU, gW

    def grad_cost(self):
        S = (self.U ** 2).sum((1, 2)) + (self.W ** 2).sum(1)
        c = self.b * S ** (self.b / 2 - 1)
        return c[:, None, None] * self.U, c[:, None] * self.W


def pack(U, W): return np.concatenate([U.ravel(), W.ravel()])


def unpack(x, m, n, p):
    U = x[:m * n * p].reshape(m, n, p); W = x[m * n * p:].reshape(m, p); return U, W


def softgamma(x, m, n, p, b, tau, out_mask=None):
    """scale-invariant smooth objective: softmin_tau of the gaps of the cost-normalised network.
    out_mask: optional (p,) projector applied to outputs (e.g. restrict to one frequency)."""
    U, W = unpack(x, m, n, p)
    if out_mask is not None: W = W @ out_mask
    net = Net(U, W, b); c = net.cost(); s = net.s
    g = net.gaps() * c ** (-s)
    gmin = g.min(); z = np.exp(-(g - gmin) / tau); Z = z.sum(); lam = z / Z
    F = gmin - tau * np.log(Z)
    gU, gW = net.grad_lin(lam)
    cU, cW = net.grad_cost()
    Eg = (lam * g).sum()
    dU = c ** (-s) * gU - s * Eg / c * cU
    dW = c ** (-s) * gW - s * Eg / c * cW
    if out_mask is not None: dW = dW @ out_mask.T
    return F, pack(dU, dW)


def optimise(U, W, b, taus=(3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6), iters=4000, out_mask=None, scale_tau=None):
    m, n, p = U.shape
    x = pack(U, W)
    for tau in taus:
        t = tau * (scale_tau if scale_tau else Bp(p, n) * (n + 1) ** (1 - (n + 1) / b))
        res = minimize(lambda z: tuple(-v for v in softgamma(z, m, n, p, b, t, out_mask)), x, jac=True,
                       method='L-BFGS-B', options={'maxiter': iters, 'ftol': 1e-15, 'gtol': 1e-12})
        x = res.x
    U, W = unpack(x, m, n, p)
    if out_mask is not None: W = W @ out_mask
    net = Net(U, W, b); c = net.cost()
    sc = c ** (-1.0 / b)
    return Net(U * sc, W * sc, b)


def if_block(p, n, j, lam=1.0, phase0=0.0):
    """Paper's nu-block at frequency j: u^{(l)}_t(a) = lam cos(2pi j a/p + alpha_l), w^{(l)}(c) = lam cos(2pi j c/p + n alpha_l)."""
    nu = n + 1; a = np.arange(p)
    U = []; W = []
    for l in range(nu):
        al = np.pi * l / nu + phase0
        U.append([lam * np.cos(2 * np.pi * j * a / p + al) for t in range(n)])
        W.append(lam * np.cos(2 * np.pi * j * a / p + n * al))
    return np.array(U), np.array(W)


def if_network(p, n, y, b=2.0):
    """IF network with amplitudes y_j (j=1..F) via nu-blocks, lam_j^nu = 2^n y_j / nu."""
    nu = n + 1; Us = []; Ws = []
    for j, yj in enumerate(y, start=1):
        if yj <= 0: continue
        lam = (2 ** n * yj / nu) ** (1.0 / nu)
        U, W = if_block(p, n, j, lam); Us.append(U); Ws.append(W)
    return Net(np.concatenate(Us), np.concatenate(Ws), b)


def gamma_if(p, n, b, Gam):
    nu = n + 1; s = nu / b
    return nu ** (1 - s) * Bp(p, n) * Gam


def freq_proj(p, k):
    """orthogonal projector (p x p) onto the frequency-k plane (k=0: constants)."""
    a = np.arange(p)
    if k == 0: return np.full((p, p), 1.0 / p)
    c = np.cos(2 * np.pi * k * a / p); s_ = np.sin(2 * np.pi * k * a / p)
    return np.outer(c, c) / (c @ c) + np.outer(s_, s_) / (s_ @ s_)


def split(net):
    """(S2) frequency split: one neuron per (i, k), k in F, all factors projected to frequency k."""
    p = net.p; F = (p - 1) // 2; Us = []; Ws = []
    for k in range(1, F + 1):
        Pk = freq_proj(p, k)
        Us.append(net.U @ Pk); Ws.append(net.W @ Pk)
    return Net(np.concatenate(Us), np.concatenate(Ws), net.b)


def out_split(net, k):
    """single-feature network: inputs of net, outputs projected to frequency k."""
    return Net(net.U.copy(), net.W @ freq_proj(net.p, k), net.b)


def prune(net, tol=1e-6):
    S = (net.U ** 2).sum((1, 2)) + (net.W ** 2).sum(1)
    keep = S > tol * S.max()
    return Net(net.U[keep], net.W[keep], net.b)


def balance(net):
    """rescale factors within each neuron to equal norms (table unchanged, cost non-increasing)."""
    U = net.U.copy(); W = net.W.copy(); nu = net.nu
    for i in range(net.m):
        norms = np.concatenate([np.linalg.norm(U[i], axis=1), [np.linalg.norm(W[i])]])
        if np.any(norms == 0): continue
        g = np.exp(np.mean(np.log(norms)))
        U[i] *= (g / norms[:-1])[:, None]; W[i] *= g / norms[-1]
    return Net(U, W, net.b)
