# General product networks with per-neuron, per-slot allowed frequency subspaces (exact linear constraints).
#   neuron i, slot t in 0..n-1 (inputs) and t = n (output): vector = B(A_it) c_it, B = real orthonormal Fourier basis of
#   the allowed frequency set A_it (subset of {0} u F).  A neuron whose slot t has no frequency j has beta_ij = 0.
#   Objective: smooth-min of normalised gaps g/P_b^s (refereed GLOBAL Net for table, gaps, gradients), optional penalty
#   mu * sum_{j in pen} |beta_j|^2 / P^{2s}; SLP (sequential LP, l_inf trust region) minimax polish.
import numpy as np, itertools
from scipy.optimize import minimize, linprog
import hlib as H
GL = H.GL


def basis(p, freqs):
    a = np.arange(p); cols = []
    for k in sorted(freqs):
        if k == 0: cols.append(np.ones(p) / np.sqrt(p))
        else:
            cols.append(np.sqrt(2.0 / p) * np.cos(2 * np.pi * k * a / p))
            cols.append(np.sqrt(2.0 / p) * np.sin(2 * np.pi * k * a / p))
    return np.array(cols).T if cols else np.zeros((p, 0))


class GNet:
    def __init__(self, p, n, b, allowed):
        """allowed: list over neurons of list over slots (n inputs + output) of frequency sets"""
        self.p, self.n, self.b = p, n, b
        self.nu = n + 1; self.s = self.nu / b; self.m = len(allowed)
        self.B = [[basis(p, A) for A in al] for al in allowed]
        self.dims = [[Bt.shape[1] for Bt in Bi] for Bi in self.B]
        self.off = np.cumsum([0] + [d for di in self.dims for d in di])
        self.nx = self.off[-1]
        self.net0 = GL.Net(np.zeros((self.m, n, p)), np.zeros((self.m, p)), b)   # for index caches

    def to_UW(self, x):
        p, n = self.p, self.n
        U = np.zeros((self.m, n, p)); W = np.zeros((self.m, p)); k = 0
        for i in range(self.m):
            for t in range(n + 1):
                v = self.B[i][t] @ x[self.off[k]:self.off[k + 1]]; k += 1
                if t < n: U[i, t] = v
                else: W[i] = v
        return U, W

    def from_UW(self, U, W):
        x = np.zeros(self.nx); k = 0
        for i in range(self.m):
            for t in range(self.n + 1):
                v = U[i, t] if t < self.n else W[i]
                x[self.off[k]:self.off[k + 1]] = self.B[i][t].T @ v; k += 1
        return x

    def back(self, gU, gW):
        g = np.zeros(self.nx); k = 0
        for i in range(self.m):
            for t in range(self.n + 1):
                v = gU[i, t] if t < self.n else gW[i]
                g[self.off[k]:self.off[k + 1]] = self.B[i][t].T @ v; k += 1
        return g

    def net(self, x):
        U, W = self.to_UW(x); return GL.Net(U, W, self.b)

    def gamma(self, x):
        nt = self.net(x); return nt.margin() / nt.cost() ** self.s

    def betas(self, x):
        nt = self.net(x); return nt.clean()

    def soft(self, x, tau, pen=(), mu=0.0):
        U, W = self.to_UW(x); nt = GL.Net(U, W, self.b)
        c = nt.cost(); s = self.s
        g = nt.gaps() * c ** (-s)
        gmin = g.min(); z = np.exp(-(g - gmin) / tau); Z = z.sum(); lam = z / Z
        Fv = gmin - tau * np.log(Z)
        gU, gW = nt.grad_lin(lam); cU, cW = nt.grad_cost()
        Eg = (lam * g).sum()
        dU = c ** (-s) * gU - s * Eg / c * cU
        dW = c ** (-s) * gW - s * Eg / c * cW
        if mu > 0 and len(pen):
            p, n, nu = self.p, self.n, self.nu
            Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1); a = np.arange(p)
            P2 = 0.0; pU = np.zeros_like(U); pW = np.zeros_like(W)
            for j in pen:
                prodall = np.prod(Uh[:, :, j], axis=1)
                bi = 2 * p ** (-nu) * np.conj(Wh[:, j]) * prodall
                bj = bi.sum(); P2 += abs(bj) ** 2
                ej = np.exp(-2j * np.pi * j * a / p)
                for t in range(n):
                    oth = np.prod(np.delete(Uh[:, :, j], t, axis=1), axis=1)
                    coef = 2 * p ** (-nu) * np.conj(Wh[:, j]) * oth                  # d bj / d Uh_itj
                    pU[:, t, :] += 2 * np.real(np.conj(bj) * coef[:, None] * ej[None, :])
                coefW = 2 * p ** (-nu) * prodall
                pW += 2 * np.real(np.conj(bj) * coefW[:, None] * np.conj(ej)[None, :])
            c2 = c ** (2 * s)
            Fv -= mu * P2 / c2
            dU -= mu * (pU / c2 - 2 * s * P2 / c2 / c * cU)
            dW -= mu * (pW / c2 - 2 * s * P2 / c2 / c * cW)
        return Fv, self.back(dU, dW)

    def normalise(self, x):
        U, W = self.to_UW(x); nt = GL.Net(U, W, self.b)
        nt = GL.balance(nt)
        k = nt.cost() ** (-1.0 / self.b)
        return self.from_UW(nt.U * k, nt.W * k)

    def lbfgs(self, x, taus=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4), iters=1500, pen=(), mu=0.0, scale=None):
        x = self.normalise(x)
        sc = scale if scale else H.Bp(self.p, self.n) * self.nu ** (1 - self.s)
        for tau in taus:
            res = minimize(lambda z: tuple(-v for v in self.soft(z, tau * sc, pen, mu)), x, jac=True, method='L-BFGS-B',
                           options={'maxiter': iters, 'ftol': 1e-15, 'gtol': 1e-13})
            x = self.normalise(res.x)
        return x

    def jac_active(self, x, idx):
        """q_k = g_k / P^s and its Jacobian in x-coordinates, for flat indices idx into (N, p-1)."""
        U, W = self.to_UW(x); nt = GL.Net(U, W, self.b)
        p, n, m = self.p, self.n, self.m
        c = nt.cost(); s = self.s; cs = c ** s
        g = nt.gaps().ravel()
        X = np.indices((p,) * n).reshape(n, -1).T
        xs, ds = np.divmod(idx, p - 1); dl = ds + 1
        y = X[xs].sum(1) % p; ym = (y - dl) % p
        k = len(idx)
        prods = np.ones((m, k))
        facs = [U[:, t, X[xs, t]] for t in range(n)]                                # (m, k)
        for t in range(n): prods *= facs[t]
        dw = W[:, y] - W[:, ym]                                                      # (m, k)
        J = np.zeros((k, self.nx)); col = 0
        cU, cW = nt.grad_cost()
        dC = self.back(cU, cW)
        for i in range(m):
            for t in range(n + 1):
                Bt = self.B[i][t]; d = Bt.shape[1]
                if d == 0: continue
                if t < n:
                    oth = dw[i].copy()
                    for u in range(n):
                        if u != t: oth = oth * facs[u][i]
                    J[:, self.off[i * (n + 1) + t]:self.off[i * (n + 1) + t + 1]] = oth[:, None] * Bt[X[xs, t]]
                else:
                    J[:, self.off[i * (n + 1) + t]:self.off[i * (n + 1) + t + 1]] = prods[i][:, None] * (Bt[y] - Bt[ym])
        gk = g[idx]
        Jq = J / cs - (s * gk / cs / c)[:, None] * dC[None, :]
        return gk / cs, Jq

    def slp(self, x, iters=150, delta0=1e-2, nact=2500, tol=1e-13):
        x = self.normalise(x); D = delta0; best = self.gamma(x)
        for it in range(iters):
            nt = self.net(x); q = nt.gaps().ravel() / nt.cost() ** self.s
            order = np.argsort(q)
            thr = q[order[0]] + 500 * D * (abs(q[order[0]]) + 1e-12)
            k = min(nact, max(60, int(np.searchsorted(q[order], thr))))
            idx = order[:k]
            qk, J = self.jac_active(x, idx)
            nv = J.shape[1]
            cvec = np.zeros(nv + 1); cvec[-1] = -1.0
            A = np.hstack([-J, np.ones((k, 1))])
            res = linprog(cvec, A_ub=A, b_ub=qk, bounds=[(-D, D)] * nv + [(None, None)], method='highs')
            if res.status != 0: D *= 0.3; continue
            xn = self.normalise(x + res.x[:nv]); val = self.gamma(xn)
            if val > best + tol * abs(best): x, best = xn, val; D = min(D * 1.6, 0.3)
            else: D *= 0.3
            if D < 1e-11: break
        return x, best


def full_allowed(p, n):
    F = (p - 1) // 2
    return [set(range(0, F + 1)) for _ in range(n + 1)]
