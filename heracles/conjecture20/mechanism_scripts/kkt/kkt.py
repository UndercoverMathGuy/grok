"""KKT (Clarke stationarity) multipliers for slot-symmetric product networks on Z_p^n -> R^p.

theta = (U, W), U[i] in R^p (same input vector in every slot), W[i] in R^p.
g(N, d) = sum_i z_i(N) (W_i(y) - W_i(y-d)),  z_i(N) = prod_a U_i(a)^{N_a},  y = sum_a a N_a mod p.
Cost N(theta) = sum_i (n||U_i||^2 + ||W_i||^2)^{b/2}.
KKT: sum_k q_k grad g_k = lam grad N, q >= 0 on active set, sum q = 1  (q_k = total mass of class k).
"""
import numpy as np, itertools, math
from scipy.optimize import nnls, linprog


def counts(n, p):
    out = []
    def rec(prefix, left, k):
        if k == p - 1:
            out.append(prefix + (left,)); return
        for v in range(left + 1):
            rec(prefix + (v,), left - v, k + 1)
    rec((), n, 0)
    return np.array(out)


def classsize(N):
    n = N.sum(); r = math.factorial(n)
    for v in N: r //= math.factorial(v)
    return r


class Net:
    def __init__(self, U, W, n, b=2.0):
        self.U = np.array(U, float); self.W = np.array(W, float)
        self.m, self.p = self.U.shape; self.n = n; self.b = b; self.nu = n + 1
        self.Nc = counts(n, self.p)
        self.y = (self.Nc @ np.arange(self.p)) % self.p
        self.cs = np.array([classsize(N) for N in self.Nc], float)

    def z(self):
        # z[k,i] = prod_a U_i(a)^{N_a}
        Z = np.ones((len(self.Nc), self.m))
        for a in range(self.p):
            Z *= self.U[:, a][None, :] ** self.Nc[:, a][:, None]
        return Z

    def gaps(self):
        Z = self.z(); p = self.p
        G = np.zeros((len(self.Nc), p))  # G[k, d], d = 0..p-1 (d=0 column = 0)
        for d in range(1, p):
            ym = (self.y - d) % p
            G[:, d] = (Z * (self.W[:, self.y].T - self.W[:, ym].T)).sum(1)
        return G

    def neuron_gaps(self):
        Z = self.z(); p = self.p
        Gi = np.zeros((len(self.Nc), p, self.m))
        for d in range(1, p):
            ym = (self.y - d) % p
            Gi[:, d, :] = Z * (self.W[:, self.y].T - self.W[:, ym].T)
        return Gi

    def cost(self):
        return ((self.n * (self.U ** 2).sum(1) + (self.W ** 2).sum(1)) ** (self.b / 2)).sum()

    def grad_cost(self):
        s = self.n * (self.U ** 2).sum(1) + (self.W ** 2).sum(1)
        c = self.b / 2 * s ** (self.b / 2 - 1)
        return np.concatenate([(c[:, None] * 2 * self.n * self.U).ravel(), (c[:, None] * 2 * self.W).ravel()])

    def grad_gap(self, k, d):
        N = self.Nc[k]; y = self.y[k]; p = self.p; m = self.m
        z = np.prod(self.U ** N[None, :], axis=1)
        dw = self.W[:, y] - self.W[:, (y - d) % p]
        gU = np.zeros((m, p))
        for a in range(p):
            if N[a] == 0: continue
            Na = N.copy(); Na[a] -= 1
            gU[:, a] = N[a] * np.prod(self.U ** Na[None, :], axis=1) * dw
        gW = np.zeros((m, p))
        gW[:, y] += z; gW[:, (y - d) % p] -= z
        return np.concatenate([gU.ravel(), gW.ravel()])

    def margin(self):
        return self.gaps()[:, 1:].min()

    def gamma(self):
        return self.margin() / self.cost() ** (self.nu / self.b)

    def active(self, rtol=1e-7):
        G = self.gaps(); h = G[:, 1:].min()
        A = [(k, d) for k in range(len(self.Nc)) for d in range(1, self.p) if G[k, d] <= h + rtol * abs(h)]
        return A, h

    def kkt(self, rtol=1e-7, A=None):
        if A is None:
            A, h = self.active(rtol)
        else:
            h = self.margin()
        J = np.array([self.grad_gap(k, d) for (k, d) in A]).T  # params x |A|
        gN = self.grad_cost()
        qt, res = nnls(J, gN, maxiter=50 * J.shape[1])
        lam = 1 / qt.sum(); q = qt * lam
        return A, q, lam, res / np.linalg.norm(gN), h


def gamma_IF(p, n, b, Gam):
    nu = n + 1; s = nu / b
    Bp = 2 * (2 * p * nu) ** (-nu / 2)
    return nu ** (1 - s) * Bp * Gam
