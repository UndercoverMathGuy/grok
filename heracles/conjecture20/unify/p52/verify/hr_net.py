"""Hostile-referee (hr) independent network library.  Written from BRIEF definitions only; imports nothing from p52.

Network: neurons (u_1..u_n, w) in (R^p)^{n+1};  f(x)[c] = sum_i w_i(c) prod_t u_it(x_t);  y(x) = sum_t x_t mod p.
g(x,d) = f(x)[y] - f(x)[y-d];  h = min_{x, d != 0} g.  Cost P_b = sum_i ||omega_i||^b;  gamma = h / P_b^{nu/b}.
hat u(k) = sum_a u(a) rho^{-k a}.
beta_{ij}   = 2 p^{-nu} conj(what_i(j)) prod_t uhat_it(j)
beta_{ij,T} = 2 p^{-nu} conj(what_i(j)) prod_{t in T} uhat_it(-j) prod_{t notin T} uhat_it(j)
Phi_l(x) = (2/p) rho^{l y(x)} sum_c f(x)[c] rho^{-l c}  (computed from the TABLE, not from modes);
phi_l(chi) = E_x Phi_l(x) rho^{-<chi, x>}.
"""
import itertools
import numpy as np


class Net:
    def __init__(self, p, n, U, W):
        self.p, self.n, self.nu = p, n, n + 1
        self.U = np.array(U, float)          # m x n x p
        self.W = np.array(W, float)          # m x p
        self.m = self.W.shape[0]
        self.rho = np.exp(2j * np.pi / p)
        self.F = list(range(1, (p - 1) // 2 + 1))
        self.Uh = np.fft.fft(self.U, axis=2)  # numpy fft: sum_a u(a) e^{-2 pi i k a/p}  == uhat(k)
        self.Wh = np.fft.fft(self.W, axis=1)
        self.X = np.array(list(itertools.product(range(p), repeat=n)))
        self.Y = self.X.sum(axis=1) % p
        self._f = None

    def table(self):
        if self._f is None:
            f = np.zeros((len(self.X), self.p))
            for i in range(self.m):
                prod = np.ones(len(self.X))
                for t in range(self.n):
                    prod = prod * self.U[i, t][self.X[:, t]]
                f += prod[:, None] * self.W[i][None, :]
            self._f = f
        return self._f

    def gaps(self):
        f = self.table(); p = self.p; N = len(self.X)
        G = np.zeros((N, p - 1))
        for d in range(1, p):
            G[:, d - 1] = f[np.arange(N), self.Y] - f[np.arange(N), (self.Y - d) % p]
        return G

    def h(self):
        return self.gaps().min()

    def norms2(self):
        return np.sum(self.U ** 2, axis=(1, 2)) + np.sum(self.W ** 2, axis=1)

    def cost(self, b):
        return np.sum(self.norms2() ** (b / 2))

    def gamma(self, b):
        return self.h() / self.cost(b) ** (self.nu / b)

    def beta_i(self, j, T=()):
        c = 2 * self.p ** (-self.nu) * np.conj(self.Wh[:, j % self.p])
        for t in range(self.n):
            c = c * (self.Uh[:, t, (-j) % self.p] if t in T else self.Uh[:, t, j % self.p])
        return c

    def beta(self, j, T=()):
        return self.beta_i(j, T).sum()

    def fields(self):
        """Phi_l(x) for l in F, from the table"""
        f = self.table(); p = self.p
        out = {}
        cs = np.arange(p)
        for l in self.F:
            out[l] = (2.0 / p) * self.rho ** (l * self.Y) * (f @ (self.rho ** (-l * cs)))
        return out

    def field_coeffs(self):
        """phi_l(chi) for all chi in Z_p^n, as dict l -> array indexed by chi (p,..,p)"""
        Phi = self.fields(); p, n = self.p, self.n
        out = {}
        for l, v in Phi.items():
            arr = v.reshape((p,) * n)
            # phi(chi) = (1/p^n) sum_x Phi(x) rho^{-<chi,x>} = fftn(arr)/p^n  (numpy fft uses e^{-2pi i k x/p})
            out[l] = np.fft.fftn(arr) / p ** n
        return out

    def modes(self, i):
        """per-neuron modes m_i(l;k) = 2 p^{-nu} what_i(l) prod_t uhat_it(k_t), dict l -> array over k"""
        p, n = self.p, self.n
        out = {}
        for l in self.F:
            arr = 2 * p ** (-self.nu) * self.Wh[i, l]
            for t in range(n):
                arr = np.multiply.outer(arr, self.Uh[i, t]) if t > 0 else arr * self.Uh[i, t]
            out[l] = np.asarray(arr)
        return out


def B_p(p, nu):
    return 2.0 * (2.0 * p * nu) ** (-nu / 2.0)


def kappa(p, j, d):
    return 1 - np.cos(2 * np.pi * j * d / p)


def rand_net(rng, p, n, m, dc=True, scale=1.0):
    U = rng.normal(size=(m, n, p)) * scale
    W = rng.normal(size=(m, p)) * scale
    if not dc:
        U -= U.mean(axis=2, keepdims=True); W -= W.mean(axis=1, keepdims=True)
    return Net(p, n, U, W)


def cos_vec(p, k, phase=0.0, amp=1.0):
    a = np.arange(p)
    return amp * np.cos(2 * np.pi * k * a / p + phase)
