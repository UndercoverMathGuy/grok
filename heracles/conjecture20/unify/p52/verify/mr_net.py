"""Math referee (mr): independent general product-network library (any odd p, any n, any width, DC and all
frequencies in every factor).  Written from the BRIEF's definitions only; does not import any p52 script.

f(x)[c] = sum_i w_i(c) prod_t u_it(x_t);  y(x) = sum x_t;  g(x,d) = f[y] - f[y-d];  h = min g.
uhat(k) = sum_a u(a) rho^{-ka}.  beta_{ij,T} = 2 p^{-nu} conj(what_i(j)) prod_{t in T} uhat_it(-j) prod_{t notin T} uhat_it(j).
Phi_l(x) = (2/p) rho^{l y(x)} sum_c f(x)[c] rho^{-lc};  phi_l(chi) = E_x Phi_l(x) rho^{-<chi,x>}.
Per-neuron modes m_i(l;k) = 2 p^{-nu} what_i(l) prod_t uhat_it(k_t), so that Phi_l(x) = sum_i sum_k m_i(l;k) rho^{<k + l 1, x>}.
"""
import itertools
import numpy as np


class Net:
    def __init__(self, p, n, U, W):
        self.p, self.n = p, n
        self.nu = n + 1
        self.U = np.asarray(U, float)     # m x n x p
        self.W = np.asarray(W, float)     # m x p
        self.m = self.W.shape[0]
        self.rho = np.exp(2j * np.pi / p)
        self.F = list(range(1, (p - 1) // 2 + 1))
        self.Uh = np.fft.fft(self.U, axis=2)
        self.Wh = np.fft.fft(self.W, axis=1)
        self._table = None

    # ------------------------------------------------------------ table, gaps, margin, cost
    def table(self):
        if self._table is None:
            p, n = self.p, self.n
            T = np.zeros((p,) * n + (p,))
            for i in range(self.m):
                prod = self.U[i, 0]
                for t in range(1, n):
                    prod = np.multiply.outer(prod, self.U[i, t])
                T += np.multiply.outer(prod, self.W[i])
            self._table = T
        return self._table

    def gaps(self):
        p, n = self.p, self.n
        f = self.table()
        idx = np.indices((p,) * n).reshape(n, -1).T          # all x
        y = idx.sum(axis=1) % p
        flat = f.reshape(-1, p)
        G = np.zeros((p ** n, p - 1))
        for d in range(1, p):
            G[:, d - 1] = flat[np.arange(p ** n), y] - flat[np.arange(p ** n), (y - d) % p]
        return idx, G

    def h(self):
        return self.gaps()[1].min()

    def norms2(self):
        return np.sum(self.U ** 2, axis=(1, 2)) + np.sum(self.W ** 2, axis=1)

    def cost(self, b):
        return np.sum(self.norms2() ** (b / 2))

    def Bp(self):
        return 2.0 * (2.0 * self.p * self.nu) ** (-self.nu / 2.0)

    def e(self):
        """per-neuron single-neuron budget e_i = B_p ||omega_i||^nu"""
        return self.Bp() * self.norms2() ** (self.nu / 2.0)

    # ------------------------------------------------------------ coefficients
    def beta_i(self, j, T=()):
        p, nu = self.p, self.nu
        out = 2 * p ** (-nu) * np.conj(self.Wh[:, j % p])
        for t in range(self.n):
            k = (-j) % p if t in T else j % p
            out = out * self.Uh[:, t, k]
        return out

    def beta(self, j, T=()):
        return self.beta_i(j, T).sum()

    def fields(self):
        p, n = self.p, self.n
        f = self.table()
        idx = np.indices((p,) * n).reshape(n, -1).T
        y = idx.sum(axis=1) % p
        flat = f.reshape(-1, p)
        out = {}
        for l in self.F:
            out[l] = (2.0 / p) * self.rho ** (l * y) * (flat @ (self.rho ** (-l * np.arange(p))))
        return idx, out

    def phi(self, l, chi):
        idx, Ph = self.fields()
        ph = self.rho ** (-(idx @ np.asarray(chi)))
        return np.mean(Ph[l] * ph)

    def modes_i(self, i):
        """dict (l, k-tuple) -> m_i(l;k) for l in F, k in Z_p^n"""
        p, nu, n = self.p, self.nu, self.n
        out = {}
        for l in self.F:
            for k in itertools.product(range(p), repeat=n):
                v = 2 * p ** (-nu) * self.Wh[i, l]
                for t in range(n):
                    v = v * self.Uh[i, t, k[t]]
                out[(l, k)] = v
        return out


def rand_net(rng, p, n, m, dc=1.0, scale=1.0):
    U = rng.normal(size=(m, n, p)) * scale + dc * rng.normal(size=(m, n, 1))
    W = rng.normal(size=(m, p)) * scale + dc * rng.normal(size=(m, 1))
    return U, W


def freq_vec(p, k, amp, ph):
    a = np.arange(p)
    return amp * np.cos(2 * np.pi * k * a / p + ph)


def kill_beta(p, n, U, W, j):
    """append one neuron supported on frequency j in every factor whose atom cancels beta_j exactly."""
    net = Net(p, n, U, W)
    b = net.beta(j)
    if abs(b) < 1e-300:
        return U, W
    # neuron with u_t = cos(2 pi j a/p + ph_t), w = c cos(2 pi j a/p + phw): uhat(j) = (p/2) e^{i ph},  what(j) = (p/2) c e^{i phw}
    # its beta = 2 p^{-nu} (p/2)^{nu} c e^{i(sum ph_t - phw)}; want = -b
    nu = n + 1
    c = abs(b) / (2 * p ** (-nu) * (p / 2.0) ** nu)
    phw = -np.angle(-b)
    newU = np.stack([freq_vec(p, j, 1.0, 0.0) for _ in range(n)])[None]
    newW = freq_vec(p, j, c, phw)[None]
    U2 = np.concatenate([U, newU]); W2 = np.concatenate([W, newW])
    assert abs(Net(p, n, U2, W2).beta(j)) < 1e-12 * (1 + abs(b))
    return U2, W2
