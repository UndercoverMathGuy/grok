"""Referee's own (independent) network library for product networks at n = 2, any odd p.

theta = list of neurons (u1, u2, w) in (R^p)^3;  f(x)[c] = sum_i w_i(c) u_i1(x1) u_i2(x2);  y(x) = x1 + x2 mod p.
gap g(x, d) = f(x)[y] - f(x)[y - d] (d = 1..p-1);  h = min g;  P = sum_i ||omega_i||^b;  gamma = h / P^s, s = 3/b.
Fields are computed from the TABLE (not from a neuron formula):
  Phi_l(x) = (2/p) rho^{l y(x)} sum_c f(x)[c] rho^{-l c};   phi_l(chi) = E_x Phi_l(x) rho^{-<chi,x>}.
  beta_l = conj(phi_l(0)),  beta_{l,T} = conj(phi_l(2 l 1_T)).
"""
import numpy as np


class Net:
    def __init__(self, p, U1, U2, W):
        self.p = p
        self.U1 = np.asarray(U1, float); self.U2 = np.asarray(U2, float); self.W = np.asarray(W, float)
        self.rho = np.exp(2j * np.pi / p)

    def table(self):
        return np.einsum('ia,ib,ic->abc', self.U1, self.U2, self.W)

    def gaps(self):
        p = self.p
        f = self.table()
        x1, x2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
        y = (x1 + x2) % p
        G = np.zeros((p, p, p - 1))
        for d in range(1, p):
            G[:, :, d - 1] = f[x1, x2, y] - f[x1, x2, (y - d) % p]
        return G

    def h(self):
        return self.gaps().min()

    def norms2(self):
        return np.sum(self.U1 ** 2, 1) + np.sum(self.U2 ** 2, 1) + np.sum(self.W ** 2, 1)

    def cost(self, b):
        return np.sum(self.norms2() ** (b / 2))

    def fields(self):
        """Phi[l] (p x p complex) for l = 1..(p-1)/2, computed from the table"""
        p = self.p; f = self.table(); rho = self.rho
        x1, x2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
        y = (x1 + x2) % p
        out = {}
        for l in range(1, (p - 1) // 2 + 1):
            fh = np.einsum('abc,c->ab', f, rho ** (-l * np.arange(p)))
            out[l] = (2 / p) * rho ** (l * y) * fh
        return out

    def phi(self):
        """phi[l][chi1, chi2] = E_x Phi_l(x) rho^{-<chi, x>}"""
        p = self.p
        out = {}
        for l, Ph in self.fields().items():
            out[l] = np.fft.fft2(Ph) / p ** 2        # fft2: sum_x Ph(x) rho^{-<chi,x>}
        return out

    def own(self, l=1):
        """(beta_l, beta_{l,{1}}, beta_{l,{2}}, beta_{l,{1,2}}) from the field coefficients"""
        p = self.p; ph = self.phi()[l]
        c = (2 * l) % p
        return np.conj(np.array([ph[0, 0], ph[c, 0], ph[0, c], ph[c, c]]))

    def neuron_beta(self, l=1):
        p = self.p
        uh1 = np.fft.fft(self.U1, axis=1); uh2 = np.fft.fft(self.U2, axis=1); wh = np.fft.fft(self.W, axis=1)
        return 2 * p ** -3.0 * np.conj(wh[:, l]) * uh1[:, l] * uh2[:, l]


def B_p(p, nu=3):
    return 2.0 * (2.0 * p * nu) ** (-nu / 2.0)


def gammaIF52(s):
    return 3 ** (1 - s) * B_p(5) * 2.5 * 2 ** (-s)


def gammaIF72(s):
    return 3 ** (1 - s) * B_p(7) * 3.5 * 3 ** (-s)


def pure_neuron(p, k, a1, a2, aw, ph1, ph2, phw):
    """neuron whose factors are pure frequency-k cosines"""
    x = np.arange(p)
    return (a1 * np.cos(2 * np.pi * k * x / p + ph1), a2 * np.cos(2 * np.pi * k * x / p + ph2),
            aw * np.cos(2 * np.pi * k * x / p + phw))


def cancel_feature(net, l):
    """append one pure frequency-l neuron that makes beta_l(net) = 0 exactly (up to float)"""
    p = net.p
    bl = net.own(l)[0]
    if abs(bl) < 1e-300:
        return net
    # pure neuron (a cos(2pi l x/p + ph)) has hat u(l) = (p a/2) e^{i ph};  beta = 2p^-3 conj(hw) hu1 hu2
    # choose a1 = a2 = aw = a, ph1 = ph2 = 0, phw = -(arg(-bl)) ... beta = 2 p^-3 (p a/2)^3 e^{-i phw}
    a = (abs(bl) / (2 * p ** -3.0 * (p / 2) ** 3)) ** (1 / 3)
    phw = -np.angle(-bl)
    u1, u2, w = pure_neuron(p, l, a, a, a, 0.0, 0.0, phw)
    out = Net(p, np.vstack([net.U1, u1]), np.vstack([net.U2, u2]), np.vstack([net.W, w]))
    return out
