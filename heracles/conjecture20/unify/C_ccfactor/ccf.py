# C_ccfactor: offset-pair / charge-rule machinery for class-C networks (on top of cclib / E_width elib).
import sys, itertools
import numpy as np
sys.path.insert(0, '/home/user/grok/heracles/conjecture20/unify/C_ccfactor')
import cclib as C
E = C.E


def price(n, s, mode='if'):
    """pi used in the tests: n=2 -> 3^{1-s} (proved, Note Thm q1n2); n>=3 -> nu^{1-s} (lower bound for pi;
    using it makes every price-type inequality STRONGER, so a violation would refute the IF form, not pi)."""
    return (n + 1) ** (1 - s)


class FeatNet:
    """class-C network with feature data at exponent s."""

    def __init__(self, net, s, lab=None, tol=1e-9):
        self.net = net; self.s = s; p, n = net.p, net.n
        self.p, self.n, self.nu = p, n, n + 1; self.F = (p - 1) // 2; self.r = 1.0 / s
        if lab is None:
            ok, lab = C.neuron_freq(net, tol)
        self.lab = lab
        self.a = net.mass()
        self.B = C.Bp(p, n)
        self.beta = net.clean()
        self.Pl = np.array([(self.a[lab == l] ** self.r).sum() for l in range(1, self.F + 1)])
        self.P = self.Pl.sum()
        self.gl = C.feature_gaps(net)                     # (F, N, p-1)
        self.g = self.gl.sum(0)
        self.h = self.g.min()
        self.kap = E.kappa(p).T                           # (F, p-1): kappa_l(delta)
        self.Rbar_l = np.real(self.beta)[:, None] * self.kap   # Re beta_l kappa_l(delta)
        self.sf = net.signflip()
        self.V = np.array([sum(abs(self.sf[(l, T)]) ** 2 for T in itertools.product([0, 1], repeat=n) if any(T))
                           for l in range(1, self.F + 1)])
        self.feat = [l for l in range(1, self.F + 1) if self.Pl[l - 1] > 0]

    def neg(self, d):  # index of -delta for index d (delta = d+1)
        return self.p - 2 - d

    def gamma(self):
        return self.h / self.P ** self.s

    def MPs(self):
        return (self.kap.T @ self.Pl ** self.s).min()

    def binding(self, tol=1e-9):
        cov = self.kap.T @ self.Pl ** self.s
        return np.where(cov <= cov.min() * (1 + tol))[0]

    def cd_ratio(self, pi):
        return self.h / (pi * self.B * self.MPs())

    def rd_ratio(self, pi, G):
        return self.h / (pi * self.B * G * self.P ** self.s)

    def excess(self, pi):
        return np.real(self.beta) - pi * self.B * self.Pl ** self.s

    # ---- depths
    def G(self, gl=None):
        g = self.g if gl is None else gl
        return g.min(0)                                   # (p-1,)

    def joint_depth(self):
        G = self.G(); Rb = self.Rbar_l.sum(0)
        return Rb - 0.5 * (G + G[::-1])                   # index d <-> -delta is p-2-d = reversed

    def pair_depth(self):
        G = self.G(); Rb = self.Rbar_l.sum(0)
        return Rb - np.minimum(G, G[::-1])

    def iso_depth(self):
        Gl = self.gl.min(1)                               # (F, p-1)
        return self.Rbar_l - 0.5 * (Gl + Gl[:, ::-1])

    # ---- charge rules: return C[l, d] with sum_l C[l,d] = (G(d)+G(-d))/2
    def charge_tm(self, rtol=1e-9):
        g = self.g; G = g.min(0); out = np.zeros((self.F, self.p - 1))
        for d in range(self.p - 1):
            idx = np.where(g[:, d] <= G[d] + rtol * max(1.0, abs(G[d])))[0]
            out[:, d] = self.gl[:, idx, d].mean(1)
        return 0.5 * (out + out[:, ::-1])

    def charge_shapley(self):
        feats = self.feat; K = len(feats); Rb = self.Rbar_l
        def v(T):
            if not T: return np.zeros(self.p - 1)
            idx = [l - 1 for l in T]
            gT = self.gl[idx].sum(0); G = gT.min(0)
            return Rb[idx].sum(0) - 0.5 * (G + G[::-1])
        cache = {}
        for k in range(K + 1):
            for T in itertools.combinations(feats, k):
                cache[T] = v(list(T))
        phi = np.zeros((self.F, self.p - 1))
        from math import factorial
        for l in feats:
            others = [k for k in feats if k != l]
            for k in range(len(others) + 1):
                wgt = factorial(k) * factorial(K - k - 1) / factorial(K)
                for T in itertools.combinations(others, k):
                    TT = tuple(sorted(T + (l,)))
                    phi[l - 1] += wgt * (cache[TT] - cache[tuple(sorted(T))])
        return Rb - phi                                   # charged contribution = mean - Shapley depth

    # ---- pair cone moment (JCM)
    def jcm(self, d):
        """returns (lhs, rhs) of (Rbar - h_d)^2 >= Ibar^2 + sum_l kappa_l cos(phi_l) V_l at offset index d."""
        p = self.p; delta = d + 1
        ls = np.arange(1, self.F + 1); phi = 2 * np.pi * ls * delta / p
        Rb = (np.real(self.beta) * self.kap[:, d]).sum()
        Ib = (-np.imag(self.beta) * np.sin(phi)).sum()
        G = self.G(); hd = min(G[d], G[self.neg(d)])
        return (Rb - hd) ** 2, Ib ** 2 + (self.kap[:, d] * np.cos(phi) * self.V).sum(), Rb, hd


def if_feature(p, n, l, lam=1.0):
    U, W = E.if_block(p, n, l, lam)
    return U, W


def perturbed_if(p, n, y, rng, eps, extra=0):
    """IF network on spt(y) (paper blocks), with class-C perturbations of size eps (amplitudes and phases of every
    factor) and `extra` additional random neurons per feature (class C)."""
    nu = n + 1; a = np.arange(p); Us = []; Ws = []; lab = []
    for l, yl in enumerate(y, start=1):
        if yl <= 0: continue
        lam = (2 ** n * yl / nu) ** (1.0 / nu)
        U, W = E.if_block(p, n, l, lam)
        for i in range(U.shape[0]):
            for t in range(n):
                amp = 1 + eps * rng.normal(); ph = eps * rng.normal()
                U[i, t] = amp * _rot(U[i, t], l, p, ph)
            W[i] = (1 + eps * rng.normal()) * _rot(W[i], l, p, eps * rng.normal())
        Us.append(U); Ws.append(W); lab += [l] * U.shape[0]
        for k in range(extra):
            amp = eps * lam * rng.uniform(0.2, 1.0, n + 1); ph = rng.uniform(0, 2 * np.pi, n + 1)
            Us.append(np.array([[amp[t] * np.cos(2 * np.pi * l * a / p + ph[t]) for t in range(n)]]))
            Ws.append(np.array([amp[n] * np.cos(2 * np.pi * l * a / p + ph[n])])); lab.append(l)
    return E.Net(np.concatenate(Us), np.concatenate(Ws)), np.array(lab)


def _rot(v, l, p, ph):
    """rotate the frequency-l plane component of v by phase ph (v is in that plane)."""
    vh = np.fft.fft(v); out = np.zeros(p, complex)
    out[l] = vh[l] * np.exp(1j * ph); out[p - l] = vh[p - l] * np.exp(-1j * ph)
    return np.real(np.fft.ifft(out))
