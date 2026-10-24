# C_fieldmap library: class-C field maps on Z_p^n.
#   A class-C feature at frequency l is a finite atom system (C_i in C, xi_i in T^n); its field is
#       Phi_l(x) = sum_i C_i prod_t (1 + xi_it rho^{2 l x_t}),
#   E_x Phi_l = sum_i C_i = conj(beta_l).  Balanced DC-free neurons are budget-tight, so in B_p units the
#   cost of the feature is Q_l = sum_i |C_i|^r (r = 1/s) and gamma = B_p h / (sum_l Q_l)^s.
#   Gaps: g(x, d) = sum_l Re(Phi_l(x) (1 - rho^{-l d})),  h = min_{x, d} g.
import numpy as np, itertools, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'E_width'))
sys.path.insert(0, os.path.join(HERE, '..', 'C_joint'))

TAU = 2 * np.pi


def grid(p, n):
    return np.indices((p,) * n).reshape(n, -1).T          # (p^n, n)


def cvec(p, l):
    d = np.arange(1, p)
    return 1 - np.exp(-1j * TAU * l * d / p)                # c_l(d), d = 1..p-1


def kap(p, l, d):
    return 1 - np.cos(TAU * l * np.asarray(d) / p)


def kappa_mat(p):
    F = (p - 1) // 2
    d = np.arange(1, p)[:, None]; l = np.arange(1, F + 1)[None, :]
    return 1 - np.cos(TAU * d * l / p)                     # (p-1, F)


def field(p, X, l, C, xi):
    """Phi_l on the points X (N,n) for atoms C (m,), xi (m,n)."""
    z = np.exp(1j * TAU * 2 * l * X / p)                   # (N, n)
    out = np.zeros(X.shape[0], complex)
    for i in range(len(C)):
        out += C[i] * np.prod(1 + xi[i][None, :] * z, axis=1)
    return out


def cost(C, r):
    return float(np.sum(np.abs(C) ** r))


class CNet:
    """class-C network: feats = {l: (C, xi)}"""
    def __init__(self, p, n, feats, X=None):
        self.p, self.n, self.nu = p, n, n + 1
        self.feats = {l: (np.asarray(C, complex), np.asarray(xi, complex)) for l, (C, xi) in feats.items()}
        self.X = grid(p, n) if X is None else X
        self.S = sorted(self.feats)
        self.Phi = {l: field(p, self.X, l, *self.feats[l]) for l in self.S}

    def G(self, l):                                          # (N, p-1): Re(Phi_l c_l(d))
        return np.real(self.Phi[l][:, None] * cvec(self.p, l)[None, :])

    def gaps(self):
        return sum(self.G(l) for l in self.S)

    def h(self):
        return float(self.gaps().min())

    def Q(self, r):
        return {l: cost(self.feats[l][0], r) for l in self.S}

    def A(self):
        return {l: complex(np.sum(self.feats[l][0])) for l in self.S}

    def profile(self):
        """m_l(d) = min_x Re(Phi_l(x) c_l(d)) (one-sided directional floor), dict l -> (p-1,)"""
        return {l: self.G(l).min(0) for l in self.S}

    def hprof(self):
        pr = self.profile()
        return float(sum(pr[l] for l in self.S).min())


def M_alloc(p, y):
    """M(y) = min_d sum_l y_l kappa_l(d) for y indexed by l = 1..F (dict or array)."""
    F = (p - 1) // 2
    if isinstance(y, dict):
        v = np.zeros(F)
        for l, t in y.items(): v[l - 1] = t
        y = v
    return float((kappa_mat(p) @ y).min())


def cm_k(z, p, k):
    """directional cone margin cm_k(z) = Re z - cot(pi k/p) |Im z|  (k in F)"""
    return np.real(z) - np.abs(np.imag(z)) / np.tan(np.pi * k / p)


def kred(p, j):
    j = j % p
    return min(j, p - j)


def if_atoms(n, A):
    """KM2 block: nu equal atoms A/nu with xi_it = e^{2 pi i k t / nu}: field == A (all sign-flip coeffs 0)."""
    nu = n + 1
    C = np.full(nu, A / nu, complex)
    xi = np.array([[np.exp(1j * TAU * k / nu)] * n for k in range(nu)])
    # use xi_it = omega^k for every slot t: B_T = (A/nu) sum_k omega^{k |T|} = 0 for 1 <= |T| <= n
    return C, xi


def singleset_atoms(n, beta, phi):
    """n atoms beta/n with xi_it = e^{2 pi i k/n} and extra phase e^{i phi} in slot 1:
    field = beta + beta e^{i phi} rho^{2 l (x_1 + ... + x_n)}  (C_joint J4 block (ii))."""
    C = np.full(n, beta / n, complex)
    xi = np.array([[np.exp(1j * TAU * k / n) * (np.exp(1j * phi) if t == 0 else 1.0) for t in range(n)]
                   for k in range(n)])
    return C, xi


def price_n2(s):
    return 3.0 ** (1 - s)


def balance(U, W):
    """rescale the nu factors of every neuron to equal norms (table unchanged, cost minimal)"""
    U = U.copy(); W = W.copy(); m, n, p = U.shape
    for i in range(m):
        norms = [np.linalg.norm(U[i, t]) for t in range(n)] + [np.linalg.norm(W[i])]
        g = np.prod(norms) ** (1.0 / (n + 1))
        for t in range(n): U[i, t] *= g / norms[t]
        W[i] *= g / norms[n]
    return U, W
