"""N2MIX library: general product networks at n = 2, any odd prime p.

Conventions (paper / Note / p52):
  rho = exp(2 pi i/p); neuron omega = (u_1, u_2, w) in (R^p)^3; f(x)[c] = sum_i w_i(c) u_i1(x_1) u_i2(x_2); y = x_1 + x_2.
  g(x,d) = f(x)[y] - f(x)[y-d];  h = min g;  P = sum_i ||omega_i||^b;  gamma = h / P^s,  s = 3/b,  r = 1/s.
  hat u(k) = sum_a u(a) rho^{-k a}  (numpy fft convention).
  Fields Phi_j(x) = (2/p) rho^{j y} sum_c f(x)[c] rho^{-j c};  g = sum_{j in F} Re(Phi_j(x) c_j(d)),  c_j(d) = 1 - rho^{-j d}.
  Mode coefficient m_i(j; k) = (2/p^3) hat w_i(j) hat u_i1(k_1) hat u_i2(k_2)  (output j in F, input k in Z_p^2).
  Field coefficient phi_j(chi) = sum_i m_i(j; chi - j 1)  (coefficient of Phi_j at rho^{<chi,x>}).
  beta_j = conj(phi_j(0)),  beta_{j,T} = conj(phi_j(2 j 1_T)).
  B_p = 2 (6 p)^{-3/2};  e(omega) = B_p ||omega||^3;  sum_{j in F} |beta_ij|^{2/3} <= e_i^{2/3}.
"""
import itertools
import numpy as np


class Grid:
    def __init__(self, p):
        self.p = p
        self.rho = np.exp(2j * np.pi / p)
        self.F = list(range(1, (p - 1) // 2 + 1))
        self.DELS = list(range(1, p))
        self.X = np.array(list(itertools.product(range(p), repeat=2)))
        self.N = p * p
        self.Bp = 2.0 * (6.0 * p) ** (-1.5)
        self.TS = [(1, 0), (0, 1), (1, 1)]      # indicator vectors of T = {1}, {2}, {1,2}

    def cfac(self, j, d):
        return 1 - self.rho ** (-j * d)

    def kappa(self, j, d):
        return 1 - np.cos(2 * np.pi * j * d / self.p)

    def cls(self, k):
        """frequency class |k| in {0} u F of residue k"""
        k %= self.p
        return min(k, self.p - k)


# ------------------------------------------------------------------ networks
def table(G, U, W):
    return np.einsum('ia,ib,ic->abc', U[:, 0, :], U[:, 1, :], W)


def gaps(G, U, W):
    p = G.p
    f = table(G, U, W)
    out = np.zeros((p, p, p - 1))
    x1, x2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
    y = (x1 + x2) % p
    for di, d in enumerate(G.DELS):
        out[:, :, di] = f[x1, x2, y] - f[x1, x2, (y - d) % p]
    return out


def cost(U, W, b):
    nrm2 = np.sum(U ** 2, axis=(1, 2)) + np.sum(W ** 2, axis=1)
    return np.sum(nrm2 ** (b / 2))


def margin(G, U, W):
    return gaps(G, U, W).min()


def gamma(G, U, W, b):
    return margin(G, U, W) / cost(U, W, b) ** (3.0 / b)


def hats(U, W):
    return np.fft.fft(U, axis=2), np.fft.fft(W, axis=1)


def atoms(G, U, W):
    """beta_ij (m x |F|), indexed by j-1"""
    p = G.p
    Uh, Wh = hats(U, W)
    return np.array([2 * p ** (-3) * np.conj(Wh[:, j]) * Uh[:, 0, j] * Uh[:, 1, j] for j in G.F]).T


def modes(G, U, W):
    """m[i, jidx, k1, k2] = (2/p^3) hat w_i(j) hat u_i1(k1) hat u_i2(k2),  j in F"""
    p = G.p
    Uh, Wh = hats(U, W)
    out = np.zeros((U.shape[0], len(G.F), p, p), dtype=complex)
    for ji, j in enumerate(G.F):
        out[:, ji] = 2 * p ** (-3) * Wh[:, j][:, None, None] * Uh[:, 0, :][:, :, None] * Uh[:, 1, :][:, None, :]
    return out


def fieldcoef(G, U, W):
    """phi[jidx, chi1, chi2] = sum_i m_i(j; chi - j 1)"""
    p = G.p
    M = modes(G, U, W).sum(axis=0)
    phi = np.zeros_like(M)
    for ji, j in enumerate(G.F):
        # phi_j(chi) = M_j(chi - j)  ->  phi_j = roll of M_j by +j in each axis
        phi[ji] = np.roll(np.roll(M[ji], j, axis=0), j, axis=1)
    return phi


def fields_from_table(G, U, W):
    """Phi_j(x) computed directly from the logit table (for checking)."""
    p = G.p
    f = table(G, U, W)
    x1, x2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
    y = (x1 + x2) % p
    c = np.arange(p)
    out = []
    for j in G.F:
        fx = np.einsum('abc,c->ab', f, G.rho ** (-j * c))
        out.append((2.0 / p) * G.rho ** (j * y) * fx)
    return np.array(out)


def own_data(G, U, W):
    """beta_j and beta_{j,T} (T in TS order) read off the field coefficients."""
    phi = fieldcoef(G, U, W)
    p = G.p
    A = np.conj(phi[:, 0, 0])
    BT = np.zeros((len(G.F), 3), dtype=complex)
    for ji, j in enumerate(G.F):
        for ti, T in enumerate(G.TS):
            BT[ji, ti] = np.conj(phi[ji, (2 * j * T[0]) % p, (2 * j * T[1]) % p])
    return A, BT


# ------------------------------------------------------------------ classification of (output j, character chi)
def classify(G, L):
    """For live set L (subset of F): returns dict (jidx, chi1, chi2) -> 'clean', 'own', 'free', 'paid', 'zero'.
    own:  j in L, chi - j1 in {+-j}^2, chi != 0;  clean: chi = 0 (j in L);  'zero': j not in L, chi = 0 (beta_j = 0 assumed)
    free: j in L, chi - j1 in (+-L)^2, not own/clean;  paid: everything else."""
    p = G.p
    pmL = set()
    for l in L:
        pmL |= {l % p, (-l) % p}
    lab = {}
    for ji, j in enumerate(G.F):
        for c1 in range(p):
            for c2 in range(p):
                k1, k2 = (c1 - j) % p, (c2 - j) % p
                if c1 == 0 and c2 == 0:
                    lab[(ji, c1, c2)] = 'clean' if j in L else 'zero'
                elif j in L and k1 in (j % p, (-j) % p) and k2 in (j % p, (-j) % p):
                    lab[(ji, c1, c2)] = 'own'
                elif j in L and k1 in pmL and k2 in pmL:
                    lab[(ji, c1, c2)] = 'free'
                else:
                    lab[(ji, c1, c2)] = 'paid'
    return lab


def mode_masses(G, U, W, L):
    """per neuron: own mass, free mass, paid mass (all modes, output in F, inputs anywhere), and b_L = (sum_{j in L} |beta_ij|^{2/3})^{3/2}"""
    p = G.p
    m = modes(G, U, W)     # i, jidx, k1, k2
    lab = classify(G, L)
    own = np.zeros(m.shape[0]); free = np.zeros(m.shape[0]); paid = np.zeros(m.shape[0])
    for ji, j in enumerate(G.F):
        for k1 in range(p):
            for k2 in range(p):
                c1, c2 = (k1 + j) % p, (k2 + j) % p
                t = lab[(ji, c1, c2)]
                v = np.abs(m[:, ji, k1, k2]) ** 2
                if t in ('own', 'clean'):
                    own += v
                elif t == 'free':
                    free += v
                else:
                    paid += v
    bet = atoms(G, U, W)
    Lidx = [j - 1 for j in L]
    bL = np.sum(np.abs(bet[:, Lidx]) ** (2.0 / 3.0), axis=1) ** 1.5
    e = G.Bp * (np.sum(U ** 2, axis=(1, 2)) + np.sum(W ** 2, axis=1)) ** 1.5
    return own, free, paid, bL, e


def random_net(G, m, rng, kind='gauss', freqs=None, dc=0.0):
    p = G.p
    if kind == 'gauss':
        U = rng.normal(size=(m, 2, p)); W = rng.normal(size=(m, p))
        return U, W
    # frequency-limited: components at listed freqs (+ optional DC)
    a = np.arange(p)
    U = np.zeros((m, 2, p)); W = np.zeros((m, p))
    for i in range(m):
        for t in range(2):
            for k in freqs:
                U[i, t] += rng.normal() * np.cos(2 * np.pi * k * a / p + rng.uniform(0, 2 * np.pi))
            U[i, t] += dc * rng.normal()
        for k in freqs:
            W[i] += rng.normal() * np.cos(2 * np.pi * k * a / p + rng.uniform(0, 2 * np.pi))
    return U, W
