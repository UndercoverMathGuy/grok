"""Library for the (p,n)=(5,2) single-feature compensation problem (task P52).

Conventions (paper / Note):
  rho = exp(2 pi i/p); neuron (u_1,u_2,w) in (R^p)^3; f(x)[c] = sum_i w_i(c) u_i1(x_1) u_i2(x_2); y(x) = x_1+x_2.
  g(x,d) = f(x)[y] - f(x)[y-d];  h = min g;  P = sum_i ||omega_i||^b;  gamma = h / P^s, s = 3/b.
  hat u(k) = sum_a u(a) rho^{-k a}.
  Fields Phi_j(x) = (2/p) sum_i hat w_i(j) prod_t u_it(x_t) rho^{j x_t};  g = sum_{j in F} Re(Phi_j(x) (1 - rho^{-j d})).
  beta_j = 2 p^{-3} sum_i conj(hat w_i(j)) prod_t hat u_it(j);  E_x Phi_j = conj(beta_j).
  Coefficient of Phi_j at character chi:  phi_j(chi) = 2 p^{-3} sum_i hat w_i(j) prod_t hat u_it(chi_t - j).
  Own characters of feature l: chi_{l,T} = 2 l 1_T,  phi_l(chi_{l,T}) = conj(beta_{l,T}).
  B_p = 2 (2 p nu)^{-nu/2};  single-neuron budget |beta| <= B_p ||omega||^3.
"""
import itertools
import numpy as np
from scipy.optimize import linprog

p = 5
n = 2
nu = 3
rho = np.exp(2j * np.pi / p)
F = [1, 2]
DELS = [1, 2, 3, 4]
B5 = 2.0 * (2.0 * p * nu) ** (-nu / 2.0)
KA = 1 - np.cos(2 * np.pi / p)      # kappa_1(+-1) = kappa_min
KB = 1 - np.cos(4 * np.pi / p)      # kappa_1(+-2)
X = np.array(list(itertools.product(range(p), repeat=n)))   # 25 x 2
TS = [(0,), (1,), (0, 1)]           # non-empty slot sets (0-based)


def e(k):
    return 1 - rho ** k


def Gamma(s):
    return max(KA, 2.5 * 2.0 ** (-s))


# ---------------------------------------------------------------- networks
def table(U, W):
    """U: m x n x p, W: m x p -> f[x1,x2,c]"""
    return np.einsum('ia,ib,ic->abc', U[:, 0, :], U[:, 1, :], W)


def gaps(U, W):
    f = table(U, W)
    G = np.zeros((p, p, p - 1))
    for x1 in range(p):
        for x2 in range(p):
            y = (x1 + x2) % p
            for di, d in enumerate(DELS):
                G[x1, x2, di] = f[x1, x2, y] - f[x1, x2, (y - d) % p]
    return G


def cost(U, W, b):
    nrm2 = np.sum(U ** 2, axis=(1, 2)) + np.sum(W ** 2, axis=1)
    return np.sum(nrm2 ** (b / 2))


def gamma(U, W, b):
    return gaps(U, W).min() / cost(U, W, b) ** (3.0 / b)


def betas(U, W):
    Uh = np.fft.fft(U, axis=2)   # hat u(k) = sum_a u(a) rho^{-ka}
    Wh = np.fft.fft(W, axis=1)
    out = {}
    for j in F:
        bi = 2 * p ** (-3) * np.conj(Wh[:, j]) * Uh[:, 0, j] * Uh[:, 1, j]
        out[j] = bi
    return out


def fields(U, W):
    """Phi_j(x) for j in F as arrays p x p"""
    Wh = np.fft.fft(W, axis=1)
    a = np.arange(p)
    out = {}
    for j in F:
        A = U[:, 0, :] * rho ** (j * a)[None, :]
        Bm = U[:, 1, :] * rho ** (j * a)[None, :]
        out[j] = (2.0 / p) * np.einsum('i,ia,ib->ab', Wh[:, j], A, Bm)
    return out


def fcoef(Phi, chi):
    """coefficient of Phi (p x p) at character chi: E_x Phi(x) rho^{-<chi,x>}"""
    ph = rho ** (-(chi[0] * X[:, 0] + chi[1] * X[:, 1]))
    return np.mean(Phi[X[:, 0], X[:, 1]] * ph)


# ---------------------------------------------------------------- free relaxation LP
_free_cache = {}


def _free_matrices(chis):
    key = tuple(chis)
    if key in _free_cache:
        return _free_cache[key]
    N = p * p
    nv = 4 * N + 1          # g(x,d) for d in DELS, then h
    # inequality: h - g(x,d) <= 0
    Aub = np.zeros((4 * N, nv))
    for di in range(4):
        for xi in range(N):
            Aub[di * N + xi, di * N + xi] = -1.0
            Aub[di * N + xi, -1] = 1.0
    rows = []
    # means: E_x g(.,d)  (4 real)
    for di in range(4):
        r = np.zeros(nv); r[di * N:(di + 1) * N] = 1.0 / N; rows.append(r)
    # own characters: sum_d rho^d ghat_d(chi) = value  (complex -> 2 real), ghat_d(chi) = E g rho^{-<chi,x>}
    for chi in chis:
        ph = rho ** (-(chi[0] * X[:, 0] + chi[1] * X[:, 1])) / N
        re = np.zeros(nv); im = np.zeros(nv)
        for di, d in enumerate(DELS):
            c = rho ** d * ph
            re[di * N:(di + 1) * N] = c.real
            im[di * N:(di + 1) * N] = c.imag
        rows.append(re); rows.append(im)
    Aeq = np.array(rows)
    _free_cache[key] = (Aub, Aeq, nv)
    return _free_cache[key]


OWN1 = [(2, 0), (0, 2), (2, 2)]   # chi_{1,T}, T = {1},{2},{1,2}


def h_free(beta1, bT, return_dual=False):
    """Free relaxation: Phi_1 own coefficients fixed (clean conj(beta1), sign-flips conj(bT[k]) at OWN1[k]),
    Phi_2 has mean 0, every other Fourier coefficient of Phi_1 and Phi_2 is free.  Returns max h."""
    Aub, Aeq, nv = _free_matrices(OWN1)
    m = [np.real(np.conj(beta1) * e(-d)) for d in DELS]   # E g(.,d) = Re(conj(beta1) c_1(d)), c_1(d) = 1 - rho^{-d}
    beq = list(m)
    for k in range(3):
        val = -(5.0 / 2.0) * np.conj(bT[k])
        beq += [val.real, val.imag]
    c = np.zeros(nv); c[-1] = -1.0
    res = linprog(c, A_ub=Aub, b_ub=np.zeros(Aub.shape[0]), A_eq=Aeq, b_eq=np.array(beq),
                  bounds=[(None, None)] * nv, method='highs')
    if res.status != 0:
        raise RuntimeError(res.message)
    if return_dual:
        return res.x[-1], res
    return res.x[-1]


def own_from_atoms(beta, xi):
    """atoms beta_i (complex), xi_i in T^2 -> (A, [B, C, D]) with B = sum beta xi_1, C = sum beta xi_2, D = sum beta xi_1 xi_2"""
    A = np.sum(beta)
    B = np.sum(beta * xi[:, 0]); C = np.sum(beta * xi[:, 1]); D = np.sum(beta * xi[:, 0] * xi[:, 1])
    return A, [B, C, D]
