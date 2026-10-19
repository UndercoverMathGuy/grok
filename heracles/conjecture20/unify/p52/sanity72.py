"""Sanity checks for the (7,2) results (task P52).

 I1  delta-Fourier isolation identity, any network (arbitrary inputs incl. DC / several frequencies, outputs on all
     frequencies):  for a symmetric probability mu on Z_7^*, a feature j, a slot set T and w in C, the function
       q(x,d) = (1/49) [ mu_d + 2 Re( w rho^{j d} rho^{-<chi_{j,T}, x>} ) ],   chi_{j,T} = 2 j 1_T,
     satisfies  sum_{x,d} q g = sum_l K_l(mu) Re beta_l - 7 Re( w conj(beta_{j,T}) ),   K_l(mu) = sum_d mu_d kappa_l(d).
 I2  with |w| = min_d mu_d / 2 the function q is a probability, hence
       h <= sum_l K_l(mu) Re beta_l - (7/2) min_d mu_d |beta_{j,T}|          (checked on random and on optimised networks)
 I3  single-neuron budget  sum_j |beta_ij|^{2/3} <= e_i^{2/3},  e_i = B_7 ||omega_i||^3.
 I4  cost ledger  3^{1-s} B_7 P^s >= ( sum_j (F_j A_j)^r )^s,  F_j = max(1 - c_1 x_j, 3^{1-s} max(1, x_j)).
 I5  Gamma_s(7) = (7/2) 3^{-s}  (LP / random search over allocations).
 I6  optimised two-feature networks (output frequencies {0,1,2} only, so beta_3 = 0):  gamma/gamma_IF <= simple bound < 1.
"""
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

p = 7; n = 2
rho = np.exp(2j * np.pi / p)
FR = [1, 2, 3]
DELS = list(range(1, p))
X = np.array(list(itertools.product(range(p), repeat=2)))
B7 = 2 * (2 * p * 3) ** (-1.5)
rng = np.random.default_rng(2026)
kap = lambda l, d: 1 - np.cos(2 * np.pi * l * d / p)


def table(U, W):
    return np.einsum('ia,ib,ic->abc', U[:, 0, :], U[:, 1, :], W)


_I1, _I2 = np.meshgrid(np.arange(p), np.arange(p), indexing='ij')
_Y = (_I1 + _I2) % p
_D = np.array(DELS)


def gaps(U, W):
    f = table(U, W)
    fy = f[_I1, _I2, _Y]                                            # p x p
    fyd = f[_I1[..., None], _I2[..., None], (_Y[..., None] - _D[None, None, :]) % p]
    return fy[..., None] - fyd


def beta_neurons(U, W, j, T=()):
    """per-neuron beta_{ij,T}: input index -j on the slots in T"""
    Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1)
    out = 2 * p ** (-3) * np.conj(Wh[:, j])
    for t in range(2):
        out = out * (Uh[:, t, (-j) % p] if t in T else Uh[:, t, j])
    return out


def own(U, W):
    A = {j: beta_neurons(U, W, j).sum() for j in FR}
    BT = {j: [beta_neurons(U, W, j, T).sum() for T in [(0,), (1,), (0, 1)]] for j in FR}
    return A, BT


def cost(U, W, b):
    nrm2 = np.sum(U ** 2, axis=(1, 2)) + np.sum(W ** 2, axis=1)
    return np.sum(nrm2 ** (b / 2))


def sym_mu(v):
    v = np.abs(v); v = v / (2 * v.sum())
    return {d: v[min(d, p - d) - 1] for d in DELS}


TS = [(0,), (1,), (0, 1)]
# ------------------------------------------------------------------ I1, I2 on random networks
worst_id = 0; worst_ineq = -np.inf; minq = np.inf
for trial in range(300):
    m = rng.integers(1, 6)
    U = rng.normal(size=(m, 2, p)); W = rng.normal(size=(m, p))
    if trial % 3 == 0:            # add strong DC / mixed-frequency content
        U += rng.normal(size=(m, 2, 1)) * 2
    G = gaps(U, W); h = G.min()
    A, BT = own(U, W)
    mu = sym_mu(rng.uniform(0.05, 1, 3))
    K = {l: sum(mu[d] * kap(l, d) for d in DELS) for l in FR}
    for j in FR:
        for k, T in enumerate(TS):
            chi = np.array([2 * j if 0 in T else 0, 2 * j if 1 in T else 0]) % p
            w = rng.normal() + 1j * rng.normal()
            q = np.zeros((p, p, p - 1))
            for di, d in enumerate(DELS):
                ph = rho ** (j * d) * rho ** (-(chi[0] * X[:, 0] + chi[1] * X[:, 1]))
                q[X[:, 0], X[:, 1], di] = (mu[d] + 2 * np.real(w * ph)) / p ** 2
            lhs = np.sum(q * G)
            rhs = sum(K[l] * np.real(A[l]) for l in FR) - p * np.real(w * np.conj(BT[j][k]))
            worst_id = max(worst_id, abs(lhs - rhs) / (1 + abs(lhs)))
            # I2: aligned test of amplitude min mu / 2
            mm = min(mu.values())
            b_ = BT[j][k]
            if abs(b_) > 0:
                w2 = 0.5 * mm * b_ / abs(b_)
                q2 = np.zeros((p, p, p - 1))
                for di, d in enumerate(DELS):
                    ph = rho ** (j * d) * rho ** (-(chi[0] * X[:, 0] + chi[1] * X[:, 1]))
                    q2[X[:, 0], X[:, 1], di] = (mu[d] + 2 * np.real(w2 * ph)) / p ** 2
                minq = min(minq, q2.min())
                bound = sum(K[l] * np.real(A[l]) for l in FR) - 0.5 * p * mm * abs(b_)
                worst_ineq = max(worst_ineq, h - bound)
print(f"I1  max relative |sum q g - (sum_l K_l Re beta_l - 7 Re(w conj beta_jT))| = {worst_id:.2e}   (300 random networks, all j, T)")
print(f"I2  min q (aligned test, |w| = min mu/2) = {minq:.2e} (>= 0 up to rounding);  max[h - bound] = {worst_ineq:.4f} (must be <= 0)")

# ------------------------------------------------------------------ I3, I4
worst_b = -np.inf; worst_c = {s: -np.inf for s in (1.0, 1.1, 1.175, 1.5)}


def F_(x, s):
    c1 = 1 - 3 ** (1 - s)
    return max(1 - c1 * x, 3 ** (1 - s) * max(1.0, x))


def cost_ledger(U, W, s):
    r = 1 / s; b = 3 / s
    P = cost(U, W, b)
    A, BT = own(U, W)
    tot = 0
    for j in FR:
        a = abs(A[j])
        if a < 1e-14:
            continue
        x = max(abs(v) for v in BT[j]) / a
        tot += (F_(x, s) * a) ** r
    return 3 ** (1 - s) * B7 * P ** s, tot ** s


def single_atom(j):
    U = np.zeros((1, 2, p)); W = np.zeros((1, p)); a = np.arange(p)
    U[0, 0] = np.cos(2 * np.pi * j * a / p + rng.uniform(0, 7)); U[0, 1] = np.cos(2 * np.pi * j * a / p + rng.uniform(0, 7))
    W[0] = np.cos(2 * np.pi * j * a / p + rng.uniform(0, 7))
    return U, W


def if_block(j, scale=1.0):
    """3 neurons, frequency j, sign-flip coefficients cancel (xi_k = zeta^k in both slots)"""
    a = np.arange(p); U = np.zeros((3, 2, p)); W = np.zeros((3, p))
    for k in range(3):
        th = 2 * np.pi * k / 3
        U[k, 0] = np.cos(2 * np.pi * j * a / p + th); U[k, 1] = np.cos(2 * np.pi * j * a / p + th)
        W[k] = np.cos(2 * np.pi * j * a / p + 2 * th)
    return U * scale, W * scale


for trial in range(400):
    kind = trial % 4
    if kind == 0:
        m = rng.integers(1, 6); U = rng.normal(size=(m, 2, p)); W = rng.normal(size=(m, p))
    elif kind == 1:
        U, W = single_atom(rng.integers(1, 4))
    elif kind == 2:
        U, W = if_block(rng.integers(1, 4))
    else:
        Us, Ws = [], []
        for j in (1, 2):
            u, w = if_block(j, rng.uniform(0.3, 1.5)) if rng.uniform() < 0.5 else single_atom(j)
            Us.append(u); Ws.append(w)
        U = np.concatenate(Us); W = np.concatenate(Ws)
        U = U + 0.1 * rng.normal(size=U.shape); W = W + 0.1 * rng.normal(size=W.shape)
    nrm3 = (np.sum(U ** 2, axis=(1, 2)) + np.sum(W ** 2, axis=1)) ** 1.5
    for i in range(U.shape[0]):
        lhs = sum(abs(beta_neurons(U[i:i + 1], W[i:i + 1], j)[0]) ** (2 / 3) for j in FR)
        worst_b = max(worst_b, lhs - (B7 * nrm3[i]) ** (2 / 3))
    for s in worst_c:
        c, l = cost_ledger(U, W, s)
        worst_c[s] = max(worst_c[s], (l - c) / c)
print(f"I3  max[sum_j |beta_ij|^(2/3) - e_i^(2/3)] = {worst_b:.2e} (must be <= 0; IF/single-atom neurons give equality)")
print("I4  max relative [ledger - cost] by s: " + ", ".join(f"s={s}: {v:.2e}" for s, v in worst_c.items()) + "  (must be <= 0)")

# ------------------------------------------------------------------ I5
for s in (1.0, 1.175, 1.5):
    r = 1 / s
    Km = np.array([[kap(j, d) for j in FR] for d in (1, 2, 3)])
    best = 0
    for t in range(300):
        y = rng.dirichlet(np.ones(3) * 0.7)
        y = y / np.sum(y ** r) ** s
        best = max(best, np.min(Km @ y))
    best = max(best, np.min(Km @ (np.ones(3) * 3 ** (-s))))
    print(f"I5  s={s}: max_y min_d sum_j kappa_j(d) y_j (random + uniform) = {best:.6f};  (7/2) 3^-s = {3.5 * 3 ** (-s):.6f}")


# ------------------------------------------------------------------ I6: optimised two-feature networks
def simple_ratio(A1, A2, M1, M2, s):
    r = 1 / s
    M = max(M1, M2)
    c = np.array([2 * (kap(1, d) * A1 + kap(2, d) * A2) for d in (1, 2, 3)] + [-3.5 * M])
    Aub = np.array([[-1, 0, 0, 1], [0, -1, 0, 1], [0, 0, -1, 1]])
    res = linprog(c, A_ub=Aub, b_ub=np.zeros(3), A_eq=np.array([[2, 2, 2, 0]]), b_eq=[1], bounds=[(0, None)] * 4, method='highs')
    num = res.fun
    den = 3.5 * 3 ** (-s) * ((F_(M1 / A1, s) * A1) ** r + (F_(M2 / A2, s) * A2) ** r) ** s
    return num / den


def proj12(W):
    Wh = np.fft.fft(W, axis=1)
    Wh[:, 3] = 0; Wh[:, 4] = 0
    return np.real(np.fft.ifft(Wh, axis=1))


def neg_soft_gamma(v, m, s, tau=200.0):
    U = v[:m * 2 * p].reshape(m, 2, p); W = proj12(v[m * 2 * p:].reshape(m, p))
    G = gaps(U, W).ravel()
    P = cost(U, W, 3 / s)
    sm = -np.log(np.mean(np.exp(-tau * (G - G.min())))) / tau + G.min()
    return -sm / P ** s


worst6 = -np.inf; best_gamma = -np.inf
for s in (1.0, 1.1, 1.175):
    gIF = 3 ** (1 - s) * B7 * 3.5 * 3 ** (-s)
    for trial in range(6):
        m = 6
        # seed: perturbed IF blocks for features 1 and 2
        Us, Ws = [], []
        for j in (1, 2):
            u, w = if_block(j, rng.uniform(0.6, 1.2)); Us.append(u); Ws.append(w)
        U0 = np.concatenate(Us) + 0.3 * rng.normal(size=(6, 2, p)); W0 = np.concatenate(Ws) + 0.3 * rng.normal(size=(6, p))
        v0 = np.concatenate([U0.ravel(), W0.ravel()])
        res = minimize(neg_soft_gamma, v0, args=(m, s), method='L-BFGS-B', options={'maxiter': 300})
        v = res.x
        U = v[:m * 2 * p].reshape(m, 2, p); W = proj12(v[m * 2 * p:].reshape(m, p))
        h = gaps(U, W).min(); gam = h / cost(U, W, 3 / s) ** s
        A, BT = own(U, W)
        A1, A2 = abs(A[1]), abs(A[2]); M1 = max(abs(b) for b in BT[1]); M2 = max(abs(b) for b in BT[2])
        assert abs(A[3]) < 1e-12
        if min(A1, A2) < 1e-9:
            continue
        sr = simple_ratio(A1, A2, M1, M2, s)
        worst6 = max(worst6, gam / gIF - sr)
        best_gamma = max(best_gamma, gam / gIF)
        print(f"I6  s={s}: optimised 2-feature net  gamma/gamma_IF = {gam / gIF:.4f}  <=  simple bound {sr:.4f}   (A2/A1={A2 / A1:.3f}, x1={M1 / A1:.3f}, x2={M2 / A2:.3f})")
print(f"I6  max[gamma/gamma_IF - simple bound] = {worst6:.4f} (must be <= 0);  best two-feature gamma/gamma_IF found = {best_gamma:.4f}")
