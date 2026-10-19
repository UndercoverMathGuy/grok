# Referee's independent library (does not import klib).
# neuron i: factors F[i][t] in R^p, t = 0..n-1 inputs, t = n output.  Shapes: U (m,n,p), W (m,p).
import numpy as np, itertools
from scipy.optimize import linprog, minimize


def Bp(p, n):
    nu = n + 1
    return 2.0 * (2.0 * p * nu) ** (-nu / 2.0)


def kap(p, l, d):
    return 1 - np.cos(2 * np.pi * l * d / p)


def grid(p, n):
    X = np.array(list(itertools.product(range(p), repeat=n)))
    return X, X.sum(1) % p


def gaps(U, W, p, n, X=None, Y=None):
    """g[x_index, delta-1], plus per-neuron pieces."""
    if X is None: X, Y = grid(p, n)
    m = U.shape[0]
    Up = np.ones((m, len(X)))
    for t in range(n): Up *= U[:, t, X[:, t]]
    G = np.zeros((len(X), p - 1))
    for d in range(1, p):
        G[:, d - 1] = (Up * (W[:, Y] - W[:, (Y - d) % p])).sum(0)
    return G


def neuron_gap(U, W, i, p, n, X, Y):
    Up = np.ones(len(X))
    for t in range(n): Up *= U[i, t, X[:, t]]
    return np.stack([Up * (W[i, Y] - W[i, (Y - d) % p]) for d in range(1, p)], 1)


def cost(U, W, b):
    sq = (U ** 2).sum((1, 2)) + (W ** 2).sum(1)
    return np.sum(sq ** (b / 2)), sq


def dual_tensor(Q, p, n, X, Y):
    """M[x_1..x_n, c] = sum_delta Q[x,delta]([c=y]-[c=y-delta])."""
    M = np.zeros((len(X), p))
    r = np.arange(len(X))
    for d in range(1, p):
        M[r, Y] += Q[:, d - 1]
        M[r, (Y - d) % p] -= Q[:, d - 1]
    return M.reshape((p,) * (n + 1))


def contract(M, vecs, skip=None):
    T = M
    for t in range(M.ndim - 1, -1, -1):
        if t == skip: continue
        T = np.tensordot(T, vecs[t], axes=([t], [0]))
    return T


def inj_norm(M, starts=60, iters=400, seed=1, extra=()):
    rng = np.random.default_rng(seed); nu = M.ndim; p = M.shape[0]
    best = -np.inf
    inits = [list(e) for e in extra] + [[rng.standard_normal(p) for _ in range(nu)] for _ in range(starts)]
    for v in inits:
        v = [a / np.linalg.norm(a) for a in v]
        old = None
        for it in range(iters):
            for k in range(nu):
                g = contract(M, v, skip=k); ng = np.linalg.norm(g)
                if ng == 0: break
                v[k] = g / ng
            val = float(contract(M, v))
            if old is not None and abs(val - old) < 1e-15: break
            old = val
        best = max(best, val)
    return best


def atoms(U, W, p):
    """beta_il = 2 p^{-nu} conj(What_i(l)) prod_t Uhat_it(l), l = 1..F."""
    n = U.shape[1]; nu = n + 1; F = (p - 1) // 2
    Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1)
    ls = np.arange(1, F + 1)
    return 2 * float(p) ** (-nu) * np.conj(Wh[:, ls]) * np.prod(Uh[:, :, ls], axis=1)


def signflip(U, W, p, l, T):
    n = U.shape[1]; nu = n + 1
    Uh = np.fft.fft(U, axis=2); Wh = np.fft.fft(W, axis=1)
    pr = np.conj(Wh[:, l])
    for t in range(n):
        pr = pr * (Uh[:, t, (-l) % p] if t in T else Uh[:, t, l])
    return 2 * float(p) ** (-nu) * pr.sum()


def grad_lin(U, W, lam, p, n, X, Y):
    """gradient of sum lam[x,delta] g(x,delta) w.r.t. U and W (own einsum implementation)."""
    M = dual_tensor(lam, p, n, X, Y)
    m = U.shape[0]
    gU = np.zeros_like(U); gW = np.zeros_like(W)
    for i in range(m):
        vecs = [U[i, t] for t in range(n)] + [W[i]]
        for k in range(n + 1):
            g = contract(M, vecs, skip=k)
            if k < n: gU[i, k] = g
            else: gW[i] = g
    return gU, gW


def programme(p, s, S=None, starts=400, seed=0):
    """Gamma^{(S)}_s by multistart on the simplex of budget shares z (y = z^s), polished by Nelder-Mead-free SLSQP
    on max t s.t. sum y kappa(d) >= t, sum y^r = 1 (in z variables).  Returns Gamma, y."""
    F = (p - 1) // 2
    S = list(range(1, F + 1)) if S is None else list(S)
    K = np.array([[kap(p, l, d) for l in S] for d in range(1, (p - 1) // 2 + 1)])
    rng = np.random.default_rng(seed)
    best = (-1, None)
    k = len(S)
    def val(z):
        z = np.maximum(z, 0); z = z / z.sum()
        return (K @ (z ** s)).min()
    # exhaustive over supports via LP vertices for small k, else multistart
    for _ in range(starts):
        z0 = rng.dirichlet(np.ones(k) * rng.choice([0.2, 1.0, 3.0]))
        cons = [{'type': 'ineq', 'fun': lambda v: K @ (np.maximum(v[:-1], 0) ** s) - v[-1]},
                {'type': 'eq', 'fun': lambda v: np.sum(np.maximum(v[:-1], 0)) - 1}]
        res = minimize(lambda v: -v[-1], np.concatenate([z0, [val(z0)]]), constraints=cons,
                       bounds=[(0, 1)] * k + [(None, None)], method='SLSQP', options={'maxiter': 500, 'ftol': 1e-15})
        z = np.maximum(res.x[:-1], 0); z /= z.sum(); v = val(z)
        if v > best[0] + 1e-13: best = (v, z)
    # vertex enumeration refinement: active rows + support
    z = best[1]; spt = np.where(z > 1e-6)[0]
    y = np.zeros(F);
    for j, l in enumerate(S): y[l - 1] = z[j] ** s
    return best[0], y


def binding_and_dual(p, s, y, tol=1e-9):
    """binding offsets of y (delta in 1..p-1) and a symmetric programme dual mu (LP: K_j(mu) = Gamma y_j^{r-1} on spt y)."""
    F = (p - 1) // 2; r = 1 / s
    ds = np.arange(1, p)
    vals = np.array([sum(y[l - 1] * kap(p, l, d) for l in range(1, F + 1)) for d in ds])
    Gam = vals.min()
    B = ds[vals <= Gam + tol * max(1, Gam)]
    spt = [l for l in range(1, F + 1) if y[l - 1] > 1e-12]
    A = np.array([[kap(p, l, d) for d in B] for l in spt])
    rhs = np.array([Gam * y[l - 1] ** (r - 1) for l in spt])
    # LP feasibility: A mu = rhs, sum mu = 1, mu >= 0 (minimise L1 residual)
    nb = len(B); k = len(spt)
    c = np.concatenate([np.zeros(nb), np.ones(2 * k)])
    Aeq = np.hstack([A, np.eye(k), -np.eye(k)])
    Aeq = np.vstack([Aeq, np.concatenate([np.ones(nb), np.zeros(2 * k)])])
    beq = np.concatenate([rhs, [1]])
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * (nb + 2 * k), method='highs')
    mu_B = res.x[:nb]
    mu = np.zeros(p - 1); mu[B - 1] = mu_B
    mu = (mu + mu[::-1]) / 2          # symmetrise (delta <-> p - delta)
    resid = np.abs(A @ mu[B - 1] - rhs).max()
    return Gam, B, mu, resid


def Kl(p, mu, l):
    d = np.arange(1, p)
    return np.sum(mu * (1 - np.exp(-2j * np.pi * l * d / p)))
