"""Torch optimiser and analysis for general (mixed) product networks, full-table (small p^n) or slot-symmetric
(count vectors).  Used by the p = 5, 7 experiments of the SPLIT angle.

theta = (U, W): U[i, t, :] (full) or U[i, :] (slot-symmetric: same vector in every slot), W[i, :].
gamma = h / P^{nu/b},  P = sum_i (sum_t ||u_it||^2 + ||w_i||^2)^{b/2}.
"""
import itertools, math
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


# ---------------------------------------------------------------- programme Gamma_s(p) (small p, brute force)
def kappa_mat(p):
    F = (p - 1) // 2
    j = np.arange(1, F + 1); d = np.arange(1, p)
    return 1 - np.cos(2 * np.pi * np.outer(d, j) / p)          # rows delta = 1..p-1, cols j


def programme(p, s, grid=4001, seed=0):
    """Gamma_s(p) for p in {3,5,7,11} by SLSQP from many starts on every support (exact enough for checks)."""
    from scipy.optimize import minimize
    K = kappa_mat(p); F = K.shape[1]; r = 1 / s
    best = (-1, None)
    rng = np.random.default_rng(seed)
    for k in range(1, F + 1):
        for S in itertools.combinations(range(F), k):
            S = list(S)
            for rep in range(6 if k > 1 else 1):
                # maximise t s.t. K[:, S] y >= t, sum y^r <= 1  -> parametrise y on the sphere y = z^s / ||z||
                def neg(z):
                    z = np.abs(z) + 1e-15
                    y = z / (z ** r).sum() ** s
                    return -(K[:, S] @ y).min()
                z0 = rng.uniform(0.2, 1.0, len(S)) if rep else np.ones(len(S))
                res = minimize(neg, z0, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 20000})
                if -res.fun > best[0]:
                    z = np.abs(res.x); y = np.zeros(F); y[S] = z / (z ** r).sum() ** s
                    best = (-res.fun, y)
    return best


def Bp(p, n):
    nu = n + 1
    return 2 * (2 * p * nu) ** (-nu / 2)


def gamma_IF(p, n, b, Gam):
    nu = n + 1; s = nu / b
    return nu ** (1 - s) * Bp(p, n) * Gam


# ---------------------------------------------------------------- full-table networks
class Full:
    def __init__(self, p, n):
        self.p, self.n = p, n
        X = np.array(list(itertools.product(range(p), repeat=n)))
        self.X = torch.tensor(X); self.y = torch.tensor(X.sum(1) % p)
        self.rows = torch.arange(len(X))

    def gaps(self, U, W):
        Z = torch.ones(len(self.X), U.shape[0])
        for t in range(self.n):
            Z = Z * U[:, t, :][:, self.X[:, t]].T
        Fl = Z @ W                                               # inputs x p
        G = torch.stack([Fl[self.rows, self.y] - Fl[self.rows, (self.y - d) % self.p] for d in range(1, self.p)], 1)
        return G

    def cost(self, U, W, b):
        return (((U ** 2).sum((1, 2)) + (W ** 2).sum(1)) ** (b / 2)).sum()


class Sym:
    """Slot-symmetric networks: U[i, :] used in every slot.  Margin over count vectors."""
    def __init__(self, p, n):
        self.p, self.n = p, n
        Nc = []
        def rec(prefix, left, k):
            if k == p - 1:
                Nc.append(prefix + (left,)); return
            for v in range(left + 1):
                rec(prefix + (v,), left - v, k + 1)
        rec((), n, 0)
        self.Nc = torch.tensor(Nc); self.y = (self.Nc @ torch.arange(p)) % p
        self.rows = torch.arange(len(Nc))

    def gaps(self, U, W):
        # Z[N, i] = prod_a U[i,a]^{N_a}  (integer powers, sign-safe)
        Z = torch.ones(len(self.Nc), U.shape[0])
        for a in range(self.p):
            pw = torch.stack([U[:, a] ** k for k in range(self.n + 1)], 0)     # (n+1) x m
            Z = Z * pw[self.Nc[:, a]]
        Fl = Z @ W
        G = torch.stack([Fl[self.rows, self.y] - Fl[self.rows, (self.y - d) % self.p] for d in range(1, self.p)], 1)
        return G

    def cost(self, U, W, b):
        return ((self.n * (U ** 2).sum(1) + (W ** 2).sum(1)) ** (b / 2)).sum()


def optimise(model, U0, W0, b, stages=(0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4), iters=400, fixed_mask=None):
    """Maximise h / P^{nu/b} by smoothed-min L-BFGS with temperature annealing (relative to the scale of the gaps)."""
    nu = model.n + 1
    U = torch.tensor(U0, requires_grad=True); W = torch.tensor(W0, requires_grad=True)
    for st in stages:
        o = torch.optim.LBFGS([U, W], lr=1, max_iter=iters, line_search_fn='strong_wolfe',
                              tolerance_grad=1e-12, tolerance_change=1e-14)
        def closure():
            o.zero_grad()
            G = model.gaps(U, W)
            P = model.cost(U, W, b)
            scale = G.detach().abs().mean() + 1e-12
            tau = st * scale
            sm = -tau * torch.logsumexp(-G.reshape(-1) / tau, 0)
            loss = -sm / P ** (nu / b)
            loss.backward()
            if fixed_mask is not None:
                U.grad[fixed_mask] = 0
            return loss
        try:
            o.step(closure)
        except Exception:
            pass
    with torch.no_grad():
        G = model.gaps(U, W); P = model.cost(U, W, b)
        return U.detach().numpy().copy(), W.detach().numpy().copy(), float(G.min()), float(P)


# ---------------------------------------------------------------- analysis (numpy)
def proj(v, k, p):
    v = np.asarray(v, float)
    if k == 0:
        return np.repeat(v.mean(-1, keepdims=True), p, axis=-1)
    a = np.arange(p)
    c = np.cos(2 * np.pi * k * a / p); s = np.sin(2 * np.pi * k * a / p)
    return np.multiply.outer(v @ c, c) / (c @ c) + np.multiply.outer(v @ s, s) / (s @ s)


def split_full(U, W, p, live_only=False, tol=1e-9):
    """U: m x n x p.  Returns split pieces (and which are live)."""
    m, n, _ = U.shape
    Us, Ws, live = [], [], []
    for i in range(m):
        for k in range(1, (p - 1) // 2 + 1):
            u = proj(U[i], k, p); w = proj(W[i], k, p)
            nrm = min(np.linalg.norm(u, axis=1).min(), np.linalg.norm(w))
            lv = nrm > tol * max(1.0, np.linalg.norm(U[i]) + np.linalg.norm(W[i]))
            if live_only and not lv:
                continue
            Us.append(u); Ws.append(w); live.append(lv)
    return np.array(Us).reshape(-1, n, p), np.array(Ws).reshape(-1, p), np.array(live)


def split_sym(U, W, p, live_only=False, tol=1e-9):
    m, _ = U.shape
    Us, Ws, live = [], [], []
    for i in range(m):
        for k in range(1, (p - 1) // 2 + 1):
            u = proj(U[i], k, p); w = proj(W[i], k, p)
            lv = min(np.linalg.norm(u), np.linalg.norm(w)) > tol * max(1.0, np.linalg.norm(U[i]) + np.linalg.norm(W[i]))
            if live_only and not lv:
                continue
            Us.append(u); Ws.append(w); live.append(lv)
    return np.array(Us).reshape(-1, p), np.array(Ws).reshape(-1, p), np.array(live)


def own(U, W, p, n, sym=False):
    """clean amplitudes beta_j and atoms."""
    if sym:
        U = np.repeat(U[:, None, :], n, axis=1)
    Uh = np.fft.fft(U, axis=-1); Wh = np.fft.fft(W, axis=-1); nu = n + 1
    out = {}
    for j in range(1, (p - 1) // 2 + 1):
        at = 2 * p ** (-nu) * np.conj(Wh[:, j]) * np.prod(Uh[:, :, j], axis=1)
        out[j] = at
    return out


def energy_stats(U, W, p, sym=False, n=None):
    """Fractions of the (b=2) energy that are DC, dead (frequency present in some factor but not live), live."""
    if sym:
        U = np.repeat(U[:, None, :], n, axis=1)
    m = U.shape[0]
    tot = dc = dead = live = second = 0.0
    shared_neurons = 0
    for i in range(m):
        fac = [W[i]] + [U[i, t] for t in range(U.shape[1])]
        Ei = sum((f ** 2).sum() for f in fac)
        tot += Ei
        dc += sum((proj(f, 0, p) ** 2).sum() for f in fac)
        nl = 0; ek = []
        for k in range(1, (p - 1) // 2 + 1):
            e = [float((proj(f, k, p) ** 2).sum()) for f in fac]
            ek.append(sum(e))
            if min(e) > 1e-12 * max(1e-300, sum(e)) and min(e) > 1e-20:
                live += sum(e); nl += 1
            else:
                dead += sum(e)
        second += sum(ek) - max(ek)          # energy outside the neuron's dominant frequency
        shared_neurons += nl >= 2
    return dict(dc=dc / tot, dead=dead / tot, live=live / tot, shared_neurons=shared_neurons, m=m,
                secondary=second / tot)
