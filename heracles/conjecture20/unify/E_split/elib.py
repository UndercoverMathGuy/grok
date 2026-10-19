"""E_split helpers: temperature bookkeeping (cost shares, entropies, free energy) on top of splitlib (a copy of
unify/split/lib.py).  Conventions as in BRIEF.md: s = nu/b, c_i = ||omega_i||^b, w_i = c_i / sum c,
gamma = h / (sum c)^s, H = Shannon entropy of w, F = log gamma + (s-1) H."""
import itertools, math
import numpy as np
from splitlib import (Fset, proj, table, gaps_from_table, margin_table, margin, Bp, own_coeffs, kappa, if_block, split)


def norms2(U, W):
    return (U ** 2).sum((1, 2)) + (W ** 2).sum(1)


def costs(U, W, b):
    return norms2(U, W) ** (b / 2)


def shares(c):
    c = np.asarray(c, float)
    return c / c.sum()


def H(w):
    w = np.asarray(w, float); w = w[w > 0]
    return float(-(w * np.log(w)).sum())


def Hrenyi(w, a):
    w = np.asarray(w, float); w = w[w > 0]
    if abs(a - 1) < 1e-12:
        return H(w)
    return float(np.log((w ** a).sum()) / (1 - a))


def escort(w, r):
    w = np.asarray(w, float); e = w ** r
    return e / e.sum()


def stats(U, W, s, h=None):
    """returns dict with h, Z, gamma, H, F, V for temperature s (b = nu/s)."""
    m, n, p = U.shape
    nu = n + 1; b = nu / s
    if h is None:
        h = margin(U, W)
    c = costs(U, W, b); Z = c.sum(); w = c / Z
    lw = np.log(w)
    Hs = float(-(w * lw).sum())
    V = float((w * lw ** 2).sum() - Hs ** 2)
    g = h / Z ** s
    return dict(h=h, Z=Z, gamma=g, H=Hs, F=math.log(g) + (s - 1) * Hs, V=V, w=w, c=c)


def dilate(U, W, j):
    """relabelled copy u'(a) = u(j a), w'(c) = w(j c)  (j a unit mod p)."""
    m, n, p = U.shape
    idx = (j * np.arange(p)) % p
    return U[:, :, idx].copy(), W[:, idx].copy()


def scale_to_cost(U, W, b, target):
    """scale every neuron by the same factor so that the total cost becomes target."""
    c = costs(U, W, b).sum()
    lam = (target / c) ** (1 / b)
    return U * lam, W * lam


def if_network(p, n, y, phases=None):
    """IF network: one Prop-9 block per frequency j with clean coefficient proportional to y_j (y indexed by F)."""
    nu = n + 1
    Us, Ws = [], []
    for j, yj in zip(Fset(p), y):
        if yj <= 0:
            continue
        U, W = if_block(p, n, j, 1.0, 0.0 if phases is None else phases[j - 1])
        Us.append(U); Ws.append(W)
    U = np.concatenate(Us); W = np.concatenate(Ws)
    # rescale blocks so that beta_j proportional to y_j: block scale lam -> beta ~ lam^nu
    beta, _, _ = own_coeffs(U, W)
    out_U, out_W = [], []
    k = 0
    for j, yj in zip(Fset(p), y):
        if yj <= 0:
            continue
        lam = (yj / abs(beta[j])) ** (1 / nu)
        out_U.append(U[k:k + nu] * lam); out_W.append(W[k:k + nu] * lam); k += nu
    return np.concatenate(out_U), np.concatenate(out_W)


def piece_energies(U, W):
    """e[i, k] = energy of neuron i in frequency plane k (k = 0 is DC), summed over all n+1 factors."""
    m, n, p = U.shape
    F = Fset(p)
    e = np.zeros((m, len(F) + 1))
    for k in [0] + F:
        e[:, k] = (proj(U, k, p) ** 2).sum((1, 2)) + (proj(W, k, p) ** 2).sum(1)
    return e


def block_split(U, W, blocks):
    """blocks: list of disjoint lists of frequencies covering {0} u F.  Each neuron -> one piece per block:
    project every factor onto the sum of the block's planes."""
    m, n, p = U.shape
    Us, Ws = [], []
    for i in range(m):
        for B in blocks:
            u = sum(proj(U[i], k, p) for k in B)
            w = sum(proj(W[i], k, p) for k in B)
            Us.append(u); Ws.append(w)
    return np.array(Us).reshape(-1, n, p), np.array(Ws).reshape(-1, p)


def block_flip(U, W, blocks, S):
    """element of the block sign group: S[bi, t] in {+-1} multiplies the block-bi component of input slot t;
    the output block-bi component is multiplied by prod_t S[bi, t]."""
    m, n, p = U.shape
    U2 = U.copy(); W2 = W.copy()
    for bi, B in enumerate(blocks):
        PB_U = sum(proj(U, k, p) for k in B)
        PB_W = sum(proj(W, k, p) for k in B)
        for t in range(n):
            if S[bi, t] < 0:
                U2[:, t] -= 2 * PB_U[:, t]
        if np.prod(S[bi]) < 0:
            W2 -= 2 * PB_W
    return U2, W2


def gaps(U, W):
    m, n, p = U.shape
    return gaps_from_table(table(U, W), p, n)   # rows x (flattened), cols delta (col 0 unused)


def profile(U, W):
    """m(delta) = min_x g(x, delta), delta = 0..p-1 (entry 0 meaningless)."""
    G = gaps(U, W)
    return G.min(0)
