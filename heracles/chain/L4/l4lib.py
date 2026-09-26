"""Helpers for L4 (structure -> coverage defect). All sets are sets of frequencies k in Z_p (sign irrelevant)."""
import itertools, numpy as np
from scipy.optimize import minimize

def dist(x, p):
    """||x||_p = distance of x/p to the nearest integer, in [0, 1/2]."""
    r = np.mod(x, p) / p
    return np.minimum(r, 1 - r)

def cov_matrix(p, H):
    """cov[D, k] = 1 - cos(2 pi k D / p), D = 0..p-1."""
    D = np.arange(p)[:, None]; k = np.asarray(H)[None, :]
    return 1 - np.cos(2 * np.pi * k * D / p)

def min_coverage(p, H, w=None):
    """min over D != 0 of sum_k w_k cov_k(D) / sum_k w_k, and the minimising D."""
    w = np.ones(len(H)) if w is None else np.asarray(w, float)
    C = cov_matrix(p, H)[1:] @ w / w.sum()
    i = int(C.argmin())
    return float(C[i]), i + 1

def box_dirichlet(p, g, Q):
    """smallest-norm D != 0 with ||g_i D||_p < 1/Q_i for all i (exists when prod Q_i < p); brute force."""
    D = np.arange(1, p)
    ok = np.ones(p - 1, bool)
    for gi, Qi in zip(g, Q):
        ok &= dist(gi * D, p) < 1.0 / Qi
    return int(D[ok][0]) if ok.any() else None

def two_sided_depth(p, H, rng, starts=10):
    """UPPER estimate of min_z max_y |Re sum_k z_k e(-2 k y / p)| / sum |z_k| (explicit flat direction)."""
    K = len(H); ze = np.exp(-2j * np.pi * 2 * np.outer(H, np.arange(p)) / p)
    f = lambda v: np.abs(np.real((v[:K] + 1j * v[K:]) @ ze)).max() / np.abs(v[:K] + 1j * v[K:]).sum()
    st = [np.r_[np.cos(np.pi * np.arange(K) ** 2 / K), np.sin(np.pi * np.arange(K) ** 2 / K)], np.r_[np.ones(K), np.zeros(K)]]
    st += [rng.normal(size=2 * K) for _ in range(starts)]
    best = 9.0
    for s in st:
        r = minimize(f, s, method='Powell', options=dict(maxiter=20000, xtol=1e-9, ftol=1e-11))
        best = min(best, r.fun)
    return best

def rudin_shapiro(m):
    """+-1 Rudin-Shapiro coefficients of length 2^m."""
    P, Q = np.array([1.0]), np.array([1.0])
    for _ in range(m):
        P, Q = np.r_[P, Q], np.r_[P, -Q]
    return P

def greedy_dissociated(p, H):
    """a maximal {-1,0,1}-dissociated subset B of H (greedy). Every k in H outside B is a signed sum of elements of B."""
    B = []
    for k in H:
        cand = B + [k]
        sums = set()
        bad = False
        for eps in itertools.product((-1, 0, 1), repeat=len(cand)):
            if not any(eps):
                continue
            if sum(e * c for e, c in zip(eps, cand)) % p == 0:
                bad = True; break
        if not bad:
            B.append(k)
    return B

def representation(p, B, k):
    """a {-1,0,1} vector eps with sum eps_b b = k (mod p), shortest support first; None if none."""
    for j in range(1, len(B) + 1):
        for idx in itertools.combinations(range(len(B)), j):
            for sg in itertools.product((-1, 1), repeat=j):
                if sum(s * B[i] for s, i in zip(sg, idx)) % p in (k % p, (-k) % p):
                    e = [0] * len(B)
                    for s, i in zip(sg, idx):
                        e[i] = s
                    return e
    return None
