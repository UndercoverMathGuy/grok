"""L6 tools: structural parameters of a frequency set H, aim depths, coverage (restricted share programme).

Conventions (chain/common.py): p odd prime, h = (p-1)/2, frequencies k in {1..h}; junk of frequency k sits at the
doubled frequency 2k (characters x -> w^{-2kx}). cov_k(D) = 1 - cos(2 pi k D / p).
"""
import itertools, json, os
import numpy as np
from scipy.optimize import minimize, linprog

HERE = os.path.dirname(os.path.abspath(__file__))


def fold(k, p):
    k %= p
    return min(k, p - k)


def cov_matrix(p):
    h = (p - 1) // 2
    D = np.arange(1, h + 1)[:, None]
    k = np.arange(1, h + 1)[None, :]
    return 1 - np.cos(2 * np.pi * k * D / p)


# ---------------------------------------------------------------- structural parameters of H

def signed_sums(p, S, coeffs=(-1, 0, 1)):
    return [sum(e * 2 * k for e, k in zip(eps, S)) % p for eps in itertools.product(coeffs, repeat=len(S))]


def distinct_sums(p, S):
    """Rudin-dissociate: all 3^K signed sums of the doubled frequencies distinct (no relation with coeffs in -2..2)."""
    seen = {0}
    for k in S:
        g = 2 * k % p
        new = set()
        for s in seen:
            for t in (s, (s + g) % p, (s - g) % p):
                if t in new:
                    return False
                new.add(t)
        seen = new
    return True


def dissociated_dimension(p, H):
    """max size of a distinct-sums subset of H (exact DFS; fine for |H| <= 16)."""
    H = list(H)
    best = [0]

    def dfs(i, chosen, sums):
        if len(chosen) + (len(H) - i) <= best[0]:
            return
        if i == len(H):
            best[0] = max(best[0], len(chosen))
            return
        g = 2 * H[i] % p
        new = set()
        ok = True
        for s in sums:
            for t in (s, (s + g) % p, (s - g) % p):
                if t in new:
                    ok = False
                    break
                new.add(t)
            if not ok:
                break
        if ok:
            dfs(i + 1, chosen + [H[i]], new)
        dfs(i + 1, chosen, sums)

    dfs(0, [], {0})
    return best[0]


def shortest_relation(p, H, jmax=6):
    """min number of nonzero coefficients c in {-2,-1,1,2} with sum c_k (2k) = 0 mod p; returns jmax+1 if none <= jmax."""
    H = list(H)
    g = np.array([2 * k % p for k in H])
    for j in range(1, min(jmax, len(H)) + 1):
        for sub in itertools.combinations(range(len(H)), j):
            gs = g[list(sub)]
            C = np.array(list(itertools.product((-2, -1, 1, 2), repeat=j)))
            if ((C @ gs) % p == 0).any():
                return j
    return jmax + 1


def energy(p, H):
    """normalised additive energy of Lambda = {+-2k}: E / |Lambda|^3 (1/|Lambda|-ish for Sidon-like, ~1 for progressions)."""
    L = sorted({2 * k % p for k in H} | {(-2 * k) % p for k in H})
    cnt = {}
    for a in L:
        for b in L:
            s = (a + b) % p
            cnt[s] = cnt.get(s, 0) + 1
    E = sum(v * v for v in cnt.values())
    return E / len(L) ** 3


# ---------------------------------------------------------------- aim depths (search = explicit directions = UPPER estimates)

def _chars(p, H):
    return np.exp(-2j * np.pi * 2 * np.outer(H, np.arange(p)) / p)


def depth_est(p, H, two_sided, rng, starts=16):
    K = len(H)
    ze = _chars(p, H)
    if two_sided:
        f = lambda v: np.abs(np.real((v[:K] + 1j * v[K:]) @ ze)).max() / (np.abs(v[:K] + 1j * v[K:]).sum() + 1e-300)
    else:
        f = lambda v: np.real((v[:K] + 1j * v[K:]) @ ze).max() / (np.abs(v[:K] + 1j * v[K:]).sum() + 1e-300)
    I = np.arange(K)
    st = [np.r_[np.cos(np.pi * I ** 2 / K), np.sin(np.pi * I ** 2 / K)], np.r_[np.ones(K), np.zeros(K)],
          np.r_[-np.ones(K), np.zeros(K)], np.r_[(-1.0) ** I, np.zeros(K)]]
    st += [rng.normal(size=2 * K) for _ in range(starts)]
    best = 9.0
    for s in st:
        best = min(best, f(s))
        r = minimize(f, s, method='Powell', options=dict(maxiter=20000, xtol=1e-9, ftol=1e-11))
        best = min(best, r.fun)
    return best


def depth_grid_lower(p, H, L, two_sided):
    """Rigorous-in-exact-arithmetic method of HOLES_aiming (fp64 here): depth >= cos(pi/L) * min over the phase grid
    (theta_1 = 0 by the cyclic symmetry) of the largest reachable radius. Also returns that min (an upper estimate)."""
    K = len(H)
    chi = np.exp(2j * np.pi * 2 * np.outer(H, np.arange(p)) / p)
    if two_sided:
        chi = np.hstack([chi, -chi])
    n = chi.shape[1]
    Ab = np.vstack([chi.real, chi.imag, np.ones(n)])
    tmin = 9.0
    for idx in itertools.product(range(L), repeat=K - 1):
        t = np.r_[0.0, 2 * np.pi * np.array(idx) / L]
        r = linprog(np.r_[np.zeros(n), -1], A_eq=np.column_stack([Ab, -np.r_[np.cos(t), np.sin(t), 0.0]]),
                    b_eq=np.r_[np.zeros(2 * K), 1], bounds=[(0, None)] * n + [(None, None)], method='highs')
        tmin = min(tmin, -r.fun)
    return np.cos(np.pi / L) * tmin, tmin


def transfer(As, Ao, n=60):
    """flip-enhanced transfer lower bound c(A*, A_s) on the l2-polyball depth (VERIFY.md 'New lemma')."""
    def betas(a):
        m = dict(zip('12F', a))
        asg = [('1', 'F', '2', As, Ao), ('1', '2', 'F', As, Ao), ('2', 'F', '1', As, Ao), ('2', '1', 'F', As, Ao),
               ('F', '1', '2', Ao, Ao), ('F', '2', '1', Ao, Ao)]
        out = []
        for o, f_, c, Ai, Aout in asg:
            x, y, w = m[f_], m[o], m[c]
            out.append(max(Ai * x + Aout * abs(y - Ai * w), Aout * (y + Ai * w) - Ai * x))
        return out
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            v = np.array([i, j, n - i - j], float) / n
            if v.any():
                pts.append(v / np.linalg.norm(v))
    B = np.array([betas(v) for v in pts])
    k = B.shape[1]
    r = linprog(np.r_[np.zeros(k), -1], A_ub=np.c_[-B, np.ones(len(B))], b_ub=np.zeros(len(B)),
                A_eq=[np.r_[np.ones(k), 0]], b_eq=[1], bounds=[(0, None)] * k + [(None, None)])
    return -r.fun


def enlarged_depth_est(p, H, rng, extra=10, flips=((1, 1), (-1, 1), (1, -1))):
    """UPPER estimate of the l2-polyball depth of the flip-enlarged body (3 patterns), by search (flip_ball.depth2)."""
    K = len(H)
    x = np.arange(p)
    ze = _chars(p, H)
    ab = (x[:, None] + x[None, :]) % p

    def h(zc):
        best = -np.inf
        for s1, s2 in flips:
            f1 = s1 * np.real(zc[0] @ ze)
            f2 = s2 * np.real(zc[1] @ ze)
            f3 = s1 * s2 * np.real(zc[2] @ ze)
            best = max(best, (f1[:, None] + f2[None, :] + f3[ab]).max())
        return best

    def obj(v):
        zc = (v[:3 * K] + 1j * v[3 * K:]).reshape(3, K)
        return h(zc) / (np.linalg.norm(zc, axis=0).sum() + 1e-300)

    I = np.arange(K)
    starts = []
    for T in range(3):
        for ph in (np.zeros(K), np.pi * np.ones(K), np.pi * (I % 2), rng.uniform(0, 2 * np.pi, K), np.pi * I ** 2 / K):
            zc = np.zeros((3, K), complex)
            zc[T] = np.exp(1j * ph)
            starts.append(np.r_[zc.real.ravel(), zc.imag.ravel()])
    starts += [rng.normal(size=6 * K) for _ in range(extra)]
    best = 9.0
    for s in starts:
        best = min(best, obj(s))
        r = minimize(obj, s + 1e-3 * rng.normal(size=6 * K), method='Powell',
                     options=dict(maxiter=30000, xtol=1e-8, ftol=1e-10))
        best = min(best, r.fun)
    return best


# ---------------------------------------------------------------- coverage: restricted share programme Gamma_H

def gamma_H(p, H, rng, starts=8, A=None):
    """max over shares u on H (sum u = 1) of min_D sum_k u_k^{3/2} cov_k(D). Multistart SLSQP: a feasible value
    (LOWER bound on Gamma_H, NUMERICAL)."""
    if A is None:
        A = cov_matrix(p)
    H = list(H)
    AH = A[:, [k - 1 for k in H]]
    K = len(H)
    best, bu = -1.0, None
    for s in range(starts):
        u0 = rng.dirichlet(np.ones(K)) if s else np.ones(K) / K
        t0 = (AH @ u0 ** 1.5).min()
        cons = [{'type': 'eq', 'fun': lambda v: v[:K].sum() - 1, 'jac': lambda v: np.r_[np.ones(K), 0.0]},
                {'type': 'ineq', 'fun': lambda v: AH @ np.clip(v[:K], 0, None) ** 1.5 - v[K],
                 'jac': lambda v: np.c_[AH * 1.5 * np.sqrt(np.clip(v[:K], 0, None))[None, :], -np.ones(AH.shape[0])]}]
        r = minimize(lambda v: -v[K], np.r_[u0, t0], jac=lambda v: np.r_[np.zeros(K), -1.0], constraints=cons,
                     bounds=[(0, 1)] * K + [(None, None)], method='SLSQP', options=dict(maxiter=500, ftol=1e-13))
        u = np.clip(r.x[:K], 0, None)
        u /= u.sum()
        val = (AH @ u ** 1.5).min()
        if val > best:
            best, bu = val, u
    return best, bu


def gamma_star(p, rng, restarts=300, steps=1500, seeds=()):
    """LOWER bound on Gamma* = Gamma_{3/2}(p): soft-min ascent over all frequencies (many restarts), SLSQP polish on the
    support, then add/remove/swap local search on the best supports. Cached in L6/gamma_star.json."""
    path = os.path.join(HERE, 'gamma_star.json')
    cache = json.load(open(path)) if os.path.exists(path) else {}
    if str(p) in cache:
        return cache[str(p)]['Gamma'], cache[str(p)]['support']
    A = cov_matrix(p)
    h = A.shape[0]
    found = []
    for S in seeds:
        found.append((gamma_H(p, S, rng, starts=4, A=A)[0], sorted(S)))
    for t in range(restarts):
        z = rng.normal(0, 2, h)
        m = np.zeros(h); v2 = np.zeros(h)
        for s in range(steps):
            tau = 0.05 * (0.01 ** (s / steps))
            b = np.exp(z - z.max()); b /= b.sum()
            val = A @ b ** 1.5
            wts = np.exp(-(val - val.min()) / tau); wts /= wts.sum()
            gb = A.T @ wts * 1.5 * np.sqrt(b)
            gz = b * (gb - (gb * b).sum())
            m = 0.9 * m + 0.1 * gz; v2 = 0.999 * v2 + 0.001 * gz ** 2
            z += 0.05 * m / (np.sqrt(v2) + 1e-12)
        b = np.exp(z - z.max()); b /= b.sum()
        supp = [i + 1 for i in range(h) if b[i] > 1e-3]
        found.append(((A[:, [k - 1 for k in supp]] @ (b[[k - 1 for k in supp]] / b[[k - 1 for k in supp]].sum()) ** 1.5).min(), supp))
    found.sort(key=lambda x: -x[0])
    best, bsupp = -1, None
    seen = set()
    for _, S in found[:6]:
        S = sorted(S)
        val = gamma_H(p, S, rng, starts=4, A=A)[0]
        improved = True
        while improved:
            improved = False
            cands = [sorted(set(S) | {k}) for k in range(1, h + 1) if k not in S] + [[x for x in S if x != k] for k in S if len(S) > 1]
            for C in cands:
                key = tuple(C)
                if key in seen:
                    continue
                seen.add(key)
                v = gamma_H(p, C, rng, starts=2, A=A)[0]
                if v > val + 1e-9:
                    val, S, improved = v, C, True
                    break
        if val > best:
            best, bsupp = val, S
    cache[str(p)] = dict(Gamma=best, support=bsupp)
    json.dump(cache, open(path, 'w'), indent=1)
    return best, bsupp


# ---------------------------------------------------------------- polar-LP depth estimates (more reliable than primal search)

def _U(theta, Ab, K):
    """LP value U(theta) = max Re sum conj(e^{i theta_k}) z_k over the polar body; depth in direction theta = 1/U."""
    c = -np.r_[np.cos(theta), np.sin(theta)]          # maximise Re z . cos + Im z . sin
    r = linprog(c, A_ub=Ab, b_ub=np.ones(Ab.shape[0]), bounds=[(None, None)] * (2 * K), method='highs')
    return -r.fun if r.status == 0 else np.inf


def depth_polar(p, H, two_sided, rng, starts=30, iters=300):
    """UPPER estimate of the K-mode depth (one- or two-sided) = 1 / max_theta U(theta), by local search over phases,
    started from structured directions and random ones. Returns (estimate, worst theta)."""
    K = len(H)
    chi = np.exp(-2j * np.pi * 2 * np.outer(H, np.arange(p)) / p)          # [K, p]; f_z(x) = Re sum z_k chi_k(x)
    rows = np.c_[chi.real.T, -chi.imag.T]                                  # Re(z chi) = Re z Re chi - Im z Im chi
    Ab = np.vstack([rows, -rows]) if two_sided else rows
    I = np.arange(K)
    th_starts = [np.zeros(K), np.pi * (I % 2), np.pi * I ** 2 / K, 2 * np.pi * I / K]
    th_starts += [rng.uniform(0, 2 * np.pi, K) for _ in range(starts)]
    bestU, bth = -np.inf, None
    for th0 in th_starts:
        f = lambda t: -_U(t, Ab, K)
        r = minimize(f, th0, method='Powell', options=dict(maxfev=iters, xtol=1e-6, ftol=1e-9))
        if -r.fun > bestU:
            bestU, bth = -r.fun, r.x
    return 1.0 / bestU, bth


def depth_osc(p, H, rng, starts=10, iters=150):
    """UPPER estimate of the oscillation depth A_osc = min_z (max f_z - min f_z) / sum|z_k|, f_z(x) = Re sum z_k w^{-2kx}
    (the polydisc inradius of the difference body M - M). Polar LP: maximise Re<e^{i theta}, z> s.t. f_z(x) <= Mx,
    f_z(x) >= mn, Mx - mn <= 1."""
    K = len(H)
    chi = np.exp(-2j * np.pi * 2 * np.outer(H, np.arange(p)) / p)
    rows = np.c_[chi.real.T, -chi.imag.T]                                   # f_z(x) = rows @ [Re z, Im z]
    nvar = 2 * K + 2
    Aub = np.vstack([np.c_[rows, -np.ones(p), np.zeros(p)], np.c_[-rows, np.zeros(p), np.ones(p)],
                     np.r_[np.zeros(2 * K), 1.0, -1.0][None, :]])
    bub = np.r_[np.zeros(2 * p), 1.0]

    def U(theta):
        c = -np.r_[np.cos(theta), np.sin(theta), 0.0, 0.0]
        r = linprog(c, A_ub=Aub, b_ub=bub, bounds=[(None, None)] * nvar, method='highs')
        return -r.fun if r.status == 0 else np.inf
    I = np.arange(K)
    ths = [np.zeros(K), np.pi * (I % 2), np.pi * I ** 2 / K, 2 * np.pi * I / K] + [rng.uniform(0, 2 * np.pi, K) for _ in range(starts)]
    best = -np.inf
    for th in ths:
        r = minimize(lambda t: -U(t), th, method='Powell', options=dict(maxfev=iters, xtol=1e-6, ftol=1e-9))
        best = max(best, -r.fun)
    return 1.0 / best
