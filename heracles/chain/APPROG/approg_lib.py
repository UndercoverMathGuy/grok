"""Floored, discounted share programme (n = 2, q = 2, s = 3/2).

u_k = budget-share amplitude (y_k / d_k); cost = sum_k u_k^{2/3}; coverage row D: sum_k d_k A[D,k] u_k,
d_k = delta on T, 1 elsewhere; share floor: u_k >= eps^{3/2} on T (budget normalised to 1).
W(delta) = min { sum u^{2/3} : A_delta u >= Gam, u >= floor }.
V(delta) >= Gam  <=>  W(delta) <= 1.   delta_crit = sup { delta : W(delta) >= 1 }.
"""
import itertools
import numpy as np
from scipy.optimize import linprog

R = 2 / 3


def cover(p):
    h = (p - 1) // 2
    D = np.arange(1, h + 1)[:, None]
    k = np.arange(1, h + 1)[None, :]
    return 1 - np.cos(2 * np.pi * k * D / p)


def _setup(p, T, delta, eps):
    A = cover(p)
    h = A.shape[0]
    d = np.ones(h)
    fl = np.zeros(h)
    for k in T:
        d[k - 1] = delta
        fl[k - 1] = eps ** 1.5
    return A * d[None, :], fl


def W_exact(p, T, delta, eps, Gam, tol=1e-9, return_u=False):
    """Complete vertex enumeration (concave cost => optimum at a vertex). Feasible for h <= ~11."""
    Ad, fl = _setup(p, T, delta, eps)
    h = Ad.shape[0]
    best, bu = np.inf, None
    for m in range(1, h + 1):
        rows = np.array(list(itertools.combinations(range(h), m)))
        for S in itertools.combinations(range(h), m):
            S = list(S)
            rest = fl.copy()
            rest[S] = 0.0
            M = Ad[rows][:, :, S]                          # (nR, m, m)
            rhs = Gam - (Ad @ rest)[rows]                 # (nR, m)
            det = np.linalg.det(M)
            ok = np.abs(det) > 1e-12
            if not ok.any():
                continue
            sol = np.linalg.solve(M[ok], rhs[ok][..., None])[..., 0]
            U = np.tile(rest, (sol.shape[0], 1))
            U[:, S] = sol
            feas = (sol >= fl[S] - tol).all(1) & ((U @ Ad.T) >= Gam - tol * max(1, Gam)).all(1)
            if not feas.any():
                continue
            c = (np.clip(U[feas], 0, None) ** R).sum(1)
            i = c.argmin()
            if c[i] < best:
                best, bu = c[i], U[feas][i]
    return (best, bu) if return_u else best


def W_heur(p, T, delta, eps, Gam, starts=40, iters=40, rng=None, seeds=()):
    """Majorise-minimise (reweighted LP) from random and seeded weights. Returns an UPPER bound on W."""
    rng = rng or np.random.default_rng(0)
    Ad, fl = _setup(p, T, delta, eps)
    h = Ad.shape[0]
    best, bu = np.inf, None
    inits = [np.asarray(s, float) for s in seeds] + [rng.exponential(1.0, h) ** 2 for _ in range(starts)]
    for g0 in inits:
        g = g0 if g0.min() > 0 else g0 + 1e-3
        prev = None
        for _ in range(iters):
            r = linprog(g, A_ub=-Ad, b_ub=-Gam * np.ones(h), bounds=list(zip(fl, [None] * h)), method='highs')
            if r.status != 0:
                break
            u = r.x
            if prev is not None and np.abs(u - prev).max() < 1e-12:
                break
            prev = u
            g = (u + 1e-12) ** (-1 / 3)
        if prev is None:
            continue
        c = (np.clip(prev, 0, None) ** R).sum()
        if c < best:
            best, bu = c, prev
    return best, bu


def gamma_star_exact(p):
    return W_exact(p, [], 1.0, 0.0, 1.0) ** -1.5


def delta_crit(Wfun, lo=1.0, hi=np.sqrt(3), steps=30):
    """sup{delta : W(delta) >= 1} on [lo, hi]; returns ('>= hi', hi) if W(hi) >= 1."""
    if Wfun(hi) >= 1:
        return hi, True
    for _ in range(steps):
        mid = (lo + hi) / 2
        if Wfun(mid) >= 1:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, False


def _mm(Ad, fl, Gam, g, iters=60):
    h = Ad.shape[0]
    prev = None
    for _ in range(iters):
        r = linprog(g, A_ub=-Ad, b_ub=-Gam * np.ones(h), bounds=list(zip(fl, [None] * h)), method='highs')
        if r.status != 0:
            return None
        u = r.x
        if prev is not None and np.abs(u - prev).max() < 1e-13:
            break
        prev = u
        g = (u + 1e-12) ** (-1 / 3)
    return prev


def W_heur2(p, T, delta, eps, Gam, starts=60, hops=40, rng=None, seeds=()):
    """MM from random-subset starts + basin hopping on the weights. UPPER bound on W."""
    rng = rng or np.random.default_rng(0)
    Ad, fl = _setup(p, T, delta, eps)
    h = Ad.shape[0]
    cost = lambda u: (np.clip(u, 0, None) ** R).sum()
    best, bu = np.inf, None
    inits = [np.asarray(s, float) for s in seeds]
    for _ in range(starts):
        m = rng.integers(2, min(h, 12) + 1)
        U = rng.choice(h, m, replace=False)
        g = np.full(h, 50.0) * rng.uniform(0.5, 2, h)
        g[U] = rng.uniform(0.5, 1.5, m)
        inits.append(g)
    for g in inits:
        u = _mm(Ad, fl, Gam, g)
        if u is not None and cost(u) < best:
            best, bu = cost(u), u
    for _ in range(hops):                                   # basin hopping around the incumbent
        g = (bu + 1e-12) ** (-1 / 3) * np.exp(rng.normal(0, 1.0, h))
        flip = rng.choice(h, rng.integers(1, 4), replace=False)
        g[flip] = rng.uniform(0.2, 1.0, len(flip)) * g.min()
        u = _mm(Ad, fl, Gam, g)
        if u is not None and cost(u) < best - 1e-14:
            best, bu = cost(u), u
    return best, bu


import heapq


def decide(p, T, delta, eps, Gam, thresh=1.0, max_nodes=200000, tol=1e-9):
    """Falk-Soland branch-and-bound for the separable concave min W = min sum u^{2/3} over {A_delta u >= Gam, u >= floor}.
    Decision version: returns ('ge', nodes) if W >= thresh is proved (all node lower bounds >= thresh - tol),
    ('lt', u) if a feasible u with cost < thresh is found, ('undecided', nodes) if max_nodes is hit.
    Lower bounds: secant (convex envelope) of u^{2/3} on each coordinate's box, then an LP (fp64, HiGHS)."""
    Ad, fl = _setup(p, T, delta, eps)
    h = Ad.shape[0]
    f = lambda x: np.clip(x, 0, None) ** R
    hi0 = np.full(h, thresh ** 1.5)                     # cost < thresh forces every u_k < thresh^{3/2}
    if (fl > hi0).any():
        return 'ge', 0

    def solve(lo, hi):
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(hi > lo + 1e-15, (f(hi) - f(lo)) / (hi - lo), 0.0)
        const = (f(lo) - s * lo).sum()
        r = linprog(s, A_ub=-Ad, b_ub=-Gam * np.ones(h), bounds=list(zip(lo, hi)), method='highs')
        if r.status != 0:
            return None
        return r.fun + const, r.x, s
    root = solve(fl.copy(), hi0)
    if root is None:
        return 'ge', 0                                    # infeasible with every u_k < thresh^{3/2}
    heap = [(root[0], 0, fl.copy(), hi0.copy(), root[1], root[2])]
    cnt = 1
    nodes = 0
    while heap:
        lb, _, lo, hi, u, s = heapq.heappop(heap)
        if lb >= thresh - tol:
            return 'ge', nodes
        nodes += 1
        if f(u).sum() < thresh - tol:
            return 'lt', u
        if nodes > max_nodes:
            return 'undecided', nodes
        sec = f(lo) + s * (u - lo)
        gap = f(u) - sec
        k = int(gap.argmax())
        cut = u[k] if hi[k] - u[k] > 1e-12 and u[k] - lo[k] > 1e-12 else (lo[k] + hi[k]) / 2
        for a, b in ((lo[k], cut), (cut, hi[k])):
            lo2, hi2 = lo.copy(), hi.copy()
            lo2[k], hi2[k] = a, b
            res = solve(lo2, hi2)
            if res is not None and res[0] < thresh - tol:
                heapq.heappush(heap, (res[0], cnt, lo2, hi2, res[1], res[2]))
                cnt += 1
    return 'ge', nodes


def delta_crit_bb(p, T, eps, Gam, lo=1.0, hi=np.sqrt(3), steps=14, **kw):
    """Bisection with the B&B decision. Returns (delta_lo, delta_hi, status): delta_crit in [delta_lo, delta_hi]
    where 'ge' at delta_lo was proved and 'lt' at delta_hi was found (or hi = sqrt3 'ge' => safe)."""
    st, _ = decide(p, T, hi, eps, Gam, **kw)
    if st == 'ge':
        return hi, hi, 'safe_at_full_discount'
    st, _ = decide(p, T, lo, eps, Gam, **kw)
    if st != 'ge':
        return lo, lo, 'pays_without_discount' if st == 'lt' else 'undecided_at_lo'
    for _ in range(steps):
        mid = (lo + hi) / 2
        st, _ = decide(p, T, mid, eps, Gam, **kw)
        if st == 'ge':
            lo = mid
        elif st == 'lt':
            hi = mid
        else:
            return lo, hi, 'undecided'
    return lo, hi, 'ok'
