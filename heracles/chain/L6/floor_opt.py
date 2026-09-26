# A stronger optimiser for the share-floored (optionally offset-dependent) discounted programme:
#   value = max_beta min_D sum_k cov_k(D) R[D,k] beta_k^{3/2},  sum beta = 1, beta_k >= eps on T,
# with R = 1/sqrt3 (block rate) off T and delta/sqrt3 on T (delta may depend on D).  ratio = value / (Gamma*/sqrt3).
# Method: SLSQP on a working support S (T always included), then add/remove local search over S, from several seeds
# (Gamma*-support, T alone, random). Still a LOWER bound on the max (NUMERICAL), but it reproduces ratio = 1.000 at
# delta = 1 on the Gamma*-support (sanity check in __main__).
import sys, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '..')
from common import cover_matrix


def _solve(AR, S, fl, rng, starts=3):
    K = len(S); M = AR[:, S]
    best, bb = -1.0, None
    for s in range(starts):
        b0 = rng.dirichlet(np.ones(K)) if s else np.ones(K) / K
        b0 = np.maximum(b0, fl); b0 /= b0.sum()
        cons = [{'type': 'eq', 'fun': lambda v: v[:K].sum() - 1, 'jac': lambda v: np.r_[np.ones(K), 0.0]},
                {'type': 'ineq', 'fun': lambda v: M @ np.clip(v[:K], 0, None) ** 1.5 - v[K],
                 'jac': lambda v: np.c_[M * 1.5 * np.sqrt(np.clip(v[:K], 0, None))[None, :], -np.ones(M.shape[0])]}]
        r = minimize(lambda v: -v[K], np.r_[b0, (M @ b0 ** 1.5).min()], jac=lambda v: np.r_[np.zeros(K), -1.0],
                     constraints=cons, bounds=[(f, 1) for f in fl] + [(None, None)], method='SLSQP',
                     options=dict(maxiter=400, ftol=1e-13))
        b = np.maximum(np.clip(r.x[:K], 0, None), fl); b /= b.sum()
        if (b < fl - 1e-9).any():
            continue
        v = (M @ b ** 1.5).min()
        if v > best:
            best, bb = v, b
    return best, bb


def floored_value(p, T, R, eps, Gam, rng, seeds=(), rounds=3):
    A = cover_matrix(p); h = A.shape[0]
    AR = A * R
    T0 = [k - 1 for k in T]
    starts = [sorted(set(T0) | {k - 1 for k in s}) for s in seeds] + [sorted(T0)]
    for _ in range(3):
        starts.append(sorted(set(T0) | set(rng.choice(h, 4, replace=False).tolist())))
    best = -1.0
    for S in starts:
        fl = np.array([eps if k in T0 else 0.0 for k in S])
        val, b = _solve(AR, S, fl, rng)
        for _ in range(rounds):
            improved = False
            for k in rng.permutation(h):
                if k in S:
                    if k in T0 or len(S) <= 1:
                        continue
                    C = [x for x in S if x != k]
                else:
                    C = sorted(S + [k])
                flC = np.array([eps if x in T0 else 0.0 for x in C])
                v, bC = _solve(AR, C, flC, rng, starts=2)
                if v > val + 1e-10:
                    val, S, b, improved = v, C, bC, True
            if not improved:
                break
        best = max(best, val)
    return best / (Gam / np.sqrt(3))


def rate_matrix(p, T, delta):
    """R[D,k]: 1/sqrt3 off T, delta/sqrt3 on T; delta scalar, per-offset vector, or [D, j] matrix."""
    h = (p - 1) // 2
    R = np.full((h, h), 3 ** -0.5)
    d = np.array(delta, float)
    if d.ndim == 0:
        d = np.full((h, len(T)), float(d))
    elif d.ndim == 1:
        d = d[:, None] * np.ones((1, len(T)))
    R[:, [k - 1 for k in T]] = d / np.sqrt(3)
    return R


if __name__ == '__main__':
    import json
    sys.path.insert(0, '.')
    from tools import gamma_star
    rng = np.random.default_rng(0)
    p = 59; Gam, S = gamma_star(p, rng)
    for eps in (0.01, 0.03):
        print('sanity delta=1 on Gamma*-support, eps', eps, round(floored_value(p, S, rate_matrix(p, S, 1.0), eps, Gam, rng, seeds=[S]), 5))
    print('sanity: single frequency {1} with discount 1.05, no floor (zero-slack lemma predicts > 1):',
          round(floored_value(p, [1], rate_matrix(p, [1], 1.05), 0.0, Gam, rng, seeds=[S]), 5))
