# The multi-offset lever. For the offset pair {D, -D} the test distribution may use INDEPENDENT row densities w+ (at D)
# and w- (at -D). Exact: the junk term of frequency k is (1/2) Re J_k [ (1 - w^{kD}) m+_k + (1 - w^{-kD}) m-_k ] where m+-
# are the densities' moments at the junk character, while the clean term is cov_k(D) Re G_k (independent of w+-).
# Same density (fable Lemma 1): junk weight cov_k (one-sided moments). Flip (same density, shifts +-D/2): -cov_k with the
# density's moments -> two-sided depth A_s. Independent densities: the reachable set is the Minkowski sum
# N_D = L+ M + L- M, which contains per-k discs of radius A* |1 - w^{kD}| = A* sqrt(2 cov_k) (relative depth
# A* sqrt(2/cov_k) >= A*), and contains the flip's set.
# pair_depth: weighted polydisc depth of N_D (single-pattern, K-mode): largest a with every target nu_k = a cov_k e^{i th_k}
# reachable; min over th by search (UPPER estimate), exact LP per direction.
import sys, numpy as np
from scipy.optimize import linprog, minimize
sys.path.insert(0, '..')
from common import cover_matrix


def _tmax(theta, p, T, D):
    K = len(T); x = np.arange(p)
    chi = np.exp(-2j * np.pi * 2 * np.outer(T, x) / p)                      # [K, p] junk characters
    wk = np.exp(2j * np.pi * np.array(T) * D / p)                           # w^{kD}
    cov = 1 - np.cos(2 * np.pi * np.array(T) * D / p)
    Lp = 0.5 * (1 - wk)[:, None] * chi; Lm = 0.5 * (1 - np.conj(wk))[:, None] * chi
    tgt = cov * np.exp(1j * theta)
    # variables: w+ (p), w- (p), t ; equalities: Re/Im of Lp w+ + Lm w- - t tgt = 0 ; sum w+ = 1 ; sum w- = 1
    Aeq = np.vstack([np.c_[Lp.real, Lm.real, -tgt.real[:, None]], np.c_[Lp.imag, Lm.imag, -tgt.imag[:, None]],
                     np.r_[np.ones(p), np.zeros(p), 0.0][None, :], np.r_[np.zeros(p), np.ones(p), 0.0][None, :]])
    beq = np.r_[np.zeros(2 * K), 1.0, 1.0]
    r = linprog(np.r_[np.zeros(2 * p), -1.0], A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * (2 * p) + [(None, None)], method='highs')
    return -r.fun if r.status == 0 else 0.0


def pair_depth(p, T, D, rng, starts=8, iters=150):
    K = len(T); I = np.arange(K)
    ths = [np.zeros(K), np.pi * np.ones(K), np.pi * (I % 2), np.pi * I ** 2 / K] + [rng.uniform(0, 2 * np.pi, K) for _ in range(starts)]
    best = 9.0
    for th in ths:
        r = minimize(lambda t: _tmax(t, p, T, D), th, method='Powell', options=dict(maxfev=iters, xtol=1e-6, ftol=1e-9))
        best = min(best, r.fun)
    return best


def floored_ratio_Ddep(p, T, deltaD, eps, Gam, rng, starts=15):
    """share-floored programme with an offset-dependent discount deltaD[D-1] (or matrix deltaD[D-1, j] for T[j]) on T (block rate elsewhere); ratio to Gamma*/sqrt3.
    NUMERICAL lower bound on the max (multistart SLSQP)."""
    A = cover_matrix(p); h = A.shape[0]; Ti = np.array(T) - 1
    dD = np.array(deltaD, float)
    if dD.ndim == 1:
        dD = dD[:, None] * np.ones((1, len(Ti)))       # uniform over T at each offset
    R = np.full((h, h), 3 ** -0.5); R[:, Ti] = dD / np.sqrt(3)    # rows = offsets D = 1..h, columns = frequencies
    AR = A * R
    cons = [{'type': 'ineq', 'fun': lambda x: AR @ np.clip(x[:h], 0, None) ** 1.5 - x[h]},
            {'type': 'eq', 'fun': lambda x: x[:h].sum() - 1}]
    bnds = [(0, 1)] * h + [(None, None)]
    for i in Ti: bnds[i] = (eps, 1)
    best = -9
    for s in range(starts):
        b0 = rng.dirichlet(np.ones(h) * rng.choice([0.2, 1.0])); b0[Ti] = np.maximum(b0[Ti], eps); b0 /= b0.sum()
        x0 = np.r_[b0, (AR @ b0 ** 1.5).min()]
        res = minimize(lambda x: -x[h], x0, constraints=cons, bounds=bnds, method='SLSQP', options=dict(maxiter=500, ftol=1e-12))
        b = np.clip(res.x[:h], 0, None)
        if abs(b.sum() - 1) < 1e-6 and (b[Ti] >= eps - 1e-7).all():
            best = max(best, (AR @ b ** 1.5).min())
    return best / (Gam / np.sqrt(3))
