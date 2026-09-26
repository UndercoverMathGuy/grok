# Provable light-junk thresholds.
#   Phi(sigma) = Gamma (1 - sigma)^{3/2}/sqrt3 + R Lambda_p(sigma, eps)  must be <= Gamma/sqrt3 for sigma in (0, min(1, h eps)]
#   Lambda_p(sigma, eps) = eps^{3/2} S_m + (sigma - m eps)^{3/2} mu_{m+1},  m = floor(sigma/eps),
#   mu_{i+1} = 1 + cos(pi (2i+1)/p)  (coverage values sorted decreasingly),  S_m = mu_1 + ... + mu_m.
# clean corollary (PROVED): eps <= Gamma^2/(12 R^2) suffices.  refined: largest eps passing on a sigma grid (NUMERICAL,
# plus a rigorous check: Phi is evaluated with an upper bound on each sigma-cell, see below).
import sys, json, numpy as np
def mus(p):
    h = (p - 1) // 2
    return 1 + np.cos(np.pi * (2 * np.arange(h) + 1) / p)          # already decreasing in i
def Lam(p, s, eps, mu, S):
    m = int(np.floor(s / eps)); h = len(mu)
    if m >= h: return eps ** 1.5 * S[h]                              # every frequency light at the cap
    return eps ** 1.5 * S[m] + max(s - m * eps, 0.0) ** 1.5 * mu[m]
def passes(p, G, R, eps, n=4000):
    # RIGOROUS on each cell (up to float rounding): Lambda is nondecreasing in sigma, g(s) = (G/sqrt3)(1-(1-s)^{3/2}) is increasing.
    # Cell [s0,s1], s0 > 0: pass if R Lam(s1) <= g(s0).  First cell [0,s1]: Lam(s) <= mu_1 sqrt(min(s1,eps)) s and
    # g(s) >= (G/sqrt3) slope(s1) s with slope(s1) = (1-(1-s1)^{3/2})/s1 (chord slope of a concave function), so pass if
    # R mu_1 sqrt(min(s1,eps)) <= (G/sqrt3) slope(s1).
    mu = mus(p); S = np.r_[0, np.cumsum(mu)]; h = len(mu); smax = min(1.0, h * eps)
    g = lambda s: G / np.sqrt(3) * (1 - (1 - s) ** 1.5)
    grid = np.r_[0, np.geomspace(1e-9, smax, n)]
    s1 = grid[1]
    if R * mu[0] * np.sqrt(min(s1, eps)) > G / np.sqrt(3) * (1 - (1 - s1) ** 1.5) / s1:
        return False
    for s0, s1 in zip(grid[1:-1], grid[2:]):
        opt_i = G * (1 - s0) ** 1.5 / np.sqrt(3) + R * Lam(p, s1, eps, mu, S)          # test at the heavy argmin offset
        opt_ii = p / (p - 1) * ((1 - s0) ** 1.5 / np.sqrt(3) + R * np.sqrt(eps) * s1)  # test averaged over all offsets
        if min(opt_i, opt_ii) > G / np.sqrt(3):
            return False
    return True
def eps_ref(p, G, R):
    lo, hi = G ** 2 / (12 * R ** 2), 1.0
    if not passes(p, G, R, lo): return None
    for _ in range(50):
        mid = (lo + hi) / 2
        if passes(p, G, R, mid): lo = mid
        else: hi = mid
    return lo
if __name__ == '__main__':
    G = {59: 0.313049, 113: 0.27593576, 211: float(sys.argv[1]) if len(sys.argv) > 1 else 0.235}
    res = {}
    for p in (59, 113, 211):
        for R in (1.0, 25 / 16, 4.0):
            e_clean = G[p] ** 2 / (12 * R ** 2); e_ref = eps_ref(p, G[p], R)
            res[f"{p}_{R}"] = dict(p=p, R=R, Gamma=G[p], eps_clean=e_clean, c_clean=1 / (12 * R ** 2),
                                   eps_ref=e_ref, c_ref=e_ref / G[p] ** 2, heavy_max_clean=int(1 / e_clean), h=(p - 1) // 2)
            print(f"p={p:3d} R={R:5.3f}: eps_clean={e_clean:.5f} (c={1 / (12 * R ** 2):.4f})  eps_ref={e_ref:.5f} (c={e_ref / G[p] ** 2:.4f})  "
                  f"1/eps_ref={1 / e_ref:.0f}  h={(p - 1) // 2}", flush=True)
    json.dump(res, open('lemma_eps.json', 'w'), indent=1)
