# Candidate global inequalities for networks WITH mixing neurons, evaluated exactly.
#  T      : gamma / gamma_sym                                   (the target; must be <= 1)
#  CA     : margin / [(2/p^3) Gamma (sum_k cost_k / 3^{1/3})^{3/2}],  cost_k = sum_i |c_ik|^{2/3}    (diagonal budget)
#  CAb    : same with cost_k -> (p/6) b_k, b_k = frequency-k share of the weight norm (imbalance slack included)
#  CAoff  : per-offset version of CA: smallest l^{2/3} budget of a clean spectrum Y~ with (2/p^3) sum_k cov_k(D) Y~_k >= m_D
#           for EVERY offset D (m_D = min over rows of the D/-D paired gap), as a ratio (budget needed / diagonal budget)^{3/2}.
import sys, itertools, numpy as np
sys.path.insert(0, '..')
from common import *
def min_budget(A, M, r=2 / 3):
    # min sum Y^r over {Y >= 0, A Y >= M}: concave minimisation -> vertex enumeration
    h = A.shape[1]; best = np.inf
    for K in range(1, h + 1):
        for S in itertools.combinations(range(h), K):
            for T in itertools.combinations(range(A.shape[0]), K):
                Msub = A[np.ix_(T, S)]
                if abs(np.linalg.det(Msub)) < 1e-12: continue
                yS = np.linalg.solve(Msub, M[list(T)])
                if (yS < -1e-12).any(): continue
                y = np.zeros(h); y[list(S)] = np.clip(yS, 0, None)
                if (A @ y < M - 1e-9 * np.abs(M).max()).any(): continue
                best = min(best, (y[y > 0] ** r).sum())
    return best
def evaluate(p, net):
    z = table(p, net); g = gaps(z); m = g[:, :, 1:].min(); P = cost2(net); h = (p - 1) // 2
    Gam, _ = programme(p, 1.5); A = cover_matrix(p)
    cost = np.zeros(h); b = np.zeros(h)
    for e1, e2, w in net:
        E1, E2, W = np.fft.fft(e1), np.fft.fft(e2), np.fft.fft(w)
        for k in range(1, h + 1):
            cost[k - 1] += abs(E1[k] * E2[k] * W[k]) ** (2 / 3)
            b[k - 1] += 2 / p * (abs(E1[k]) ** 2 + abs(E2[k]) ** 2 + abs(W[k]) ** 2)
    Bdiag = cost.sum() / 3 ** (1 / 3); Bb = (p / 6) * b.sum() / 3 ** (1 / 3)
    T = m / P ** 1.5 / gamma_sym(p, Gam)
    CA = m / (2 / p ** 3 * Gam * Bdiag ** 1.5); CAb = m / (2 / p ** 3 * Gam * Bb ** 1.5)
    mD = np.array([0.5 * (g[:, :, D] + g[:, :, p - D]).min() for D in range(1, h + 1)])
    need = min_budget(A, mD * p ** 3 / 2) if (mD > 0).all() else np.nan
    CAoff = (need / Bdiag) ** 1.5 if np.isfinite(need) else np.nan
    return dict(T=T, CA=CA, CAb=CAb, CAoff=CAoff, m=m, mD=mD)
if __name__ == '__main__':
    for p, N, seeds in ((5, 9, [3]), (7, 12, [1, 3, 4, 5, 6, 7, 9])):
        for s in seeds:
            v = np.load(f'../../claude0923/endpoint_p{p}_s{s}.npy').reshape(3, N, p)
            r = evaluate(p, [(v[0, i], v[1, i], v[2, i]) for i in range(N)])
            print(f"p={p} endpoint s{s}: T {r['T']:.4f}  CA {r['CA']:.4f}  CAb {r['CAb']:.4f}  CAoff {r['CAoff']:.4f}")
