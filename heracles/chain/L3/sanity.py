# Sanity check of the L3 construction on explicit sets: build the actual Riesz densities for random targets z,
# check non-negativity, the moment identity, that the achieved value >= bound * |z|_1, and that the true
# one-sided max_x Re P_z(x) (two-sided max|P_z|) is >= the achieved value.
import numpy as np
from fractions import Fraction as Fr
from riesz_bound import bounds
rng = np.random.default_rng(1)
def check(p, H, trials=300):
    (one, r1), (two, r2) = bounds(p, H)[:2]
    one, two, r1, r2 = float(one), float(two), float(r1), float(r2)
    x = np.arange(p); m = np.array([(2 * k) % p for k in H]); chi = np.exp(2j * np.pi * np.outer(m, x) / p)   # w^{m_k x}
    worst1 = worst2 = 9; negmin = 9
    for _ in range(trials):
        z = rng.normal(size=len(H)) + 1j * rng.normal(size=len(H)); z *= rng.exponential(size=len(H))
        def riesz(lam):
            R = np.prod(1 + 2 * np.real(lam[:, None] * np.conj(chi)), axis=0)
            return R, R.mean()
        lam1 = r1 * np.conj(z) / np.abs(z); R, M = riesz(lam1); negmin = min(negmin, R.min())
        mu = (R[None, :] * chi).mean(1) / M
        val1 = np.real(z @ mu)
        lam2 = r2 * np.conj(z) / np.abs(z); Ra, Ma = riesz(lam2); Rb, Mb = riesz(-lam2)
        nu = (Ma * (Ra[None, :] * chi).mean(1) / Ma - Mb * (Rb[None, :] * chi).mean(1) / Mb) / (Ma + Mb)
        val2 = np.real(z @ nu)
        P = np.real(z @ chi); nz = np.abs(z).sum()
        worst1 = min(worst1, (val1 - one * nz) / nz, (P.max() - val1) / nz)
        worst2 = min(worst2, (val2 - two * nz) / nz, (np.abs(P).max() - val2) / nz)
    return one, two, worst1, worst2, negmin
for p, H in ((1009, [1, 5, 17, 60, 211, 450]), (1009, [3, 7, 10, 41, 99, 300, 402]), (1009, [1, 2, 3, 4, 5, 6]),
             (10007, [11, 250, 261, 3000, 4100, 777, 1234, 4321])):
    one, two, w1, w2, nm = check(p, H)
    print(f"p={p} H={H}: bound1 {one:.4f} bound2 {two:.4f} | min slack (achieved-bound, truth-achieved) 1-sided {w1:+.2e} 2-sided {w2:+.2e} | min Riesz value {nm:.2e}")
