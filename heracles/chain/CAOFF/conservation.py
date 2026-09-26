# Junk-visibility conservation (any network, mixing neurons allowed).
# Row-mode (-2k,0) coefficient of the gap at offset D:  Phi(D) = sum_ga Zhat(-2k-ga, -ga, ga) (1 - w^{-ga D}).
# Claims: (i) J_{k,1} = -(1/p) sum_{D != 0} Phi(D) w^{-kD}  (the X-blind functional), so sum_D |Phi(D)| >= p |J|;
#         (ii) min over X of sum_D |Phi(D)|^2 = p^2/(p-1) |J|^2  (Gram of the offset profiles = p (I + 11^T)).
import sys, numpy as np
sys.path.insert(0, '..')
from common import random_net, table, gaps, spectrum
rng = np.random.default_rng(0); worst = 0; ratios = []
for p in (7, 11, 13):
    w = np.exp(2j * np.pi / p); a = np.arange(p)
    for t in range(6):
        z = table(p, random_net(p, 10, rng, pure=False)); g = gaps(z); G, J = spectrum(z)
        for k in range(1, (p - 1) // 2 + 1):
            u = (-2 * k) % p
            Phi = np.array([p ** 3 * (g[:, :, D] * w ** (-u * a)[:, None]).sum() / p ** 2 for D in range(1, p)])
            rec = -(1 / p) * sum(Phi[D - 1] * w ** (-k * D) for D in range(1, p))
            worst = max(worst, abs(rec - J['1'][k]) / max(1e-12, abs(J['1'][k])))
            if abs(J['1'][k]) > 1e-9:
                ratios.append(np.abs(Phi).sum() / (p * abs(J['1'][k])))
print(f"(i) reconstruction J = -(1/p) sum_D Phi(D) w^(-kD): max rel. error {worst:.1e};  sum|Phi| / (p|J|) min over 120 cases = {min(ratios):.3f} (>= 1 claimed)")
for p in (5, 7, 11, 13):   # (ii) Gram check
    D = np.arange(1, p); gams = [g for g in range(1, p)]
    F = np.array([1 - np.exp(-2j * np.pi * g * D / p) for g in gams])     # rows: profiles
    Gram = F.conj() @ F.T
    k = 1; idx = gams.index((-k) % p)
    others = np.delete(F, idx, 0)
    coef, *_ = np.linalg.lstsq(others.T, F[idx], rcond=None)
    resid = np.linalg.norm(F[idx] - others.T @ coef) ** 2
    print(f"(ii) p={p}: Gram = p(I+11^T)? {np.allclose(Gram, p * (np.eye(p - 1) + 1))};  min_X sum|Phi|^2 / |J|^2 = {resid:.4f} vs p^2/(p-1) = {p * p / (p - 1):.4f};  no-X value 2p = {2 * p}")
