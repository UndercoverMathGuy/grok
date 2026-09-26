# Check the proved universal pair-offset floors:
#  (F1) single-pattern junk: (dip_D + dip_-D)/2 = osc(F)/2 >= ||F||_2 = (sum_k cov_k(D) |J_k|^2)^{1/2}
#  (F2) 3-pattern junk: (dip_D + dip_-D)/2 >= max_T (sum_k cov_k(D) |J_kT|^2)^{1/2}
import numpy as np, torch
from mo import Body, cov
rng = np.random.default_rng(5); worst1 = worst2 = 9; tight = 0
for p in (11, 17, 23, 31):
    h = (p - 1) // 2
    for t in range(60):
        K = int(rng.integers(1, min(h, 7) + 1)); T = sorted(rng.choice(np.arange(1, h + 1), K, replace=False).tolist())
        D = int(rng.integers(1, p)); c = cov(p, np.array(T), np.array([D]))[:, 0]
        body = Body(p, T, [D, p - D], [0.5, 0.5])
        J = torch.tensor(rng.normal(size=(3, K)) + 1j * rng.normal(size=(3, K)))
        if t % 2 == 0: J[1:] = 0                          # single pattern
        exp = float((body.lamt * body.dips(J)).sum())
        floors = [np.sqrt((c * np.abs(J[T_].numpy()) ** 2).sum()) for T_ in range(3)]
        r = exp / max(floors)
        if t % 2 == 0: worst1 = min(worst1, r)
        else: worst2 = min(worst2, r)
print(f"min exposure/floor: single-pattern {worst1:.4f}, 3-pattern {worst2:.4f} (both must be >= 1)")
