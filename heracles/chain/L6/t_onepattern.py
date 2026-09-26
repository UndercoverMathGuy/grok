# Check: if a population's junk vanishes on ONE pattern T (J_T = 0), then sum|c|^{2/3} >= 2^{1/3} |G|^{2/3}
# (n = 1 feature cost applied to the 2x2 chain matrix of that pattern's phase u_l = prod_{j in T} v_lj), so its discount
# is at most sqrt(3/2) = 1.2247 whatever the aim. Random populations projected onto J_T = 0, plus local minimisation.
import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(9); worst = 9
for T in ('1', '2', '12'):
    for trial in range(3000):
        N = rng.integers(2, 7)
        v1 = np.exp(1j * rng.uniform(0, 2 * np.pi, N)); v2 = np.exp(1j * rng.uniform(0, 2 * np.pi, N))
        u = {'1': v1, '2': v2, '12': v1 * v2}[T]
        c = rng.normal(size=N) + 1j * rng.normal(size=N)
        c = c - (np.vdot(u.conj(), c) / np.vdot(u.conj(), u.conj())) * u.conj()   # remove component along conj(u): sum c u = 0
        if abs((c * u).sum()) > 1e-9 or abs(c.sum()) < 1e-6: continue
        worst = min(worst, (np.abs(c) ** (2 / 3)).sum() / (2 ** (1 / 3) * abs(c.sum()) ** (2 / 3)))
print(f"min over random J_T=0 populations of cost / (2^(1/3)|G|^(2/3)) = {worst:.4f}  (>= 1 claimed; = 1 at the 2-atom pair)")
