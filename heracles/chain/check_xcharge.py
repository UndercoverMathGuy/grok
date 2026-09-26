# Independent checks of X_CHARGE claims: (1) transfer constant at (A*, A_s) = (1/4, 1/2); (2) the p = 5 local max beats the
# per-frequency ceiling margin <= (2/p^3) Gamma* (sum_k cost_k / 3^{1/3})^{3/2}, cost_k = sum_i |c_ik|^{2/3} (diagonal atoms).
import numpy as np
from common import *
from transfer_flip import depth
c, lam = depth(0.5, 0.25, n=120)
print(f"transfer constant c(A_s=1/2, A*=1/4) ~ {c:.4f} (claim: >= 0.29, fails at 0.292)")
p, N = 5, 9
v = np.load('../claude0923/endpoint_p5_s3.npy').reshape(3, N, p)
net = [(v[0, i], v[1, i], v[2, i]) for i in range(N)]
m = margin(table(p, net)); Gam = programme(p, 1.5)[0]
cost = np.zeros(3)
for e1, e2, w in net:
    E1, E2, W = np.fft.fft(e1), np.fft.fft(e2), np.fft.fft(w)
    for k in (1, 2):
        cost[k] += abs(E1[k] * E2[k] * np.conj(W[k])) ** (2 / 3)
ceiling = 2 / p ** 3 * Gam * (cost[1:].sum() / 3 ** (1 / 3)) ** 1.5
print(f"p=5 local max: margin {m:.6f}, per-frequency ceiling {ceiling:.6f}, ratio {m / ceiling:.4f} (claim 1.018)")
print(f"sanity: budget (B) sum_k cost_k = {cost[1:].sum():.5f} <= (p/6) P = {p / 6 * cost2(net):.5f}")
y = (cost[1:] / 3 ** (1 / 3)) ** 1.5
own = 2 / p ** 3 * (cover_matrix(p) @ y).min()
print(f"own-spectrum per-frequency bound (2/p^3) min_D sum cov_k y_k, y_k = (cost_k/3^(1/3))^(3/2): {own:.6f}, margin/bound = {m / own:.4f}")
