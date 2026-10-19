# R8(a) check: union of relabelled single features theta_j (u -> u(j.), w -> w(j.)), each scaled to cost P_j P,
#   has margin >= B_p P^s min_delta sum_j P_j^s m_{theta_j}(j delta), with m_theta(l) = min_x g(x,l)/(B_p P_theta^s).
#   Also relabelling identity g'(x,d) = g(jx, jd) and cost invariance. Random (non-IF) single features, arbitrary inputs.
import math, numpy as np
from vnet import *

rng = np.random.default_rng(8)
worst_id = 0; worst_union = np.inf
for trial in range(60):
    p = int(rng.choice([5, 7, 11])); n = int(rng.choice([2, 3])) if p < 11 else 2; b = rng.uniform(2, n + 1); s = (n + 1) / b
    F = (p - 1) // 2
    nets = []
    for j in range(1, F + 1):
        m = int(rng.integers(2, 5))
        U0, W0 = if_block(p, n, 1, 1.0, rng.uniform(0, 1))
        U = U0[:m] if m <= n + 1 else np.concatenate([U0, rng.normal(size=(m - n - 1, n, p))])
        W = W0[:m] if m <= n + 1 else np.concatenate([W0, rng.normal(size=(m - n - 1, p))])
        U = U + 0.2 * rng.normal(size=U.shape)
        W = W + 0.2 * (rng.normal(size=(len(W), 2)) @ freq_basis(p, 1))
        nets.append((U, W))
    Pw = rng.dirichlet(np.ones(F)); Ptot = 3.7
    Us, Ws, prof = [], [], []
    for j, (U, W) in enumerate(nets, start=1):
        Pj = cost(U, W, b); c = (Pw[j - 1] * Ptot / Pj) ** (1 / b)       # scale each factor by c: cost scales by c^b
        U = U * c; W = W * c
        G = gaps(table(U, W)).reshape(-1, p - 1)
        prof.append(G.min(0) / (B_p(p, n) * cost(U, W, b) ** s))
        perm = (j * np.arange(p)) % p
        Ur = U[:, :, perm]; Wr = W[:, perm]
        # identity check g'(x,d) = g(jx, jd)
        Gr = gaps(table(Ur, Wr)); Go = gaps(table(U, W))
        if n == 2:
            for x1 in range(p):
                for x2 in range(p):
                    for d in range(1, p):
                        worst_id = max(worst_id, abs(Gr[x1, x2, d - 1] - Go[(j * x1) % p, (j * x2) % p, (j * d) % p - 1]))
        Us.append(Ur); Ws.append(Wr)
    U = np.concatenate(Us); W = np.concatenate(Ws)
    h = margin(table(U, W)); P = cost(U, W, b)
    pred = min(sum(Pw[j - 1] ** s * prof[j - 1][(j * d) % p - 1] for j in range(1, F + 1)) for d in range(1, p))
    worst_union = min(worst_union, h / (B_p(p, n) * P ** s) - pred)
    assert abs(P - Ptot) < 1e-9
print(f"relabelling identity max error (n=2 cases): {worst_id:.2e}")
print(f"min over trials of [gamma(union)/B_p - min_d sum_j P_j^s m_j(j d)] = {worst_union:.3e}  (>= 0 required)")
