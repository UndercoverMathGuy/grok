# Check the own-share bound gamma <= B V_s(beta) (pure nets; beta_k = sum_{i at k} |theta_i|^2 / P) and the
# whole-heavy-set lemma V_s(beta) <= heavy(D0) + light(D0) <= heavy(D0) + 2 sqrt(eps) sigma_L, on positive-margin junky nets.
import sys, numpy as np
sys.path.insert(0, '..')
from common import *
rng = np.random.default_rng(21); worst_own = -9; worst_split = -9; npos = 0
for p in (7, 11, 13):
    Gam, ystar = programme(p, 1.5); h = (p - 1) // 2; A = cover_matrix(p); B = 2 * (6 * p) ** -1.5
    for t in range(150):
        y = ystar * rng.uniform(0.3, 1.7, h) + (rng.uniform(size=h) < 0.3) * rng.uniform(0, 0.4, h)
        net = block_net(p, y) + [tuple(rng.uniform(0.1, 0.8) * v for v in n) for n in random_net(p, int(rng.integers(1, 5)), rng, pure=True)]
        z = table(p, net); m = margin(z); P = cost2(net)
        if m <= 0: continue
        npos += 1
        beta = np.zeros(h)
        for e1, e2, w in net:
            k = np.argmax(np.abs(np.fft.fft(e1))[1:h + 1]) + 1           # the neuron's frequency (pure nets)
            beta[k - 1] += e1 @ e1 + e2 @ e2 + w @ w
        beta /= P
        Vs = (A @ beta ** 1.5).min()
        worst_own = max(worst_own, (m / P ** 1.5) / (B * Vs) - 1)
        for eps in (0.02, 0.05, 0.1):
            H = beta >= eps; L = ~H
            for D0 in range(h):
                split = (A[D0, H] * beta[H] ** 1.5).sum() + 2 * np.sqrt(eps) * beta[L].sum()
                worst_split = max(worst_split, Vs - split)
print(f"positive-margin nets: {npos};  max(gamma/(B V_s) - 1) = {worst_own:.3e} (<= 0 needed);  max(V_s - split bound) = {worst_split:.3e} (<= 0 needed)")
