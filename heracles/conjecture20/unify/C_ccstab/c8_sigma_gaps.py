# Exact five-gap decomposition of the per-neuron misalignment (budget chain of SPLIT E7 / E3):
#   B_p a_i - sum_l Re beta_il = G_amgm + G_dc + G_mix + G_hol + G_phase, all >= 0, each vanishing to 2nd order at
#   aligned single-frequency balanced DC-free neurons.  c = 2 (2p)^{-nu/2}; A_t(j) = |vhat_t(j)| / (sqrt(p/2) ||v_t||).
import numpy as np, math
from sclib import *
rng = np.random.default_rng(4); worst_id = 0; worst_neg = 0
for (p, n) in [(5, 2), (7, 2), (5, 3), (11, 2), (7, 4)]:
    nu = n + 1; F = (p - 1) // 2; c = 2 * (2 * p) ** (-nu / 2); B = Bp(p, n)
    for t in range(200):
        U = rng.normal(size=(1, n, p)); W = rng.normal(size=(1, p))
        if t % 3 == 0:   # near a single-frequency aligned neuron
            x = np.arange(p); ph = rng.uniform(0, 6, n)
            U[0] = np.array([np.cos(2 * np.pi * x / p + ph[k]) for k in range(n)]) + 0.05 * U[0]
            W[0] = np.cos(2 * np.pi * x / p + ph.sum()) + 0.05 * W[0]
        net = Net(U, W, 2.0); a = masses(net)[0]
        facs = list(U[0]) + [W[0]]; norms = np.array([np.linalg.norm(v) for v in facs])
        At = np.array([[abs(np.fft.fft(v)[j]) / (math.sqrt(p / 2) * np.linalg.norm(v)) for j in range(1, F + 1)] for v in facs])
        prod_norm = norms.prod()
        G_amgm = B * a - c * prod_norm
        G_dc = c * prod_norm * (1 - np.prod(np.linalg.norm(At, axis=1)))
        G_mix = c * prod_norm * (np.prod(np.linalg.norm(At, axis=1)) - np.prod(np.sum(At ** nu, axis=1) ** (1 / nu)))
        G_hol = c * prod_norm * (np.prod(np.sum(At ** nu, axis=1) ** (1 / nu)) - np.sum(np.prod(At, axis=0)))
        at = net.atoms()[0]
        G_phase = np.abs(at).sum() - np.real(at).sum()
        tot = G_amgm + G_dc + G_mix + G_hol + G_phase
        worst_id = max(worst_id, abs(tot - (B * a - np.real(at).sum())) / (B * a))
        worst_neg = min(worst_neg, min(G_amgm, G_dc, G_mix, G_hol, G_phase) / (B * a))
        assert abs(c * prod_norm * np.sum(np.prod(At, axis=0)) - np.abs(at).sum()) < 1e-10 * B * a
print('max |sum of gaps - sigma_i B_p a_i| / (B_p a_i) =', worst_id)
print('min gap / (B_p a_i)  (>= 0)                    =', worst_neg)
