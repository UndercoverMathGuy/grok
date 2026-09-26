# V1: Step 2 split gamma/gamma_sym = rho * tau * eta^{3/2}, plus the facts it leans on:
#     Lemma 2 (rho <= 1), programme definition (tau <= 1), budget (B) per neuron, eta <= 3^{1/3} always, eta <= 1 if junk-free.
import numpy as np
from common import *
rng = np.random.default_rng(2)
r = 2 / 3
worst = dict(ident=0, rho=-9, tau=-9, B=-9, eta=-9, eta_jf=-9)
for p in (5, 7, 11, 13):
    Gam, ystar = programme(p, 1.5)
    A = cover_matrix(p)
    for trial in range(60):
        kind = trial % 3
        if kind == 0:   # block + small junky neurons (positive margin)
            net = block_net(p, ystar) + [tuple(0.3 * v for v in n) for n in random_net(p, 3, rng, pure=bool(trial % 2))]
        elif kind == 1:  # random junk-free: blocks with random shares and random common phases
            y = rng.uniform(0, 1, len(ystar)) * (rng.uniform(size=len(ystar)) < 0.6)
            net = block_net(p, y + 1e-3)
        else:            # fully random
            net = random_net(p, 8, rng, pure=bool(trial % 2))
        z = table(p, net)
        m = margin(z)
        G, J = spectrum(z)
        aG = np.abs(G[1:])
        H = (aG ** r).sum()
        F = (A @ aG).min()
        P = cost2(net)
        Q = (p / 6) * P
        rho = m / (2 * p ** -3 * F)
        tau = F / (Gam * H ** 1.5)
        eta = 3 ** (1 - r) * H / Q
        lhs = gamma(p, net) / gamma_sym(p, Gam)
        worst['ident'] = max(worst['ident'], abs(lhs - rho * tau * eta ** 1.5) / max(1, abs(lhs)))
        worst['rho'] = max(worst['rho'], rho)
        worst['tau'] = max(worst['tau'], tau)
        worst['eta'] = max(worst['eta'], eta / 3 ** (1 / 3))
        if kind == 1:
            worst['eta_jf'] = max(worst['eta_jf'], eta)
        for e1, e2, w in net:   # budget (B): sum_k |c_ik|^{2/3} <= (p/6) |theta_i|^2
            c = np.array([np.fft.fft(e1)[k] * np.fft.fft(e2)[k] * np.conj(np.fft.fft(w)[k]) for k in range(1, (p - 1) // 2 + 1)])
            worst['B'] = max(worst['B'], (np.abs(c) ** r).sum() / ((p / 6) * (e1 @ e1 + e2 @ e2 + w @ w)))
print({k: float(f"{v:.3e}") for k, v in worst.items()})
print("need: ident~0, rho<=1, tau<=1, B<=1, eta/3^(1/3)<=1, eta_jf<=1")
