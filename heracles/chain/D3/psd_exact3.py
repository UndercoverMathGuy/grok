# Replace Hoelder (sum lam^{2/3} >= S1^{4/3} / S2^{1/3}) by the exact minimum over 3 eigenvalues lam >= 0 with sum = S1, sum sq = S2.
# Parametrise: lam = S1/3 (1 + r cos(th + 2 pi j/3) * sqrt2) ... the circle {sum t = 0, sum t^2 = rho^2}.
import numpy as np
TH = np.linspace(0, 2 * np.pi, 20001)
def exact_min(S1, S2):
    m = S1 / 3; rho2 = S2 - S1 ** 2 / 3                    # sum (lam - m)^2
    if rho2 <= 0: return 3 * m ** (2 / 3)
    rho = np.sqrt(rho2 * 2 / 3)                             # lam_j = m + rho cos(th - 2 pi j / 3): sum sq dev = 3 rho^2 / 2
    lam = m + rho * np.cos(TH[None, :] - 2 * np.pi * np.arange(3)[:, None] / 3)
    ok = (lam >= -1e-12).all(0)
    if not ok.any(): return np.inf
    return (np.clip(lam[:, ok], 0, None) ** (2 / 3)).sum(0).min()
def cost_lb(e, h, cross):
    u = np.sqrt(2 * e * (e - 1)); B = 3 * e ** 2 + 2 * h ** 2 + cross * h * u + 2 * u ** 2
    B = min(B, 9 * e ** 2)                                   # tr M^2 <= (tr M)^2 for PSD
    return max(e ** (2 / 3), 3 ** (-2 / 3) * exact_min(3 * e, B))
def kap(a, cross, E=np.linspace(1, np.sqrt(3), 161), n_eta=241):
    best = np.inf
    for e in E:
        for h in np.linspace(0, np.sqrt(3) * e, n_eta):
            rhs = max(1 - a * h, 0) ** (2 / 3)
            if rhs > 0: best = min(best, cost_lb(e, h, cross) / rhs)
    return best
# sanity: Hoelder vs exact at a sample
print("sample e=1.03, eta=1.4:", 3 ** (1 / 3) * 1.03 ** (4 / 3) * (3 / (3 * 1.03 ** 2 + 2 * 1.96 + 2 * np.sqrt(2) * 1.4 * np.sqrt(2 * 1.03 * .03) + 2 * 2 * 1.03 * .03)) ** (1 / 3), cost_lb(1.03, 1.4, 2 * np.sqrt(2)))
for cross, name in ((4.0, 'single order'), (2 * np.sqrt(2), 'two orders')):
    lo, hi = 0.2, 0.3
    for _ in range(14):
        m = (lo + hi) / 2
        if kap(m, cross) >= 3 ** (1 / 3) - 1e-9: hi = m
        else: lo = m
    worst = min(kap(a, cross) / min(3 ** (1 / 3), (1 - np.sqrt(3) * a) ** (-2 / 3)) for a in (0.05, 0.1, 0.15, 0.2, 0.22, 0.24, 0.25))
    print(f"{name}, exact 3-eigenvalue min: threshold {hi:.4f}; worst kappa/kappa_lone {worst:.4f}", flush=True)
