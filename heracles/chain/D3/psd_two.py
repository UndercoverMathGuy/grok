# Two chain orders (slot 1 first / slot 2 first) give two PSD matrices; each carries one of J1, J2 exactly and the other up to u.
# min(B1, B2) <= 3e^2 + 2eta^2 + 2 sqrt2 u eta + 2u^2  (vs fable's single-order 4 u eta). Recompute the curve and threshold.
import numpy as np
def Lc(e, h, cross):
    u = np.sqrt(2 * e * (e - 1)); B = 3 * e ** 2 + 2 * h ** 2 + cross * h * u + 2 * u ** 2
    return np.maximum(e ** (2 / 3), 3 ** (1 / 3) * e ** (4 / 3) * (3 / B) ** (1 / 3))
E = np.linspace(1, np.sqrt(3), 1601)[:, None]; H = np.linspace(0, 1, 2401)[None, :] * np.sqrt(3) * E
def kap(a, cross):
    rhs = np.clip(1 - a * H, 0, None) ** (2 / 3)
    with np.errstate(divide='ignore'):
        return np.where(rhs > 0, Lc(E, H, cross) / rhs, np.inf).min()
for cross, name in ((4.0, 'fable single order'), (2 * np.sqrt(2), 'two orders')):
    # threshold: smallest a with kappa_PSD(a) >= 3^{1/3}
    lo, hi = 0.2, 0.3
    for _ in range(40):
        m = (lo + hi) / 2
        if kap(m, cross) >= 3 ** (1 / 3) - 1e-12: hi = m
        else: lo = m
    worst = min(kap(a, cross) / min(3 ** (1 / 3), (1 - np.sqrt(3) * a) ** (-2 / 3)) for a in np.linspace(0.01, 0.27, 27))
    print(f"{name:20s}: eta-form threshold {hi:.4f} (sharp would be {(np.sqrt(3) - 1) / 3:.4f}); worst kappa_PSD/kappa_lone on [0.01,0.27] = {worst:.4f}")
