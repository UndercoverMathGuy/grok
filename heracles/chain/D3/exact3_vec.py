# Threshold of the two-order matrix route when Hoelder is replaced by the exact minimum of sum lam^{2/3} over 3 eigenvalues
# (lam >= 0, sum = 3e, sum sq = B). Vectorised; float exploration only.
import numpy as np
TH = np.linspace(0, 2 * np.pi, 1441)
def exact_min(S1, S2):                       # arrays
    m = S1 / 3; rho2 = np.maximum(S2 - S1 ** 2 / 3, 0); rho = np.sqrt(rho2 * 2 / 3)
    out = np.full(S1.shape, np.inf)
    for j0 in range(0, len(TH), 90):
        th = TH[j0:j0 + 90]
        lam = m[..., None, None] + rho[..., None, None] * np.cos(th[None, None, None, :] - 2 * np.pi * np.arange(3)[None, None, :, None] / 3)
        ok = (lam >= -1e-12).all(-2)
        val = np.where(ok, (np.clip(lam, 0, None) ** (2 / 3)).sum(-2), np.inf)
        out = np.minimum(out, val.min(-1))
    return out
E = np.linspace(1, np.sqrt(3), 241)[:, None]; H = np.linspace(0, 1, 361)[None, :] * np.sqrt(3) * E
u = np.sqrt(2 * E * (E - 1))
for name, cross in (('two orders + Hoelder', None), ('two orders + exact 3-eig', 2 * np.sqrt(2))):
    B = np.minimum(3 * E ** 2 + 2 * H ** 2 + 2 * np.sqrt(2) * H * u + 2 * u ** 2, 9 * E ** 2)
    if cross is None:
        Lv = np.maximum(E ** (2 / 3), 3 ** (1 / 3) * E ** (4 / 3) * (3 / B) ** (1 / 3))
    else:
        Lv = np.maximum(E ** (2 / 3), 3 ** (-2 / 3) * exact_min(np.broadcast_to(3 * E, B.shape).copy(), B))
    def kap(a):
        rhs = np.clip(1 - a * H, 0, None) ** (2 / 3)
        with np.errstate(divide='ignore'):
            return np.where(rhs > 0, Lv / rhs, np.inf).min()
    lo, hi = 0.2, 0.3
    for _ in range(30):
        mid = (lo + hi) / 2
        if kap(mid) >= 3 ** (1 / 3) - 1e-12: hi = mid
        else: lo = mid
    print(f"{name}: eta-form threshold ~ {hi:.4f}", flush=True)
