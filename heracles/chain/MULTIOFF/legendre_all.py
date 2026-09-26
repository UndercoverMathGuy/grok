# Obstruction check: the Legendre direction on the dense set is flat under EVERY twist, so even uniform-over-all-offsets
# multi-offset aiming only exposes O(1/sqrt p) of it.  (Single pattern; the ratio is the exact exposure/denominator.)
import numpy as np, torch
from mo import Body
for p in (59, 113, 211, 1009, 4001):
    h = (p - 1) // 2; T = list(range(1, h + 1))
    leg = np.array([1.0 if pow(k, (p - 1) // 2, p) == 1 else -1.0 for k in T])
    Ds = np.arange(1, p); lam = np.ones(p - 1) / (p - 1)
    body = Body(p, T, Ds, lam, mode='1')
    best = 9
    for z in (leg, 1j * leg):
        J = torch.tensor(np.stack([z, 0 * z, 0 * z]).astype(complex))
        best = min(best, float(body.ratio(J)))
    print(f"p={p}: uniform-all multi-offset exposure of the Legendre direction = {best:.4f}   (2/sqrt(p) = {2 / np.sqrt(p):.4f})", flush=True)
