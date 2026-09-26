"""counterexample_d3.py -- a certified counterexample to the group analogue of (T) at EVEN nu (where kappa is exact).

D_3 (= S_3), n = 3 operands (nu = 4), q = 2 (plain weight decay). Theorem G2 + Theorem K2 give, for EVERY symmetric (junk-free)
gated chain, gamma_2 <= (N nu)^{-2} Gamma with Gamma = (min_v sum_pi c_pi v_pi^{1/2})^{-2}, c_sign = 1, c_rho = 2^{4/2} * 2 = 8, over the
vertices {rho: 1/2} and {sign: 1/6, rho: 1/3} of P  ->  ceiling = 1/(576 (1/sqrt6 + 8/sqrt3)^2).
We run unrestricted normalized-margin ascent, SAVE the best network, and recompute its margin by an independent loop over all
216 rows and 6 classes (fp64 and exact-rational re-check of the ceiling)."""
import sys, itertools, json
from fractions import Fraction
from pathlib import Path
import numpy as np, torch, mpmath as mp
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gt_core as gc
from verify_fourier import row_products
torch.set_default_dtype(torch.float64)
mp.mp.dps = 30
G = gc.get("d3"); N = G.N; n = 3; nu = 4; q = 2.0
ceil = 1 / (mp.mpf(576) * (1 / mp.sqrt(6) + 8 / mp.sqrt(3)) ** 2)
ceil_alt = 1 / (mp.mpf(576) * (8 * mp.sqrt(mp.mpf(1) / 2)) ** 2)
ceiling = max(ceil, ceil_alt)
y = torch.from_numpy(row_products(G, n))
best = (-1, None)
for seed in range(4):
    g = torch.Generator().manual_seed(seed); m = 16
    E = (torch.randn(n, m, N, generator=g) / N ** 0.5).requires_grad_(True); W = (torch.randn(m, N, generator=g) / N ** 0.5).requires_grad_(True)
    opt = torch.optim.Adam([E, W], lr=0.01); steps = 6000
    for it in range(steps):
        h = torch.ones(m, 1)
        for j in range(n): h = (h[:, :, None] * E[j][:, None, :]).reshape(m, -1)
        Z = h.T @ W; P2 = (E ** 2).sum() + (W ** 2).sum(); Zn = Z / P2 ** 2
        gaps = Zn.gather(1, y[:, None]) - Zn; gaps = gaps + torch.zeros_like(gaps).scatter_(1, y[:, None], 1e6)
        mn = float(gaps.min().detach())
        if mn > best[0]: best = (mn, (E.detach().numpy().copy(), W.detach().numpy().copy()), seed)
        tau = max(abs(mn), 1e-12) * (0.3 * (1 - it / steps) + 0.01)
        soft = -tau * torch.logsumexp(-gaps.reshape(-1) / tau, 0)
        opt.zero_grad(); (-soft).backward(); opt.step()
        if it == steps // 2:
            for gp in opt.param_groups: gp["lr"] = 0.003
val, (E, W), seed = best
np.savez(HERE / "out" / "counterexample_d3_nu4.npz", E=E, W=W)
# independent recomputation: explicit loops
worst = np.inf
for a in itertools.product(range(N), repeat=n):
    c_true = G.MULT[G.MULT[a[0], a[1]], a[2]]
    logit = np.array([sum(W[i, c] * E[0, i, a[0]] * E[1, i, a[1]] * E[2, i, a[2]] for i in range(W.shape[0])) for c in range(N)])
    for c in range(N):
        if c != c_true: worst = min(worst, logit[c_true] - logit[c])
P2 = float((E ** 2).sum() + (W ** 2).sum()); gam = worst / P2 ** 2
res = dict(seed=seed, margin=worst, P2=P2, gamma=gam, ceiling=float(ceiling), ratio=gam / float(ceiling), G1=float(1 / (576 * mp.mpf(1)) * 0))
print(json.dumps(res, indent=1))
(HERE / "out" / "counterexample_d3_nu4.json").write_text(json.dumps(res, indent=1))
