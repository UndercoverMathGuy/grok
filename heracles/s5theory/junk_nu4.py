"""junk_nu4.py -- the even-nu test of (T_G): at nu = 4 the symmetric-class ceiling is EXACT (Theorem K2), so any network above it
proves that same-irrep junk pays. Unrestricted normalized-margin ascent, n = 3 (nu = 4), q = 2, D_3 and S4, plus the
cyclic control Z_7 (where (T) is believed true)."""
import sys, json, itertools
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gt_core as gc
from programme import kap_fn
from verify_fourier import row_products
torch.set_default_dtype(torch.float64); torch.set_num_threads(4)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
def ceil(G, nu, q):
    irr, _ = gc.programme_data(G); cov = np.array([p["cov"] for p in irr]); verts = gc.vertices(cov, exact=False)
    out = {}
    for mode in ("school", "unres"):
        c = gc.prices(irr, nu, q, kap_fn(mode, nu, q)); Gam, k, _ = gc.solve(verts, c, q / nu)
        out[mode] = ((G.N * nu) ** (-nu / 2) * Gam, gc.support_names(irr, verts[k]))
    return out
def ascent(G, n, m, q, steps, seed, lr=0.01):
    N = G.N; nu = n + 1; y = torch.from_numpy(row_products(G, n))
    g = torch.Generator().manual_seed(seed)
    E = (torch.randn(n, m, N, generator=g) / N ** 0.5).requires_grad_(True); W = (torch.randn(m, N, generator=g) / N ** 0.5).requires_grad_(True)
    opt = torch.optim.Adam([E, W], lr=lr); best = -np.inf
    for it in range(steps):
        h = torch.ones(m, 1)
        for j in range(n): h = (h[:, :, None] * E[j][:, None, :]).reshape(m, -1)
        Z = h.T @ W; th2 = (E ** 2).sum((0, 2)) + (W ** 2).sum(1); Pq = (th2 ** (q / 2)).sum(); Zn = Z / Pq ** (nu / q)
        gaps = Zn.gather(1, y[:, None]) - Zn; gaps = gaps + torch.zeros_like(gaps).scatter_(1, y[:, None], 1e6)
        mn = float(gaps.min().detach()); best = max(best, mn)
        tau = max(abs(mn), 1e-12) * (0.3 * (1 - it / steps) + 0.01)
        soft = -tau * torch.logsumexp(-gaps.reshape(-1) / tau, 0)
        opt.zero_grad(); (-soft).backward(); opt.step()
        if it == steps // 2:
            for gp in opt.param_groups: gp["lr"] = lr * 0.3
    return best
def main():
    for gname, n, widths, seeds, steps in [("d3", 3, [16, 32], 3, 5000), ("z7", 3, [16, 32], 3, 5000), ("s4", 3, [96], 2, 3000)]:
        G = gc.get(gname) if gname != "z7" else gc.groups_mod.abelian((7,), "z7")
        ce = ceil(G, n + 1, 2.0); gs, gu = ce["school"][0], ce["unres"][0]
        log(f"== {gname} n={n} (nu {n+1}) q=2: symmetric ceiling (EXACT at even nu) {gs:.6e} on {ce['school'][1]}; G1 {gu:.6e}")
        for m in widths:
            for sd in range(seeds):
                v = ascent(G, n, m, 2.0, steps, sd)
                log(f"   m {m} seed {sd}: gamma/ceiling {v / gs:.5f}   gamma/G1 {v / gu:.5f}")
        (HERE / "out" / "junk_nu4.log").write_text("\n".join(LOG) + "\n")
if __name__ == "__main__":
    main()
