"""junk_a5_nu4.py -- (T_G) at even nu (kappa exact) from the symmetric optimum: does unrestricted ascent ESCAPE upward from the
schoolbook construction? A5 (simple: no sign character, no index-2 coset) and S4, n = 3 (nu = 4), q = 2.
Start: the exact symmetric optimum (Theorem G2 + K2) + 16 extra small neurons + relative noise eps; Adam on the soft-min of gamma."""
import sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import gt_core as gc
from junk_nu4 import ceil
from construct_verify import build_net
from verify_fourier import row_products
torch.set_default_dtype(torch.float64); torch.set_num_threads(8)
def ascent_init(G, n, E0, W0, steps, lr, q=2.0):
    nu = n + 1; y = torch.from_numpy(row_products(G, n)); m = W0.shape[0]
    E = torch.from_numpy(E0).clone().requires_grad_(True); W = torch.from_numpy(W0).clone().requires_grad_(True)
    opt = torch.optim.Adam([E, W], lr=lr); best = -np.inf; first = None
    for it in range(steps):
        h = torch.ones(m, 1)
        for j in range(n): h = (h[:, :, None] * E[j][:, None, :]).reshape(m, -1)
        Z = h.T @ W; P2 = (E ** 2).sum() + (W ** 2).sum(); Zn = Z / P2 ** (nu / 2)
        gaps = Zn.gather(1, y[:, None]) - Zn; gaps = gaps + torch.zeros_like(gaps).scatter_(1, y[:, None], 1e6)
        mn = float(gaps.min().detach()); best = max(best, mn); first = mn if first is None else first
        tau = max(abs(mn), 1e-14) * (0.1 * (1 - it / steps) + 0.005)
        soft = -tau * torch.logsumexp(-gaps.reshape(-1) / tau, 0)
        opt.zero_grad(); (-soft).backward(); opt.step()
    return first, best
lines = []
def log(s):
    print(s, flush=True); lines.append(s); (HERE / "out" / "junk_a5_nu4.log").write_text("\n".join(lines) + "\n")
for gname, amps in [("s4", None), ("a5", None)]:
    G = gc.get(gname); ce = ceil(G, 4, 2.0); gs = ce["school"][0]
    irr, _ = gc.programme_data(G); cov = np.array([p["cov"] for p in irr]); verts = gc.vertices(cov, exact=False)
    from programme import kap_fn
    c = gc.prices(irr, 4, 2.0, kap_fn("school", 4, 2.0)); Gam, k, _ = gc.solve(verts, c, 0.5)
    amps = {irr[i]["name"]: float(x) for i, x in enumerate(verts[k]) if x > 0}
    E0, W0 = build_net(G, 3, amps)
    log(f"== {gname} n=3 q=2: exact symmetric ceiling {gs:.6e} on {amps} ({W0.shape[0]} neurons)")
    rng = np.random.default_rng(0)
    for eps in ((0.02,) if gname == "a5" else (0.0, 0.02, 0.1)):
        pad = 16; sc = np.abs(E0).max()
        E = np.concatenate([E0, rng.normal(size=(3, pad, G.N)) * 0.05 * sc], 1); W = np.concatenate([W0, rng.normal(size=(pad, G.N)) * 0.05 * sc], 0)
        E = E + eps * sc * rng.normal(size=E.shape); W = W + eps * sc * rng.normal(size=W.shape)
        f, b = ascent_init(G, 3, E, W, 700 if gname == "a5" else 3000, 0.002)
        log(f"   eps {eps}: start gamma/ceiling {f / gs:.5f} -> best after ascent {b / gs:.5f}")
