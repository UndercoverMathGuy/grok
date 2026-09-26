"""junk_ascent.py -- evidence for / against the group analogue of (T): can an UNRESTRICTED gated chain (junk allowed) beat the
symmetric-class ceiling (Theorem G2 with schoolbook kappa)?

Direct normalized-margin ascent (Adam on a temperature-annealed soft-min of gamma_q, fp64, full table), n = 2 (nu = 3), q = 2,
real weights, from (a) random init at several widths and (b) the schoolbook optimum + perturbation (local escape test).
Reports gamma / gamma_school(G2) and gamma / gamma_unres(G1). A value > 1 of the first would REFUTE (T_G) (or kappa = schoolbook).
Groups: S3 (irreps sign, 2), S4, and (cheap) D_7.
"""
import sys, json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gt_core as gc  # noqa: E402
from programme import kap_fn  # noqa: E402
from construct_verify import build_net  # noqa: E402

torch.set_default_dtype(torch.float64)
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def ceilings(G, nu, q):
    irr, _ = gc.programme_data(G); cov = np.array([p["cov"] for p in irr]); verts = gc.vertices(cov, exact=False); r = q / nu
    out = {}
    for mode in ("school", "unres"):
        c = gc.prices(irr, nu, q, kap_fn(mode, nu, q)); Gam, k, _ = gc.solve(verts, c, r)
        out[mode] = ((G.N * nu) ** (-nu / 2) * Gam, {irr[i]["name"]: float(x) for i, x in enumerate(verts[k]) if x > 0})
    return out


def ascent(G, m, q, steps, seed, init=None, lr=0.01):
    N = G.N; nu = 3
    y = torch.from_numpy(G.MULT.reshape(-1))
    g = torch.Generator().manual_seed(seed)
    if init is None:
        E = torch.randn(2, m, N, generator=g) / N ** 0.5; W = torch.randn(m, N, generator=g) / N ** 0.5
    else:
        E, W = [torch.from_numpy(x).clone() for x in init]
    E.requires_grad_(True); W.requires_grad_(True)
    opt = torch.optim.Adam([E, W], lr=lr)
    best = (-np.inf, None)
    for it in range(steps):
        h = (E[0][:, :, None] * E[1][:, None, :]).reshape(E.shape[1], -1)
        Z = h.T @ W
        th2 = (E ** 2).sum((0, 2)) + (W ** 2).sum(1); Pq = (th2 ** (q / 2)).sum()
        Zn = Z / Pq ** (nu / q)
        gaps = Zn.gather(1, y[:, None]) - Zn
        gaps = gaps + torch.zeros_like(gaps).scatter_(1, y[:, None], 1e6)
        mn = float(gaps.min().detach())
        if mn > best[0]:
            best = (mn, (E.detach().numpy().copy(), W.detach().numpy().copy()))
        frac = it / steps
        tau = max(abs(mn), 1e-9) * (0.3 * (1 - frac) + 0.01)
        soft = -tau * torch.logsumexp(-gaps.reshape(-1) / tau, 0)
        opt.zero_grad(); (-soft).backward(); opt.step()
        if it == steps // 2:
            for gp in opt.param_groups: gp["lr"] = lr * 0.3
    return best


def main():
    out = {}
    for gname, widths, seeds, steps in [("d3", [6, 12, 24], 4, 6000), ("d7", [16, 32], 3, 6000), ("s4", [40, 90], 3, 6000)]:
        G = gc.get(gname); q = 2.0
        ce = ceilings(G, 3, q)
        gs, gu = ce["school"][0], ce["unres"][0]
        log(f"== {gname}: gamma_school(G2) {gs:.6e} on {ce['school'][1]}; gamma_unres(G1) {gu:.6e}")
        rows = []
        for m in widths:
            for sd in range(seeds):
                val, _ = ascent(G, m, q, steps, sd)
                rows.append(dict(m=m, seed=sd, ratio_school=val / gs, ratio_unres=val / gu))
                log(f"   random init m {m:3d} seed {sd}: gamma/school {val / gs:.6f}  gamma/unres {val / gu:.6f}")
        # local escape from the schoolbook optimum: build it, pad with zero-ish neurons, perturb
        amps = ce["school"][1]
        E0, W0 = build_net(G, 2, amps)
        rng = np.random.default_rng(0)
        for eps in (1e-3, 1e-2, 5e-2):
            pad = 8
            E = np.concatenate([E0, rng.normal(size=(2, pad, G.N)) * 1e-3], 1); W = np.concatenate([W0, rng.normal(size=(pad, G.N)) * 1e-3], 0)
            sc = np.abs(E0).max()
            E = E + eps * sc * rng.normal(size=E.shape); W = W + eps * sc * rng.normal(size=W.shape)
            val, _ = ascent(G, E.shape[1], q, 4000, 0, init=(E, W), lr=0.002)
            rows.append(dict(start=f"school+{eps}", ratio_school=val / gs))
            log(f"   schoolbook + noise {eps}: gamma/school after ascent {val / gs:.6f}")
        out[gname] = dict(school=gs, unres=gu, rows=rows)
    (HERE / "out" / "junk_ascent.json").write_text(json.dumps(out, indent=1))
    (HERE / "out" / "junk_ascent.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
