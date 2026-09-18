"""audit_ascent.py -- try to BEAT the claimed bound by direct margin ascent on the exact families at small p.
Objective: softmin_rows,c (z_y - z_c) / P_q^{nu/q},  P_q = sum_i ||theta_i||^q  (scale invariant), annealed softmin, Adam.
Bound (derived in PROOF_AUDIT.md from the doc's own chain):  gamma_q <= 2 c_n (2p)^{-nu/2} Gamma_free(p, s), s = nu/max(q,2).
Reports best gamma/bound over seeds (random init and warm start from the junk-free construction), the clean-spectrum
count, and the junk fraction.   uv run python heracles/audit_ascent.py"""
import json, math, sys, itertools
import numpy as np, torch
from audit_common import rows, logits, margin, clean, clean_table, Gmat, cn, Pq
from audit_program import free, ladder
from audit_construct import build


def ascend(p, n, act, q, m, seed, steps=12000, warm=None):
    torch.manual_seed(seed); nu = n + 1
    ops, y = rows(p, n)
    if warm is None:
        E = torch.randn(n, m, p) * 0.5; W = torch.randn(m, p) * 0.5
    else:
        E, W = warm; E = E + 0.02 * torch.randn_like(E); W = W + 0.02 * torch.randn_like(W)
    E.requires_grad_(True); W.requires_grad_(True)
    opt = torch.optim.Adam([E, W], lr=0.01)
    oh = torch.nn.functional.one_hot(y, p).bool()
    best = -1e9; bestEW = None
    for t in range(steps):
        z = logits(E, W, ops, act)
        d = (z.gather(1, y[:, None]) - z)[~oh] / Pq(E, W, q) ** (nu / q)
        sc = d.detach().abs().mean() + 1e-30
        beta = 2.0 * (1 + 400 * (t / steps) ** 2)
        sm = -torch.logsumexp(-beta * d / sc, 0) / beta * sc
        opt.zero_grad(); (-sm / sc).backward(); opt.step()
        if t % 200 == 0 or t == steps - 1:
            v = float(d.min())
            if v > best: best, bestEW = v, (E.detach().clone(), W.detach().clone())
        if t == int(0.7 * steps):
            for g_ in opt.param_groups: g_["lr"] = 0.002
    return best, bestEW


def describe(E, W, p, n, act):
    ops, y = rows(p, n); z = logits(E, W, ops, act); g = clean(z, y, p)
    zc = z - z.mean(1, keepdim=True); gt = clean_table(g, y, p); gt = gt - gt.mean(1, keepdim=True)
    junk = float((zc - gt).square().sum() / zc.square().sum())
    a = np.abs(np.fft.fft(g.numpy()) / p)[1:(p - 1) // 2 + 1]
    n2 = (E.square().sum((0, 2)) + W.square().sum(1)).numpy()
    return junk, (a / a.max()).round(3).tolist(), int((n2 > 0.01 * n2.max()).sum())


if __name__ == "__main__":
    out = {}
    cfgs = [(11, 2, "prod", q) for q in (1.5, 2.0, 2.5, 3.0)] + [(11, 2, "pow", q) for q in (1.5, 2.0, 2.5, 3.0)] + \
           [(7, 3, "prod", q) for q in (2.0, 3.0, 4.0)] + [(7, 3, "pow", 2.0)]
    for p, n, act, q in cfgs:
        nu = n + 1; s = nu / max(q, 2.0)
        lad = ladder(p, s, (p - 1) // 2)
        if s > 1:
            Gf, x = free(p, s, [lad[K][2] for K in lad], seeds=1500, steps=1500)
        else:                                            # q = nu: LP, flat spectrum, value p/(p-1)
            Gf = p / (p - 1)
        bound = 2 * cn(n, act) * (2 * p) ** (-nu / 2) * Gf
        res = []
        for seed in range(4):
            b, EW = ascend(p, n, act, q, 40, seed)
            res.append((b / bound, *describe(*EW, p, n, act)))
        # warm start from the cheapest junk-free construction on the equal-ladder K*
        Ks = max(lad, key=lambda K: lad[K][0]); S = lad[Ks][2]
        ph = [tuple([math.pi * l / (n + 1)] * n) for l in range(n + 1)] if act == "prod" else \
             ([tuple(2 * math.pi * l * c / 6 for c in (1, 4)) for l in range(6)] if n == 2 else
              [tuple(2 * math.pi * l * c / 18 for c in (1, 3, 12)) for l in range(18)])
        b, EW = ascend(p, n, act, q, None, 99, warm=build(p, n, S, ph, act))
        res.append((b / bound, *describe(*EW, p, n, act)))
        key = f"p{p}_n{n}_{act}_q{q}"
        out[key] = dict(bound=bound, Gamma_free=float(Gf), K_equal=Ks, runs=res)
        print(key, f"Gamma_free={Gf:.4f} K_eq={Ks}", flush=True)
        for r in res:
            print(f"    gamma/bound={r[0]:+.4f} junk={r[1]:.3f} live_neurons={r[3]} clean_spec={r[2]}", flush=True)
    json.dump(out, open(sys.path[0] + "/audit_ascent.json", "w"), indent=1)
