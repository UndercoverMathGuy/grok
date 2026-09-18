"""audit_program.py -- is the FREE-amplitude sparse programme
    Gamma_s(p) = max { min_D sum_k x_k g_k(D) : sum_k x_k^{1/s} <= 1, x >= 0 },   s = nu/q,
really equal to the equal-amplitude ladder max_K M_K K^{-s}, and is its support K*?  (THEORY_nsum sec 3 asserts
'free amplitudes change the argmax by <= 1 rung'; l2_theorem sec 3 asserts 'free amplitudes do not improve it'.)
Batched multistart softmin ascent + SLSQP polish.  uv run python heracles/audit_program.py"""
import itertools, json, sys
import numpy as np, torch
from scipy.optimize import minimize
from audit_common import Gmat

torch.set_default_dtype(torch.float64)


def ladder(p, s, kmax):
    g = Gmat(p); nf = g.shape[0]; out = {}
    for K in range(1, min(kmax, nf) + 1):
        best, bS = -1, None
        for S in itertools.combinations(range(1, nf), K - 1):
            S = (0,) + S
            m = g[list(S)].sum(0).min()
            if m > best: best, bS = m, S
        out[K] = (best * K ** (-s), best, [k + 1 for k in bS])
    return out


def free(p, s, starts, seeds=4000, steps=3000):
    g = torch.tensor(Gmat(p)); nf = g.shape[0]
    th = torch.randn(seeds, nf) * 3
    for i, S in enumerate(starts):                       # equal-amplitude ladder points as warm starts
        th[i] = -8.0; th[i, [k - 1 for k in S]] = 0.0
    th.requires_grad_(True)
    opt = torch.optim.Adam([th], lr=0.05)
    for t in range(steps):
        beta = 5 * (1 + 60 * t / steps)
        b = torch.softmax(th, 1); x = b ** s
        v = x @ g                                         # (seeds, p-1)
        sm = -torch.logsumexp(-beta * v / v.detach().mean(1, keepdim=True), 1) / beta
        opt.zero_grad(); (-sm.sum()).backward(); opt.step()
    with torch.no_grad():
        b = torch.softmax(th, 1); x = b ** s; val = (x @ g).min(1).values
    order = val.argsort(descending=True)[:8]
    best = (-1, None)
    gn = g.numpy()
    for j in order:                                       # SLSQP polish: max t s.t. x g_D >= t, sum x^{1/s} <= 1
        x0 = np.concatenate([x[j].numpy(), [val[j].item()]])
        cons = [dict(type="ineq", fun=lambda v, gn=gn: v[:-1] @ gn - v[-1]),
                dict(type="ineq", fun=lambda v: 1 - np.sum(np.clip(v[:-1], 0, None) ** (1 / s)))]
        r = minimize(lambda v: -v[-1], x0, constraints=cons, bounds=[(0, None)] * nf + [(None, None)],
                     method="SLSQP", options=dict(maxiter=500, ftol=1e-14))
        xx = np.clip(r.x[:-1], 0, None); xx = xx / np.sum(xx ** (1 / s)) ** s
        vv = (xx @ gn).min()
        if vv > best[0]: best = (vv, xx)
    return best


if __name__ == "__main__":
    res = {}
    for p in (23, 31, 47):
        for s in (1.25, 1.5, 2.0, 2.5):
            lad = ladder(p, s, 7 if p < 47 else 6)
            Ks = max(lad, key=lambda K: lad[K][0])
            val, x = free(p, s, [lad[K][2] for K in lad])
            supp = [(k + 1, round(float(v / x.max()), 3)) for k, v in enumerate(x) if v > 1e-3 * x.max()]
            res[f"p{p}_s{s}"] = dict(K_equal=Ks, gamma_equal=lad[Ks][0], gamma_free=float(val), ratio=float(val / lad[Ks][0]),
                                     K_free=len(supp), supp=supp)
            print(p, s, "equal K*", Ks, f"{lad[Ks][0]:.5f}", "| free", f"{val:.5f}", f"x{val / lad[Ks][0]:.4f}", "K_free", len(supp), supp, flush=True)
    json.dump(res, open(sys.path[0] + "/audit_program.json", "w"), indent=1)
