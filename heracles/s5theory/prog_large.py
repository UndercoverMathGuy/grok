"""prog_large.py -- the weighted programme for groups with many irreps (Z_120, Z_11 x Z_11, Z_p, D_n), and the
complex-weight closure.

    Gamma = max_{b in simplex} min_{C != e} sum_pi b_pi^s w_pi cov_pi(C),   w_pi = c_pi^{-s},  s = nu/q
(equivalent to the y-form in programme.py). No vertex enumeration here (too many vertices): support search on the
equal-share relaxation (weighted coverage, swap local search, many restarts) followed by free-share SLSQP polish on the
best supports per K, plus a dense-start free-share run. COMPUTER-ASSISTED, no optimality certificate (same status as
the cyclic Gamma_s(p) of THEORY_nsum / VERIFY.md).

Pricing modes (per real irrep pi; eps = 1 self-dual, 2 conjugate pair; d = complex dim):
  real_H     real weights, symmetric class (H): complex pair kappa = nu^{1-r} (ELEGANT), real character kappa = 1,
             d-dim real type kappa = d^{nu(1-r)} (schoolbook; exact at even nu)
  real_unres real weights, every network: kappa = 1
  complex    complex weights + real-part readout (Theorem C): every character priced 1 (one neuron), pairs one-sided
"""
import argparse, itertools, json, math, sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gt_core as gc  # noqa: E402

OUT = HERE / "out"; OUT.mkdir(exist_ok=True)


def weights(irr, nu, q, mode):
    r = q / nu; s = nu / q; w = []
    for p in irr:
        d, eps = p["d"], p["eps"]
        if mode == "complex":
            c = 1.0 * (d ** (nu * (1 - r)) if d > 1 else 1.0) * d ** (q / 2)
        else:
            if eps == 2:
                kap = nu ** (1 - r) if mode == "real_H" else 1.0
            elif d == 1:
                kap = 1.0
            else:
                kap = d ** (nu * (1 - r)) if mode == "real_H" else 1.0
            c = kap * eps ** (q / 2 - r) * d ** (q / 2)
        w.append(c ** (-s))
    return np.array(w)


def value(g, b, s):
    return float(((b ** s) @ g).min())


def polish(g, S, s, restarts=6, rng=None, b0=None):
    rng = rng or np.random.default_rng(0)
    Gs = g[list(S)]; K = len(S)
    best_b = np.full(K, 1 / K) if b0 is None else b0.copy()
    best = value(Gs, best_b, s)
    if K == 1:
        return best, best_b
    for t in range(restarts):
        y0 = best_b if t == 0 else rng.dirichlet(np.ones(K))
        z0 = np.append(y0, value(Gs, y0, s))
        cons = [dict(type="ineq", fun=lambda z: (np.abs(z[:-1]) ** s) @ Gs - z[-1]),
                dict(type="eq", fun=lambda z: np.abs(z[:-1]).sum() - 1)]
        try:
            res = minimize(lambda z: -z[-1], z0, method="SLSQP", constraints=cons, bounds=[(0, 1)] * K + [(None, None)],
                           options=dict(maxiter=500, ftol=1e-15))
        except Exception:
            continue
        b = np.abs(res.x[:-1]); b /= b.sum(); v = value(Gs, b, s)
        if v > best:
            best, best_b = v, b
    return best, best_b


def support_search(g, K, s, rng, restarts=60):
    """swap local search on the equal-share value K^{-s} min_C sum_{S} g."""
    R = g.shape[0]; out = {}
    for _ in range(restarts):
        S = sorted(rng.choice(R, K, replace=False).tolist()); cur = g[S].sum(0).min(); improved = True
        while improved:
            improved = False
            for i in range(K):
                base = g[[x for j, x in enumerate(S) if j != i]].sum(0)
                cand = (base[None, :] + g).min(1); cand[S] = -1; j = int(cand.argmax())
                if cand[j] > cur + 1e-12:
                    S[i] = j; cur = cand[j]; improved = True
            S = sorted(S)
        out[tuple(S)] = cur * K ** (-s)
    return sorted(out.items(), key=lambda kv: -kv[1])


def solve(G, nu, q, mode, kmax=14, restarts=60, top=4, seed=0):
    irr, classes = gc.programme_data(G)
    s = nu / q
    w = weights(irr, nu, q, mode)
    g = w[:, None] * np.array([p["cov"] for p in irr])
    rng = np.random.default_rng(seed)
    rows = []
    exhaustive_small = len(irr) <= 12
    for K in range(1, min(kmax, len(irr)) + 1):
        if exhaustive_small:
            cands = [(S, None) for S in itertools.combinations(range(len(irr)), K)]
        else:
            cands = support_search(g, K, s, rng, restarts)[:top]
        for S, _ in cands:
            if g[list(S)].sum(0).min() <= 1e-12:
                continue
            v, b = polish(g, S, s, rng=rng)
            if b.min() < 1e-4 and K > 1:
                continue                                   # a zero share = a smaller support
            rows.append(dict(K=K, support=[irr[i]["name"] for i in S], val=v, shares=[round(float(x), 4) for x in b]))
    # dense start: all irreps, free shares
    v, b = polish(g, list(range(len(irr))), s, restarts=3, rng=rng)
    keep = [i for i in range(len(irr)) if b[i] > 1e-3]
    v2, b2 = polish(g, keep, s, restarts=3, rng=rng)
    rows.append(dict(K=len(keep), support=[irr[i]["name"] for i in keep], val=v2, shares=[round(float(x), 4) for x in b2], dense_start=True))
    rows.sort(key=lambda r: -r["val"])
    top_v = rows[0]["val"]
    byK = {}
    for r_ in rows:
        byK.setdefault(r_["K"], r_)
    return dict(argmax=rows[0], byK={K: dict(rel=round(v_["val"] / top_v, 4), support=v_["support"], shares=v_["shares"]) for K, v_ in sorted(byK.items())},
                Gamma=top_v, gamma_bound=(G.N * nu) ** (-nu / 2) * top_v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="z120,z11x11,z113,d113")
    ap.add_argument("--cells", default="3:2,4:2,6:2")
    ap.add_argument("--modes", default="real_H,real_unres,complex")
    ap.add_argument("--restarts", type=int, default=60)
    a = ap.parse_args()
    out = {}
    for gname in a.groups.split(","):
        G = gc.get(gname) if gname != "z113" else gc.groups_mod.get_group("z113")
        for cell in a.cells.split(","):
            nu, q = int(cell.split(":")[0]), float(cell.split(":")[1])
            for mode in a.modes.split(","):
                if gname.startswith("d") and mode == "complex":
                    continue
                res = solve(G, nu, q, mode, restarts=a.restarts)
                key = f"{gname}_nu{nu}_q{q:g}_{mode}"; out[key] = res
                am = res["argmax"]
                print(f"{key}: Gamma {res['Gamma']:.6g} gamma {res['gamma_bound']:.4e} argmax K={am['K']} {am['support']} shares {am['shares']}", flush=True)
                print("      byK: " + "  ".join(f"{K}:{v['rel']:.3f}" for K, v in res["byK"].items()), flush=True)
    (OUT / "prog_large.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
