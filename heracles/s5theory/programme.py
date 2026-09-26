"""programme.py -- exact solution of the group programme by vertex enumeration, and the kappa-robustness test.

For a group G, degree nu = n + 1, penalty exponent q in [2, nu], r = q / nu:
    Gamma(c) = max { min_{C != e} sum_pi y_pi cov_pi(C) : y >= 0, sum_pi c_pi y_pi^r <= 1 }
             = ( min_{v in vert P} sum_pi c_pi v_pi^r )^{-1/r},     P = { y >= 0 : cov^T y >= 1 }     (Lemma V)
with prices c_pi = kappa_pi eps_pi^{q/2 - r} d_pi^{q/2}. gamma_q <= (N nu)^{-nu/2} Gamma.
kappa modes:
    unres   kappa = 1                                        (master bound, every network)
    school  kappa = d^{nu (1 - r)} real type, 1 real char    (schoolbook feature; EXACT at even nu, Theorem K2)
    lower   kappa = d^{(nu - 1)(1 - r)}                      (proven lower bound at odd nu, Theorem K3)
    box     every corner of prod_d [lower_d, school_d]       (robustness: argmax constant on the box iff constant at all corners)
Complex-type pairs (abelian) get kappa = nu^{1 - r} (ELEGANT Theorem 1); real characters kappa = 1.
Usage: uv run python programme.py
"""
import itertools, json, sys
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gt_core as gc  # noqa: E402

OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def kap_fn(mode, nu, q, box_choice=None):
    r = q / nu

    def k(p):
        if p["eps"] == 2:
            return 1.0 if mode == "unres" else nu ** (1 - r)   # complex-type d = 1 (abelian): ELEGANT in the (H) class
        d = p["d"]
        if d == 1:
            return 1.0
        if mode == "unres":
            return 1.0
        if mode == "school":
            return d ** (nu * (1 - r))
        if mode == "lower":
            return d ** ((nu - 1) * (1 - r)) if nu % 2 == 1 else d ** (nu * (1 - r))
        if mode == "box":
            lo, hi = d ** ((nu - 1) * (1 - r)), d ** (nu * (1 - r))
            return hi if box_choice[d] else lo
        if mode == "custom":
            return box_choice[d]
        raise ValueError(mode)
    return k


def describe(irr, v, c, r):
    names = gc.support_names(irr, v)
    y = np.array([float(x) for x in v])
    lead = max((i for i in range(len(irr)) if y[i] > 0), key=lambda i: irr[i]["d"] * y[i] / irr[i]["eps"])
    A = [irr[i]["d"] * y[i] / irr[i]["eps"] for i in range(len(irr))]      # GX amplitude A = d G (coefficient of chi/d)
    rel = {irr[i]["name"]: round(A[i] / A[lead], 4) for i in range(len(irr)) if y[i] > 0}
    shares = c * np.where(y > 0, y, 0) ** r; shares = shares / shares.sum()
    return dict(support=names, K=len(names), amp_rel=rel, shares={irr[i]["name"]: round(float(shares[i]), 4) for i in range(len(irr)) if y[i] > 0})


def ladder(irr, verts, c, r, top=10):
    """best vertex per support (fixed-support values), sorted."""
    V = np.array([[float(x) for x in v] for v in verts])
    costs = (c[None, :] * np.where(V > 0, V, 0) ** r).sum(1)
    best = {}
    for k, v in enumerate(verts):
        key = tuple(gc.support_names(irr, v))
        if key not in best or costs[k] < best[key][0]:
            best[key] = (costs[k], k)
    # fixed-support value of S = best over vertices with support subset of S -> keep only the minimal description
    rows = sorted(((cst ** (-1 / r), key, k) for key, (cst, k) in best.items()), reverse=True)
    g0 = rows[0][0]
    return [dict(support=list(key), rel=round(g / g0, 4), **{k2: v2 for k2, v2 in describe(irr, verts[k], c, r).items() if k2 in ("amp_rel",)}) for g, key, k in rows[:top]]


def run_group(gname, cells, exact=True):
    G = gc.get(gname)
    irr, classes = gc.programme_data(G)
    cov = np.array([p["cov"] for p in irr])
    if exact:
        covq = np.array([[Fraction(x).limit_denominator(10 ** 6) for x in row] for row in cov], dtype=object)
        assert np.allclose(covq.astype(float), cov)
        verts = gc.vertices(covq, exact=True)
    else:
        verts = gc.vertices(cov, exact=False)
    log(f"\n== {gname}: |G| {G.N}; irreps {[(p['name'], p['d'], p['eps']) for p in irr]}; classes {[G.class_names[c] for c in classes]}; {len(verts)} vertices of P")
    log("   cov (d - chi):  " + "; ".join(f"{p['name']}: {list(np.round(p['cov'], 3))}" for p in irr))
    res = {}
    ds = sorted(set(p["d"] for p in irr if p["eps"] == 1 and p["d"] > 1))
    for nu, q in cells:
        r = q / nu; cell = {}
        for mode in ("unres", "school", "lower"):
            if mode == "lower" and nu % 2 == 0:
                continue
            c = gc.prices(irr, nu, q, kap_fn(mode, nu, q))
            Gam, k, costs = gc.solve(verts, c, r)
            gam = (G.N * nu) ** (-nu / 2) * Gam
            d = describe(irr, verts[k], c, r)
            cell[mode] = dict(Gamma=Gam, gamma_bound=gam, argmax=d, ladder=ladder(irr, verts, c, r))
            log(f"  nu {nu} q {q:g} [{mode:6s}] Gamma {Gam:.6g} gamma {gam:.4e}  argmax K={d['K']} {d['support']} amp_rel {d['amp_rel']}")
            log("        ladder: " + "; ".join(f"{'+'.join(x['support'])} {x['rel']:.3f}" for x in cell[mode]["ladder"][:8]))
        if nu % 2 == 1 and ds:
            winners = {}
            for bits in itertools.product([0, 1], repeat=len(ds)):
                ch = dict(zip(ds, bits))
                c = gc.prices(irr, nu, q, kap_fn("box", nu, q, ch))
                Gam, k, _ = gc.solve(verts, c, r)
                winners[bits] = tuple(gc.support_names(irr, verts[k]))
            distinct = sorted(set(winners.values()))
            robust = len(distinct) == 1
            # dense grid in log-kappa on the box: record every winner and its size
            grid_w = {}
            for t in itertools.product(np.linspace(0, 1, 9), repeat=len(ds)):
                ch = {d_: np.exp((1 - tt) * np.log(d_ ** ((nu - 1) * (1 - r))) + tt * np.log(d_ ** (nu * (1 - r)))) for d_, tt in zip(ds, t)}
                c = gc.prices(irr, nu, q, kap_fn("custom", nu, q, ch))
                Gam, k, _ = gc.solve(verts, c, r)
                key = tuple(gc.support_names(irr, verts[k])); grid_w[key] = grid_w.get(key, 0) + 1
            cell["box"] = dict(corner_winners={str(b): list(w) for b, w in winners.items()}, robust=robust,
                               grid_winners={"+".join(k): v for k, v in grid_w.items()},
                               sizes=sorted(set(len(k) for k in grid_w)))
            log(f"  nu {nu} q {q:g} [box   ] kappa_d in [d^((nu-1)(1-r)), d^(nu(1-r))] for d in {ds}: corner winners {distinct}; ROBUST={robust}; "
                f"grid (9^{len(ds)}) winners {grid_w}; support sizes {cell['box']['sizes']}")
        res[f"nu{nu}_q{q:g}"] = cell
    return res, irr, verts


def main():
    cells = [(3, 2.0), (3, 2.5), (4, 2.0), (4, 3.0), (5, 2.0), (6, 2.0), (6, 4.0)]
    out = {}
    for g, ex in [("s5", True), ("s4", True), ("a5", False)]:
        res, irr, verts = run_group(g, cells, exact=ex)
        out[g] = res
        out[g + "_vertices"] = [dict(support=gc.support_names(irr, v), y=[str(x) for x in v]) for v in verts]
    (OUT / "programme.json").write_text(json.dumps(out, indent=1, default=float))
    (OUT / "programme.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
