"""construct_verify.py -- attainment check for Theorem G2 (symmetric class) by brute force on the full table.

For a group G, n operands (nu = n + 1), q, and a vertex y of the programme polyhedron, build the SCHOOLBOOK network:
for each irrep pi with y_pi > 0 (real type, G_pi = y_pi), d^nu neurons, atom (k_0..k_n):
    W = e_{k0 k1},  E_j = e_{k_{j+1} k_j}  (k_{n+1} = k_0),   every factor scaled by (G_pi / d^nu)^{1/nu},
realised as functions e(x) = (d/N) tr(E rho(x)), w(c) = (d/N) tr(W rho(c)) with an explicit real orthogonal rho.
Then compute the true full-table margin, P_q, gamma_q, and compare with the closed form (N nu)^{-nu/2} (sum_pi c_pi y_pi^r)^{-1/r}
with the schoolbook price c = d^{nu(1-r)} d^{q/2}. Also reports the max same-irrep junk (must be 0) and the logit's
deviation from a class function of P(a) c^{-1} (must be 0).
"""
import itertools, sys
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gt_core as gc  # noqa: E402
from verify_fourier import logits, row_products  # noqa: E402

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def build_net(G, n, amps):
    """amps: dict irrep_name -> G_pi (> 0). Returns E (n, m, N), W (m, N)."""
    N = G.N; nu = n + 1
    Es, Ws = [], []
    for nm, g in amps.items():
        r = G.irrep_names.index(nm); d = int(G.irrep_dims[r])
        rho = gc.real_irrep(G, r) if d > 1 else G.CHAR[r][G.class_of][:, None, None].astype(float)
        s = (g / d ** nu) ** (1 / nu)
        for ks in itertools.product(range(d), repeat=nu):          # ks = (k0, k1, ..., kn)
            k = list(ks) + [ks[0]]
            Wm = np.zeros((d, d)); Wm[k[0], k[1]] = s
            Ej = []
            for j in range(1, n + 1):
                M = np.zeros((d, d)); M[k[j + 1], k[j]] = s; Ej.append(M)
            f = lambda M: (d / N) * np.einsum("ab,xba->x", M, rho)   # (d/N) tr(M rho(x))
            Ws.append(f(Wm)); Es.append([f(M) for M in Ej])
    E = np.array(Es).transpose(1, 0, 2); W = np.array(Ws)
    return E, W


def check(gname, n, q, amps, c_price):
    G = gc.get(gname); N = G.N; nu = n + 1; r = q / nu
    E, W = build_net(G, n, amps)
    z = logits(G, E, W, n); P = row_products(G, n)
    zz = z.copy(); zy = zz[np.arange(len(P)), P]; zz[np.arange(len(P)), P] = -np.inf
    margin = float((zy - zz.max(1)).min())
    # class-function check: z(a, c) depends only on class of P c^{-1}
    D = G.MULT[P[:, None], G.INV[None, :]]; cls = G.class_of[D]
    dev = max(float(np.ptp(z[cls == C])) for C in range(len(G.class_names)))
    th2 = (E ** 2).sum((0, 2)) + (W ** 2).sum(1)
    Pq = float((th2 ** (q / 2)).sum())
    gam = margin / Pq ** (nu / q)
    irr, _ = gc.programme_data(G)
    y = {p["name"]: 0.0 for p in irr}; y.update(amps)
    cost = sum(c_price[p["name"]] * y[p["name"]] ** r for p in irr if y[p["name"]] > 0)
    # the vertex is normalised so that min_C sum y cov = 1
    mincov = min(sum(y[p["name"]] * p["cov"][j] for p in irr) for j in range(len(irr[0]["cov"])))
    pred = (N * nu) ** (-nu / 2) * mincov * cost ** (-1 / r)
    log(f"  {gname} n={n} q={q:g} support {list(amps)} ({W.shape[0]} neurons): margin {margin:.6e} (pred N^-nu*mincov {mincov / N ** nu:.6e}), "
        f"class-fn deviation {dev:.1e}, gamma_q {gam:.8e} vs closed form {pred:.8e}, ratio {gam / pred:.12f}")
    return gam / pred


def main():
    worst = 0
    # S5 nu 3: the schoolbook argmax std_sign (vertex y = 1/3), and the runner-up vertex {sign, std} (y_std 1/3, y_sign 1/6)
    for q in (2.0, 2.5):
        nu = 3; r = q / nu
        price = lambda d: d ** (nu * (1 - r)) * d ** (q / 2)
        G = gc.get("s5"); cp = {nm: price(int(G.irrep_dims[i])) for i, nm in enumerate(G.irrep_names)}
        worst = max(worst, abs(1 - check("s5", 2, q, {"std_sign": 1 / 3}, cp)))
        worst = max(worst, abs(1 - check("s5", 2, q, {"std": 1 / 3, "sign": 1 / 6}, cp)))
    # A5 nu 3 q 2: argmax 3a + 3b (take the vertex from programme.json)
    import json
    pj = json.loads((HERE / "out" / "programme.json").read_text())
    for g, cell, n in [("a5", "nu3_q2", 2), ("s4", "nu3_q2", 2), ("s4", "nu4_q2", 3), ("s4", "nu4_q3", 3)]:
        key = "school"
        vert_list = pj[g + "_vertices"]
        sup = pj[g][cell][key]["argmax"]["support"]
        v = next(x for x in vert_list if x["support"] == sup)
        irr, _ = gc.programme_data(gc.get(g))
        amps = {irr[i]["name"]: float(Fraction(s)) if "/" in s or s.isdigit() else float(s) for i, s in enumerate(v["y"]) if float(Fraction(s) if ("/" in s or s.isdigit()) else s) > 0}
        nu = n + 1; q = float(cell.split("_q")[1]); r = q / nu
        G = gc.get(g); cp = {nm: int(G.irrep_dims[i]) ** (nu * (1 - r)) * int(G.irrep_dims[i]) ** (q / 2) for i, nm in enumerate(G.irrep_names)}
        worst = max(worst, abs(1 - check(g, n, q, amps, cp)))
    log(f"  worst |gamma / closed form - 1| = {worst:.2e}")
    (HERE / "out" / "construct_verify.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
