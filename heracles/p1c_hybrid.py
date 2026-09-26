"""P1C / step 4: the HYBRID bound.  A PROOF instrument, not a search.

Notation.  A (KL) system with |A| = 1; x_1 >= x_2 >= ... are the sorted atom moduli |c_l|;
S = the n+1 largest atoms;  R := sum of the moduli of all the OTHER atoms ("tail mass").
r = q/nu in (0,1].  Target: F := sum_l x_l^r >= (n+1)^{1-r}.

Three proved ingredients.

(a) KEY LEMMA WITH A TAIL  (new here; agent 1's Key Lemma is the case R = 0).
    Substituting u_j = v_{sigma(j),j} in   sum_l c_l prod_j (u_j - v_lj) = A prod_j u_j
    for a bijection sigma from the n slots onto S minus {l0} kills every atom of S except
    l0 and leaves each outside atom multiplied by a factor of modulus <= 2^n:
        |c_l0| * prod_j |v_{sigma(j),j} - v_{l0,j}|  >=  |A| - 2^n R  =: A'.
    Geometric mean over the n! bijections, product over l0 in S, Hadamard
    (prod_{l<m}|z_l - z_m| <= (n+1)^{(n+1)/2} for n+1 points of the unit circle), AM-GM:
        sum_{l in S} x_l^r  >=  (n+1)^{1-r} max(A', 0)^r  =: A_G(R).
    Plain reading: the Key Lemma survives a tail, with an error linear in the tail mass.

(b) THE LADDER (agent 3, Theorem E, unchanged): sum_{l>h} x_l >= D_h(n), h = 0..n, over
    ALL atoms; splitting off the tail, sum_{h < l <= n+1} x_l + R >= D_h(n).

(c) TAIL COST.  Every tail atom has modulus <= x_{n+1} =: s, and t^r >= t s^{r-1} for
    0 <= t <= s, so   sum_{l > n+1} x_l^r  >=  max( R^r , R s^{r-1} ).
    Plain reading: mass parked below the (n+1)-st atom is charged at least at the marginal
    price of that atom -- which blows up as the atom gets small.

Hence, for every R >= 0,

    F  >=  V(R) := min over x_1 >= ... >= x_{n+1} = s >= 0 of
                    [ sum_{l<=n+1} x_l^r + max(R^r, R s^{r-1}) ]
           subject to (b) and sum_{l<=n+1} x_l^r >= A_G(R),

and  F >= Lhyb := min over R >= 0 of V(R).  Since min max >= max min, the script reports
the (valid, slightly weaker) V(R) >= max(V1(R), V2(R)) with the two tail terms used
separately, each solved by SLSQP multistart.
"""
import argparse, json, os
from itertools import combinations
import numpy as np
from scipy.optimize import minimize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1C")

# ---- agent 3's constants (heracles/GLOBAL_OPT3.md Sec 3), reproduced ----
S_D = {1: 1.0, 2: 2.0 / np.sqrt(3.0), 3: 9.0 / 7.0,
       4: 1.380900987, 5: 1.452990673, 6: 1.509590371}
KAPPA = {d: (1.0 - 2.0 ** (-d)) / S_D[d] for d in S_D}


def D_h(h, n):
    if h == 0:
        return 1.0
    best = 0.0

    def rec(rem, parts, prod):
        nonlocal best
        if len(parts) == h:
            if rem == 0:
                best = max(best, prod)
            return
        for d in range(1, rem - (h - len(parts) - 1) + 1):
            rec(rem - d, parts + [d], prod * KAPPA[d])
    rec(n, [], 1.0)
    return best


def A_G(R, n, r):
    return (n + 1) ** (1 - r) * max(1.0 - 2 ** n * R, 0.0) ** r


def V_tail(R, n, r, mode, starts=24, seed=0):
    """min over x (length n+1, sorted decreasing, >= 0) of sum x^r + tail(mode),
    subject to the ladder-with-tail constraints and sum x^r >= A_G(R)."""
    m = n + 1
    Ag = A_G(R, n, r)
    rng = np.random.default_rng(seed)

    def obj(x):
        base = float(np.sum(np.maximum(x, 0.0) ** r))
        if mode == "flat":
            return base + R ** r
        s = max(x[-1], 1e-14)
        return base + R * s ** (r - 1.0)

    cons = []
    for h in range(0, n + 1):
        cons.append(dict(type="ineq",
                         fun=(lambda x, h=h: float(np.sum(x[h:]) + R - D_h(h, n)))))
    for i in range(m - 1):
        cons.append(dict(type="ineq", fun=(lambda x, i=i: float(x[i] - x[i + 1]))))
    cons.append(dict(type="ineq", fun=lambda x: float(np.sum(np.maximum(x, 0.0) ** r) - Ag)))
    bnds = [(0.0, 3.0)] * m

    best = np.inf
    for t in range(starts):
        if t == 0:
            x0 = np.full(m, max(1.0 - R, 1e-6) / m)
        else:
            x0 = np.sort(rng.uniform(0.0, 0.7, m))[::-1]
        res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons,
                       options=dict(maxiter=500, ftol=1e-14))
        if not res.success:
            continue
        x = np.maximum(res.x, 0.0)
        ok = all(np.sum(x[h:]) + R >= D_h(h, n) - 1e-8 for h in range(n + 1))
        ok = ok and all(x[i] >= x[i + 1] - 1e-8 for i in range(m - 1))
        ok = ok and (np.sum(x ** r) >= Ag - 1e-8)
        if ok:
            best = min(best, obj(x))
    return best


def hybrid(n, q, ngrid=241, starts=24, Rmax=0.6):
    nu = n + 1
    r = q / nu
    target = nu ** (1 - r)
    Rs = np.concatenate([np.linspace(0.0, 0.08, ngrid // 2),
                         np.linspace(0.08, Rmax, ngrid - ngrid // 2)])
    vals, v1s, v2s = [], [], []
    for R in Rs:
        v1 = V_tail(R, n, r, "flat", starts=starts)
        v2 = V_tail(R, n, r, "marg", starts=starts) if R > 0 else v1
        vals.append(max(v1, v2)); v1s.append(v1); v2s.append(v2)
    vals = np.array(vals)
    k = int(np.argmin(vals))
    return dict(n=n, q=q, r=r, target=target, Rstar=float(Rs[k]),
                Lhyb=float(vals[k]), ratio=float(vals[k] / target),
                ladder_only=float(V_tail(0.0, n, r, "flat", starts=starts)),
                curve=[[float(a), float(b), float(c), float(d)]
                       for a, b, c, d in zip(Rs, vals, v1s, v2s)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["2:2", "2:2.5", "3:2", "3:3", "4:2"])
    ap.add_argument("--ngrid", type=int, default=241)
    ap.add_argument("--starts", type=int, default=24)
    ap.add_argument("--Rmax", type=float, default=0.6)
    ap.add_argument("--tag", default="hybrid")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for cell in a.cells:
        n, q = cell.split(":")
        n, q = int(n), float(q)
        rec = hybrid(n, q, ngrid=a.ngrid, starts=a.starts, Rmax=a.Rmax)
        rows.append(rec)
        # ladder-alone ratio for comparison
        print(f"n={n} nu={n+1} q={q} r={rec['r']:.4f}: target={rec['target']:.9f}  "
              f"HYBRID={rec['Lhyb']:.9f}  ratio={rec['ratio']:.9f}  at R*={rec['Rstar']:.6f}",
              flush=True)
        cur = np.array(rec["curve"])
        lo = cur[np.argsort(cur[:, 1])[:6]]
        for row in lo:
            print(f"    R={row[0]:.5f}  V={row[1]:.9f} (flat {row[2]:.6f}, marginal {row[3]:.6f})",
                  flush=True)
    with open(os.path.join(OUT, f"{a.tag}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
