"""P1C / step 2: a census of LOCAL minima of the (KL) cost, and the stationarity picture.

For each cell (n, q, N) this runs many random-start SLSQP minimisations of
F = sum_l |c_l|^r subject to (KL) with A = 1, and for every converged point records

  * F / (n+1)^{1-r}              -- anything below 1 refutes P1
  * the support size s = #{l : |c_l| > tol} and the sorted moduli
  * the spread  max|c_l| / min|c_l|  over the support  -- the RIGIDITY question:
    is every stationary point equal-modulus on its support?
  * the stationarity residuals of conditions (i) and (ii) of p1c_core
  * sum_l |Lambda(v_l)|^{-sigma}  against n+1   -- P1 at stationary points is exactly
    the statement that this is >= n+1.

Why the rigidity question is the whole problem: if every atom of the support has the same
|c_l| = m, then sum_l c_l = A forces N m >= |A| and hence
F = N m^r = N^{1-r} (N m)^r >= (n+1)^{1-r} |A|^r, using only N >= n+1 (Theorem A).
So "equal moduli at every stationary point" implies P1 outright.
"""
import argparse, json, os
import numpy as np
from p1c_core import minimise_F, stationarity_report, cost

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1C")


def run_cell(n, q, N, starts, seed, tol):
    nu = n + 1
    r = q / nu
    target = nu ** (1 - r)
    recs = []
    for s in range(starts):
        phi, c, F, res = minimise_F(N, n, r, seed=seed + 1000 * s)
        if res > 1e-8 or not np.isfinite(F):
            continue
        rep = stationarity_report(phi, c, r, tol=tol)
        mods = np.sort(np.abs(c))[::-1]
        sup = mods[mods > tol]
        recs.append(dict(
            F=F, ratio=F / target, support=int(len(sup)),
            spread=float(sup.max() / sup.min()) if len(sup) else np.inf,
            mods=mods.tolist(), res_kl=res,
            res_i=rep["res_i"], res_ii=rep["res_ii"],
            lam_empty_abs=float(abs(rep["lam_empty"])),
            sumL=rep.get("sum_Lambda_pow", None),
            F_id=rep.get("F_from_identity", None)))
    return dict(n=n, q=q, N=N, r=r, target=target, recs=recs)


def summarise(cell):
    recs = cell["recs"]
    if not recs:
        return "  (no converged runs)"
    ratios = np.array([x["ratio"] for x in recs])
    best = recs[int(np.argmin(ratios))]
    # distinct local minima by rounded value
    vals = sorted(set(round(x["ratio"], 6) for x in recs))
    lines = [f"  runs={len(recs)}  min ratio={ratios.min():.12f}  "
             f"distinct local values={ [f'{v:.6f}' for v in vals[:8] ] }"]
    lines.append(f"  argmin: support={best['support']}  spread={best['spread']:.6f}  "
                 f"mods={['%.6f' % m for m in best['mods']]}")
    lines.append(f"  argmin stationarity: res_i={best['res_i']:.2e} res_ii={best['res_ii']:.2e} "
                 f"|lam_empty|={best['lam_empty_abs']:.9f}  "
                 f"sum|L|^-sigma={best['sumL'] if best['sumL'] is None else round(best['sumL'],9)} "
                 f"(need >= {cell['n']+1})")
    # rigidity over ALL converged local minima that are genuine (res_i small)
    good = [x for x in recs if x["res_i"] < 1e-5 and x["res_ii"] < 1e-4]
    if good:
        sp = np.array([x["spread"] for x in good])
        lines.append(f"  rigidity over {len(good)} clean stationary points: "
                     f"max spread={sp.max():.6f}  (1.0 = equal moduli)")
        bad = [x for x in good if x["spread"] > 1.001]
        if bad:
            w = min(bad, key=lambda x: x["ratio"])
            lines.append(f"    UNEQUAL example: ratio={w['ratio']:.9f} support={w['support']} "
                         f"mods={['%.6f' % m for m in w['mods'] if m > 1e-7]}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["2:2:4", "2:2:5", "2:2:6", "2:2.5:4",
                                                   "3:2:5", "3:2:6"])
    ap.add_argument("--starts", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--tag", default="census")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    allc = []
    for cell in a.cells:
        n, q, N = cell.split(":")
        n, N, q = int(n), int(N), float(q)
        c = run_cell(n, q, N, a.starts, a.seed, a.tol)
        allc.append(c)
        print(f"n={n} nu={n+1} q={q} N={N} r={c['r']:.4f} target={c['target']:.9f}", flush=True)
        print(summarise(c), flush=True)
    with open(os.path.join(OUT, f"{a.tag}.json"), "w") as f:
        json.dump(allc, f, indent=1)


if __name__ == "__main__":
    main()
