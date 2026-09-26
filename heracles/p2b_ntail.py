"""P2B step 6: the N > n+1 axis (P2's unfinished F2-2), at n = 2.

Let c_1..c_N (N > 3) have C_empty = 1 and junk energy <= eta^2.  Sort |c_1| >= ... >= |c_N|,
let S be the top three atoms and tau := sum_{l not in S} |c_l| the tail mass.  Then:

  * the restricted population has C'_T = C_T - sum_{l not in S} c_l prod_{j in T} v_lj, so
    |C'_T - C_T| <= tau for every T, hence |C'_empty| >= 1 - tau and
    ||C'_{!=empty}||_2 <= eta + sqrt3 tau ;
  * normalising by C'_empty, its junk ratio is  eta'(tau) <= (eta + sqrt3 tau)/(1 - tau) ;
  * THEOREM P2B-1 applied to the normalised triple (valid iff eta'(tau) <= eta_0) gives
    prod_{l in S}|c_l| >= ((1-tau)/3)^3 exp(-C eta'^2), so by AM-GM
        sum_{l in S} |c_l|^r >= 3^{1-r}(1-tau)^r exp(-r C eta'^2/3) ;
  * the tail costs at least  sum_{l not in S}|c_l|^r >= tau^r  (r <= 1, subadditivity).

So for tau <= tau_max(eta) := (eta_0 - eta)/(sqrt3 + eta_0),

    sum_l |c_l|^r  >=  g(tau) := 3^{1-r}(1-tau)^r exp(-r C eta'(tau)^2/3) + tau^r .

This script checks that g is minimised at tau = 0 on that range (so that, as long as the tail
is small, N > 3 is genuinely no harder than N = 3), and reports the tau at which the
guarantee stops -- the honest boundary of the reduction.
"""
import argparse, json, os
import numpy as np

OUT = "FINAL_RUNS/P2B"
SQ3 = np.sqrt(3.0)


def g(tau, eta, r, C, eta0):
    ep = (eta + SQ3 * tau) / (1 - tau)
    if ep > eta0:
        return None
    return 3 ** (1 - r) * (1 - tau) ** r * np.exp(-r * C * ep ** 2 / 3) + tau ** r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta0", type=float, default=0.15)
    ap.add_argument("--C", type=float, default=5.036206)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = {"eta0": a.eta0, "C": a.C}
    print(f"eta_0 = {a.eta0}, C = {a.C}", flush=True)
    print(f"{'r':>8} {'eta':>8} {'tau_max':>10} {'min g / g(0)':>16} {'argmin tau':>12}",
          flush=True)
    rows = []
    for r in (0.4, 0.5, 2 / 3, 0.75, 0.9, 1.0):
        for eta in (0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.14):
            if eta > a.eta0:
                continue
            tmax = (a.eta0 - eta) / (SQ3 + a.eta0)
            ts = np.linspace(0, tmax, 200001)
            vals = np.array([g(t, eta, r, a.C, a.eta0) for t in ts], dtype=float)
            v0 = vals[0]
            k = int(np.nanargmin(vals))
            rows.append({"r": r, "eta": eta, "tau_max": float(tmax),
                         "ratio": float(vals[k] / v0), "argmin": float(ts[k])})
            print(f"{r:>8.4f} {eta:>8.3f} {tmax:>10.5f} {vals[k]/v0:>16.12f} "
                  f"{ts[k]:>12.6f}", flush=True)
    res["rows"] = rows
    worst = min(x["ratio"] for x in rows)
    res["worst_ratio"] = worst
    print(f"\nworst min g / g(0) over all cells = {worst:.12f}  "
          f"({'tau = 0 is the minimiser everywhere' if worst >= 1 - 1e-12 else 'NOT minimised at 0'})",
          flush=True)
    json.dump(res, open(f"{OUT}/ntail.json", "w"), indent=1, default=float)
    print(f"wrote {OUT}/ntail.json", flush=True)


if __name__ == "__main__":
    main()
