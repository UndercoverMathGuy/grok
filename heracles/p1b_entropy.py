"""P1B: the `r -> 1` window as ONE r-free extremal constant, and why the max-weight route is capped.

P1A's reduction (their P1A.md): with p = x/S, S = sum_l |c_l|, Renyi H_r >= Shannon H for r <= 1,
    sum_l x_l^r = S^r exp((1-r) H_r) >= S^r exp((1-r) H),
so   H(p) + (r/(1-r)) log S >= log(n+1)   implies P1 at that r (and, being monotone in r, on [r,1]).

THIS FILE turns that into a single number.  Define

    a*(n) := sup over (KL) systems with S > 1 of  [ log(n+1) - H(x/S) ] / log S .

Then the criterion holds exactly for  r >= a*/(1 + a*)  -- i.e. a*(n) is the whole content of the
`r -> 1` end of P1.  Because the two-branch / certified arguments already cover r <= r*(n), the
window closes as soon as

    a*(n)  <=  r*(n) / (1 - r*(n))        ( = 5.37 at n = 2 with r* = 0.843,
                                             1.68 at n = 3 with r* = 0.627 ).

Blocks:
  A   measure a*(n) globally (search over configurations)
  D   the delta-scaling of the two competing routes near the block:
      (i) max_l x_l - 1/(n+1)  vs  delta = S - 1        -- grows like sqrt(delta): the MAX-WEIGHT
          route is capped, because bounding H by -log(max p) is a first-order bound while H is
          quadratically flat at the uniform point;
      (ii) log(n+1) - H(x/S)   vs  log S                -- grows LINEARLY, with a small slope.
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize

import p1b_core as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1B")


def shannon(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 1e-300]
    return float(-(p * np.log(p)).sum())


def profile(theta, n, N):
    """The sorted modulus profile of the (KL) rule on N atoms with these phases, or None."""
    v = np.exp(1j * theta.reshape(N, n))
    M = C.moment_matrix(v)
    rhs = np.zeros(1 << n, dtype=complex); rhs[0] = 1.0
    if N == (1 << n):
        try:
            z = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return None
    else:
        z, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        if np.linalg.norm(M @ z - rhs) > 1e-9:
            return None
    if not np.all(np.isfinite(z)):
        return None
    x = np.abs(z)
    return x if x.sum() < 1e7 else None


def a_ratio(x, n):
    S = float(x.sum())
    if S <= 1 + 1e-13:
        return None
    return (np.log(n + 1) - shannon(x / S)) / np.log(S), S


def measure_astar(n, Ns, trials=40000, seed=11):
    """Lower bound on a*(n) by direct sampling (uniform-random and block-perturbed nodes).

    Sampling, not local optimisation: the objective has a 0/0 ridge as S -> 1 and Nelder-Mead
    chases it without terminating, whereas the extremisers turn out to be far from the block
    (near-cancelling pairs at S = 2.8 / 49), which sampling finds immediately."""
    rng = np.random.default_rng(seed)
    best, bx, tot = -np.inf, None, 0
    for N in Ns:
        for t in range(trials):
            if t % 2 == 0:
                v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n)))
            else:
                sc = rng.choice([0.03, 0.1, 0.3, 0.8, 1.6])
                v = np.vstack([C.block(n) * np.exp(1j * rng.normal(0, sc, (n + 1, n))),
                               np.exp(1j * rng.uniform(0, 2 * np.pi, (N - n - 1, n)))])
            x = profile(np.angle(v).ravel(), n, N)
            if x is None:
                continue
            rr = a_ratio(x, n)
            if rr is None:
                continue
            tot += 1
            if rr[0] > best:
                best, bx = rr[0], (np.sort(x)[::-1][:6].tolist(), rr[1], N)
    rc = best / (1 + best)
    return dict(n=n, systems=tot, a_star_lower=float(best), r_crit=float(rc),
                q_crit=float((n + 1) * rc), argmax_profile=bx[0], argmax_S=bx[1], argmax_N=bx[2])


def measure_scaling(n, N, bands, trials=6000, seed=0):
    """Near the block: the sqrt(delta) growth of max_l x_l and the linear growth of log(n+1) - H."""
    rng = np.random.default_rng(seed)
    rows = []
    for lo in bands:
        hi = lo * np.sqrt(10.0)
        best_mx, best_dh = -np.inf, -np.inf
        at = None
        for t in range(trials):
            th = rng.uniform(0, 2 * np.pi, N * n)

            def obj(th):
                x = profile(th, n, N)
                if x is None:
                    return 1e3
                d = float(x.sum()) - 1.0
                if d < lo or d > hi:
                    return 1e3 + 100 * abs(np.log(max(d, 1e-18) / np.sqrt(lo * hi)))
                return -(x.max() - 1.0 / (n + 1))
            if t % 12 == 0:
                th = minimize(obj, th, method="Nelder-Mead",
                              options=dict(maxiter=800, maxfev=800, xatol=1e-11, fatol=1e-13)).x
            x = profile(th, n, N)
            if x is None:
                continue
            d = float(x.sum()) - 1.0
            if not (lo <= d <= hi):
                continue
            S = 1.0 + d
            best_mx = max(best_mx, (x.max() - 1.0 / (n + 1)) / np.sqrt(d))
            dh = (np.log(n + 1) - shannon(x / S)) / np.log(S)
            if dh > best_dh:
                best_dh, at = dh, d
        rows.append(dict(band=[lo, hi], max_excess_over_sqrt_delta=float(best_mx),
                         entropy_deficit_over_logS=float(best_dh), at_delta=at))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="AD")
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--tag", default="entropy")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = {}

    if "A" in args.blocks:
        out["A"] = []
        for n, Ns in [(2, [4, 5, 6]), (3, [8, 9])]:
            row = measure_astar(n, Ns, trials=args.trials * 100)
            need = {2: 0.84333, 3: 0.62678}.get(n)
            row["threshold_to_close"] = need / (1 - need) if need else None
            out["A"].append(row)
            print("A  n=%d: a* >= %.5f over %d systems => criterion covers r >= %.5f (q >= %.3f); "
                  "closes if a* <= %.3f -> margin x%.1f"
                  % (n, row["a_star_lower"], row["systems"], row["r_crit"], row["q_crit"],
                     row["threshold_to_close"], row["threshold_to_close"] / row["a_star_lower"]))
            print("     argmax S=%.6f N=%d profile %s"
                  % (row["argmax_S"], row["argmax_N"], np.round(row["argmax_profile"], 6)))

    if "D" in args.blocks:
        out["D"] = measure_scaling(2, 4, [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                                   trials=max(args.trials * 10, 4000))
        print("D  n=2 near-block scaling (delta = S - 1):")
        for row in out["D"]:
            print("     delta ~ %.0e : (max|c| - 1/3)/sqrt(delta) = %.4f   "
                  "(log3 - H)/log S = %.4f"
                  % (row["band"][0], row["max_excess_over_sqrt_delta"],
                     row["entropy_deficit_over_logS"]))

    with open(os.path.join(OUT, args.tag + ".json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
