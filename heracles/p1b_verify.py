"""P1B: step-by-step verification of every link of the two-branch proof.

Each check is run on random (KL) systems (square case N = 2^n, where the rule is unique)
and on block perturbations (which is the only way to reach a small tail E).

  V1  the kill identity      sum_l c_l prod_j (u_j - v_lj) = A prod_j u_j   for all u in C^n
  V2  the per-(l0,sigma) inequality   x_{l0} P_sigma(l0) + sum_{tail} x_l Q_sigma(l) >= |A|
  V3  the bookkeeping identity   prod_{l0 in B} prod_sigma P_sigma(l0) = prod_j V_j^{2 (n-1)!}
      (each ordered pair (m, l0) occurs in (n-1)! bijections at each slot; the geometric mean
      over the n! bijections is what the proof actually uses, and it is the (n-1)!/n! = 1/n
      power, which is why the n-th root appears in step (ii) of the Key Lemma)
  V4  Hadamard                 V_j <= (n+1)^{(n+1)/2}
  V5  the product bound        prod_{l in B} x_l >= (1 - 2^n E)^{n+1} / (n+1)^{n+1}
  V6  BRANCH 1                 cost >= (n+1)^{1-r} (1 - 2^n E)^r + E^r
  V7  the peeling ladder       sum_{l>h} x_l >= D_h(n) |A|,  h = 0..n   (GLOBAL_OPT3 Thm E)
  V8  P1 itself                cost >= (n+1)^{1-r} |A|^r
"""

import argparse
import itertools
import json
import os

import numpy as np

import p1b_core as C
import p1b_proof as P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1B")


def sample(n, rng, mode):
    """A (KL) system with N = 2^n atoms (unique rule)."""
    N = 1 << n
    if mode == "random":
        v = np.exp(1j * rng.uniform(0, 2 * np.pi, (N, n)))
    else:
        s = float(mode)
        vb = C.block(n) * np.exp(1j * rng.normal(0, s, (n + 1, n)))
        extra = np.exp(1j * rng.uniform(0, 2 * np.pi, (N - n - 1, n)))
        v = np.vstack([vb, extra])
    try:
        z = C.rule_from_config(v)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(z)) or np.sum(np.abs(z)) > 1e7:
        return None
    return v, z


def run(n, r, trials=1500, seed=0):
    rng = np.random.default_rng(seed)
    modes = ["random", "0.3", "0.1", "0.03", "0.01", "0.003", "0.001"]
    res = {k: np.inf for k in ("V2", "V5", "V6", "V7", "V8")}
    maxerr = {"V1": 0.0, "V3": 0.0, "V4": 0.0}
    act = used = 0
    for i in range(trials):
        s = sample(n, rng, modes[i % len(modes)])
        if s is None:
            continue
        v, z = s
        N = z.size
        used += 1
        x_all = np.abs(z)
        order = np.argsort(-x_all)
        B = list(order[:n + 1])
        tail = list(order[n + 1:])
        E = float(np.sum(x_all[tail]))
        xs = x_all[order]

        # V1: kill identity at random complex u
        u = rng.normal(size=n) + 1j * rng.normal(size=n)
        lhs = np.sum(z * np.prod(u[None, :] - v, axis=1))
        maxerr["V1"] = max(maxerr["V1"], abs(lhs - np.prod(u)))

        # V2 / V3 / V4 over all (l0, sigma)
        logprodP = 0.0
        for l0 in B:
            others = [m for m in B if m != l0]
            for perm in itertools.permutations(range(n)):
                # slot j killed at atom others[perm[j]]
                uu = np.array([v[others[perm[j]], j] for j in range(n)])
                Pv = float(np.prod(np.abs(uu - v[l0])))
                logprodP += np.log(max(Pv, 1e-300))
                Q = np.array([np.prod(np.abs(uu - v[l])) for l in tail]) if tail else np.zeros(0)
                lhs2 = x_all[l0] * Pv + float(np.dot(x_all[tail], Q))
                res["V2"] = min(res["V2"], lhs2)
        logVj = 0.0
        for j in range(n):
            vv = 1.0
            for a, b in itertools.combinations(B, 2):
                vv *= abs(v[a, j] - v[b, j])
            maxerr["V4"] = max(maxerr["V4"], vv / (n + 1) ** ((n + 1) / 2.0))
            logVj += np.log(max(vv, 1e-300))
        import math
        expo = 2 * math.factorial(n - 1)
        maxerr["V3"] = max(maxerr["V3"], abs(logprodP - expo * logVj) / max(abs(logprodP), 1.0))

        # V5 / V6
        t = 1.0 - (2 ** n) * E
        if t > 0:
            act += 1
            res["V5"] = min(res["V5"], float(np.prod(x_all[B])) /
                            (t ** (n + 1) / (n + 1) ** (n + 1)))
            cost = float(np.sum(x_all ** r))
            res["V6"] = min(res["V6"], cost / ((n + 1) ** (1 - r) * t ** r + E ** r))

        # V7 / V8
        for h in range(0, min(n, N - 1) + 1):
            res["V7"] = min(res["V7"], float(np.sum(xs[h:])) / P.D_h(n, h))
        res["V8"] = min(res["V8"], float(np.sum(x_all ** r)) / (n + 1) ** (1 - r))

    return dict(n=n, r=r, used=used, branch1_active=act,
                V1_max_identity_error=maxerr["V1"],
                V3_max_rel_error=maxerr["V3"],
                V4_max_Vj_over_Hadamard=maxerr["V4"],
                V2_min_lhs_over_A=res["V2"],
                V5_min_ratio=res["V5"], V6_min_ratio=res["V6"],
                V7_min_ratio=res["V7"], V8_min_ratio=res["V8"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1500)
    ap.add_argument("--tag", default="verify")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for (n, q) in [(1, 1.2), (2, 2.0), (2, 2.5), (3, 2.0), (3, 3.0), (4, 2.0), (5, 2.0)]:
        row = run(n, q / (n + 1), trials=args.trials, seed=11)
        rows.append(row)
        print("n=%d r=%.4f used=%d (branch1 active %d)" % (n, row["r"], row["used"], row["branch1_active"]))
        print("   V1 kill-identity max error      %.3e   (must be ~0)" % row["V1_max_identity_error"])
        print("   V3 log-identity max rel err     %.3e   (must be ~0)" % row["V3_max_rel_error"])
        print("   V4 max V_j/(n+1)^((n+1)/2)      %.9f   (must be <= 1)" % row["V4_max_Vj_over_Hadamard"])
        print("   V2 min lhs (A = 1)              %.9f   (must be >= 1)" % row["V2_min_lhs_over_A"])
        print("   V5 min prod/bound               %.9f   (must be >= 1)" % row["V5_min_ratio"])
        print("   V6 min cost/branch1             %.9f   (must be >= 1)" % row["V6_min_ratio"])
        print("   V7 min peel/D_h                 %.9f   (must be >= 1)" % row["V7_min_ratio"])
        print("   V8 min cost/target  (P1)        %.9f   (must be >= 1)" % row["V8_min_ratio"])
    with open(os.path.join(OUT, args.tag + ".json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
