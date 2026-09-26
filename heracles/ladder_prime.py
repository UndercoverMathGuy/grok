"""ladder_prime.py -- the exact prime axis: K*(p) from the programme, no training.

For every prime p in the list and every programme exponent s = nu/q:
  M_K(p)      = max over |S| = K of min_{D != 0} sum_{k in S} (1 - cos 2 pi k D / p)
  K*_eq(p, s) = argmax_K  M_K(p) * K^{-s}           (equal shares: the kstar.py ladder)
  K*_free(p,s)= number of non-zero shares at the free-amplitude optimum
                (= the first K whose free-amplitude value reaches Gamma_s(p))
Also records the RESOLUTION of the argmax: the ratio of the runner-up rung to the winner, so that
near-ties (where the argmax is numerically fragile) are visible rather than hidden.

Support search: exhaustive over all K-sets containing 1 while C(nf-1, K-1) <= 3e6 (provably best),
otherwise swap local search with `--restarts` restarts (heuristic; the reported M_K is a lower
bound on the true M_K, so a missed support can only move K* toward smaller K).

Usage: uv run python heracles/ladder_prime.py --out FINAL_RUNS/LADDER/json/prime_axis.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ladder_core import OUT, coverage_topK, gmat, solve_support  # noqa: E402

S_LIST = [1.5, 2.0, 2.5, 1.2, 4 / 3, 5 / 3]     # nu/q: (3,2) (4,2) (5,2) (3,2.5) (4,3) (5,3)
S_TAG = {1.5: "nu3_q2 / nu4_q8-3 / nu5_q10-3", 2.0: "nu4_q2", 2.5: "nu5_q2",
         1.2: "nu3_q2.5", 4 / 3: "nu4_q3", 5 / 3: "nu5_q3"}


def primes_upto(n):
    sieve = np.ones(n + 1, bool)
    sieve[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return np.nonzero(sieve)[0].tolist()


def ladder_for(p, kmax, restarts, shortlist):
    g = gmat(p)
    M, Sbest, allc = {}, {}, {}
    for K in range(1, kmax + 1):
        cands, how = coverage_topK(g, K, shortlist=max(shortlist, 20), restarts=restarts,
                                   exhaustive_limit=3_000_000)
        Sbest[K] = cands[0]
        allc[K] = cands
        M[K] = float(g[np.array(cands[0]) - 1].sum(0).min())
    return g, M, Sbest, allc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--restarts", type=int, default=120)
    ap.add_argument("--shortlist", type=int, default=20)
    ap.add_argument("--pmax", type=int, default=1200)
    ap.add_argument("--dense_upto", type=int, default=300)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--grow", type=int, default=40, help="grow-the-winner candidates per rung")
    ap.add_argument("--slist", type=float, nargs="+", default=None, help="programme exponents s (default: all six)")
    ap.add_argument("--out", default=str(OUT / "json" / "prime_axis.json"))
    a = ap.parse_args()
    slist = a.slist if a.slist else S_LIST

    ps = [p for p in primes_upto(a.pmax) if p >= 11]
    sel = [p for p in ps if p <= a.dense_upto] + [p for i, p in enumerate(ps) if p > a.dense_upto and i % a.stride == 0]
    print(f"{len(sel)} primes: {sel[:8]} ... {sel[-4:]}", flush=True)

    res = {}
    for p in sel:
        t0 = time.time()
        kmax = min(a.kmax, (p - 1) // 2)
        g, M, Sbest, cov_all = ladder_for(p, kmax, a.restarts, a.shortlist)
        row = dict(p=p, nf=(p - 1) // 2, M={str(K): M[K] for K in M},
                   S={str(K): Sbest[K] for K in Sbest}, per_s={})
        for s in slist:
            gh = {K: M[K] * K ** (-s) for K in M}
            keq = max(gh, key=gh.get)
            order = sorted(gh, key=lambda K: -gh[K])
            runner = order[1] if len(order) > 1 else keq
            # free-amplitude ladder: coverage shortlist + grow-the-winner, then the ACTIVE-SUPPORT
            # REPAIR of ladder_polish.py (a solution with k < K active shares is a valid rung-k net,
            # and the ladder must be non-decreasing in K because a K-support contains every smaller one)
            fv, prev, pool = {}, None, []
            for K in range(1, kmax + 1):
                cands = [list(S) for S in cov_all[K][:a.shortlist]]
                if prev is not None:
                    cands += [sorted(prev + [k]) for k in range(1, (p - 1) // 2 + 1) if k not in prev][:a.grow]
                best = (-1.0, None, None)
                for S in cands:
                    v, b = solve_support(g, S, s, restarts=2)
                    if v > best[0]:
                        best = (v, S, b)
                bb = np.asarray(best[2])
                act = [k for k, x in zip(best[1], bb) if x > 1e-4]
                ba = bb[bb > 1e-4]
                ba = ba / ba.sum()
                pool.append((len(act), float(((ba ** s) @ g[np.array(act) - 1]).min()), act))
                pool.append((len(best[1]), best[0], list(best[1])))
                cand = [c for c in pool if c[0] <= K]
                top = max(cand, key=lambda c: c[1])
                fv[K] = dict(value=top[1], S=top[2], n_active=top[0])
                prev = list(best[1])
            kbest = max(fv, key=lambda K: fv[K]["value"])
            kfree = fv[kbest]["n_active"]
            row["per_s"][f"{s:.6f}"] = dict(
                s=s, K_star_eq=keq, gamma_hat={str(K): gh[K] for K in gh},
                runner_up=runner, resolution=gh[runner] / gh[keq],
                Gamma=fv[kbest]["value"], K_star_free=kfree, K_plateau_start=kbest,
                S_free=fv[kbest]["S"], free_values={str(K): fv[K]["value"] for K in fv},
                free_active={str(K): fv[K]["n_active"] for K in fv})
        res[str(p)] = row
        line = "  ".join(f"s{s:.2f}: eq {row['per_s'][f'{s:.6f}']['K_star_eq']}"
                         f"/free {row['per_s'][f'{s:.6f}']['K_star_free']}"
                         f" (res {row['per_s'][f'{s:.6f}']['resolution']:.3f})" for s in slist)
        print(f"p {p:5d}  {line}   ({time.time() - t0:.1f}s)", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
