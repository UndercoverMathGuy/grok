"""K*(p, nu, q): the size-constrained max-margin ladder under an l_q geometry.

M_K(p) = max_{|S|=K} min_{D != 0} sum_{k in S} (1 - cos 2 pi k D / p)   (coverage)
gamma_hat(K; nu, q) = M_K(p) * K^(-nu/q)                                  (equal share, decoupled pricing)
K*(p, nu, q) = argmax_K gamma_hat(K)          q = 2: weight decay / GD;  q -> inf: sign descent (dense)
Sparse iff q < nu. q = nu is the Morwani L_{2,nu} boundary (linear budget, dense).

WLOG 1 in S (the unit group acts transitively on frequencies and permutes D).
Exhaustive while C(nf-1, K-1) <= 4e6, else local search (swap moves, restarts).

Usage: uv run python heracles/kstar.py 59 71 97 113 127 --out heracles/kstar.json
"""
import argparse, itertools, json, math
from math import comb

import numpy as np

NUS = (3, 4, 5, 6)
QS = (2, 3, 4, 6, float("inf"))


def G(p):
    k = np.arange(1, (p - 1) // 2 + 1); d = np.arange(1, p)
    return 1 - np.cos(2 * np.pi * np.outer(k, d) / p)


def best_subset(p, K, rng, exhaustive_limit=4_000_000, restarts=300):
    g = G(p); nf = g.shape[0]; others = np.arange(1, nf)
    if K == 1:
        return float(g[0].min()), [1], "exact"
    if comb(nf - 1, K - 1) <= exhaustive_limit:
        best, bestS = -1.0, None
        it = itertools.combinations(others, K - 1)
        while True:
            block = list(itertools.islice(it, 200_000))
            if not block:
                break
            idx = np.array(block)
            m = (g[0][None, :] + g[idx].sum(1)).min(1)
            j = int(m.argmax())
            if m[j] > best:
                best, bestS = float(m[j]), [1] + [int(x) + 1 for x in idx[j]]
        return best, bestS, "exhaustive"
    best, bestS = -1.0, None
    for _ in range(restarts):
        S = [0] + rng.choice(others, K - 1, replace=False).tolist()
        cur = g[S].sum(0).min(); improved = True
        while improved:
            improved = False
            for i in range(1, K):
                base = g[[x for j, x in enumerate(S) if j != i]].sum(0)
                cand = (base[None, :] + g).min(1)            # swap S[i] -> every frequency
                cand[S] = -1
                j = int(cand.argmax())
                if cand[j] > cur + 1e-12:
                    S[i] = j; cur = cand[j]; improved = True
        if cur > best:
            best, bestS = float(cur), sorted(int(x) + 1 for x in S)
    return best, bestS, "local"


def ladder(p, kmax=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = {}
    for K in range(1, kmax + 1):
        M, S, how = best_subset(p, K, rng)
        rows[K] = dict(M=M, S=S, how=how)
    out = dict(p=p, nf=(p - 1) // 2, M_K={K: r["M"] for K, r in rows.items()},
               S={K: r["S"] for K, r in rows.items()}, how={K: r["how"] for K, r in rows.items()}, pred={})
    for nu in NUS:
        for q in QS:
            s = 0.0 if q == float("inf") else nu / q
            gh = {K: rows[K]["M"] * K ** (-s) for K in rows}
            gmax = max(gh.values()); ks = max(gh, key=gh.get)
            r = {K: gh[K] / gmax for K in gh}
            # dense reference: all nf frequencies at equal amplitude, coverage nf/2 ... exact value:
            # sum_k (1 - cos) over all k = nf + 1/2 for every D != 0
            dense = (out["nf"] + 0.5) * out["nf"] ** (-s)
            key = f"nu{nu}_q{'inf' if q == float('inf') else q}"
            out["pred"][key] = dict(nu=nu, q=q, s=s, K_star=ks, gamma_hat=gh, r=r,
                                    plateau74=[K for K in r if r[K] >= 0.74], plateau84=[K for K in r if r[K] >= 0.84],
                                    dense_over_sparse=dense / gmax,
                                    K_star_or_dense=("dense" if dense >= gmax or ks == kmax else ks))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("primes", nargs="*", type=int, default=[59, 71, 97, 113, 127])
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = {}
    for p in args.primes:
        res[p] = ladder(p, args.kmax)
        print(f"p={p}  M_K: " + " ".join(f"{K}:{res[p]['M_K'][K]:.3f}" for K in res[p]["M_K"]))
        for nu in NUS:
            line = []
            for q in QS:
                key = f"nu{nu}_q{'inf' if q == float('inf') else q}"
                pr = res[p]["pred"][key]
                line.append(f"q={q if q != float('inf') else 'inf'}: K*={pr['K_star_or_dense']} plat74={pr['plateau74'][0] if pr['plateau74'] else '-'}..{pr['plateau74'][-1] if pr['plateau74'] else '-'} dense/sparse={pr['dense_over_sparse']:.2f}")
            print(f"  nu={nu} | " + " | ".join(line))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1, default=str)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
