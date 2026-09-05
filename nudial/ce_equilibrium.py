"""CE+wd equilibrium ranking of committee sizes (the finite-scale functional).

Diagnosis target: gamma_hat ranks K by HARD min-margin (the infinite-scale
Lyu-Li limit), but trained nets minimize CE + weight decay at a FINITE
equilibrium logit scale, where the ranking on the plateau is a ~2% race and
soft-min corrections can flip it. For a pure-cosine committee with
amplitudes A_k on support S the trained objective is EXACT per training
pair (every (a,b) has the same per-example loss):

    L = log(1 + sum_{D=1..p-1} exp(-sum_k A_k g_k(D))) + lam * sum_k A_k^(2/nu)
    g_k(D) = 1 - cos(2 pi k D / p)

First term = cross-entropy (a softmin over the p-1 wrong offsets).
Second = cheapest parameter norm realizing the amplitudes (AM-GM chain
pricing through nu multiplicative layers; architecture constant absorbed
into lam). Hard-margin gamma_hat is the lam->0 limit of argmin.

No fitted constants: lam is scanned over decades AND pinned by the observed
train-CE floor (~1.9e-7 for the vanilla ReLU recipe) — an observable
independent of the K histograms.

Stages per prime:
  1. pool of top hard-margin sets per K (exhaustive when feasible, else
     local search; WLOG freq 1 in S, unit-group symmetry).
  2. equal-amplitude scan: minimize L over amplitude grid, per set, per lam;
     report argmin K vs lam.
  3. free-amplitude polish at pinned lam (Adam on log-amps): per-K optimum,
     effective support (amps within 1e-3 of leader), democracy check.
  4. unconstrained descent from wide supports: the support size the
     objective itself selects.

Usage: uv run python nudial/ce_equilibrium.py [primes...]  (default all)
Output: notes/ce_equilibrium.json + stdout tables.
"""
import numpy as np, itertools, json, sys, heapq
from math import comb

NU = 6.0
PRIMES = [97, 113, 127, 211, 1009]
KRANGE = list(range(2, 9))
LAMS = np.logspace(-9, -3, 25)
CE_FLOOR = 1.9e-7  # measured train-CE floor, vanilla ReLU recipe (independent observable)
rng = np.random.default_rng(0)


def G(p):
    k = np.arange(1, (p - 1) // 2 + 1)
    d = np.arange(1, p)
    return 1 - np.cos(2 * np.pi * np.outer(k, d) / p)


def pool_subsets(p, K, g, npool=20, exhaustive_limit=3_000_000, restarts=150):
    """Top-npool subsets of size K by hard min-margin (indices into g rows)."""
    nf = g.shape[0]
    others = np.arange(1, nf)
    if K == 1:
        return [(float(g[0].min()), (0,))]
    if comb(nf - 1, K - 1) <= exhaustive_limit:
        top = []
        it = itertools.combinations(others, K - 1)
        block_sz = max(20_000, int(2e7 / (p - 1)))
        while True:
            block = list(itertools.islice(it, block_sz))
            if not block:
                break
            idx = np.array(block)
            m = (g[0][None, :] + g[idx].sum(1)).min(1)
            for j in np.argsort(m)[-npool:]:
                item = (float(m[j]), (0,) + tuple(int(x) for x in idx[j]))
                if len(top) < npool:
                    heapq.heappush(top, item)
                elif item[0] > top[0][0]:
                    heapq.heapreplace(top, item)
        return sorted(top, reverse=True)
    found = {}
    for _ in range(restarts):
        S = set([0]) | set(rng.choice(others, K - 1, replace=False).tolist())
        cur = g[list(S)].sum(0).min()
        improved = True
        while improved:
            improved = False
            for out in list(S):
                if out == 0:
                    continue
                base = g[[x for x in S if x != out]].sum(0)
                cand = (base[None, :] + g).min(1)
                cand[list(S)] = -1
                j = int(cand.argmax())
                if cand[j] > cur + 1e-12:
                    S.remove(out); S.add(j); cur = float(cand[j]); improved = True
                    break
        found[tuple(sorted(S))] = float(cur)
    return sorted([(m, s) for s, m in found.items()], reverse=True)[:npool]


def equal_amp(g, sets, lams, nA=1400):
    """Minimize L over a log-amplitude grid, per set per lam.
    Returns dict lam -> (best L, set index, A*, CE*)."""
    Gm = np.stack([g[list(S)].sum(0) for S in sets])  # (nsets, p-1)
    A = np.logspace(-1.0, 3.8, nA)
    ce = np.empty((len(sets), nA))
    chunk = max(1, int(4e7 / (Gm.shape[1] * nA)))
    for i in range(0, len(sets), chunk):
        E = np.exp(-A[None, None, :] * Gm[i:i + chunk, :, None])
        ce[i:i + chunk] = np.log1p(E.sum(1))
    K = len(sets[0])
    pen = K * A ** (2 / NU)
    out = {}
    for lam in lams:
        L = ce + lam * pen[None, :]
        i, a = np.unravel_index(int(L.argmin()), L.shape)
        out[float(lam)] = (float(L[i, a]), int(i), float(A[a]), float(ce[i, a]))
    return out


def free_amp(gS, lam, A0, steps=6000, lr=0.02):
    """Adam on x = log A. Returns (L, amps)."""
    x = np.log(np.maximum(A0, 1e-8))
    m = np.zeros_like(x); v = np.zeros_like(x)
    b1, b2 = 0.9, 0.999
    for t in range(1, steps + 1):
        Aamp = np.exp(x)
        z = Aamp @ gS
        e = np.exp(-z)
        s = e.sum()
        gA = -(gS @ e) / (1 + s) + lam * (2 / NU) * Aamp ** (2 / NU - 1)
        gx = gA * Aamp
        m = b1 * m + (1 - b1) * gx
        v = b2 * v + (1 - b2) * gx * gx
        x -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-12)
        x = np.maximum(x, -25.0)
    Aamp = np.exp(x)
    z = Aamp @ gS
    L = float(np.log1p(np.exp(-z).sum()) + lam * (Aamp ** (2 / NU)).sum())
    return L, Aamp


def run_prime(p):
    g = G(p)
    print(f"\n=== p={p} (nf={g.shape[0]}) ===", flush=True)
    pools = {}
    for K in KRANGE:
        pools[K] = pool_subsets(p, K, g)
        print(f"  pool K={K}: best hard M={pools[K][0][0]:.4f} "
              f"gamma_hat={pools[K][0][0]/K**(NU/2):.6f} ({len(pools[K])} sets)", flush=True)
    hard_argmax = max(KRANGE, key=lambda K: pools[K][0][0] / K ** (NU / 2))

    scans = {K: equal_amp(g, [s for _, s in pools[K]], LAMS) for K in KRANGE}
    lam_rows = []
    for lam in LAMS:
        Ls = {K: scans[K][float(lam)] for K in KRANGE}
        order = sorted(KRANGE, key=lambda K: Ls[K][0])
        lam_rows.append((float(lam), order, {K: Ls[K] for K in KRANGE}))
        top = ",".join(str(K) for K in order[:3])
        Kb = order[0]
        print(f"  lam={lam:9.2e} argmin K={Kb} (top {top}) "
              f"A*={Ls[Kb][2]:8.2f} CE*={Ls[Kb][3]:9.2e} L*={Ls[Kb][0]:9.3e}", flush=True)

    # pin lam by CE floor at the argmin-K optimum
    lam_star = min(LAMS, key=lambda lam: abs(np.log(
        max(scans[min(KRANGE, key=lambda K: scans[K][float(lam)][0])][float(lam)][3], 1e-300))
        - np.log(CE_FLOOR)))
    print(f"  lam* (CE floor {CE_FLOOR:g}) = {lam_star:.2e}", flush=True)

    # free-amp polish at lam* (and one decade either side): top-3 sets per K
    free_rows = {}
    for lam in [lam_star / 10, lam_star, lam_star * 10]:
        row = {}
        for K in KRANGE:
            best = None
            for _, S in pools[K][:3]:
                gS = g[list(S)]
                _, _, A0, _ = scans[K][float(min(LAMS, key=lambda l: abs(l - lam)))]
                L, amps = free_amp(gS, lam, np.full(K, A0))
                if best is None or L < best[0]:
                    eff = int((amps > 1e-3 * amps.max()).sum())
                    demo = float(amps.min() / amps.max())
                    best = (L, eff, demo, S, amps.tolist())
            row[K] = best
        order = sorted(KRANGE, key=lambda K: row[K][0])
        free_rows[float(lam)] = {K: row[K][:4] for K in KRANGE}
        print(f"  free-amp lam={lam:9.2e}: argmin K={order[0]} "
              f"order={order[:4]} eff={[row[K][1] for K in KRANGE]} "
              f"min/max_amp@argmin={row[order[0]][2]:.3f}", flush=True)

    # unconstrained descent: start from best K=8 pool set and from a random
    # wide support (16 freqs), let the concave penalty kill members
    wide = {}
    for tag, S0 in [("best8", pools[8][0][1]),
                    ("rand16", tuple(sorted(rng.choice(g.shape[0], 16, replace=False).tolist())))]:
        gS = g[list(S0)]
        _, _, A0, _ = scans[8][float(lam_star)]
        L, amps = free_amp(gS, lam_star, np.full(len(S0), A0), steps=12000)
        amps = np.array(amps)
        eff = int((amps > 1e-3 * amps.max()).sum())
        wide[tag] = (float(L), eff)
        print(f"  wide-start {tag}: survivors={eff} L={L:.3e}", flush=True)

    return dict(p=p, hard_argmax=hard_argmax, lam_star=float(lam_star),
                lam_rows=[(l, o) for l, o, _ in lam_rows],
                free=free_rows, wide=wide,
                pools={K: pools[K][0] for K in KRANGE})


if __name__ == "__main__":
    primes = [int(x) for x in sys.argv[1:]] or PRIMES
    out = {}
    for p in primes:
        out[p] = run_prime(p)
    with open("notes/ce_equilibrium.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("\nwrote notes/ce_equilibrium.json")
