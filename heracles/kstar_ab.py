"""Ladder + constructed ceilings for abelian group tasks (Z_120, Z_11 x Z_11; Z_113 as the mod-add cross-check).

Ladder (equal share, q = 2): M_K = max_{|S| = K} min_{x != e} sum_{r in S} cov[r, x], cov = 1 - chi_r(x)/d_r over the
nontrivial real irreps (2-dim: 1 - cos(2 pi j.x / n); order-2 characters: 1 - (-1)^..); gamma_hat(K) = M_K / K^{nu/2};
K* = argmax. Search: exhaustive when C(R, K) <= 4e6 else local search with swap moves and restarts.
Constructed ceiling: the exact product-grid committee net of nudial/construct_gamma.py with the phase table
theta_r(x) = 2 pi j.x / n (2-dim irreps) — same blocks, same closed-form l_2 balance, amplitude polish — evaluated on the
group's own label table; gamma_2 per K for the ladder's S*_K, plus the free-run comparison denominator.

Usage: uv run python heracles/kstar_ab.py --groups z120,z11x11,z113 --out heracles/kstar_ab.json
"""
import argparse, itertools, json, math, sys
from math import comb
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from groups import get_group

NUS = (4, 6); M = 512


def coverage(G):
    cov = G.coverage(); nt = [r for r in range(G.R) if G.irrep_names[r] != "trivial"]
    x = [i for i in range(G.N) if i != G.e]
    return cov[np.ix_(nt, x)], nt


def best_subset(g, K, rng, exhaustive_limit=4_000_000, restarts=200):
    R = g.shape[0]
    if comb(R, K) <= exhaustive_limit:
        best, bestS = -1.0, None
        it = itertools.combinations(range(R), K)
        while True:
            block = list(itertools.islice(it, 200_000))
            if not block: break
            idx = np.array(block); m = g[idx].sum(1).min(1); j = int(m.argmax())
            if m[j] > best: best, bestS = float(m[j]), [int(x) for x in idx[j]]
        return best, bestS, "exhaustive"
    best, bestS = -1.0, None
    for _ in range(restarts):
        S = rng.choice(R, K, replace=False).tolist(); cur = g[S].sum(0).min(); improved = True
        while improved:
            improved = False
            for i in range(K):
                base = g[[x for j, x in enumerate(S) if j != i]].sum(0)
                cand = (base[None, :] + g).min(1); cand[S] = -1; j = int(cand.argmax())
                if cand[j] > cur + 1e-12: S[i] = j; cur = cand[j]; improved = True
        if cur > best: best, bestS = float(cur), sorted(S)
    return best, bestS, "local"


# ---------------------------------------------------------------- constructed committee nets on a group (layout (1,1))
def phase_table(G, r):
    """theta_r(x) for a 2-dim irrep r = the angle of the complex character chi_j(x), recovered from the label."""
    name = G.irrep_names[r]; assert name.startswith("f"), name
    j = [int(t) for t in name[1:].split("+")]
    orders = _orders(G); els = list(itertools.product(*[range(n) for n in orders]))
    return np.array([2 * np.pi * sum(y * x / n for x, y, n in zip(a, j, orders)) for a in els])


def _orders(G):
    return {"z120": (120,), "z11x11": (11, 11), "z113": (113,)}[G.name]


def committee_net(G, S, amps, m, n_a, n_b):
    N = G.N; K = len(S); pairs = np.array(list(itertools.product(range(n_a), range(n_b)))) * 2 * np.pi / np.array([n_a, n_b])
    n = len(pairs); assert K * n <= m, (K, n, m)
    d = 2 * K; E = np.zeros((d, N)); U = np.zeros((m, d)); V = np.zeros((m, d)); W1 = np.zeros((d, m)); W = np.zeros((N, d))
    for j, r in enumerate(S):
        th = phase_table(G, r); s = amps[j] ** 0.25; rows = slice(j * n, (j + 1) * n); phi, psi = pairs[:, 0], pairs[:, 1]
        E[2 * j] = s * np.cos(th); E[2 * j + 1] = s * np.sin(th)
        U[rows, 2 * j] = s * np.cos(phi); U[rows, 2 * j + 1] = -s * np.sin(phi)
        V[rows, 2 * j] = s * np.cos(psi); V[rows, 2 * j + 1] = -s * np.sin(psi)
        W1[2 * j, rows] = s * np.cos(phi + psi) / n; W1[2 * j + 1, rows] = -s * np.sin(phi + psi) / n
        W[:, 2 * j] = s * np.cos(th); W[:, 2 * j + 1] = s * np.sin(th)
    return E, U, V, W1, W


def full_margin(G, E, U, V, W1, W):
    N = G.N; a = np.repeat(np.arange(N), N); b = np.tile(np.arange(N), N)
    H = np.maximum((U @ E)[:, a] + (V @ E)[:, b], 0.0); L = (W @ (W1 @ H)).T
    y = G.MULT[a, b]; idx = np.arange(N * N); zy = L[idx, y]; L[idx, y] = -np.inf
    return float((zy - L.max(1)).min())


def gamma_balanced(G, E, U, V, W1, W, nu):
    """closed-form l_2 balance over the B scale groups (nu 4: E, U|V, W1, W; nu 6 adds identity-extended E2 and W2)."""
    g = full_margin(G, E, U, V, W1, W)
    blocks = [E, np.concatenate([U.ravel(), V.ravel()]), W1, W]
    if nu == 6:
        blocks = [E, np.ones(E.shape[0]), blocks[1], W1, np.ones(W1.shape[0]), W]
    B = len(blocks); norms = [np.linalg.norm(np.asarray(b)) for b in blocks]
    return g / (B ** (B / 2) * float(np.prod(norms))), g


def polish(G, S, m, n_a, n_b, nu, iters=25):
    K = len(S); amps = np.ones(K)
    ev = lambda am: gamma_balanced(G, *committee_net(G, S, am, m, n_a, n_b), nu)[0]
    best = ev(amps); step = 0.15
    for _ in range(iters):
        improved = False
        for j in range(K):
            for sgn in (1, -1):
                cand = amps.copy(); cand[j] *= math.exp(sgn * step); v = ev(cand)
                if v > best * (1 + 1e-12): best, amps, improved = v, cand, True
        if not improved:
            step *= 0.5
            if step < 0.02: break
    return best, amps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--groups", default="z120,z11x11,z113"); ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--out", default=str(HERE / "kstar_ab.json")); ap.add_argument("--polish-iters", type=int, default=20)
    a = ap.parse_args(); rng = np.random.default_rng(0); out = {}
    for gname in a.groups.split(","):
        G = get_group(gname); g, nt = coverage(G); res = {"N": G.N, "ladder": {}, "construct": {}}
        print(f"\n== {gname}: |G| {G.N}, {len(nt)} nontrivial irreps")
        for K in range(1, a.kmax + 1):
            MK, S, how = best_subset(g, K, rng); names = [G.irrep_names[nt[i]] for i in S]
            res["ladder"][K] = dict(M=MK, S=[int(nt[i]) for i in S], names=names, how=how,
                                    gamma_hat={nu: MK / K ** (nu / 2) for nu in NUS})
            print(f"  K {K:2d}  M_K {MK:.4f}  {how:10}  S = {names}")
        for nu in NUS:
            gh = {K: v["gamma_hat"][nu] for K, v in res["ladder"].items()}; Ks = max(gh, key=gh.get); top = gh[Ks]
            res[f"kstar_nu{nu}"] = Ks; res[f"plateau74_nu{nu}"] = [K for K, v in gh.items() if v >= 0.74 * top]
            print(f"  nu {nu}: K* = {Ks}, plateau74 = {res[f'plateau74_nu{nu}']}, gamma_hat(K*+1)/gamma_hat(K*) = {gh.get(Ks + 1, 0) / top:.3f}")
        # constructed ceilings for the ladder's sets (2-dim irreps only; a set containing an order-2 character is skipped)
        for K, v in res["ladder"].items():
            S = v["S"]
            if any(G.irrep_dims[r] != 2 for r in S): res["construct"][K] = None; continue
            n_a = n_b = max(2, int(math.floor(math.sqrt(M // K)))); row = {"n_a": n_a}
            for nu in NUS:
                g_eq, _ = gamma_balanced(G, *committee_net(G, S, np.ones(K), M, n_a, n_b), nu)
                g_pol, amps = polish(G, S, M, n_a, n_b, nu, a.polish_iters) if a.polish_iters else (g_eq, np.ones(K))
                row[f"nu{nu}"] = dict(eq=g_eq, pol=g_pol, amps=[float(x) for x in amps])
            res["construct"][K] = row
            print(f"  construct K {K:2d}: " + "  ".join(f"nu{nu} {row[f'nu{nu}']['eq']:.3g} -> {row[f'nu{nu}']['pol']:.3g}" for nu in NUS), flush=True)
        for nu in NUS:
            col = {K: r[f"nu{nu}"]["pol"] for K, r in res["construct"].items() if r}
            if col:
                Kc = max(col, key=col.get); res[f"construct_argmax_nu{nu}"] = Kc; res[f"ceiling_nu{nu}"] = col[Kc]
                print(f"  constructed argmax nu {nu}: K = {Kc} at gamma_2 {col[Kc]:.3g}")
        out[gname] = res
    json.dump(out, open(a.out, "w"), indent=1, default=float); print("wrote", a.out)


if __name__ == "__main__":
    main()
