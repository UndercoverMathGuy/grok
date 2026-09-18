"""nu4_program.py -- the sparse programme with FREE amplitudes (no equal-share assumption).
  gamma_prog(S; nu) = max_{b >= 0, sum_{k in S} b_k = 1}  min_{D != 0} sum_k b_k^{nu/2} (1 - cos 2 pi k D / p)
(b_k = share of the l2 budget on frequency k; amplitude a_k = b_k^{nu/2}; equal shares give M_K / K^{nu/2}.)
Solved by SLSQP on the epigraph form with restarts. Reports, for each ladder set S_K (kstar.py), the equal-share value,
the free-amplitude value, and for S = all frequencies the best support found (restarts from every S_K and random sparse b).
Run ON A POD (CPU-only, light):  python3 heracles/nu4_program.py --p 71 --nu 3 4 5 --out heracles/nu4_program_p71.json"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kstar import ladder, G  # noqa: E402


def solve(g, e, b0):
    """g: (K, nD) coverage rows; e = nu/2. max t s.t. sum_k b_k^e g_k(D) >= t, sum b = 1, b >= 0."""
    K = g.shape[0]
    f = lambda x: -x[-1]; fj = lambda x: np.r_[np.zeros(K), -1.0]
    c = lambda x: (np.clip(x[:K], 0, None) ** e) @ g - x[-1]
    cj = lambda x: np.c_[(e * np.clip(x[:K], 1e-300, None) ** (e - 1))[None, :] * g.T, -np.ones(g.shape[1])]
    x0 = np.r_[b0, ((b0 ** e) @ g).min()]
    r = minimize(f, x0, jac=fj, method="SLSQP", bounds=[(0, 1)] * K + [(None, None)],
                 constraints=[dict(type="ineq", fun=c, jac=cj), dict(type="eq", fun=lambda x: x[:K].sum() - 1, jac=lambda x: np.r_[np.ones(K), 0.0])],
                 options=dict(maxiter=500, ftol=1e-14))
    b = np.clip(r.x[:K], 0, None); b /= b.sum()
    return float(((b ** e) @ g).min()), b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=71); ap.add_argument("--nu", type=float, nargs="+", default=[3, 4, 5])
    ap.add_argument("--kmax", type=int, default=12); ap.add_argument("--restarts", type=int, default=200); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p = a.p; g = G(p)[:, :(p - 1) // 2]; nf = g.shape[0]; lad = ladder(p, a.kmax); rng = np.random.default_rng(0); res = {}
    for nu in a.nu:
        e = nu / 2; rows = {}
        print(f"\np {p} nu {nu:g}:  K | equal-share M_K/K^e | free-amplitude | gain | shares b_k", flush=True)
        for K in range(2, a.kmax + 1):
            S = np.array(lad["S"][K]) - 1
            eq = lad["M_K"][K] / K ** e
            best, bb = max((solve(g[S], e, b0) for b0 in [np.full(K, 1 / K)] + [rng.dirichlet(np.full(K, 3.0)) for _ in range(8)]), key=lambda t: t[0])
            rows[K] = dict(S=lad["S"][K], equal=eq, free=best, b=bb.tolist(), n_active=int((bb > 1e-3).sum()))
            print(f"  {K:2d} | {eq:.5f} | {best:.5f} | {best / eq:.3f} | " + " ".join(f"{x:.3f}" for x in bb), flush=True)
        full = []
        starts = [(f"S_{K}", np.array(lad["S"][K]) - 1) for K in range(2, a.kmax + 1)] + [(f"rand{i}", rng.choice(nf, rng.integers(3, 10), replace=False)) for i in range(a.restarts)]
        for name, S in starts:
            b0 = np.full(nf, 1e-4); b0[S] = 1.0; b0 /= b0.sum()
            v, b = solve(g, e, b0); full.append((v, int((b ** e > 0.05 * (b ** e).max()).sum()), name, b))
        full.sort(key=lambda t: -t[0])
        print("  full support (all %d freqs), top local optima: " % nf + "; ".join(f"{v:.5f} (K_active {k}, from {nm})" for v, k, nm, _ in full[:8]), flush=True)
        ks = [k for v, k, _, _ in full if v >= 0.98 * full[0][0]]
        print(f"  supports within 2% of the best: K_active {min(ks)}..{max(ks)}", flush=True)
        res[str(nu)] = dict(ladder=rows, full_top=[dict(value=v, K_active=k, start=nm, b=b.tolist()) for v, k, nm, b in full[:20]],
                            eq_argmax=max(rows, key=lambda K: rows[K]["equal"]), free_argmax=max(rows, key=lambda K: rows[K]["free"]))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
