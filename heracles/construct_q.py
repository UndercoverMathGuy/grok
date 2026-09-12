"""H6 for the geometry dial: constructed gamma_q(K) ceilings, p 113, m 512, nu {4, 6}, q {2, 3, 4, 6, inf}.

Reuses nudial/construct_gamma.py's exact product-grid K-committee nets (layout (1,1); relu-6 via identity-extended
stacks) and generalises the closed-form block balance to l_q:
  max_alpha  g prod(alpha_i) / ||(alpha_i block_i)||_q^B  =  g / (B^{B/q} prod_i ||block_i||_q)      (q = inf: g / prod_i max|block_i|)
K-sets come from heracles/kstar.json (equal-share ladder, K <= 12); "dense" = all 56 frequencies, 9 neurons each.
Per-frequency amplitudes are polished by coordinate ascent on the exact q-normalised margin.

Usage: uv run python heracles/construct_q.py [--polish-iters 25] -> heracles/construct_q.json + table
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "nudial"))
from construct_gamma import committee_net, full_margin, P   # noqa: E402

QS = (2.0, 3.0, 4.0, 6.0, float("inf"))
TRAINED = {  # median over 8 seeds, grokked runs, cross-norm matrix (this session): {nu: {q: (gamma_q, K mode)}}
    4: {2.0: (4.27e-6, "4-5"), 3.0: (9.23e-3, 5), 4.0: (0.472, 5), 6.0: (24.1, 5), float("inf"): (4.54e4, 6)},
    6: {2.0: (3.96e-8, 4), 3.0: (6.07e-3, 4), 4.0: (2.24, 4), 6.0: (682, 4)},
}


def qnorm(x, q):
    x = np.abs(np.asarray(x, dtype=np.float64))
    return x.max() if q == float("inf") else (x ** q).sum() ** (1 / q)


def gamma_q_balanced(E, U, V, W1, W, nu, q):
    g = full_margin(E, U, V, W1, W)
    blocks = [E, np.concatenate([U.ravel(), V.ravel()]), W1, W]
    if nu == 6:                       # identity-extended E2 (d x d) and W2 (2K x 2K) as their own scale groups
        d = E.shape[0]; blocks = [E, np.ones(d), blocks[1], W1, np.ones(W1.shape[0]), W]
    B = len(blocks); norms = [qnorm(b, q) for b in blocks]
    pref = 1.0 if q == float("inf") else B ** (B / q)
    return g / (pref * float(np.prod(norms))), g


def polish(S, m, n_a, n_b, nu, q, iters):
    K = len(S); amps = np.ones(K)
    ev = lambda am: gamma_q_balanced(*committee_net(S, am, m, n_a, n_b), nu, q)[0]
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
    ap = argparse.ArgumentParser(); ap.add_argument("--polish-iters", type=int, default=25); ap.add_argument("--nus", default="4,6")
    a = ap.parse_args(); m = 512
    sets = {int(k): tuple(v) for k, v in json.load(open(HERE / "kstar.json"))["113"]["S"].items()}
    sets["dense"] = tuple(range(1, P // 2 + 1))
    out = {}
    for nu in [int(x) for x in a.nus.split(",")]:
        out[str(nu)] = {}
        print(f"\n== nu {nu} ==  columns: K | " + " | ".join(f"g{('inf' if q == float('inf') else int(q))} eq / polished" for q in QS), flush=True)
        for K, S in sets.items():
            n_a = n_b = max(2, int(math.floor(math.sqrt(m // len(S)))))
            row = {"S": list(S), "n_a": n_a}; line = f" K={K!s:>5} n/f={n_a*n_b:>2}:"
            for q in QS:
                g_eq, _ = gamma_q_balanced(*committee_net(S, np.ones(len(S)), m, n_a, n_b), nu, q)
                g_pol = g_eq
                if K != "dense" and a.polish_iters > 0:
                    g_pol, amps = polish(S, m, n_a, n_b, nu, q, a.polish_iters)
                qs = "inf" if q == float("inf") else f"{q:g}"; row[f"g{qs}_eq"] = g_eq; row[f"g{qs}_pol"] = g_pol
                line += f"  {g_eq:9.3g}/{g_pol:9.3g}"
            out[str(nu)][str(K)] = row; print(line, flush=True)
        print(" trained medians (gamma_q, K mode): " + ", ".join(f"q{('inf' if q == float('inf') else int(q))}: {v[0]:.3g} @K{v[1]}" for q, v in TRAINED.get(nu, {}).items()))
        for q in QS:
            qs = "inf" if q == float("inf") else f"{q:g}"; col = {K: r[f"g{qs}_pol"] for K, r in out[str(nu)].items()}
            best = max(col, key=col.get); tr = TRAINED.get(nu, {}).get(q)
            print(f"   q {qs}: argmax_K constructed = {best} (gamma {col[best]:.3g}); trained {tr[0]:.3g} = {tr[0]/col[best]:.2f} of that" if tr else f"   q {qs}: argmax_K constructed = {best}")
    (HERE / "construct_q.json").write_text(json.dumps(out, indent=1)); print("\nwrote heracles/construct_q.json")


if __name__ == "__main__":
    main()
