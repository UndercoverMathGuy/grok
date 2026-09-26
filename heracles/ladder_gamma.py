"""ladder_gamma.py -- is Gamma_s(p) a local-optimiser artefact?  Two independent answers.

The programme
    Gamma_s(p) = max_{b >= 0, sum_k b_k = 1}  min_{D != 0}  sum_k b_k^s g_k(D),   g_k(D) = 1 - cos(2 pi k D/p)
is a NON-CONVEX max-min: for s > 1 each inner term is convex in b, so the inner min is a min of
convex functions, which is neither convex nor concave.  Every value any solver returns is only a
LOWER bound (a feasible point).  This file adds the two things that were missing.

(A) AN INDEPENDENT SOLVER.  Substitute x_k = b_k^s.  Then the objective  f(x) = min_D sum_k x_k g_k(D)
    is CONCAVE (a min of linear functions) and the constraint set  X = {x >= 0 : sum_k x_k^{1/s} <= 1}
    is a star-shaped "sparse ball" (non-convex for s > 1).  Projected supergradient ascent in x with
    radial renormalisation onto the boundary of X, many random restarts, is algorithmically unrelated
    to the SLSQP epigraph solve in nu4_program.py.  Agreement between the two is evidence, not proof.

(B) A CERTIFIED UPPER BOUND on a FIXED support S, by branch and bound.  On a box l <= x <= u the
    concave function t -> t^{1/s} lies ABOVE its chord, so
        sum_k chord_k(x_k) <= sum_k x_k^{1/s} <= 1
    is a valid LINEAR relaxation of the budget, and
        UB(box) = max { t : t <= sum_k x_k g_k(D) for all D != 0,  sum_k chord_k(x_k) <= 1,  l <= x <= u }
    is a linear program whose value upper-bounds the programme restricted to that box.  Splitting the
    widest coordinate and discarding boxes whose UB is below the best feasible value found gives a
    certified interval [LB, UB] for Gamma_s(S).  This certifies the SHARE problem on a given support;
    the choice of support is a separate, combinatorial search (exhaustive where stated).

Usage: uv run python heracles/ladder_gamma.py --primes 59 71 113 --s 1.5 2.0 2.5
"""
import argparse
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ladder_core import OUT, gmat, solve_support  # noqa: E402


# ----------------------------------------------------------------- (A) independent solver
def ascend(gS, s, x0, iters=4000, eta0=0.3, beta0=30.0, beta1=4000.0):
    """Projected supergradient ascent on f(x) = min_D (x . g(.,D)) over {x>=0, sum x^{1/s} <= 1}.
    Softmin with annealed sharpness beta; radial renormalisation keeps the budget tight."""
    x = np.maximum(x0, 1e-12)
    x = x / (x ** (1 / s)).sum() ** s
    best = -1.0
    bx = x.copy()
    for t in range(iters):
        beta = beta0 * (beta1 / beta0) ** (t / max(iters - 1, 1))
        v = x @ gS                                   # (nD,)
        w = np.exp(-beta * (v - v.min()))
        w /= w.sum()
        grad = gS @ w
        eta = eta0 * (1 - t / iters) * max(x.sum(), 1e-9)
        x = np.maximum(x + eta * grad / (np.linalg.norm(grad) + 1e-12), 1e-14)
        x = x / (x ** (1 / s)).sum() ** s            # back onto the boundary of the star set
        f = float((x @ gS).min())
        if f > best:
            best, bx = f, x.copy()
    return best, bx


def gamma_independent(g, S, s, restarts=40, seed=0):
    gS = g[np.asarray(S) - 1]
    rng = np.random.default_rng(seed)
    K = len(S)
    best, bx = -1.0, None
    for i in range(restarts):
        b0 = np.full(K, 1.0 / K) if i == 0 else rng.dirichlet(np.full(K, 1.0 + 3 * rng.random()))
        x0 = b0 ** s
        f, x = ascend(gS, s, x0)
        if f > best:
            best, bx = f, x
    b = bx ** (1 / s)
    b = b / b.sum()
    return best, b


# ----------------------------------------------------------------- (B) certified upper bound
def _lp_ub(gS, s, lo, hi):
    """LP relaxation on the box [lo, hi]: chord minorant of x^{1/s} for the budget."""
    K = len(lo)
    a = np.empty(K)
    c0 = 0.0
    for k in range(K):
        l, u = lo[k], hi[k]
        if u - l < 1e-15:
            a[k] = 0.0
            c0 += l ** (1 / s)
        else:
            a[k] = (u ** (1 / s) - l ** (1 / s)) / (u - l)
            c0 += l ** (1 / s) - a[k] * l
    if c0 > 1 + 1e-12:
        return -np.inf, None
    # variables (x_1..x_K, t); maximise t
    A_ub = np.zeros((gS.shape[1] + 1, K + 1))
    b_ub = np.zeros(gS.shape[1] + 1)
    A_ub[:gS.shape[1], :K] = -gS.T
    A_ub[:gS.shape[1], K] = 1.0
    A_ub[gS.shape[1], :K] = a
    b_ub[gS.shape[1]] = 1.0 - c0
    cvec = np.zeros(K + 1)
    cvec[K] = -1.0
    r = linprog(cvec, A_ub=A_ub, b_ub=b_ub, bounds=[(lo[k], hi[k]) for k in range(K)] + [(None, None)],
                method="highs")
    if not r.success:
        return -np.inf, None
    return float(-r.fun), r.x[:K]


def _feasible_value(gS, s, x):
    x = np.maximum(x, 0.0)
    nrm = (x ** (1 / s)).sum()
    if nrm <= 0:
        return -np.inf
    x = x / nrm ** s
    return float((x @ gS).min())


def certify(g, S, s, tol=1e-3, max_nodes=20000, time_budget=60.0):
    """Branch and bound: returns (LB, UB, nodes, secs).  LB is attained by an explicit share vector."""
    gS = g[np.asarray(S) - 1]
    K = len(S)
    lb, b0 = gamma_independent(g, S, s, restarts=8)
    lo, hi = np.zeros(K), np.ones(K)
    ub, x = _lp_ub(gS, s, lo, hi)
    heap = [(-ub, 0, lo.tolist(), hi.tolist())]
    t0 = time.time()
    nodes = 0
    cnt = 1
    while heap and nodes < max_nodes and time.time() - t0 < time_budget:
        negub, _, lo, hi = heapq.heappop(heap)
        ub_node = -negub
        if ub_node <= lb * (1 + tol):
            ub = max(lb * (1 + tol), ub_node)
            break
        lo, hi = np.array(lo), np.array(hi)
        u2, xc = _lp_ub(gS, s, lo, hi)
        if xc is not None:
            lb = max(lb, _feasible_value(gS, s, xc))
        if u2 <= lb * (1 + tol):
            continue
        j = int(np.argmax(hi - lo))
        mid = 0.5 * (lo[j] + hi[j])
        for a_, b_ in ((lo[j], mid), (mid, hi[j])):
            l2, h2 = lo.copy(), hi.copy()
            l2[j], h2[j] = a_, b_
            uu, xx = _lp_ub(gS, s, l2, h2)
            if xx is not None:
                lb = max(lb, _feasible_value(gS, s, xx))
            if uu > lb * (1 + tol):
                cnt += 1
                heapq.heappush(heap, (-uu, cnt, l2.tolist(), h2.tolist()))
        nodes += 1
    ub_final = max([lb] + [-h[0] for h in heap]) if heap else max(lb, lb * (1 + tol))
    return lb, ub_final, nodes, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primes", type=int, nargs="+", default=[59, 71, 113])
    ap.add_argument("--s", type=float, nargs="+", default=[1.5, 2.0, 2.5])
    ap.add_argument("--dk", type=int, default=1, help="certify K* - dk .. K* + dk")
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--tbudget", type=float, default=90.0)
    ap.add_argument("--out", default=str(OUT / "json" / "gamma_check.json"))
    a = ap.parse_args()
    fl = json.loads((OUT / "json" / "free_ladders_polished.json").read_text())
    res = []
    for p in a.primes:
        g = gmat(p)
        for s in a.s:
            lad = fl[f"{p}_{s:.6f}"]
            Gam = max(r["value"] for r in lad.values())
            Ks = min(int(K) for K, r in lad.items() if r["value"] >= Gam * (1 - 1e-3))
            for K in range(max(1, Ks - a.dk), Ks + a.dk + 1):
                if str(K) not in lad:
                    continue
                S = [k for k, b in zip(lad[str(K)]["S"], lad[str(K)]["b"]) if b > 1e-3] or lad[str(K)]["S"]
                v_slsqp = lad[str(K)]["value"]
                v_ind, b_ind = gamma_independent(g, S, s, restarts=40)
                lb, ub, nodes, secs = certify(g, S, s, max_nodes=a.nodes, time_budget=a.tbudget)
                row = dict(p=p, s=s, K=K, K_star=Ks, S=S, slsqp=v_slsqp, independent=v_ind,
                           rel_disagreement=abs(v_ind - v_slsqp) / max(v_slsqp, 1e-300),
                           bnb_lb=lb, bnb_ub=ub, bnb_gap=ub / max(lb, 1e-300) - 1, nodes=nodes, secs=secs)
                res.append(row)
                print(f"p {p} s {s:.3f} K {K} |S| {len(S)}: SLSQP {v_slsqp:.8f}  independent {v_ind:.8f}  "
                       f"(rel diff {row['rel_disagreement']:.2e})  certified [{lb:.6f}, {ub:.6f}] "
                       f"gap {row['bnb_gap'] * 100:.2f}%  ({nodes} nodes, {secs:.0f}s)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
