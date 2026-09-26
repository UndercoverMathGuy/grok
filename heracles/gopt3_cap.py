"""gopt3_cap.py -- the CAPPED-MASS conjecture (C'), a proposed r-free route to the Key Lemma.

Background.  The "Key Lemma" of LEMMA_ATTEMPT.md is the finite complex statement

    c_1..c_N in C,  v_lj on the unit circle (l = 1..N, j = 1..n),
    sum_l c_l prod_{j in T} v_lj = A [T empty]   for every T subset of {1..n}     (KL-constraints)
    ==>   sum_l |c_l|^r >= (n+1)^{1-r} |A|^r     for every r in (0,1].

It is proved for N = n+1 (Vandermonde/Hadamard) and OPEN for N > n+1 ("G1").

This script tests a stronger, r-FREE statement that implies the Key Lemma for every r at once:

    (C')  for every theta > 0:   sum_l min(|c_l|, theta)  >=  min( (n+1) theta , |A| ).

Why (C') implies the Key Lemma.  For x >= 0 and t > 0 one has x^r >= t^r min(x/t, 1)
(if x <= t then x^r = t^r (x/t)^r >= t^r (x/t); if x >= t then x^r >= t^r).  Take t = |A|/(n+1):

    sum_l |c_l|^r >= t^r sum_l min(|c_l|/t, 1) = t^r (1/t) sum_l min(|c_l|, t)
                  >= t^{r-1} (n+1) t = (n+1)^{1-r} |A|^r .

Both steps are equalities at the (n+1)-block (all |c_l| = |A|/(n+1)), so (C') is exactly as tight.

Plain reading of (C'): "you cannot hide the clean amplitude in a few big atoms".  Capping every
atom at theta and adding up, you still have at least (n+1) theta of mass -- i.e. the clean amplitude
A must be carried by at least n+1 atoms' worth of mass at EVERY scale theta <= A/(n+1).

The script minimises  sum_l min(|c_l|, theta)  over the KL-constraint manifold (free c AND free
phases v) by SLSQP from many random starts, for a grid of theta, n and N.  fp64.  Local CPU.

Usage: uv run python heracles/gopt3_cap.py --out heracles/FINAL_RUNS/GOPT3
"""
import argparse, itertools, json, math, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
from scipy.optimize import minimize

DELTA = 1e-18   # smoothing of |c| near 0
DELTA2 = 1e-18  # smoothing of the kink of min(.,theta)


def subsets(n):
    out = [()]
    for k in range(1, n + 1):
        out += list(itertools.combinations(range(n), k))
    return out


def unpack(x, N, n):
    c = x[:N] + 1j * x[N:2 * N]
    eta = x[2 * N:].reshape(N, n)
    return c, eta


def cons_and_jac(x, N, n, TS):
    """Residual vector (real) of all KL-constraints, and its Jacobian."""
    c, eta = unpack(x, N, n)
    v = np.exp(1j * eta)
    res = np.empty(2 * len(TS))
    J = np.zeros((2 * len(TS), x.size))
    for t, T in enumerate(TS):
        w = np.ones(N, dtype=complex)
        for j in T:
            w = w * v[:, j]
        val = np.sum(c * w) - (1.0 if len(T) == 0 else 0.0)
        res[2 * t], res[2 * t + 1] = val.real, val.imag
        # d/d Re c_l = w_l ;  d/d Im c_l = i w_l
        J[2 * t, :N] = w.real
        J[2 * t + 1, :N] = w.imag
        J[2 * t, N:2 * N] = -w.imag
        J[2 * t + 1, N:2 * N] = w.real
        for j in T:                                   # d/d eta_lj = i * c_l w_l
            d = 1j * c * w
            J[2 * t, 2 * N + np.arange(N) * n + j] = d.real
            J[2 * t + 1, 2 * N + np.arange(N) * n + j] = d.imag
    return res, J


def obj_and_grad(x, N, n, theta):
    c = x[:N] + 1j * x[N:2 * N]
    a = np.sqrt(c.real ** 2 + c.imag ** 2 + DELTA)          # ~ |c_l|
    u = a - theta
    s = np.sqrt(u ** 2 + DELTA2)
    f = np.sum((a + theta - s) / 2.0)                        # ~ sum min(a, theta)
    dfda = (1.0 - u / s) / 2.0                               # in [0,1]
    g = np.zeros_like(x)
    g[:N] = dfda * c.real / a
    g[N:2 * N] = dfda * c.imag / a
    return f, g


def solve_cell(n, N, theta, starts, seed):
    TS = subsets(n)
    rng = np.random.default_rng(seed)
    cons = [{"type": "eq",
             "fun": lambda x: cons_and_jac(x, N, n, TS)[0],
             "jac": lambda x: cons_and_jac(x, N, n, TS)[1]}]
    best, bx = np.inf, None
    for s in range(starts):
        if s == 0:                                            # the (n+1)-block, extras at 0
            c0 = np.zeros(N, dtype=complex); c0[:n + 1] = 1.0 / (n + 1)
            eta0 = np.zeros((N, n))
            eta0[:n + 1] = (-2 * math.pi * np.arange(n + 1) / (n + 1))[:, None]
            eta0[n + 1:] = rng.uniform(0, 2 * math.pi, (max(N - n - 1, 0), n))
        else:
            c0 = rng.normal(size=N) + 1j * rng.normal(size=N)
            c0 = c0 / np.sum(c0) if abs(np.sum(c0)) > 1e-9 else c0
            eta0 = rng.uniform(0, 2 * math.pi, (N, n))
        x0 = np.concatenate([c0.real, c0.imag, eta0.ravel()])
        try:
            r = minimize(lambda x: obj_and_grad(x, N, n, theta), x0, jac=True, method="SLSQP",
                         constraints=cons, options=dict(maxiter=600, ftol=1e-14))
        except Exception:
            continue
        res = cons_and_jac(r.x, N, n, TS)[0]
        if np.max(np.abs(res)) > 1e-9:
            continue
        c = r.x[:N] + 1j * r.x[N:2 * N]
        val = float(np.sum(np.minimum(np.abs(c), theta)))     # exact value, not the smoothed one
        if val < best:
            best, bx = val, r.x
    return best, bx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--extra", type=int, default=4, help="N runs from n+1 to n+1+extra")
    ap.add_argument("--starts", type=int, default=120)
    ap.add_argument("--out", default=str(ROOT / "FINAL_RUNS" / "GOPT3"))
    a = ap.parse_args()

    rows = []
    worst = 1.0
    for n in a.ns:
        for N in range(n + 1, n + 2 + a.extra):
            for frac in (0.2, 0.5, 0.8, 1.0, 1.3, 2.0):
                theta = frac / (n + 1)
                t0 = time.time()
                best, _ = solve_cell(n, N, theta, a.starts, seed=1000 * n + 10 * N + int(100 * frac))
                tgt = min((n + 1) * theta, 1.0)
                ratio = best / tgt
                worst = min(worst, ratio)
                rows.append(dict(n=n, N=N, theta=theta, frac=frac, best=best, target=tgt, ratio=ratio))
                print(f"n {n}  N {N}  theta {theta:.4f} (={frac:g}/(n+1))  min sum min(|c|,th) = {best:.8f}  "
                      f"target {tgt:.8f}  ratio {ratio:.6f}  [{time.time()-t0:.1f}s]", flush=True)
    print(f"\nWORST ratio over all cells: {worst:.6f}   ((C') holds iff >= 1)")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "cap.json").write_text(json.dumps(dict(worst=worst, rows=rows), indent=1))
