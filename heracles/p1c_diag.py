"""P1C / step 1: the DIAGONAL RELAXATION of the sharp-constant problem (P1).

Setting.
--------
(KL) is: N complex "atoms" c_l and unit-modulus phases v_lj (l = 1..N, j = 1..n) with

    sum_l c_l prod_{j in T} v_lj = A * [T empty]   for every T subset of {1..n},

equivalently the generating identity

    h(y) := sum_l c_l prod_j (1 + y_j v_lj)  ==  A      for every y in C^n.

P1 claims  sum_l |c_l|^r >= (n+1)^{1-r} |A|^r  for every N >= n+1, r = q/nu in (0,1].

The relaxation.
---------------
Put every slot to the SAME value, y_j = y.  Then, with

    p_l(y) := prod_j (y + conj(v_lj))      (monic, degree n, ALL ROOTS ON THE UNIT CIRCLE),
    d_l    := c_l * prod_j v_lj            (so |d_l| = |c_l|),

the identity becomes the ONE-VARIABLE polynomial identity

    (DIAG)     sum_l d_l p_l(y)  ==  A     for every y in C,

i.e. n+1 complex linear constraints on d (one per power of y) instead of 2^n.
Every (KL) system yields a (DIAG) system with the same moduli, so

    min over (DIAG) of sum |d_l|^r   <=   min over (KL) of sum |c_l|^r.

So if the (DIAG) minimum were still (n+1)^{1-r}, P1 would follow -- and (DIAG) is a
one-variable problem about monic polynomials with roots on the circle, which is a much
more classical object.  THIS SCRIPT DECIDES THAT.

What is computed
----------------
For each (n, q, N): minimise sum_l |d_l|^r over the phases (n per atom) and, when
N > n+1, over the (N-n-1)-complex-dimensional null space of the constraint matrix.
Report min / (n+1)^{1-r}.  A value < 1 kills the route (and is recorded exactly).

Local CPU, float64, no pods.
"""
import argparse, itertools, json, os, sys
import numpy as np
from scipy.optimize import minimize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1C")


def poly_coeffs(u):
    """Coefficients (highest power first) of prod_j (y + u_j), u = array of n unit numbers."""
    c = np.array([1.0 + 0.0j])
    for uj in u:
        c = np.convolve(c, np.array([1.0 + 0.0j, uj]))
    return c


def build_M(phases, n, N):
    """phases: (N, n) real. Returns (n+1, N) complex matrix; row i = coeff of y^{n-i}."""
    M = np.zeros((n + 1, N), dtype=complex)
    for l in range(N):
        M[:, l] = poly_coeffs(np.exp(1j * phases[l]))
    return M


def cost(d, r, eps):
    return float(np.sum((np.abs(d) ** 2 + eps) ** (r / 2.0)))


def solve_d(M, n, N, t=None):
    """Particular solution of M d = b (b = A e_last, A = 1) plus null-space component t."""
    b = np.zeros(n + 1, dtype=complex)
    b[n] = 1.0
    # least squares particular solution + null space basis
    d0, *_ = np.linalg.lstsq(M, b, rcond=None)
    if N > n + 1:
        _, s, Vh = np.linalg.svd(M)
        ns = Vh[(n + 1):].conj().T  # (N, N-n-1)
        if t is not None:
            d0 = d0 + ns @ t
        return d0, ns
    return d0, None


def objective(x, n, N, r, eps):
    ph = x[: N * n].reshape(N, n)
    M = build_M(ph, n, N)
    if N > n + 1:
        k = N - n - 1
        t = x[N * n: N * n + k] + 1j * x[N * n + k: N * n + 2 * k]
        d, _ = solve_d(M, n, N, t)
    else:
        d, _ = solve_d(M, n, N)
    # feasibility guard (square / overdetermined cases): penalise residual
    b = np.zeros(n + 1, dtype=complex); b[n] = 1.0
    res = np.abs(M @ d - b).sum()
    return cost(d, r, eps) + 1e4 * res


def run_cell(n, N, q, starts, seed):
    nu = n + 1
    r = q / nu
    rng = np.random.default_rng(seed)
    nvar = N * n + (2 * (N - n - 1) if N > n + 1 else 0)
    best = np.inf
    bestx = None
    for s in range(starts):
        x0 = np.zeros(nvar)
        x0[: N * n] = rng.uniform(0, 2 * np.pi, N * n)
        if N > n + 1:
            x0[N * n:] = rng.normal(scale=0.3, size=nvar - N * n)
        x = x0
        for eps in (1e-4, 1e-8, 1e-12, 1e-16):
            res = minimize(objective, x, args=(n, N, r, eps), method="Nelder-Mead",
                           options=dict(maxiter=8000, maxfev=20000, xatol=1e-12, fatol=1e-14))
            x = res.x
        v = objective(x, n, N, r, 0.0)
        if v < best:
            best, bestx = v, x.copy()
    target = nu ** (1 - r)
    # report the minimiser
    ph = bestx[: N * n].reshape(N, n)
    M = build_M(ph, n, N)
    if N > n + 1:
        k = N - n - 1
        t = bestx[N * n: N * n + k] + 1j * bestx[N * n + k: N * n + 2 * k]
        d, _ = solve_d(M, n, N, t)
    else:
        d, _ = solve_d(M, n, N)
    b = np.zeros(n + 1, dtype=complex); b[n] = 1.0
    return dict(n=n, N=N, q=q, r=r, best=best, target=target, ratio=best / target,
                moduli=sorted(np.abs(d).tolist(), reverse=True),
                phases=(ph % (2 * np.pi)).tolist(),
                residual=float(np.abs(M @ d - b).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["2:2:3", "2:2:4", "2:2:5", "3:2:4", "3:2:5"],
                    help="n:q:N triples")
    ap.add_argument("--starts", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="diag")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for cell in a.cells:
        n, q, N = cell.split(":")
        n, N = int(n), int(N)
        q = float(q)
        row = run_cell(n, N, q, a.starts, a.seed + 17 * n + N)
        rows.append(row)
        print(f"n={n} nu={n+1} q={q} N={N}  r={row['r']:.4f}  "
              f"min={row['best']:.9f}  target={row['target']:.9f}  "
              f"ratio={row['ratio']:.9f}  moduli={['%.5f' % m for m in row['moduli']]}",
              flush=True)
    with open(os.path.join(OUT, f"{a.tag}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
