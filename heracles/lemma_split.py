"""lemma_split.py -- THE SPLITTING PROGRAMME (atom form of the (n+1)-block conjecture).

Reduction (derived in LEMMA_ATTEMPT.md Sec. 2).  For the gated chain restricted to neurons that are
PURE single-frequency cosines (the equality class of the per-neuron AM-GM + Parseval steps), the whole
question "is the (n+1)-neuron common-phase block a global maximiser of gamma_q?" collapses, per
frequency, to the following finite complex programme with NO reference to p, to the task, or to the net:

    variables : c_1..c_N in C  (one per neuron: c_l = (prod_j r_lj) rho_l e^{i psi_l}, the signed
                                clean amplitude of neuron l),
                v_lj  on the unit circle (j = 1..n) (v_lj = e^{-2 i alpha_lj}, alpha_lj the phase of
                                neuron l's j-th embedding).
    constraints: sum_l c_l prod_{j in T} v_lj = 0   for every NONEMPTY T subset of {1..n}   (junk-free)
                 sum_l c_l = A = 1                                                          (clean amplitude)
    objective  : minimise  F_r = sum_l |c_l|^r ,  r = q/nu in (0, 1].

Plain language: each neuron is one rank-one term; the 2^n numbers indexed by T are the 2^n Fourier
"sign patterns" the neuron emits at its frequency (T = the set of operands whose sign is flipped);
T = empty is the useful (clean) one, the other 2^n - 1 are junk.  We must cancel all junk while keeping
the clean sum equal to 1, and we pay sum |c_l|^r because a balanced neuron of clean amplitude |c_l|
costs ||theta_l||^q = const * |c_l|^{q/nu}.

CONJECTURE (equivalent to the target inequality on this class):  min F_r = (n+1)^{1-r}.
The (n+1)-neuron common-phase block is c_l = 1/(n+1), v_lj = e^{-2 pi i l/(n+1)}, giving F_r = (n+1)^{1-r}.

This script minimises F_r by SLSQP from many random starts, for n = 1..4, N = n+1 .. n+6, several r.
fp64 throughout.  Local CPU, seconds.

Usage: uv run python heracles/lemma_split.py --out heracles/FINAL_RUNS/LEMMA
"""
import argparse, itertools, json, math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

DELTA = 1e-30  # smoothing of |c|^r at c = 0 (never active at the optima found)


def subsets(n):
    """All T subset of {0..n-1} as tuples, empty first."""
    out = [()]
    for k in range(1, n + 1):
        out += list(itertools.combinations(range(n), k))
    return out


def unpack(x, N, n):
    c = x[:N] + 1j * x[N:2 * N]
    eta = x[2 * N:].reshape(N, n)
    return c, eta


def cons_vec(x, N, n, TS):
    """Real vector of all constraint residuals: Re/Im of  sum_l c_l prod_{j in T} v_lj - A delta_{T,0}."""
    c, eta = unpack(x, N, n)
    v = np.exp(1j * eta)                      # (N, n)
    out = []
    for T in TS:
        w = np.ones(N, dtype=complex)
        for j in T:
            w = w * v[:, j]
        s = (c * w).sum() - (1.0 + 0j if len(T) == 0 else 0.0)
        out += [s.real, s.imag]
    return np.array(out)


def obj(x, N, n, r):
    c, _ = unpack(x, N, n)
    return float(((np.abs(c) ** 2 + DELTA) ** (r / 2)).sum())


def obj_grad(x, N, n, r):
    c, _ = unpack(x, N, n)
    a2 = np.abs(c) ** 2 + DELTA
    d = r * (a2 ** (r / 2 - 1))               # d F / d(a2)  * 2 ... careful: dF/d a2 = (r/2) a2^{r/2-1}
    g = np.zeros_like(x)
    g[:N] = d * c.real                        # dF/dRe c = (r/2) a2^{r/2-1} * 2 Re c
    g[N:2 * N] = d * c.imag
    return g


def solve(n, N, r, rng, tries=400):
    TS = subsets(n)
    best = (np.inf, None)
    for _ in range(tries):
        x0 = np.concatenate([rng.normal(0, 1 / N, 2 * N), rng.uniform(0, 2 * np.pi, N * n)])
        res = minimize(obj, x0, args=(N, n, r), jac=obj_grad, method="SLSQP",
                       constraints=[{"type": "eq", "fun": cons_vec, "args": (N, n, TS)}],
                       options={"maxiter": 600, "ftol": 1e-14})
        v = cons_vec(res.x, N, n, TS)
        if np.abs(v).max() < 1e-10 and res.fun < best[0]:
            best = (float(res.fun), res.x.copy())
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="heracles/FINAL_RUNS/LEMMA")
    ap.add_argument("--tries", type=int, default=400)
    ap.add_argument("--nmax", type=int, default=4)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for n in range(1, a.nmax + 1):
        nu = n + 1
        for q in (2.0, 2.5, 3.0):
            if q > nu:
                continue
            r = q / nu
            target = (n + 1) ** (1 - r)
            for N in range(2, n + 7):
                f, x = solve(n, N, r, rng, a.tries)
                amps = sorted(np.abs(unpack(x, N, n)[0]).round(8), reverse=True) if x is not None else None
                rows.append(dict(n=n, nu=nu, q=q, r=r, N=N, F=f, target=target,
                                 ratio=(f / target if np.isfinite(f) else None), amps=amps))
                print(f"n={n} nu={nu} q={q:g} r={r:.4f} N={N:2d}  min F_r = {f:.8f}  "
                      f"(n+1)^(1-r) = {target:.8f}  ratio {f/target if np.isfinite(f) else float('nan'):.6f}"
                      f"  amps {amps if amps is None else [f'{u:.4f}' for u in amps[:6]]}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "split.json").write_text(json.dumps(rows, indent=1))
