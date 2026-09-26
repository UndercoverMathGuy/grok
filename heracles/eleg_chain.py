"""eleg_chain.py -- numerical verification of the CHAIN-FRAME proof of P1.

The claim under test (P1):  for every N >= n+1 and every (KL) system

    (KL)   sum_l c_l prod_{j in T} v_lj = A [T empty]   for every T subset of {1..n},
           |v_lj| = 1,  c_l in C,

one has   sum_l |c_l|^r  >=  (n+1)^{1-r} |A|^r   for every r in (0,1].

Five steps, each checked here in fp64.

  S1  chain map:   m_l := (1, v_l1, v_l1 v_l2, ..., v_l1...v_ln) in C^{n+1},  ||m_l||^2 = n+1
                   M := sum_l c_l m_l m_l^*  is UPPER TRIANGULAR with every diagonal entry A,
                   because (m_l m_l^*)_{k,k'} = prod_{j=k'+1..k} v_lj for k > k' (an interval
                   sign pattern) and = 1 for k = k'.  Only the binom(n+1,2) INTERVAL constraints
                   of (KL) are used.  (M is NOT A*I: its strictly upper part is unconstrained.)
  S2  spectrum:    M triangular with constant diagonal A  ==>  E_k(M) = binom(n+1,k) A^k.
  S3  Cauchy-Binet / Sylvester:
                   E_k(M) = sum_{|L|=k} (prod_{l in L} d_l) G_L,  d_l := (n+1) c_l / A,
                   G_L := det Gram(m_l/sqrt(n+1) : l in L) in [0,1] (Hadamard)
                   ==>  e_k(x) >= binom(n+1,k)  for k = 0..n+1,  x_l := |d_l|.
  S4  generating:  prod_l (1 + x_l t) >= (1+t)^{n+1}  for all t >= 0.
  S5  integral:    x^r = (r sin(pi r)/pi) int_0^inf log(1+xt) t^{-1-r} dt
                   integrate S4 against t^{-1-r} dt  ==>  sum_l x_l^r >= n+1, i.e. P1.

Run:  uv run --with scipy python eleg_chain.py
"""

import itertools
import json
import math
import os

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "ELEG")
os.makedirs(OUT, exist_ok=True)
RS = (0.05, 0.2, 1 / 3, 0.5, 2 / 3, 0.75, 0.9, 0.95, 0.99, 1.0)


# ----------------------------------------------------------------- basic objects
def chain_vectors(v):
    """v: (N, n) unimodular -> m: (N, n+1), m[l,k] = prod_{j<k} v[l,j]."""
    N, n = v.shape
    m = np.ones((N, n + 1), dtype=complex)
    for k in range(1, n + 1):
        m[:, k] = m[:, k - 1] * v[:, k - 1]
    return m


def subsets(n):
    out = []
    for k in range(n + 1):
        out.extend(itertools.combinations(range(n), k))
    return out


def intervals(n):
    """non-empty interval subsets {s..t} of {0..n-1}: there are binom(n+1,2) of them."""
    return [tuple(range(s, t + 1)) for s in range(n) for t in range(s, n)]


def moment_matrix(v, sets):
    M = np.ones((len(sets), v.shape[0]), dtype=complex)
    for i, T in enumerate(sets):
        for j in T:
            M[i] *= v[:, j]
    return M


def elem_sym_all(x):
    """coefficients of prod_l (1 + x_l t), low order first: e_0, e_1, ..."""
    p = np.array([1.0])
    for xi in x:
        p = np.convolve(p, np.array([1.0, xi]))
    return p


# ----------------------------------------------------------------- (KL) samplers
def _solve(v, sets, rng, jitter=0.0):
    M = moment_matrix(v, sets)
    rhs = np.zeros(len(sets), dtype=complex)
    rhs[0] = 1.0
    c, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    if jitter > 0:
        _, s, Vh = np.linalg.svd(M)
        tol = max(M.shape) * np.finfo(float).eps * (s[0] if len(s) else 1.0)
        ns = Vh[int(np.sum(s > tol)):].conj().T
        if ns.shape[1] > 0:
            z = rng.normal(size=ns.shape[1]) + 1j * rng.normal(size=ns.shape[1])
            c = c + jitter * (ns @ z)
    return c, float(np.max(np.abs(M @ c - rhs)))


def sample_full(n, N, rng, jitter=0.0):
    v = np.exp(2j * np.pi * rng.random((N, n)))
    c, res = _solve(v, subsets(n), rng, jitter)
    return None if res > 1e-9 else (v, c)


def sample_interval(n, N, rng, jitter=0.0):
    """impose ONLY the interval constraints -- the weaker hypothesis the proof uses."""
    v = np.exp(2j * np.pi * rng.random((N, n)))
    c, res = _solve(v, [()] + intervals(n), rng, jitter)
    return None if res > 1e-9 else (v, c)


def sample_pertblock(n, N, rng, jitter=0.0, scale=0.5):
    """block phases + random extras, full system: feasible even when N < 2^n."""
    nb = n + 1
    base = np.array([[np.exp(-2j * np.pi * l / nb)] * n for l in range(nb)])
    v = base[:N] if N <= nb else np.vstack(
        [base, np.exp(2j * np.pi * rng.random((N - nb, n)))])
    v = v * np.exp(1j * scale * rng.normal(size=v.shape))
    v = v / np.abs(v)
    c, res = _solve(v, subsets(n), rng, jitter)
    return None if res > 1e-9 else (v, c)


def sample_nearcancel(n, N, rng, jitter=0.0):
    """two nearly-cancelling big atoms: the family that refutes the domination lemma."""
    got = sample_full(n, N, rng) if N >= 2 ** n else sample_pertblock(n, N, rng)
    if got is None:
        return None
    v, c = got
    M = moment_matrix(v, subsets(n))
    _, s, Vh = np.linalg.svd(M)
    tol = max(M.shape) * np.finfo(float).eps * s[0]
    ns = Vh[int(np.sum(s > tol)):].conj().T
    if ns.shape[1] == 0:
        return v, c
    z = rng.normal(size=ns.shape[1]) + 1j * rng.normal(size=ns.shape[1])
    return v, c + (10.0 ** rng.uniform(-1, 2)) * (ns @ z)


# ----------------------------------------------------------------- the five steps
def Ek_expansion(d, u, k):
    """sum_{|L|=k} (prod_{l in L} d_l) det Gram(u_L), brute force."""
    tot = 0.0 + 0j
    for L in itertools.combinations(range(len(d)), k):
        UL = u[list(L)]
        tot += np.prod(d[list(L)]) * np.linalg.det(UL.conj() @ UL.T)
    return tot


def check_all(v, c, do_expansion=True):
    n = v.shape[1]
    D = n + 1
    A = c.sum()
    m = chain_vectors(v)
    u = m / np.sqrt(D)
    d = D * c / A
    x = np.abs(d)
    M = np.einsum("l,lk,lj->kj", c, m, m.conj())

    o = {"n": n, "N": len(c), "A": abs(A)}
    o["S1_lower_max"] = float(np.max(np.abs(np.tril(M, -1))))   # must be ~0
    o["S1_diag_err"] = float(np.max(np.abs(np.diag(M) - A)))    # must be ~0
    o["S1_upper_max"] = float(np.max(np.abs(np.triu(M, 1))))    # unconstrained; info only
    o["S2_eig_err"] = float(np.max(np.abs(np.linalg.eigvals(M) - A)))
    e = elem_sym_all(x)
    idt, ratio = 0.0, np.inf
    for k in range(1, D + 1):
        target = math.comb(D, k)
        if do_expansion:
            idt = max(idt, abs(Ek_expansion(d, u, k) - target))
        ratio = min(ratio, (e[k] if k < len(e) else 0.0) / target)
    o["S3_identity_err"] = float(idt)
    o["S3_ek_ratio_min"] = float(ratio)
    ts = np.logspace(-6, 6, 600)
    o["S4_min_slack"] = float(np.min(np.sum(np.log1p(np.outer(ts, x)), axis=1)
                                     - D * np.log1p(ts)))
    o["P1_ratio"] = {f"{r:.3f}": float(np.sum(np.abs(c) ** r)
                                       / ((n + 1) ** (1 - r) * abs(A) ** r)) for r in RS}
    return o


def check_integral_rep():
    worst = 0.0
    for r in (0.15, 1 / 3, 0.5, 2 / 3, 0.8, 0.95):
        for x in (0.01, 0.3, 1.0, 2.5, 17.0, 1e3):
            a, _ = quad(lambda t: np.log1p(x * t) * t ** (-1 - r), 0, 1,
                        limit=300, epsabs=1e-13, epsrel=1e-13)
            b, _ = quad(lambda t: np.log1p(x * t) * t ** (-1 - r), 1, np.inf,
                        limit=300, epsabs=1e-13, epsrel=1e-13)
            approx = r * np.sin(np.pi * r) / np.pi * (a + b)
            worst = max(worst, abs(approx - x ** r) / x ** r)
    return worst


# ------------------------------------------- adversarial: the e_k relaxation itself
def relaxation_min(D, r, N, rng, starts=60):
    """min sum x^r s.t. e_k(x) >= binom(D,k), k=1..D, x >= 0.  Proof says the answer is D."""
    combs = [float(math.comb(D, k)) for k in range(D + 1)]

    def obj(z):
        return float(np.sum((z ** 2) ** r))

    def cons(z):
        e = elem_sym_all(z ** 2)
        return np.array([(e[k] if k < len(e) else 0.0) - combs[k] for k in range(1, D + 1)])

    best, bestx = np.inf, None
    for _ in range(starts):
        z0 = rng.random(N) * 2.0
        res = minimize(obj, z0, constraints=[{"type": "ineq", "fun": cons}],
                       method="SLSQP", options={"maxiter": 400, "ftol": 1e-14})
        if res.success and np.min(cons(res.x)) > -1e-9 and res.fun < best:
            best, bestx = float(res.fun), np.sort(res.x ** 2)[::-1]
    return best, bestx


def main():
    rng = np.random.default_rng(20260921)
    report = {}

    print("=== S5: integral representation of x^r ===", flush=True)
    w = check_integral_rep()
    print(f"  max relative error: {w:.3e}", flush=True)
    report["integral_rep_max_rel_err"] = w

    print("\n=== S1-S4 + conclusion, random (KL) systems ===", flush=True)
    cells = []
    samplers = {"full": sample_full, "interval": sample_interval,
                "pertblock": sample_pertblock, "nearcancel": sample_nearcancel}
    for n in (1, 2, 3, 4, 5):
        for sname, sfun in samplers.items():
            for N in range(n + 1, min(2 ** n + 4, 12) + 1):
                agg = {"S1_lower_max": 0.0, "S1_diag_err": 0.0, "S2_eig_err": 0.0,
                       "S3_identity_err": 0.0, "S1_upper_max": 0.0,
                       "S3_ek_ratio_min": np.inf, "S4_min_slack": np.inf,
                       "P1_min": np.inf, "count": 0}
                trials = 40 if n <= 3 else 15
                for _ in range(trials):
                    got = sfun(n, N, rng, jitter=0.3)
                    if got is None:
                        continue
                    v, c = got
                    if abs(c.sum()) < 1e-6 or not np.all(np.isfinite(c)):
                        continue
                    o = check_all(v, c, do_expansion=(N <= 10))
                    for key in ("S1_lower_max", "S1_diag_err", "S2_eig_err",
                                "S3_identity_err", "S1_upper_max"):
                        agg[key] = max(agg[key], o[key])
                    agg["S3_ek_ratio_min"] = min(agg["S3_ek_ratio_min"], o["S3_ek_ratio_min"])
                    agg["S4_min_slack"] = min(agg["S4_min_slack"], o["S4_min_slack"])
                    agg["P1_min"] = min(agg["P1_min"], min(o["P1_ratio"].values()))
                    agg["count"] += 1
                if agg["count"] == 0:
                    continue
                agg.update({"n": n, "N": N, "sampler": sname})
                cells.append(agg)
                print(f"  n={n} N={N:2d} {sname:10s} k={agg['count']:3d} | "
                      f"tri {agg['S1_lower_max']:.1e} diag {agg['S1_diag_err']:.1e} "
                      f"eig {agg['S2_eig_err']:.1e} CB {agg['S3_identity_err']:.1e} | "
                      f"e_k {agg['S3_ek_ratio_min']:.6f}  S4 {agg['S4_min_slack']:+.2e}  "
                      f"P1 {agg['P1_min']:.9f}", flush=True)
    report["cells"] = cells

    print("\n=== adversarial: is the e_k relaxation alone enough? ===", flush=True)
    rel = []
    for D in (2, 3, 4, 5, 6):
        for r in (1 / 3, 0.5, 2 / 3, 0.9):
            for N in (D, D + 1, D + 3, 2 * D):
                best, bx = relaxation_min(D, r, N, rng)
                rel.append({"D": D, "r": r, "N": N, "min": best, "ratio": best / D,
                            "argmin": None if bx is None else
                            [float(t) for t in np.round(bx, 6)]})
                print(f"  D={D} r={r:.3f} N={N:2d}: min={best:.9f} "
                      f"ratio={best / D:.9f}", flush=True)
    report["relaxation"] = rel

    with open(os.path.join(OUT, "chain.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(f"\nwrote {os.path.join(OUT, 'chain.json')}", flush=True)


if __name__ == "__main__":
    main()
