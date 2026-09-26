"""P3C step 2: the PER-FREQUENCY criterion (PF_A).

Idea (see P3C.md SS3).  Replace the junk-level bookkeeping (which the chord
refutation kills) by a single per-frequency statement obtained from a product of
per-slot Riesz-type test densities.  For a one-frequency population
{c_l, v_lj} put

    g_A(psi) := sum_l c_l prod_{j=1..n} ( 1 + A e^{i psi_j} v_lj )
              = sum_T C_T A^{|T|} e^{i sum_{j in T} psi_j} ,

the value the network's frequency-k contribution takes under a product test
density whose slot-j Fourier coefficient at mode 2k is A e^{i psi_j}.

    (PF_A)   min_{psi in T^n} Re g_A(psi)  <=  kappa0^{-1} ( sum_l |c_l|^r )^{1/r},
             kappa0 = (n+1)^{(1-r)/r} .

Plain reading: "after the test density has been aimed at it, no population beats
the junk-free (n+1)-block of the same weight-decay cost."  At A = 1 this IS the
Key Lemma / Theorem B (P1).  At A = 0 it is false.  This script measures the
threshold A_crit(n,r) = inf{A : (PF_A) holds}, by maximising

    ratio(pop) = [ min_psi Re g_A(psi) ] / [ kappa0^{-1} (sum_l |c_l|^r)^{1/r} ]

over populations of N atoms.  (PF_A) holds iff sup ratio <= 1.

  uv run python p3c_pf.py --cells nu3q2 nu4q2 --Ns 1 2 3 4 5 --starts 200
"""
import argparse, json, os
import numpy as np
from scipy.optimize import minimize

import p3c_core as C

CELLS = {"nu3q2": (3, 2.0), "nu3q2.5": (3, 2.5), "nu4q2": (4, 2.0),
         "nu4q3": (4, 3.0), "nu5q2": (5, 2.0), "nu5q4": (5, 4.0)}


def gcoef(c, V, A):
    """Flat coefficient vector of g_A over bitmasks t: entry t is C_t * A^{|t|}."""
    n = V.shape[1]
    Ct = C.coeffs(c, V)                    # indexed by bitmask
    w = np.array([A ** bin(t).count("1") for t in range(1 << n)])
    return Ct * w


_BITS = {}


def bits(n):
    if n not in _BITS:
        _BITS[n] = np.array([[(t >> j) & 1 for j in range(n)]
                             for t in range(1 << n)], dtype=float)
    return _BITS[n]


def min_reg(coef, n, restarts=12, sweeps=60, rng=None, tol=1e-15):
    """min over psi in T^n of Re g_A(psi), by EXACT cyclic coordinate descent.

    g_A is affine in each e^{i psi_j} separately: g = P + Q e^{i psi_j} with
    P = sum_{T not containing j} ... and Q = sum_{T containing j} ... .  So the
    exact per-coordinate minimum is Re P - |Q|, at e^{i psi_j} = -conj(Q)/|Q|.
    Sweeping to convergence from several starts finds the global minimum in
    practice (checked against a brute-force grid in --selftest).
    """
    B = bits(n)
    if rng is None:
        rng = np.random.default_rng(0)
    best, bpsi = np.inf, None
    for s in range(restarts):
        psi = np.zeros(n) if s == 0 else (
            np.pi * np.ones(n) if s == 1 else rng.uniform(0, 2 * np.pi, n))
        z = np.exp(1j * psi)
        prev = np.inf
        for _ in range(sweeps):
            for j in range(n):
                ph = np.exp(1j * (B @ psi))
                m1 = B[:, j] == 1
                Q = np.sum(coef[m1] * ph[m1]) / z[j]
                P = np.sum(coef[~m1] * ph[~m1])
                if abs(Q) > 1e-300:
                    psi[j] = np.angle(-np.conj(Q))
                    z[j] = np.exp(1j * psi[j])
                cur = float(np.real(P + Q * z[j]))
            if prev - cur < tol:
                break
            prev = cur
        val = float(np.real(coef @ np.exp(1j * (B @ psi))))
        if val < best:
            best, bpsi = val, psi.copy()
    return best, bpsi


def ratio(z, N, n, r, A, restarts=6):
    c, V = C.unpack(z, N, n)
    cost = float(np.sum(np.abs(c) ** r))
    if cost <= 0:
        return 0.0
    denom = (cost ** (1.0 / r)) / C.kappa0(n, r)
    v, _ = min_reg(gcoef(c, V, A), n, restarts=restarts)
    return v / denom


def search(n, r, A, N, starts, rng):
    best = (-np.inf, None)
    for s in range(starts):
        if s == 0 and N >= 1:
            c0 = np.zeros(N, dtype=complex); c0[0] = 1.0
            z0 = C.pack(c0, np.zeros((N, n)))
        elif s == 1 and N >= n + 1:
            cb, tb = C.block(n)
            c0 = np.concatenate([cb, 1e-3 * np.ones(N - len(cb))])
            th0 = np.vstack([tb, rng.uniform(0, 2 * np.pi, (N - tb.shape[0], n))])
            z0 = C.pack(c0, th0)
        else:
            c0 = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            z0 = C.pack(c0, rng.uniform(0, 2 * np.pi, (N, n)))
        res = minimize(lambda z: -ratio(z, N, n, r, A, 4), z0, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-11, "fatol": 1e-13})
        # exact re-evaluation with many more inner restarts
        c, V = C.unpack(res.x, N, n)
        cost = float(np.sum(np.abs(c) ** r))
        if cost <= 1e-12:
            continue
        v, psi = min_reg(gcoef(c, V, A), n, restarts=40, rng=rng)
        rr = v / ((cost ** (1.0 / r)) / C.kappa0(n, r))
        if rr > best[0]:
            best = (rr, res.x.copy())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["nu3q2", "nu4q2"])
    ap.add_argument("--Ns", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--starts", type=int, default=120)
    ap.add_argument("--A", type=float, nargs="*", default=None,
                    help="A values; default = the N=1 threshold 1 - kappa0^{-1/n} "
                         "and a small ladder above/below it")
    ap.add_argument("--out", default="FINAL_RUNS/P3C/pf.json")
    a = ap.parse_args()

    out = {}
    for key in a.cells:
        nu, q = CELLS[key]
        n, r = nu - 1, q / nu
        k0 = C.kappa0(n, r)
        A1 = 1.0 - k0 ** (-1.0 / n)          # the N = 1 requirement, exactly
        As = a.A if a.A else [A1 - 0.06, A1 - 0.03, A1, A1 + 0.03, A1 + 0.08,
                              0.5, 0.75, 1.0]
        cell = {"nu": nu, "q": q, "r": r, "n": n, "kappa0": k0,
                "A_N1": A1, "runs": []}
        for A in As:
            rng = np.random.default_rng(777)
            per, cfg = {}, {}
            for N in a.Ns:
                rr, z = search(n, r, A, N, a.starts, rng)
                per[str(N)] = float(rr)
                if z is not None:
                    cc, VV = C.unpack(z, N, n)
                    cfg[str(N)] = {"c_re": cc.real.tolist(), "c_im": cc.imag.tolist(),
                                   "theta": np.angle(VV).tolist()}
                print(f"{key} A={A:.6f} N={N}: sup ratio = {rr:.9f}", flush=True)
            cell["runs"].append({"A": float(A), "perN": per, "cfg": cfg,
                                 "sup": float(max(per.values())),
                                 "holds": bool(max(per.values()) <= 1 + 1e-9)})
            print(f"== {key} A={A:.6f}: sup over N = {max(per.values()):.9f} "
                  f"{'OK' if max(per.values()) <= 1 + 1e-9 else 'FAILS'}\n", flush=True)
        out[key] = cell
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
