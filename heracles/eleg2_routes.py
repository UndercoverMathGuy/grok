"""eleg2_routes.py -- the elegant candidate routes that FAIL, with exact breaking points.

R1  DETERMINANT form.   |det(I + s M)| >= (1 + s m)^D,  M = sum_l c_l m_l m_l^*  -- REFUTED.
    (It would give the general-N theorem in two lines: Cauchy-Binet + Hadamard bound
     prod_l (1 + s D |c_l|) >= |det(I + sM)|, then Bernstein.)
    Minimal witness search at n = 1 (D = 2), where det M = A^2 - C_1 Cbar_1.
R2  SPECTRUM.           Re lambda >= m for every eigenvalue of M  -- REFUTED (implies R1).
R3  NUMERICAL RANGE.    Herm(M) = (M + M^*)/2 >= m I               -- REFUTED (implies R2).
R4  DIAGONAL restriction.  Replacing the torus by the diagonal u_1 = ... = u_n = z reduces the
    junk-free system to n equations, and the constant (n+1) is then FALSE -- explicit witness.
R5  GAUGE SYMMETRISATION is circular: averaging the population over the D rotations
    v -> omega^s v kills every junk pattern but multiplies the cost by exactly D^{1-r}.
R6  The eta-parametrisation: what Theorem A implies for Cl2_r(eta) is only the LINEAR bound
    of P2 Theorem P2-1, i.e. Theorem A is not a sharpening of the junk-cost curve -- it says
    the curve was the wrong object.

Run: uv run --with scipy python eleg2_routes.py
"""
import json
import os
import sys
import numpy as np
from math import comb
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eleg2_core import (chain_matrix, min_re_h, rand_pop, CT, subsets, elem_sym, h_at)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "ELEG2")
os.makedirs(OUT, exist_ok=True)
rec = {}


# ---------------------------------------------------------------- R1/R2/R3
def detratio(x, N, n, rng, starts=40):
    c = np.exp(x[:N]) * np.exp(1j * x[N:2 * N])
    v = np.exp(1j * x[2 * N:].reshape(N, n))
    m, _ = min_re_h(c, v, 1.0, starts=starts, rng=rng)
    if m <= 1e-9:
        return 1e6
    D = n + 1
    M = chain_matrix(c, v)
    return float(abs(np.linalg.det(M)) / m ** D)


def r1(nmax=3, Nmax=6, restarts=14):
    rng = np.random.default_rng(5)
    best = {}
    for n in range(1, nmax + 1):
        for N in range(2, Nmax + 1):
            bv, bx = np.inf, None
            dim = 2 * N + N * n
            for t in range(restarts):
                x0 = np.concatenate([0.4 * rng.normal(size=N), 2 * np.pi * rng.random(N),
                                     2 * np.pi * rng.random(N * n)])
                res = minimize(lambda z: detratio(z, N, n, rng, 8), x0,
                               method="Nelder-Mead",
                               options=dict(maxiter=400 + 60 * dim, maxfev=400 + 60 * dim,
                                            xatol=1e-9, fatol=1e-12))
                val = detratio(res.x, N, n, rng, 50)
                if val < bv:
                    bv, bx = val, res.x.copy()
            c = np.exp(bx[:N]) * np.exp(1j * bx[N:2 * N])
            v = np.exp(1j * bx[2 * N:].reshape(N, n))
            best[f"n{n}_N{N}"] = dict(ratio=bv,
                                      c=[[float(z.real), float(z.imag)] for z in c],
                                      v=[[float(z.real), float(z.imag)] for z in v.ravel()])
            print(f"  R1 n={n} N={N}: min |det M| / m^D = {bv:.9f}")
    return best


def r2r3(trials=400):
    rng = np.random.default_rng(6)
    out = {}
    for n in [1, 2, 3]:
        ws, wh = np.inf, np.inf
        for N in [n + 1, n + 2, 2 * (n + 1)]:
            for kind in ["generic", "junkfree", "nearjf"]:
                for t in range(trials):
                    c, v = rand_pop(rng, N, n, kind)
                    c = c / max(np.abs(c).max(), 1e-30)
                    m, _ = min_re_h(c, v, 1.0, starts=14, rng=rng)
                    if m <= 1e-9:
                        continue
                    M = chain_matrix(c, v)
                    ws = min(ws, float(np.linalg.eigvals(M).real.min() / m))
                    wh = min(wh, float(np.linalg.eigvalsh((M + M.conj().T) / 2).min() / m))
        out[n] = dict(min_Re_spec_over_m=ws, min_eig_Herm_over_m=wh)
        print(f"  R2/R3 n={n}: min Re(spec)/m = {ws:.6f}   lam_min(Herm)/m = {wh:.6f}")
    return out


# ---------------------------------------------------------------- R4 diagonal
def r4(n=2, restarts=40):
    """Minimise sum|c_l|^r over populations whose DIAGONAL restriction h(z,..,z) is constant.

    The diagonal conditions are sum_l c_l e_k(v_l) = 0 for k = 1..n (only n equations, not
    2^n - 1), so N = n atoms already suffice and the constant (n+1) is false.
    """
    rng = np.random.default_rng(7)
    r = 2 / 3
    N = n                                    # strictly fewer than n+1 atoms
    best = np.inf
    for t in range(restarts):
        b = 2 * np.pi * rng.random(N * n)

        def f(bb):
            v = np.exp(1j * bb.reshape(N, n))
            # e_k(v_l) for k=0..n, per atom -> (n+1, N); solve rows 1..n = 0, row 0 = 1
            Emat = np.array([elem_sym_complex(v[l], n) for l in range(N)]).T  # (n+1, N)
            M = Emat[: n + 1]
            rhs = np.zeros(n + 1, dtype=complex); rhs[0] = 1.0
            c, *_ = np.linalg.lstsq(M, rhs, rcond=None)
            resid = np.linalg.norm(M @ c - rhs)
            return float(np.sum(np.abs(c) ** r)) + 50.0 * resid
        res = minimize(f, b, method="Nelder-Mead",
                       options=dict(maxiter=8000, maxfev=8000, xatol=1e-10, fatol=1e-13))
        best = min(best, res.fun)
    return best, (n + 1) ** (1 - r)


def elem_sym_complex(z, n):
    e = np.zeros(n + 1, dtype=complex)
    e[0] = 1.0
    for zi in z:
        for k in range(n, 0, -1):
            e[k] += e[k - 1] * zi
    return e


# ---------------------------------------------------------------- R5 symmetrisation
def r5(n=2, N=4, trials=60):
    """The D-fold rotation average kills all junk but multiplies the cost by exactly D^{1-r}."""
    rng = np.random.default_rng(8)
    D = n + 1
    w = 0.0
    wcost = 0.0
    for t in range(trials):
        c, v = rand_pop(rng, N, n, "generic")
        om = np.exp(2j * np.pi / D)
        cs, vs = [], []
        for s in range(D):
            cs.append(c / D)
            vs.append(v * om ** s)
        cs = np.concatenate(cs); vs = np.concatenate(vs, axis=0)
        Cavg = CT(cs, vs)
        w = max(w, float(np.max(np.abs(Cavg[1:]))))           # junk of the average: must be 0
        for r in [1/3, 2/3]:
            ratio = np.sum(np.abs(cs) ** r) / (D ** (1 - r) * np.sum(np.abs(c) ** r))
            wcost = max(wcost, abs(ratio - 1.0))
    return w, wcost


# ---------------------------------------------------------------- R6 eta implication
def r6(n=2, r=2/3, etas=(0.02, 0.05, 0.1, 0.2, 0.3)):
    """Theorem A implies Cl2_r(eta) >= (n+1)^{1-r} (1 - sqrt(2^n-1) eta)_+^r, i.e. exactly
    P2's LINEAR Theorem P2-1 -- the curve is not improved, the variable is replaced."""
    D = n + 1
    return {float(e): float(D ** (1 - r) * max(0.0, 1 - np.sqrt(2 ** n - 1) * e) ** r)
            for e in etas}


if __name__ == "__main__":
    print("R1  determinant form")
    rec["R1"] = r1()
    print("R2/R3  spectrum and numerical range")
    rec["R2R3"] = r2r3(trials=250)
    print("R4  diagonal restriction")
    b, tgt = r4()
    rec["R4"] = dict(min_cost_with_n_atoms=b, target_if_true=tgt)
    print(f"  R4 n=2, r=2/3: cost with only n=2 atoms = {b:.9f}  vs (n+1)^(1-r) = {tgt:.9f}")
    print("R5  gauge symmetrisation")
    j, cst = r5()
    rec["R5"] = dict(max_junk_of_average=j, max_cost_ratio_minus_one=cst)
    print(f"  R5: junk of the D-fold average = {j:.3e};  cost / (D^(1-r) cost) - 1 = {cst:.3e}")
    rec["R6"] = r6()
    print("R6  the eta bound Theorem A implies:", {k: round(v, 6) for k, v in rec["R6"].items()})
    with open(os.path.join(OUT, "routes.json"), "w") as f:
        json.dump(rec, f, indent=1)
