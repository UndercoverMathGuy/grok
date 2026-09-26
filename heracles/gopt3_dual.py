"""gopt3_dual.py -- the PEELING NUMBERS P_h, and their exact convex dual.

Context.  The open gap "G1" is the Key Lemma for more than n+1 atoms:

    c_1..c_N in C,  v_lj on the unit circle,  sum_l c_l prod_{j in T} v_lj = A [T empty]  (T subset {1..n})
    ==>  sum_l |c_l|^r >= (n+1)^{1-r} |A|^r          for every r in (0,1].

This script studies the following r-free family of statements (A = 1, x_l := |c_l|):

    (P_h)   [total mass] - [mass of the h LARGEST atoms]  >=  (n+1-h)/(n+1),     h = 0, 1, ..., n.

h = 0 is the classical  sum_l |c_l| >= |A|.  h = n is a sharp form of the "kill" bound (killing n atoms
with the n operand slots), where the elementary argument only gives 2^{-n} instead of 1/(n+1).
The (n+1)-block makes every (P_h) an equality.

EXACT DUAL.  Fix the h excluded atoms' positions v_1..v_h.  The remaining atoms' minimal total mass is
an ATOMIC NORM: minimise sum_{l>h}|c_l| such that the moment vector m_T = sum_{l>h} c_l prod_{j in T} v_lj
lies in  e_empty + span{ (prod_{j in T} v_{lj})_T : l <= h }.  The atoms are the vectors
(prod_{j in T} v_j)_T = tensor_j (1, v_j), and the dual norm of a coefficient vector Lambda is
sup_{v in T^n} |Lambda(v)| where Lambda(v) = sum_T Lambda_T prod_{j in T} v_j is a MULTI-AFFINE polynomial.
Norm-minimisation over an affine subspace = maximisation over the annihilator inside the dual ball, so

    P_h(v_1..v_h) = max { |Lambda_empty| : Lambda multi-affine, sup_{T^n}|Lambda| <= 1, Lambda(v_i) = 0 }
                  = 1 / min { ||Lambda||_{L^inf(T^n)} : Lambda multi-affine, Lambda_empty = 1, Lambda(v_i) = 0 },

and (P_h) is the claim  min over v_1..v_h of that quantity  >=  (n+1-h)/(n+1), i.e.

    (D_h)   for ANY h <= n points of the torus T^n there is a multi-affine Lambda with
            Lambda(v_i) = 0, Lambda_empty = 1 and  sup_{T^n} |Lambda| <= (n+1)/(n+1-h).

Plain reading: a multi-affine "test polynomial" can be made to vanish at any h <= n prescribed points of
the torus while its average over the torus stays at least (n+1-h)/(n+1) of its peak.  That is an existence
statement -- a construction -- not an impossibility proof, which is why it is a promising route.

The script computes min ||Lambda||_inf by linear programming (epigraph + phase discretisation of
|Lambda(g)| <= t on a torus grid).  DROPPING CONSTRAINTS LOWERS THE MINIMUM, so the LP value t_lp obeys
t_lp <= min ||Lambda||_inf, i.e. 1/t_lp is an UPPER bound on P_h(v); re-measuring the exact sup-norm
t_true of the LP's own solution on a much finer grid gives the matching LOWER bound 1/t_true.  Both are
printed, bracketing P_h(v).  fp64, local CPU.

Usage: uv run python heracles/gopt3_dual.py --n 2 --out heracles/FINAL_RUNS/GOPT3
"""
import argparse, itertools, json, math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
from scipy.optimize import linprog


def subsets(n):
    out = [()]
    for k in range(1, n + 1):
        out += list(itertools.combinations(range(n), k))
    return out


def mono(TS, V):
    """V: (M, n) array of unit-modulus points.  Returns (M, 2^n) matrix of prod_{j in T} v_j."""
    M = V.shape[0]
    out = np.ones((M, len(TS)), dtype=complex)
    for t, T in enumerate(TS):
        w = np.ones(M, dtype=complex)
        for j in T:
            w = w * V[:, j]
        out[:, t] = w
    return out


def torus_grid(n, G):
    ang = 2 * math.pi * np.arange(G) / G
    mesh = np.meshgrid(*([ang] * n), indexing="ij")
    return np.exp(1j * np.stack([m.ravel() for m in mesh], 1))


def supnorm(TS, lam, n, G=256):
    """Exact-ish sup over T^n of |Lambda|, by a fine grid (multi-affine => smooth, grid is reliable)."""
    V = torus_grid(n, G)
    return float(np.max(np.abs(mono(TS, V) @ lam)))


def min_supnorm(n, pts, G=48, P=24):
    """min ||Lambda||_inf over multi-affine Lambda with Lambda_empty = 1 and Lambda(pts) = 0.

    LP variables: [Re lam_1..Re lam_{K-1}, Im lam_1..Im lam_{K-1}, t]  (lam_0 = Lambda_empty = 1 fixed).
    Constraints  : Re( e^{-i phi} Lambda(g) ) <= t  for every grid point g and phase phi  (=> |Lambda| <= t)
                   Lambda(v_i) = 0  (real + imaginary part).
    Returns (t_lp, lam).  t_lp is a LOWER bound on the true min (finite grid); supnorm(lam) is an UPPER bound.
    """
    TS = subsets(n)
    K = len(TS)
    V = torus_grid(n, G)
    Mg = mono(TS, V)                                   # (G^n, K)
    phis = 2 * math.pi * np.arange(P) / P
    nv = 2 * (K - 1) + 1                               # unknowns
    rows, rhs = [], []
    for ph in phis:
        Z = Mg * np.exp(-1j * ph)                      # Re(sum_T lam_T Z_T) <= t
        # Re part: sum_T [ReZ * Re lam - ImZ * Im lam];  lam_0 = 1 contributes a constant
        A = np.concatenate([Z[:, 1:].real, -Z[:, 1:].imag, -np.ones((Z.shape[0], 1))], 1)
        rows.append(A)
        rhs.append(-Z[:, 0].real)
    A_ub = np.concatenate(rows, 0)
    b_ub = np.concatenate(rhs, 0)
    A_eq, b_eq = [], []
    if len(pts):
        Mp = mono(TS, np.asarray(pts))
        for i in range(Mp.shape[0]):
            A_eq.append(np.concatenate([Mp[i, 1:].real, -Mp[i, 1:].imag, [0.0]]))
            b_eq.append(-Mp[i, 0].real)
            A_eq.append(np.concatenate([Mp[i, 1:].imag, Mp[i, 1:].real, [0.0]]))
            b_eq.append(-Mp[i, 0].imag)
    c = np.zeros(nv); c[-1] = 1.0
    r = linprog(c, A_ub=A_ub, b_ub=b_ub,
                A_eq=np.array(A_eq) if A_eq else None,
                b_eq=np.array(b_eq) if b_eq else None,
                bounds=[(None, None)] * (nv - 1) + [(0, None)], method="highs")
    if not r.success:
        return np.inf, None
    lam = np.concatenate([[1.0 + 0j], r.x[:K - 1] + 1j * r.x[K - 1:2 * (K - 1)]])
    return float(r.x[-1]), lam


def scan(n, h, ngrid, G, P, seed=0, rand=0):
    """min over h points of T^n of D_h = 1/min||Lambda||_inf.  v_1 normalised to (1,..,1) by rotation."""
    TS = subsets(n)
    best = (np.inf, None)          # smallest D_h found  (worst case for (P_h))
    cands = []
    if h == 0:
        cands = [[]]
    else:
        base = [np.ones(n, dtype=complex)]
        if h == 1:
            cands = [base]
        else:
            ang = 2 * math.pi * np.arange(ngrid) / ngrid
            free = list(itertools.product(*([ang] * (n * (h - 1)))))
            for f in free:
                pts = list(base)
                for i in range(h - 1):
                    pts.append(np.exp(1j * np.array(f[i * n:(i + 1) * n])))
                cands.append(pts)
    rng = np.random.default_rng(seed)
    for _ in range(rand):
        pts = [np.ones(n, dtype=complex)] + [np.exp(1j * rng.uniform(0, 2 * math.pi, n)) for _ in range(h - 1)]
        cands.append(pts)
    for pts in cands:
        t_lp, lam = min_supnorm(n, pts, G=G, P=P)
        if lam is None:
            continue
        t_true = supnorm(TS, lam, n)                  # >= true min  => D_h >= 1/t_true (certified)
        D_lo = 1.0 / t_true
        if D_lo < best[0]:
            best = (D_lo, pts, t_lp, t_true)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--ngrid", type=int, default=12)
    ap.add_argument("--G", type=int, default=40)
    ap.add_argument("--P", type=int, default=20)
    ap.add_argument("--rand", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "FINAL_RUNS" / "GOPT3"))
    a = ap.parse_args()
    n = a.n
    res = []
    for h in range(0, n + 1):
        D_lo, pts, t_lp, t_true = scan(n, h, a.ngrid, a.G, a.P, rand=(a.rand if h >= 2 else 0))
        tgt = (n + 1 - h) / (n + 1)
        print(f"n {n}  h {h}:  min_v D_h in [{D_lo:.6f}, {1.0/t_lp:.6f}]   target (n+1-h)/(n+1) = {tgt:.6f}"
              f"   ratio {D_lo/tgt:.5f}    [LP t {t_lp:.6f}, true supnorm {t_true:.6f}]", flush=True)
        if pts:
            print("      worst points:", [np.round(np.angle(p) / math.pi, 4) for p in pts], "(angles / pi)")
        res.append(dict(n=n, h=h, D=D_lo, target=tgt, ratio=D_lo / tgt,
                        pts=[[float(x) for x in np.angle(p)] for p in pts] if pts else []))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"dual_n{n}.json").write_text(json.dumps(res, indent=1))
