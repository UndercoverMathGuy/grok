r"""P1A: LEMMA R'' -- closing the window in m = min_l |c_l| by the complete family
of point-substitution constraints, with EXACT chord geometry.

THE SETUP.  n = 2, N = 4, |A| = 1.  The identity

    sum_{l=1}^4 c_l (u_1 - x_l)(u_2 - y_l) = A u_1 u_2      for all (u_1, u_2)

is exact.  Substituting u_1 = x_s and u_2 = y_t (s, t in {1,2,3,4}) kills atom s
(its first factor vanishes) and atom t (its second factor vanishes), and the
right-hand side has modulus 1 because |x_s| = |y_t| = 1.  Hence the COMPLETE
family of 16 constraints

    (PS_{s,t})   sum_{l not in {s,t}}  |c_l| |x_s - x_l| |y_t - y_l|  >=  1 .

Relabel so that atom 4 is the smallest, |c_4| = m, and gauge-fix x_4 = y_4 = 1.
Write w_l := |c_l| for l = 1,2,3; every w_l >= m.  Then

    Psi(m) := m^r  +  min over the slot angles (psi_1..psi_3, chi_1..chi_3)
                      of  [ min over w in P(m; psi, chi) of sum_{l<=3} w_l^r ] ,

    P(m; psi, chi) := { w >= m : (PS_{s,t}) for all s,t }  (a polytope in R^3),

is a PROVED lower bound for sum_l |c_l|^r.  The inner problem minimises a
CONCAVE function over a polytope, so its minimum is at a vertex and is found by
exact vertex enumeration (3 variables); the outer problem is a 6-angle global
minimisation.

WHAT THIS BUYS.  sum_l |c_l|^r >= 4 m^r already beats 3^{1-r} once
m >= m_triv = (3^{1-r}/4)^{1/r} = 0.2165 at r = 2/3, and Lemma R' (p1a_lemmaR.py)
covers m <= 0.055.  Only the window m in [0.055, 0.2165] is left; this script
measures how much of it Psi covers.

Plain reading: "pin each atom down by shooting the other atoms out of the
identity, one slot value at a time, and keep the true chord lengths instead of
bounding them by 2."

Run:  uv run python p1a_window.py --starts 400
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
from scipy.optimize import minimize

from p1a_core import atoms_cramer, bigM, points

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1A")


def chord(t):
    return 2.0 * np.abs(np.sin(0.5 * t))


def constraints(m, psi, chi):
    """Rows (a, rhs) of  a . w >= rhs  for w = (w_1,w_2,w_3), from (PS_{s,t}).
    Angles: psi[3] = slot-1 angles of atoms 1..3 (atom 4 at 0), chi[3] likewise."""
    P = np.concatenate((psi, [0.0]))   # slot-1 angles, atom 4 last
    C = np.concatenate((chi, [0.0]))   # slot-2 angles
    rows, rhs = [], []
    for s in range(4):
        for t in range(4):
            a = np.zeros(3)
            const = 0.0
            for l in range(4):
                if l == s or l == t:
                    continue
                coef = chord(P[s] - P[l]) * chord(C[t] - C[l])
                if l < 3:
                    a[l] += coef
                else:
                    const += m * coef
            if np.all(a == 0.0):
                continue          # constraint involves no free variable
            rows.append(a)
            rhs.append(1.0 - const)
    # triangle inequality  sum_l |c_l| >= |A| = 1
    rows.append(np.ones(3))
    rhs.append(1.0 - m)
    return np.array(rows), np.array(rhs)


_TRIPLE_CACHE: dict[int, np.ndarray] = {}


def _triples(nrow):
    if nrow not in _TRIPLE_CACHE:
        _TRIPLE_CACHE[nrow] = np.array(list(itertools.combinations(range(nrow), 3)))
    return _TRIPLE_CACHE[nrow]


def inner_min(m, psi, chi, r):
    """min over the polytope { w >= m, A w >= b } of sum_l w_l^r, by exact vertex
    enumeration (concave objective on a pointed polyhedron => minimum at a
    vertex).  Batched: all C(nrow,3) candidate vertices are solved at once."""
    A, b = constraints(m, psi, chi)
    A = np.vstack((A, np.eye(3)))
    b = np.concatenate((b, np.full(3, m)))
    nrow = A.shape[0]
    T = _triples(nrow)
    Ms = A[T]                       # (nT, 3, 3)
    bs = b[T]                       # (nT, 3)
    dets = np.linalg.det(Ms)
    ok = np.abs(dets) > 1e-9
    if not np.any(ok):
        return 3.0 * m ** r
    W = np.full((T.shape[0], 3), np.nan)
    W[ok] = np.linalg.solve(Ms[ok], bs[ok][:, :, None])[:, :, 0]
    feas = ok & np.all(W >= m - 1e-9, axis=1) & np.all(
        (W @ A.T) >= b[None, :] - 1e-9, axis=1)
    if not np.any(feas):
        return 3.0 * m ** r
    return float(np.min(np.sum(np.maximum(W[feas], 0.0) ** r, axis=1)))


def Psi2(m, r, starts=200, seed=0):
    """m^r + min over the 6 slot angles of the inner vertex minimum."""
    rng = np.random.default_rng(seed)

    def f(z):
        return inner_min(m, z[:3], z[3:], r)

    best = np.inf
    bx = None
    for _ in range(starts):
        z0 = rng.uniform(0, 2 * np.pi, 6)
        res = minimize(f, z0, method="Nelder-Mead",
                       options=dict(maxiter=1500, xatol=1e-8, fatol=1e-11))
        if res.fun < best:
            best, bx = float(res.fun), res.x.copy()
    return m ** r + best, bx


def verify(m_tol, r, trials=8000, seed=31):
    """Check (PS_{s,t}) and the whole Lemma R'' chain on real configurations."""
    rng = np.random.default_rng(seed)
    worst_ps = np.inf
    worst_chain = np.inf
    for _ in range(trials):
        al = rng.uniform(0, 2 * np.pi, 4)
        be = rng.uniform(0, 2 * np.pi, 4)
        a, b = points(al, be)
        if abs(np.linalg.det(bigM(a, b))) < 1e-4:
            continue
        c = atoms_cramer(a, b)
        xs = np.abs(c)
        for s in range(4):
            for t in range(4):
                tot = sum(xs[l] * abs(a[s] - a[l]) * abs(b[t] - b[l])
                          for l in range(4) if l not in (s, t))
                worst_ps = min(worst_ps, tot - 1.0)
        k = int(np.argmin(xs))
        m = xs[k]
        order = [i for i in range(4) if i != k]
        psi = np.angle(a[order] / a[k])
        chi = np.angle(b[order] / b[k])
        lhs = float(np.sum(xs ** r))
        worst_chain = min(worst_chain, lhs - (m ** r + inner_min(m, psi, chi, r)))
    return worst_ps, worst_chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, default=2.0 / 3.0)
    ap.add_argument("--starts", type=int, default=200)
    ap.add_argument("--lo", type=float, default=0.04)
    ap.add_argument("--hi", type=float, default=0.23)
    ap.add_argument("--npts", type=int, default=20)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    r = a.r
    tgt = 3.0 ** (1 - r)
    m_triv = (tgt / 4.0) ** (1.0 / r)
    print(f"r = {r:.10f}  target = {tgt:.12f}  m_triv = {m_triv:.9f}")

    if a.verify:
        wps, wch = verify(0.0, r)
        print(f"  verification on real configurations: min slack in (PS_st) = {wps:+.3e},"
              f"  min slack in the Lemma R'' chain = {wch:+.3e}  (both must be >= 0)")

    rows = []
    worst = (np.inf, None)
    for m in np.linspace(a.lo, a.hi, a.npts):
        v, z = Psi2(m, r, starts=a.starts, seed=int(1000 * m) + 1)
        v = max(v, 4.0 * m ** r)
        rows.append(dict(m=float(m), psi2=float(v), ratio=float(v / tgt)))
        flag = "OK " if v >= tgt - 1e-12 else "GAP"
        print(f"   m = {m:.6f}   Psi'' = {v:.9f}   ratio {v/tgt:.9f}   {flag}")
        if v / tgt < worst[0]:
            worst = (v / tgt, float(m))
    print(f"\n  worst ratio over the window: {worst[0]:.9f} at m = {worst[1]:.6f}")
    covered = all(row["ratio"] >= 1.0 - 1e-12 for row in rows)
    print(f"  window [{a.lo}, {a.hi}] fully covered: {covered}")
    with open(os.path.join(OUT, f"window_r{r:.4f}.json"), "w") as fh:
        json.dump(dict(r=r, target=tgt, m_triv=float(m_triv), rows=rows,
                       worst_ratio=float(worst[0]), worst_m=float(worst[1]),
                       covered=bool(covered)), fh, indent=1)
    print(f"wrote {os.path.join(OUT, f'window_r{r:.4f}.json')}")


if __name__ == "__main__":
    main()
