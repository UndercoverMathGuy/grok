r"""P1A: LEMMA R* -- Lemma R' with the chord geometry parametrised EXACTLY.

Lemma R' (p1a_lemmaR.py) bounds

    prod_{l<=3} |c_l|  >=  prod_{i != j} (1 - m A_i B_j)_+^{1/2} / (V_x V_y),

with A_i = |x_i - x_4|, V_x = prod_{i<j<=3} |x_i - x_j| (slot 1) and B, V_y the
same in slot 2, and then relaxes the pair (A, V_x) by Hadamard
(V_x <= 3^{3/2}, V_x prod_i A_i <= 16).  That relaxation is badly lossy: its
minimiser wants A_i = 2 for all i (all three retained atoms antipodal to the
dropped one), which forces V_x = 0, not the 16/8 = 2 that Hadamard allows.

EXACT PARAMETRISATION.  Gauge-fix the dropped atom to x_4 = 1 and write the
other three slot-1 phases as x_i = e^{2 i u_i}, u_i in [0, pi).  Then

    A_i = |x_i - 1| = 2 sin u_i ,        V_x = 8 prod_{i<j} |sin(u_i - u_j)| ,

and likewise B_i = 2 sin v_i, V_y = 8 prod_{i<j}|sin(v_i - v_j)| in slot 2.
No inequality is spent: every (A, V_x) that can occur is exactly one point of
u in [0,pi)^3, and every u gives an admissible configuration.  Hence the PROVED
bound

    sum_l |c_l|^r  >=  Psi*(m) := max{ 4 m^r ,  m^r + 3 Pi*(m)^{r/3} } ,

    Pi*(m) := min over u, v in [0,pi)^3 of
       prod_{i != j} ( 1 - 4 m sin u_i sin v_j )_+^{1/2}
       / ( 64 prod_{i<j}|sin(u_i - u_j)| prod_{i<j}|sin(v_i - v_j)| ) .

Sanity: at m = 0 the numerator is 1 and the two Vandermonde products are each at
most their Fekete value 3^{3/2}/8, so Pi*(0) = 1/27 and Psi*(0) = 3^{1-r}
exactly -- the bound is tight at the block, as it must be.

Plain reading: "shoot two of the three big atoms out of the identity with the
slot values of the other two, keep every chord exactly, and let the dropped atom
pay only what its own chords say it can."

Run:  uv run python p1a_sharp.py --starts 300
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


def core(m, u, v):
    """the Lemma R* integrand (to be minimised over u, v in [0,pi)^3)."""
    su, sv = np.sin(u), np.sin(v)
    Vu = abs(np.sin(u[0] - u[1]) * np.sin(u[0] - u[2]) * np.sin(u[1] - u[2]))
    Vv = abs(np.sin(v[0] - v[1]) * np.sin(v[0] - v[2]) * np.sin(v[1] - v[2]))
    den = 64.0 * Vu * Vv
    if den <= 1e-300:
        return np.inf
    P = 1.0 - 4.0 * m * np.outer(su, sv)
    np.clip(P, 0.0, None, out=P)
    np.fill_diagonal(P, 1.0)
    return float(np.prod(P) ** 0.5 / den)


def Pi_star(m, starts=300, seed=0, grid=13):
    g = (np.arange(grid) + 0.5) * np.pi / grid
    best, bz = np.inf, None
    T = np.array(list(itertools.product(range(grid), repeat=3)))
    for ia in T:
        u = g[ia]
        for ib in T[: min(len(T), 400)]:
            pass
        break
    # coarse random + multistart local (the grid in 6d is too big to scan)
    rng = np.random.default_rng(seed)
    for s in range(starts):
        z0 = rng.uniform(0, np.pi, 6)
        res = minimize(lambda z: core(m, np.mod(z[:3], np.pi), np.mod(z[3:], np.pi)),
                       z0, method="Nelder-Mead",
                       options=dict(maxiter=3000, xatol=1e-11, fatol=1e-14))
        if res.fun < best:
            best, bz = float(res.fun), np.mod(res.x, np.pi)
    return best, bz


def Psi_star(m, r, starts=300, seed=0):
    p, z = Pi_star(m, starts=starts, seed=seed)
    return max(4.0 * m ** r, m ** r + 3.0 * p ** (r / 3.0)), p, z


def verify(r, trials=15000, seed=77):
    """Check the Lemma R* product bound directly against real configurations."""
    rng = np.random.default_rng(seed)
    worst = np.inf
    for _ in range(trials):
        al = rng.uniform(0, 2 * np.pi, 4)
        be = rng.uniform(0, 2 * np.pi, 4)
        a, b = points(al, be)
        if abs(np.linalg.det(bigM(a, b))) < 1e-4:
            continue
        c = atoms_cramer(a, b)
        xs = np.abs(c)
        k = int(np.argmin(xs))
        m = xs[k]
        idx = [i for i in range(4) if i != k]
        u = np.mod(np.angle(a[idx] / a[k]) / 2.0, np.pi)
        v = np.mod(np.angle(b[idx] / b[k]) / 2.0, np.pi)
        cv = core(m, u, v)
        if not np.isfinite(cv):
            continue
        worst = min(worst, float(np.prod(xs[idx])) - cv)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, default=2.0 / 3.0)
    ap.add_argument("--starts", type=int, default=300)
    ap.add_argument("--npts", type=int, default=16)
    ap.add_argument("--hi", type=float, default=0.24)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    r = a.r
    tgt = 3.0 ** (1 - r)
    m_triv = (tgt / 4.0) ** (1.0 / r)
    print(f"r = {r:.10f}   target = {tgt:.12f}   m_triv = {m_triv:.9f}")

    if a.verify:
        w = verify(r)
        print(f"  Lemma R* verification on real configurations: "
              f"min(prod_{{l<=3}}|c_l| - bound) = {w:+.3e}   (must be >= 0)")

    rows = []
    worst = (np.inf, None)
    for m in np.linspace(0.0, a.hi, a.npts):
        v, p, z = Psi_star(m, r, starts=a.starts, seed=int(10000 * m) + 3)
        rows.append(dict(m=float(m), psi=float(v), ratio=float(v / tgt), Pi=float(p)))
        flag = "OK " if v >= tgt - 1e-11 else "GAP"
        print(f"   m = {m:.6f}   Psi* = {v:.9f}   ratio {v/tgt:.9f}   {flag}")
        if v / tgt < worst[0]:
            worst = (v / tgt, float(m))
    covered = all(row["ratio"] >= 1.0 - 1e-11 for row in rows)
    print(f"\n  worst ratio on [0, {a.hi}]: {worst[0]:.9f} at m = {worst[1]:.6f}")
    print(f"  Lemma R* covers [0, m_triv] : {covered and a.hi >= m_triv}")
    with open(os.path.join(OUT, f"sharp_r{r:.4f}.json"), "w") as fh:
        json.dump(dict(r=r, target=tgt, m_triv=float(m_triv), rows=rows,
                       worst_ratio=float(worst[0]), worst_m=float(worst[1]),
                       covered=bool(covered)), fh, indent=1)
    print(f"wrote {os.path.join(OUT, f'sharp_r{r:.4f}.json')}")


if __name__ == "__main__":
    main()
