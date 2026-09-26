r"""P1A: LEMMA R and LEMMA R' -- drop the smallest atom and re-run the Key Lemma.

SETTING.  n = 2 (so nu = 3, r = q/3), N = 4 atoms.  (KL):
    sum_l c_l (u_1 - x_l)(u_2 - y_l) = A u_1 u_2   for all u,
with x_l, y_l on the unit circle ("slot phases") and |A| = 1.  Write
x_l := |c_l| where convenient; m := min_l |c_l| is the smallest atom.

-----------------------------------------------------------------------------
LEMMA R (proved; every n, N = n+2).
    sum_l |c_l|^r  >=  m^r + (n+1)^{1-r} (1 - 2^n m)_+^r .
Proof: relabel so |c_N| = m.  Subtract atom N from the identity: the first n+1
atoms satisfy the same identity with right-hand side A prod u_j - c_N prod_j
(u_j - v_Nj), whose modulus on the torus is at least 1 - 2^n m.  Then run the
Key Lemma verbatim (bijection identity -> geometric mean over the n! bijections
-> Vandermonde -> Hadamard), which only ever uses a LOWER bound on that
right-hand side.  Finally add |c_N|^r = m^r.

Plain reading: "the smallest atom is junk for the other three; junk of size m
costs at most 2^n m of clean amplitude."

-----------------------------------------------------------------------------
LEMMA R' (proved; n = 2, N = 4) -- the same argument keeping the CHORDS.
Let A_i := |x_i - x_4|, B_i := |y_i - y_4| (i = 1,2,3) be the slot chords from
the dropped atom to the other three, V_x := prod_{i<j<=3}|x_i - x_j|, likewise
V_y.  For each l_0 in {1,2,3} with {l_1,l_2} the other two, putting
(u_1,u_2) = (x_{l_1}, y_{l_2}) kills atoms l_1 and l_2 and leaves

    |c_{l_0}| |x_{l_1}-x_{l_0}| |y_{l_2}-y_{l_0}|  >=  1 - m A_{l_1} B_{l_2}.

Geometric mean over the two orderings, then the product over l_0 = 1,2,3
(which assembles exactly V_x V_y on the left), gives

    prod_{l<=3} |c_l|  >=  prod_{i != j} (1 - m A_i B_j)_+^{1/2} / (V_x V_y),

and AM-GM gives sum_{l<=3}|c_l|^r >= 3 (that product)^{r/3}.  The two Hadamard
facts  V_x <= 3^{3/2},  V_x * prod_i A_i <= 4^{4/2} = 16  (Fekete/Vandermonde
for 3 and for 4 points of the unit circle) cap the denominator.  Minimising the
right-hand side over all (A, B) in [0,2]^3 x [0,2]^3 subject to those caps gives
a bound Psi(m) that depends on m alone.

WHY THIS CAN GO FURTHER THAN THE PEELING LADDER.  Agent 3 (GLOBAL_OPT3, F3-2)
proved that the ladder -- which sees only the SORTED TAIL SUMS of |c| -- can
never reach 3^{1-r}.  Lemma R' is not a ladder bound: it uses the chord geometry
and Hadamard for the FOUR points, information the tail sums do not carry.

TRIVIAL COMPLEMENT.  Every atom is at least m, so sum_l |c_l|^r >= 4 m^r, which
already beats 3^{1-r} once m >= m_triv(r) := (3^{1-r}/4)^{1/r}
(= 0.216... at r = 2/3).  So Lemma R' only has to cover m in (0, m_triv).

Run:  uv run python p1a_lemmaR.py --r 0.6666666666666666
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

H3 = 3.0 ** 1.5   # Hadamard, 3 points on the unit circle
H4 = 16.0         # Hadamard, 4 points on the unit circle


# ---------------------------------------------------------------- Lemma R


def lemmaR(m, r, n=2):
    j = 2.0 ** n
    return m ** r + (n + 1) ** (1 - r) * max(0.0, 1 - j * m) ** r


def threshold(fun, r, n=2, hi=3.0, grid=400000):
    """largest t such that fun(m) >= (n+1)^{1-r} for every m in [0, t]."""
    tgt = (n + 1) ** (1 - r)
    ms = np.linspace(0.0, hi, grid)
    vals = np.array([fun(m, r) for m in ms])
    bad = np.where(vals < tgt - 1e-14)[0]
    if len(bad) == 0:
        return hi
    return float(ms[bad[0] - 1])


# ---------------------------------------------------------------- Lemma R'


def Rprime_core(m, A, B):
    """prod_{i!=j}(1 - m A_i B_j)_+^{1/2} / (Vx Vy) with Vx, Vy at their caps."""
    pA, pB = float(np.prod(A)), float(np.prod(B))
    Vx = min(H3, H4 / pA) if pA > 0 else H3
    Vy = min(H3, H4 / pB) if pB > 0 else H3
    num = 1.0
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            num *= max(0.0, 1.0 - m * A[i] * B[j]) ** 0.5
    return num / (Vx * Vy)


def _grid_triples(grid):
    g = np.linspace(0.0, 2.0, grid)
    return np.array(list(itertools.product(g, repeat=3)))


def _core_vec(m, TA, TB):
    """Rprime_core for every pair of rows of TA (A-triples) and TB (B-triples)."""
    pA = np.prod(TA, axis=1)
    pB = np.prod(TB, axis=1)
    Vx = np.minimum(H3, np.where(pA > 0, H4 / np.maximum(pA, 1e-300), H3))
    Vy = np.minimum(H3, np.where(pB > 0, H4 / np.maximum(pB, 1e-300), H3))
    # M[a, b, i, j] = (1 - m A_i B_j)_+   with the i == j entries set to 1
    M = 1.0 - m * TA[:, None, :, None] * TB[None, :, None, :]
    np.clip(M, 0.0, None, out=M)
    eye = np.eye(3, dtype=bool)
    M[:, :, eye] = 1.0
    num = np.prod(M, axis=(2, 3)) ** 0.5
    return num / (Vx[:, None] * Vy[None, :])


def Psi(m, r, starts=24, seed=0, grid=9):
    """Lemma R' bound: m^r + 3 * (min over admissible (A,B) of Rprime_core)^{r/3},
    and never worse than the trivial 4 m^r."""
    TA = _grid_triples(grid)
    vals = _core_vec(m, TA, TA)
    k = int(np.argmin(vals))
    ia, ib = divmod(k, TA.shape[0])
    best = float(vals[ia, ib])
    bestAB = (TA[ia].copy(), TA[ib].copy())
    rng = np.random.default_rng(seed)
    for s in range(starts):
        z0 = np.concatenate(bestAB) if s == 0 else rng.uniform(0, 2, 6)
        res = minimize(lambda z: Rprime_core(m, np.clip(z[:3], 0, 2), np.clip(z[3:], 0, 2)),
                       z0, method="Nelder-Mead",
                       options=dict(maxiter=3000, xatol=1e-10, fatol=1e-13))
        if res.fun < best:
            best, bestAB = float(res.fun), (np.clip(res.x[:3], 0, 2), np.clip(res.x[3:], 0, 2))
    return max(m ** r + 3.0 * best ** (r / 3.0), 4.0 * m ** r), bestAB


def Psi_fun(r, cache={}):
    def f(m, rr):
        key = (round(m, 9), rr)
        if key not in cache:
            cache[key] = Psi(m, rr, starts=12)[0]
        return cache[key]
    return f


# ------------------------------------------------- numerical verification


def verify_lemmas(r, trials=20000, seed=9):
    """Check Lemma R and Lemma R' against genuine 4-atom configurations."""
    rng = np.random.default_rng(seed)
    worstR = np.inf
    worstRp = np.inf
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
        lhs = float(np.sum(xs ** r))
        worstR = min(worstR, lhs - lemmaR(m, r))
        idx = [i for i in range(4) if i != k]
        A = np.abs(a[idx] - a[k])
        B = np.abs(b[idx] - b[k])
        Vx = float(np.prod([abs(a[i] - a[j]) for i, j in itertools.combinations(idx, 2)]))
        Vy = float(np.prod([abs(b[i] - b[j]) for i, j in itertools.combinations(idx, 2)]))
        num = 1.0
        for i in range(3):
            for j in range(3):
                if i != j:
                    num *= max(0.0, 1.0 - m * A[i] * B[j]) ** 0.5
        if Vx * Vy > 0:
            prod_bound = num / (Vx * Vy)
            worstRp = min(worstRp, float(np.prod(xs[idx])) - prod_bound)
    return worstR, worstRp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, nargs="*", default=[2.0 / 3.0])
    ap.add_argument("--grid", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rec = {}

    for r in a.r:
        tgt = 3.0 ** (1 - r)
        m_triv = (tgt / 4.0) ** (1.0 / r)
        tR = threshold(lemmaR, r)
        print(f"\nr = {r:.10f}   target 3^(1-r) = {tgt:.12f}")
        print(f"  trivial cover  m >= m_triv = {m_triv:.9f}   (4 m^r >= target)")
        print(f"  Lemma R  threshold t*  = {tR:.9f}"
              f"   -> covers m in [0, {tR:.6f}]")
        ms = np.linspace(0.0, min(0.30, m_triv * 1.25), 31)
        rows = []
        worst = (np.inf, None)
        for m in ms:
            v, AB = Psi(m, r, starts=10, grid=a.grid)
            rows.append(dict(m=float(m), psi=float(v), ratio=float(v / tgt)))
            if v / tgt < worst[0]:
                worst = (v / tgt, float(m), AB)
        covered = all(row["psi"] >= tgt - 1e-12 for row in rows)
        print(f"  Lemma R' bound Psi(m) on [0, {ms[-1]:.4f}]:"
              f"  min ratio {worst[0]:.9f} at m = {worst[1]:.6f}")
        for row in rows[::3]:
            print(f"     m = {row['m']:.5f}   Psi = {row['psi']:.9f}"
                  f"   ratio {row['ratio']:.9f}   (Lemma R {lemmaR(row['m'], r)/tgt:.6f})")
        print(f"  ==> Lemma R' covers the whole gap (0, m_triv): "
              f"{'YES -- S1 at n=2,N=4 PROVED' if covered and ms[-1] >= m_triv else 'NO'}")
        rec[f"r{r:.6f}"] = dict(target=tgt, m_triv=float(m_triv), lemmaR_threshold=float(tR),
                                rows=rows, worst_ratio=float(worst[0]),
                                worst_m=float(worst[1]), covered=bool(covered))

        wR, wRp = verify_lemmas(r)
        print(f"  numerical check on real configurations: "
              f"min(LHS - LemmaR) = {wR:+.3e}, min(prod - R' bound) = {wRp:+.3e}"
              f"   (both must be >= 0)")
        rec[f"r{r:.6f}"]["verify"] = dict(lemmaR_slack=float(wR), lemmaRp_slack=float(wRp))

    with open(os.path.join(OUT, "lemmaR.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"\nwrote {os.path.join(OUT, 'lemmaR.json')}")


if __name__ == "__main__":
    main()
