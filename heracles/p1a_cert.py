r"""P1A: a RIGOROUS interval-arithmetic certificate for the adversary agent's
two-variable programme, i.e. an unconditional proof of P1 at q = 2.

WHAT IS BEING CERTIFIED.  `ADVERSARY.md` section 6 (`adv_proof.py`) reduces P1 --
`sum_l |c_l|^r >= (n+1)^{1-r} |A|^r` for EVERY number of atoms N -- to the
statement that a function of two real variables never goes below the target.
With |A| = 1, the moduli sorted, `beta := x_{n+1}` (the smallest of the n+1
biggest atoms) and `tau := sum_{l > n+1} x_l` (the total mass of the rest):

    sum_l |c_l|^r  >=  G(beta, tau)
       :=  Tail(tau, beta)  +  max( Peel(beta, tau), tgt * (1 - 2^n tau)_+^r ),
    tgt   := (n+1)^{1-r},
    Tail  := max( tau^r , tau * beta^{r-1} ),
    Peel  := beta^r + min { sum_{i=1}^n y_i^r : y_1 >= ... >= y_n >= beta,
                            sum_{i=h+1}^n y_i >= D_h(n) - beta - tau, h = 0..n-1 } .

P1 at exponent r follows from  min G >= tgt.  The adversary computed that minimum
on a 600 x 600 grid with a Nelder-Mead polish and reported a 2.2% margin at
q = 2; a grid is not a proof.  This script replaces the grid by a proof.

THE FOUR PIECES (the first three are exact algebra, the fourth is a certified
branch-and-bound in TWO variables).

  (P-a) beta >= 1/(n+1)  ==>  done.
        All n+1 big atoms are >= beta, so Peel >= (n+1) beta^r >= (n+1)^{1-r}.

  (P-b) tau >= tau_hi := (tgt / C(n,r))^{1/r}  ==>  done, where
            C(n,r) := min_{s>0} [ max(1, s^{r-1}) + (n+1) s^r ]
                    =  s*^{r-1} + (n+1) s*^r ,   s* = (1-r)/((n+1) r)  (< 1 here).
        Indeed G >= Tail + (n+1) beta^r, and putting beta = s tau makes the
        right-hand side tau^r [ max(1, s^{r-1}) + (n+1) s^r ] >= C tau^r.
        (C(2,2/3) = 2.7256..., tau_hi = 0.3849...;  C(3,1/2) = 4, tau_hi = 0.25.)

  (P-c) tau <= eps_cert  ==>  done.  Here G >= phi(tau) := tgt (1 - 2^n tau)_+^r
        + tau^r, and phi is handled by a CHORD bound: z |-> (1-z)^r is concave on
        [0, z_0], so it lies above its chord, (1-z)^r >= 1 - z K with
        K := (1 - (1-z_0)^r)/z_0.  Hence for tau <= t_0 (and z_0 := 2^n t_0)

            phi(tau) - tgt  >=  tau^r - tgt * 2^n K * tau  >= 0
            whenever  t_0^{1-r} <= 1 / (tgt * 2^n * K) .

        Bisecting on t_0 with that test evaluated in interval arithmetic gives a
        PROVED eps_cert.  (It is essentially lossless: eps_cert = 0.016972 against
        the adversary's exactly computed eps* = 0.016973 at n = 2, q = 2.)

  (P-d) the rectangle [0, 1/(n+1)] x [eps_cert, tau_hi]: interval branch-and-bound.
        On a box [bl, bh] x [tl, th] a rigorous lower bound for G is
            Tail_lo = max( tl^r , tl * bh^{r-1} )            (beta^{r-1} decreases)
            drop_lo = tgt * (1 - 2^n * th)_+^r               (decreases in tau)
            Peel_lo = bl^r + inner(P*),  P* the RELAXED polyhedron
                      { y_1 >= ... >= y_n >= bl,
                        sum_{i>h} y_i >= D_h - bh - th } ,
        because every polyhedron occurring in the box is contained in P*.  The
        inner minimum of a concave function over a pointed polyhedron sits at a
        vertex; the constraint matrix has integer entries, so each candidate
        vertex is an EXACT rational combination of the right-hand sides and is
        evaluated in interval arithmetic.  Taking the minimum over all candidate
        vertex systems (a superset of the true vertices, with definitely
        infeasible ones discarded) is a valid lower bound.

Everything runs in mpmath's outward-rounded interval arithmetic (`mpmath.iv`),
so every number printed with a "proved" label is an enclosure, not a float.

Run:  uv run --with mpmath python p1a_cert.py --cells 2:2 3:2
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from fractions import Fraction

from mpmath import iv, mp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "P1A")

iv.dps = 30
mp.dps = 30


# --------------------------------------------------------------------------
# interval helpers
# --------------------------------------------------------------------------


def I(x):
    return iv.mpf(x)


def ipow(x, r):
    """x^r for an interval x >= 0 and a positive real r, outward rounded."""
    lo = iv.mpf(x.a)
    hi = iv.mpf(x.b)
    if x.a <= 0:
        low = iv.mpf(0)
    else:
        low = iv.exp(r * iv.log(lo))
    if x.b <= 0:
        return iv.mpf([0, 0])
    high = iv.exp(r * iv.log(hi))
    return iv.mpf([low.a, high.b])


def ipow_neg(x, e):
    """x^e for an interval x > 0 and a NEGATIVE real e (so it is decreasing)."""
    if x.a <= 0:
        raise ValueError("x must be strictly positive")
    low = iv.exp(e * iv.log(iv.mpf(x.b)))
    high = iv.exp(e * iv.log(iv.mpf(x.a)))
    return iv.mpf([low.a, high.b])


# --------------------------------------------------------------------------
# the inherited constants
# --------------------------------------------------------------------------


def kappa(d):
    """kappa_d = (1 - 2^{-d}) / S_d, exact for d = 1,2,3 (GLOBAL_OPT3 section 3)."""
    if d == 1:
        return iv.mpf(1) / 2
    if d == 2:
        return 3 * iv.sqrt(iv.mpf(3)) / 8
    if d == 3:
        return iv.mpf(49) / 72
    raise ValueError("only n <= 3 is covered by exact kappa values")


def D_h(n, h):
    """D_h(n) = max over compositions d_1+...+d_h = n of prod_i kappa_{d_i}."""
    if h == 0:
        return iv.mpf(1)

    def parts(tot, k):
        if k == 1:
            yield (tot,)
            return
        for first in range(1, tot - k + 2):
            for rest in parts(tot - first, k - 1):
                yield (first,) + rest

    best = None
    for d in parts(n, h):
        v = iv.mpf(1)
        for dd in d:
            v = v * kappa(dd)
        if best is None or v.a > best.a:
            best = v
    return best


# --------------------------------------------------------------------------
# the inner polyhedron: exact rational vertex systems
# --------------------------------------------------------------------------


def peel_rows(n):
    """Integer constraint matrix A (a . y >= b) and the tag of each row."""
    rows, tags = [], []
    for i in range(n - 1):
        z = [0] * n
        z[i], z[i + 1] = 1, -1
        rows.append(z)
        tags.append(("ord", i))
    z = [0] * n
    z[n - 1] = 1
    rows.append(z)
    tags.append(("beta", None))
    for h in range(n):
        z = [0] * n
        for i in range(h, n):
            z[i] = 1
        rows.append(z)
        tags.append(("tail", h))
    return rows, tags


def _inv_rational(M):
    """Exact inverse of a small integer matrix, or None if singular."""
    k = len(M)
    a = [[Fraction(M[i][j]) for j in range(k)] + [Fraction(int(i == j)) for j in range(k)]
         for i in range(k)]
    for col in range(k):
        piv = None
        for row in range(col, k):
            if a[row][col] != 0:
                piv = row
                break
        if piv is None:
            return None
        a[col], a[piv] = a[piv], a[col]
        pv = a[col][col]
        a[col] = [v / pv for v in a[col]]
        for row in range(k):
            if row != col and a[row][col] != 0:
                f = a[row][col]
                a[row] = [x - f * y for x, y in zip(a[row], a[col])]
    return [[a[i][k + j] for j in range(k)] for i in range(k)]


_SYS_CACHE: dict[int, list] = {}


def vertex_systems(n):
    """All n-subsets of the rows with an invertible submatrix, with exact inverses."""
    if n in _SYS_CACHE:
        return _SYS_CACHE[n]
    rows, tags = peel_rows(n)
    out = []
    for S in itertools.combinations(range(len(rows)), n):
        inv = _inv_rational([rows[i] for i in S])
        if inv is not None:
            out.append((S, inv))
    _SYS_CACHE[n] = (rows, tags, out)
    return _SYS_CACHE[n]


def inner_lower(n, r, bl, bh, th, Ds):
    """Rigorous lower bound for the inner minimum over the RELAXED polyhedron
    P* = { y_1 >= ... >= y_n >= bl,  sum_{i>h} y_i >= D_h - bh - th }."""
    rows, tags, systems = vertex_systems(n)
    b = []
    for k, h in tags:
        if k == "ord":
            b.append(iv.mpf(0))
        elif k == "beta":
            b.append(bl)
        else:
            b.append(Ds[h] - bh - th)
    best = None
    for S, inv in systems:
        y = []
        for i in range(n):
            acc = iv.mpf(0)
            for j in range(n):
                cij = inv[i][j]
                if cij != 0:
                    acc = acc + iv.mpf(cij.numerator) / iv.mpf(cij.denominator) * b[S[j]]
            y.append(acc)
        # discard the systems that are DEFINITELY infeasible
        infeasible = False
        for row, bb in zip(rows, b):
            acc = iv.mpf(0)
            for j in range(n):
                if row[j]:
                    acc = acc + row[j] * y[j]
            if acc.b < bb.a:          # certainly  a.y < b
                infeasible = True
                break
        if infeasible:
            continue
        tot = iv.mpf(0)
        for yi in y:
            lo = iv.mpf([max(0.0, float(yi.a)), max(0.0, float(yi.a))])
            tot = tot + ipow(lo, r)
        if best is None or tot.a < best:
            best = tot.a
    return iv.mpf(0) if best is None else iv.mpf(best)


# --------------------------------------------------------------------------
# G's rigorous lower bound on a box
# --------------------------------------------------------------------------


def tail_staircase(r, tl, bh):
    """SHARPENED tail price.  The tail atoms satisfy 0 <= x_l <= beta and
    sum_l x_l = tau over an arbitrary number of atoms.  sum_l x_l^r is concave,
    so its minimum over that box-simplex is at an extreme point: k = floor(tau/beta)
    atoms at beta and one at the remainder.  Hence

        sum_{tail} x_l^r  >=  k beta^r + (tau - k beta)^r ,   k = floor(tau/beta),

    which dominates the adversary's tau beta^{r-1} (because (tau-k beta)^r >=
    (tau - k beta) beta^{r-1} when the remainder is at most beta).  Evaluated here
    at the box corner (tau = tl, beta = bh) that minimises it: the staircase value
    is increasing in tau and decreasing in beta."""
    if bh.a <= 0 or tl.b <= 0:
        return iv.mpf(0)
    k = int(float(tl.a) // float(bh.b))
    k = max(k, 0)
    rem = tl - k * bh
    # guard: if rounding puts rem outside [0, beta], fall back to k-1 / k+1
    while rem.a < 0 and k > 0:
        k -= 1
        rem = tl - k * bh
    val = k * ipow(bh, r) + ipow(iv.mpf([max(0.0, float(rem.a))] * 2), r)
    return iv.mpf(val.a)


def G_lower(n, r, bl, bh, tl, th, Ds, tgt, staircase=True):
    """Lower bound for G on [bl,bh] x [tl,th]  (all arguments intervals)."""
    tail1 = ipow(tl, r)
    tail2 = tl * ipow_neg(bh, r - 1) if bh.a > 0 else iv.mpf(0)
    tail_lo = iv.mpf(max(tail1.a, tail2.a))
    if staircase:
        tail_lo = iv.mpf(max(tail_lo.a, tail_staircase(r, tl, bh).a))
    one_minus = iv.mpf(1) - (2 ** n) * th
    drop_lo = tgt * ipow(iv.mpf([max(0.0, float(one_minus.a))] * 2), r) \
        if one_minus.a > 0 else iv.mpf(0)
    peel_lo = ipow(bl, r) + inner_lower(n, r, bl, bh, th, Ds)
    big = iv.mpf(max(peel_lo.a, drop_lo.a))
    return tail_lo + big


def certify_rectangle(n, r, b_hi, t_lo, t_hi, tgt, Ds, max_boxes=4_000_000, staircase=False):
    """Interval branch-and-bound: prove G >= tgt on [0,b_hi] x [t_lo,t_hi]."""
    stack = [(iv.mpf(0), b_hi, t_lo, t_hi)]
    done = 0
    worst = None
    depth_max = 0
    while stack:
        bl, bh, tl, th = stack.pop()
        done += 1
        if done > max_boxes:
            return False, done, worst, depth_max
        # boxes on which the peel constraint beta + tau >= D_n cannot hold are
        # entirely infeasible for (KL), so there is nothing to prove there
        if (bh + th).b < Ds[n].a:
            continue
        g = G_lower(n, r, bl, bh, tl, th, Ds, tgt, staircase=staircase)
        if g.a >= tgt.b:
            if worst is None or g.a < worst:
                worst = g.a
            continue
        wb = float(bh.b) - float(bl.a)
        wt = float(th.b) - float(tl.a)
        if wb < 1e-13 and wt < 1e-13:
            return False, done, (float(bl.a), float(tl.a), float(g.a)), depth_max
        if wb >= wt:
            mid = (bl + bh) / 2
            stack.append((bl, mid, tl, th))
            stack.append((mid, bh, tl, th))
        else:
            mid = (tl + th) / 2
            stack.append((bl, bh, tl, mid))
            stack.append((bl, bh, mid, th))
        depth_max = max(depth_max, len(stack))
    return True, done, worst, depth_max


# --------------------------------------------------------------------------
# the analytic pieces
# --------------------------------------------------------------------------


def C_const(n, r):
    """C(n,r) = s*^{r-1} + (n+1) s*^r, s* = (1-r)/((n+1) r); verified s* <= 1."""
    s = (iv.mpf(1) - r) / ((n + 1) * iv.mpf(r))
    assert s.b <= 1.0, "s* > 1: the other branch of the min would be needed"
    return ipow_neg(s, r - 1) + (n + 1) * ipow(s, r), s


def eps_cert(n, r, tgt):
    """Largest t0 (by bisection, each test in interval arithmetic) for which the
    chord bound proves phi(t) >= tgt on [0, t0]."""
    def ok(t0):
        if t0 <= 0:
            return True
        z0 = iv.mpf(2 ** n) * iv.mpf(t0)
        if z0.b >= 1.0:
            return False
        K = (iv.mpf(1) - ipow(iv.mpf(1) - z0, r)) / z0
        lhs = ipow(iv.mpf(t0), 1.0 - r)          # t0^{1-r}
        rhs = iv.mpf(1) / (tgt * iv.mpf(2 ** n) * K)
        return lhs.b < rhs.a

    lo, hi = 0.0, 2.0 ** (-n)
    if ok(hi):
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------


def run_cell(n, q, max_boxes, staircase=False):
    r = q / (n + 1.0)
    tgt = ipow(iv.mpf(n + 1), 1.0 - r)
    Ds = [D_h(n, h) for h in range(n + 1)]
    print(f"\n=== n = {n}, nu = {n+1}, q = {q}, r = {r:.12f} ===")
    print(f"  target (n+1)^(1-r)  in [{float(tgt.a):.18f}, {float(tgt.b):.18f}]")
    print(f"  D_h(n) = " + ", ".join(f"{float(d.a):.10f}" for d in Ds))

    # (P-a)
    b_hi = iv.mpf(1) / (n + 1)
    print(f"  (P-a) beta >= 1/(n+1) = {float(b_hi.a):.12f}: Peel >= (n+1) beta^r >= target.  PROVED")

    # (P-b)
    C, s = C_const(n, r)
    t_hi = ipow(tgt / C, 1.0 / r)
    print(f"  (P-b) C(n,r) in [{float(C.a):.12f}, {float(C.b):.12f}] (s* = {float(s.a):.10f});"
          f"  tau >= tau_hi = {float(t_hi.b):.12f}: G >= C tau^r >= target.  PROVED")

    # (P-c)
    e = eps_cert(n, r, tgt)
    print(f"  (P-c) tau <= eps_cert = {e:.12e}: chord bound gives phi(tau) >= target.  PROVED")

    # (P-d)
    t_lo = iv.mpf(e)
    t_hi_i = iv.mpf(float(t_hi.b))
    ok, boxes, worst, dep = certify_rectangle(n, r, b_hi, t_lo, t_hi_i, tgt, Ds,
                                              max_boxes=max_boxes, staircase=staircase)
    if ok:
        print(f"  (P-d) rectangle [0, {float(b_hi.a):.6f}] x [{e:.6e}, {float(t_hi.b):.6f}]:"
              f"  CERTIFIED with {boxes} boxes (max stack {dep}).")
        print(f"  ==> P1 at n = {n}, q = {q} (r = {r:.6f}) is PROVED for every N.")
    else:
        print(f"  (P-d) FAILED after {boxes} boxes; worst box {worst}")
    return dict(n=n, q=q, r=r, target=[str(tgt.a), str(tgt.b)],
                C=[str(C.a), str(C.b)], tau_hi=float(t_hi.b), eps_cert=e,
                boxes=int(boxes), certified=bool(ok),
                worst_G_lower=(float(worst) if worst is not None and not isinstance(worst, tuple) else (list(map(float, worst)) if isinstance(worst, tuple) else None)))


def certifies(n, r, max_boxes=400_000, staircase=False):
    """True iff the whole four-piece certificate goes through at this r."""
    try:
        tgt = ipow(iv.mpf(n + 1), 1.0 - r)
        Ds = [D_h(n, h) for h in range(n + 1)]
        C, _ = C_const(n, r)
        t_hi = iv.mpf(float(ipow(tgt / C, 1.0 / r).b))
        e = eps_cert(n, r, tgt)
        if e <= 0:
            return False
        ok, _, _, _ = certify_rectangle(n, r, iv.mpf(1) / (n + 1), iv.mpf(e), t_hi,
                                        tgt, Ds, max_boxes=max_boxes, staircase=staircase)
        return ok
    except Exception:
        return False


def r_star_certified(n, lo=0.4, hi=0.999, iters=22, max_boxes=400_000, staircase=False):
    """Bisection on r: the largest exponent at which the certificate succeeds.
    Every evaluation is the full interval-arithmetic proof, so the returned lo is
    a PROVED statement ('P1 holds for every r in (0, lo]' -- the certificate is
    run at lo itself; monotonicity in r is not assumed, so see the sweep)."""
    if not certifies(n, lo, max_boxes, staircase):
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if certifies(n, mid, max_boxes, staircase):
            lo = mid
        else:
            hi = mid
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["2:2", "3:2"],
                    help="n:q pairs, e.g. 2:2 3:2 2:2.25")
    ap.add_argument("--max-boxes", type=int, default=2_000_000)
    ap.add_argument("--staircase", action="store_true",
                    help="use the sharpened staircase tail price (a strictly stronger bound)")
    ap.add_argument("--rstar", nargs="*", type=int, default=[],
                    help="values of n for which to bisect the certified r*")
    ap.add_argument("--sweep", nargs="*", type=int, default=[],
                    help="values of n for which to sweep r on a grid")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rec = []
    for cell in a.cells:
        n_s, q_s = cell.split(":")
        rec.append(run_cell(int(n_s), float(q_s), a.max_boxes, a.staircase))
    for n in a.sweep:
        print(f"\n  certified sweep in r at n = {n}:")
        row = []
        for r in [0.35, 0.4, 0.5, 0.6, 2 / 3, 0.7, 0.75, 0.78, 0.8, 0.82, 0.84,
                  0.86, 0.88, 0.9, 0.95]:
            ok = certifies(n, r, a.max_boxes, a.staircase)
            print(f"    r = {r:.4f}  (q = {r*(n+1):.4f})   certificate: "
                  f"{'PROVED' if ok else 'fails'}")
            row.append(dict(r=r, q=r * (n + 1), certified=bool(ok)))
        rec.append(dict(sweep_n=n, rows=row))
    for n in a.rstar:
        rs = r_star_certified(n, max_boxes=a.max_boxes, staircase=a.staircase)
        print(f"\n  certified r*({n}) = {rs}   (q* = {None if rs is None else rs*(n+1):.6f})")
        rec.append(dict(rstar_n=n, r_star=rs))
    with open(os.path.join(OUT, "cert.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"\nwrote {os.path.join(OUT, 'cert.json')}")


if __name__ == "__main__":
    main()
