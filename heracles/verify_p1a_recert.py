r"""VERIFY / P1A -- an INDEPENDENT rigorous re-certification of the two-variable
programme G(beta, tau) >= (n+1)^{1-r}.

Written from the formulas in P1A.md section 2 / ADVERSARY.md section 6, NOT by calling
p1a_cert.py.  Differences from p1a_cert.py, all in the direction of more rigour:

  * r is carried as an INTERVAL enclosing the exact rational q/(n+1) (p1a_cert uses the
    float q/(n+1.0) and never says so), and every power is exp(R * log(x)) with R an
    interval, so a successful run proves the statement for EVERY r in that enclosure.
  * no float() is ever applied to an interval endpoint (p1a_cert does that in four
    places: inner_lower, tail_staircase, G_lower's drop term, run_cell's tau_hi --
    float() rounds to NEAREST and can therefore round a lower bound UP).
  * the large-tau region is closed by the trivial  G >= tau^r >= tgt  for
    tau >= tgt^{1/r} instead of by the constant C(n,r), removing one calculus step;
    C(n,r) is separately re-certified by a 1-D interval branch-and-bound.
  * the whole quadrant [0,inf) x [0,inf) is covered explicitly, and the coverage of the
    four regions is checked arithmetically at the junctions.

Regions (beta = x_{n+1}, tau = sum_{l>n+1} x_l, |A| = 1, moduli sorted decreasing):

  (A) beta >= 1/(n+1)          : Peel >= (n+1) beta^r >= (n+1)^{1-r} = tgt.
  (B) tau  >= tau_max          : Tail >= tau^r >= tgt,  tau_max := tgt^{1/r}.
  (C) tau  <= eps              : chord bound on phi(t) = tgt (1-2^n t)_+^r + t^r.
  (D) the residual rectangle [0, 1/(n+1)] x [eps, tau_max] : interval branch and bound.

Run:  uv run --with mpmath python verify_p1a_recert.py
      uv run --with mpmath python verify_p1a_recert.py --sweep
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from fractions import Fraction

from mpmath import iv, mp

iv.dps = 40
mp.dps = 40

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_RUNS", "VERIFY")


# ---------------------------------------------------------------- interval helpers


def EX(x):
    """exact degenerate interval from an mpf endpoint (verified exact: iv.mpf(mpf) keeps it)."""
    return iv.mpf([x, x])


def ipow(x, R):
    """enclosure of { xx^rr : xx in x, rr in R },  x >= 0, R any real interval.
    For x.a <= 0 the lower end is taken as 0 (valid because R > 0 wherever used with
    a possibly-zero base)."""
    if x.b <= 0:
        return iv.mpf([0, 0])
    hi = iv.exp(R * iv.log(EX(x.b)))
    if x.a <= 0:
        return iv.mpf([0, hi.b])
    lo = iv.exp(R * iv.log(EX(x.a)))
    return iv.mpf([min(lo.a, hi.a), max(lo.b, hi.b)])


def ipow_pos(x, R):
    """same, but x.a must be > 0 (used for negative exponents too)."""
    if x.a <= 0:
        raise ValueError("base must be strictly positive")
    lo = iv.exp(R * iv.log(EX(x.a)))
    hi = iv.exp(R * iv.log(EX(x.b)))
    return iv.mpf([min(lo.a, hi.a), max(lo.b, hi.b)])


# ---------------------------------------------------------------- inherited constants


def kappa(d):
    """GLOBAL_OPT3 Theorem E peeling constants (INHERITED, not re-derived here)."""
    if d == 1:
        return iv.mpf(1) / 2
    if d == 2:
        return 3 * iv.sqrt(iv.mpf(3)) / 8
    if d == 3:
        return iv.mpf(49) / 72
    raise ValueError("only n <= 3")


def D_h(n, h, le=False):
    """D_h(n) = max prod kappa_{d_i} over h positive parts with sum = n  (le=False)
    or sum <= n (le=True, P1B's correction).  Returned as an interval."""
    if h == 0:
        return iv.mpf(1)

    def parts(tot, k):
        if k == 1:
            yield (tot,)
            return
        for first in range(1, tot - k + 2):
            for rest in parts(tot - first, k - 1):
                yield (first,) + rest

    totals = range(h, n + 1) if le else [n]
    best = None
    for tot in totals:
        for d in parts(tot, h):
            v = iv.mpf(1)
            for dd in d:
                v = v * kappa(dd)
            if best is None or v.a > best.a:
                best = v
    return best


# ---------------------------------------------------------------- inner polyhedron


def peel_rows(n):
    rows, tags = [], []
    for i in range(n - 1):
        z = [0] * n
        z[i], z[i + 1] = 1, -1
        rows.append(z); tags.append(("ord", i))
    z = [0] * n; z[n - 1] = 1
    rows.append(z); tags.append(("beta", None))
    for h in range(n):
        z = [0] * n
        for i in range(h, n):
            z[i] = 1
        rows.append(z); tags.append(("tail", h))
    return rows, tags


def _inv(M):
    k = len(M)
    a = [[Fraction(M[i][j]) for j in range(k)] + [Fraction(int(i == j)) for j in range(k)]
         for i in range(k)]
    for col in range(k):
        piv = next((row for row in range(col, k) if a[row][col] != 0), None)
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


_CACHE: dict = {}


def systems(n):
    if n not in _CACHE:
        rows, tags = peel_rows(n)
        out = [(S, _inv([rows[i] for i in S]))
               for S in itertools.combinations(range(len(rows)), n)
               if _inv([rows[i] for i in S]) is not None]
        _CACHE[n] = (rows, tags, out)
    return _CACHE[n]


def inner_lo(n, R, bl, bh, th, Ds):
    """lower bound of  min { sum_i y_i^r : y_1>=..>=y_n>=beta, sum_{i>h} y_i >= D_h-beta-tau }
    over the whole box, via the RELAXED polyhedron and exact-rational vertex systems."""
    rows, tags, syss = systems(n)
    b = []
    for k, h in tags:
        if k == "ord":
            b.append(iv.mpf(0))
        elif k == "beta":
            b.append(bl)
        else:
            b.append(Ds[h] - bh - th)
    best = None
    for S, inv in syss:
        y = []
        for i in range(n):
            acc = iv.mpf(0)
            for j in range(n):
                cij = inv[i][j]
                if cij:
                    acc = acc + (iv.mpf(cij.numerator) / iv.mpf(cij.denominator)) * b[S[j]]
            y.append(acc)
        bad = False
        for row, bb in zip(rows, b):
            acc = iv.mpf(0)
            for j in range(n):
                if row[j]:
                    acc = acc + row[j] * y[j]
            if acc.b < bb.a:
                bad = True
                break
        if bad:
            continue
        tot = iv.mpf(0)
        for yi in y:
            lo = yi.a if yi.a > 0 else mp.mpf(0)
            tot = tot + ipow(EX(lo), R)
        if best is None or tot.a < best:
            best = tot.a
    return iv.mpf(0) if best is None else EX(best)


# ---------------------------------------------------------------- G's lower bound


def staircase_lo(R, tl, bh):
    """Lemma T: min of sum x^r over {0<=x<=beta, sum x = tau} = k beta^r + (tau-k beta)^r,
    k = floor(tau/beta).  Decreasing in beta and increasing in tau (both proved by the
    variational characterisation: enlarging beta enlarges the feasible set; lowering one
    atom lowers the objective).  Evaluated rigorously at (tau, beta) = (tl, bh)."""
    if bh.a <= 0 or tl.b <= 0:
        return iv.mpf(0)
    ratio = tl / bh
    k = int(mp.floor(ratio.a))          # k <= floor(tau/beta); under-estimating k is safe
    if k < 0:
        k = 0
    rem = tl - k * bh
    while k > 0 and rem.a < 0:
        k -= 1
        rem = tl - k * bh
    lo = rem.a if rem.a > 0 else mp.mpf(0)
    return EX((k * ipow(bh, R) + ipow(EX(lo), R)).a)


def G_lo(n, R, bl, bh, tl, th, Ds, tgt, staircase=True):
    t1 = ipow(tl, R)
    t2 = tl * ipow_pos(bh, R - 1) if bh.a > 0 else iv.mpf(0)
    tail = max(t1.a, t2.a)
    if staircase:
        tail = max(tail, staircase_lo(R, tl, bh).a)
    om = iv.mpf(1) - (2 ** n) * th
    drop = (tgt * ipow(EX(om.a), R)).a if om.a > 0 else mp.mpf(0)
    peel = (ipow(bl, R) + inner_lo(n, R, bl, bh, th, Ds)).a
    return EX(tail) + EX(max(peel, drop))


def certify_rect(n, R, b_hi, t_lo, t_hi, tgt, Ds, max_boxes=200_000, staircase=True,
                 use_skip=True):
    stack = [(iv.mpf(0), b_hi, t_lo, t_hi)]
    done = 0
    worst = None
    while stack:
        bl, bh, tl, th = stack.pop()
        done += 1
        if done > max_boxes:
            return False, done, worst
        if use_skip and (bh + th).b < Ds[n].a:
            continue
        g = G_lo(n, R, bl, bh, tl, th, Ds, tgt, staircase=staircase)
        if g.a >= tgt.b:
            if worst is None or g.a < worst:
                worst = g.a
            continue
        wb = float(bh.b) - float(bl.a)
        wt = float(th.b) - float(tl.a)
        if wb < 1e-12 and wt < 1e-12:
            return False, done, (float(bl.a), float(tl.a), float(g.a))
        if wb >= wt:
            mid = (bl + bh) / 2
            stack += [(bl, mid, tl, th), (mid, bh, tl, th)]
        else:
            mid = (tl + th) / 2
            stack += [(bl, bh, tl, mid), (bl, bh, mid, th)]
    return True, done, worst


# ---------------------------------------------------------------- analytic regions


def eps_chord(n, R, tgt):
    """largest dyadic t0 for which the chord test certifies phi >= tgt on [0,t0],
    for EVERY r in R.  Returns an mpf (exact endpoint), 0 if none."""
    def ok(t0):
        if t0 <= 0:
            return True
        z0 = iv.mpf(2 ** n) * EX(t0)
        if z0.b >= 1:
            return False
        K = (iv.mpf(1) - ipow(iv.mpf(1) - z0, R)) / z0
        lhs = ipow(EX(t0), iv.mpf(1) - R)
        rhs = iv.mpf(1) / (tgt * iv.mpf(2 ** n) * K)
        return lhs.b < rhs.a

    lo, hi = mp.mpf(0), mp.mpf(2) ** (-n)
    if ok(hi):
        return hi
    for _ in range(140):
        mid = (lo + hi) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def tau_max(R, tgt):
    """smallest certified t with t^r >= tgt for every r in R  (so Tail >= tgt there)."""
    t = iv.exp(iv.log(tgt) / R).b
    for _ in range(200):
        if ipow(EX(t), R).a >= tgt.b:
            return t
        t = t * mp.mpf("1.0000001")
    raise RuntimeError("tau_max not certified")


def C_const_certified(n, R, nsteps=200_000):
    """1-D interval branch-and-bound lower bound for
       C(n,r) = min_{s>0} [ max(1, s^{r-1}) + (n+1) s^r ].
    Independent re-derivation of (P-b)'s constant (no calculus)."""
    # s >= 1 branch: value >= 1 + (n+1) >= n+2
    best_hi = iv.mpf(n + 2)
    # s in (0,1]: f(s) = s^{r-1} + (n+1) s^r ; for s <= s_small, f >= s_small^{r-1}
    # choose s_small so that s_small^{r-1} >= n+2
    s_small = iv.exp(iv.log(iv.mpf(n + 2)) / (R - 1)).b     # (n+2)^{1/(r-1)}
    lo = None
    stack = [(EX(s_small), iv.mpf(1))]
    done = 0
    while stack:
        a, b = stack.pop()
        done += 1
        if done > nsteps:
            break
        box = iv.mpf([a.a, b.b])
        f = ipow_pos(box, R - 1) + (n + 1) * ipow(box, R)
        if lo is not None and f.a >= lo:
            continue
        if float(b.b) - float(a.a) < 1e-12:
            lo = f.a if lo is None else min(lo, f.a)
            continue
        m = (a + b) / 2
        stack += [(a, m), (m, b)]
    # refine: the true min is >= min over all leaves; recompute by a clean sweep
    N = 40000
    vals = []
    s0, s1 = mp.mpf(s_small), mp.mpf(1)
    for i in range(N):
        a = s0 + (s1 - s0) * i / N
        b = s0 + (s1 - s0) * (i + 1) / N
        box = iv.mpf([a, b])
        f = ipow_pos(box, R - 1) + (n + 1) * ipow(box, R)
        vals.append(f.a)
    return min(min(vals), best_hi.a)


# ----------------------------------------------------------------


def run(n, qnum, qden=1, staircase=True, use_skip=True, max_boxes=200_000, verbose=True):
    """r = qnum/(qden*(n+1)) as an exact rational, carried as a tight interval."""
    fr = Fraction(qnum, qden * (n + 1))
    R = iv.mpf(fr.numerator) / iv.mpf(fr.denominator)
    tgt = ipow(iv.mpf(n + 1), iv.mpf(1) - R)
    Ds = [D_h(n, h) for h in range(n + 1)]
    # P1B's "sum d_i <= n" variant must give the same constants
    Ds_le = [D_h(n, h, le=True) for h in range(n + 1)]
    same = all(abs(float(a.a) - float(b.a)) < 1e-25 for a, b in zip(Ds, Ds_le))
    b_hi = iv.mpf(1) / (n + 1)
    eps = eps_chord(n, R, tgt)
    tmx = tau_max(R, tgt)
    if verbose:
        print(f"\n=== n={n}  nu={n+1}  q={Fraction(qnum,qden)}  r={fr} = {float(fr):.15f} ===")
        print(f"  r enclosure width      {float(R.b-R.a):.3e}")
        print(f"  target (n+1)^(1-r)  in [{mp.nstr(tgt.a,20)}, {mp.nstr(tgt.b,20)}]")
        print(f"  D_h(n) (sum=n)       {[mp.nstr(d.a,10) for d in Ds]}")
        print(f"  D_h(n) (sum<=n)      identical: {same}")
        print(f"  (A) beta >= 1/(n+1) = {float(b_hi.a):.12f}   PROVED (exact)")
        print(f"  (B) tau  >= tau_max = {mp.nstr(tmx,15)}      PROVED (Tail >= tau^r)")
        print(f"  (C) tau  <= eps     = {mp.nstr(eps,15)}      PROVED (chord)")
    ok, boxes, worst = certify_rect(n, R, b_hi, EX(eps), EX(tmx), tgt, Ds,
                                    max_boxes=max_boxes, staircase=staircase,
                                    use_skip=use_skip)
    if verbose:
        if ok:
            print(f"  (D) rectangle [0,{float(b_hi.a):.6f}] x [{float(eps):.6e},{float(tmx):.6f}]:"
                  f"  CERTIFIED in {boxes} boxes")
            print(f"  ==> min G >= tgt on the whole quadrant. P1 holds for EVERY N at r={fr}.")
        else:
            print(f"  (D) FAILED after {boxes} boxes; worst {worst}")
    return ok, boxes, dict(eps=float(eps), tau_max=float(tmx), Dh_le_same=bool(same))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--max-boxes", type=int, default=200_000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rec = {}

    for n in (2, 3):
        ok, boxes, info = run(n, 2, 1, staircase=True, use_skip=True, max_boxes=a.max_boxes)
        rec[f"main_n{n}_q2"] = dict(certified=bool(ok), boxes=boxes, **info)
        # is the D_n(n) skip load bearing?
        ok2, b2, _ = run(n, 2, 1, staircase=True, use_skip=False,
                         max_boxes=a.max_boxes, verbose=False)
        print(f"  [no D_n(n) skip]      {'CERTIFIED' if ok2 else 'FAILS'} ({b2} boxes)")
        rec[f"main_n{n}_q2"]["certified_without_Dn_skip"] = bool(ok2)
        # is Lemma T's staircase load bearing?
        ok3, b3, _ = run(n, 2, 1, staircase=False, use_skip=True,
                         max_boxes=a.max_boxes, verbose=False)
        print(f"  [no Lemma T staircase] {'CERTIFIED' if ok3 else 'FAILS'} ({b3} boxes)")
        rec[f"main_n{n}_q2"]["certified_without_staircase"] = bool(ok3)
        # independent C(n,r)
        fr = Fraction(2, n + 1)
        R = iv.mpf(fr.numerator) / iv.mpf(fr.denominator)
        C = C_const_certified(n, R)
        print(f"  [independent C(n,r) >= {mp.nstr(C,15)}]  (p1a_cert claims "
              f"{'2.725680889248' if n==2 else '4'})")
        rec[f"main_n{n}_q2"]["C_lower_independent"] = float(C)

    if a.sweep:
        for n in (2, 3):
            print(f"\n  certified r-sweep at n = {n} (exact rational r = num/den):")
            row = []
            for num, den in [(7, 20), (2, 5), (1, 2), (3, 5), (2, 3), (7, 10), (3, 4),
                             (39, 50), (4, 5), (41, 50), (21, 25), (43, 50), (22, 25),
                             (9, 10), (19, 20)]:
                fr = Fraction(num, den)
                R = iv.mpf(num) / iv.mpf(den)
                tgt = ipow(iv.mpf(n + 1), iv.mpf(1) - R)
                Ds = [D_h(n, h) for h in range(n + 1)]
                eps = eps_chord(n, R, tgt)
                good = False
                if eps > 0:
                    tmx = tau_max(R, tgt)
                    good, _, _ = certify_rect(n, R, iv.mpf(1) / (n + 1), EX(eps), EX(tmx),
                                              tgt, Ds, max_boxes=30_000, staircase=True)
                print(f"    r = {float(fr):.4f} = {fr}   (q = {float(fr)*(n+1):.4f})   "
                      f"{'PROVED' if good else 'fails'}")
                row.append(dict(r=float(fr), q=float(fr) * (n + 1), certified=bool(good)))
            rec[f"sweep_n{n}"] = row

    with open(os.path.join(OUT, "p1a_recert.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"\nwrote {os.path.join(OUT,'p1a_recert.json')}")


if __name__ == "__main__":
    main()
